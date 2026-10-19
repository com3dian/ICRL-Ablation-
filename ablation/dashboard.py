"""Local, read-only dashboard for append-only ablation runs."""

import argparse
import fcntl
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from flask import Flask, abort, jsonify, render_template_string, request

from .events import OUTPUT_ROOT

app = Flask(__name__)
_metric_cache = {}
_metric_lock = Lock()


def _score_stats():
    return {"count": 0, "sum": 0.0, "sum_sq": 0.0}


def _add_score(stats: dict, value: float):
    stats["count"] += 1
    stats["sum"] += value
    stats["sum_sq"] += value * value


def _mean_sem(stats: dict) -> dict:
    count = stats["count"]
    if not count:
        return {"count": 0, "mean": None, "sem": None}
    mean = stats["sum"] / count
    variance = max(0.0, (stats["sum_sq"] - count * mean * mean) / (count - 1)) if count > 1 else 0.0
    return {"count": count, "mean": mean, "sem": math.sqrt(variance / count)}


def _round_stats():
    return {"reward": _score_stats(), "success": _score_stats(),
            "steps": [_score_stats() for _ in range(4)]}


def _update_metrics(path: Path, state: dict):
    """Read only complete new JSONL records; retain compact aggregates, not prompts."""
    with path.open("rb") as stream:
        stream.seek(state["offset"])
        while True:
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                break
            state["offset"] = stream.tell()
            event = json.loads(line)
            round_idx = event.get("round")
            if not isinstance(round_idx, int) or round_idx < 0:
                continue
            kind = event.get("event")
            if kind not in ("eval_step", "attempt_end"):
                continue
            bucket = state["rounds"].get(round_idx)
            if bucket is None:
                bucket = state["rounds"][round_idx] = _round_stats()
            if kind == "eval_step":
                step = event.get("step")
                if state["task"] == "game24" and isinstance(step, int) and 1 <= step <= 4:
                    _add_score(bucket["steps"][step - 1], float(event["reward"]))
            else:
                _add_score(bucket["reward"], float(event["reward"]))
                if state["task"] == "game24":
                    # Figure 2 reports correctness, not the LLM judge's reward.
                    accuracy = event.get("accuracy_reward")
                    fmt = event.get("format_reward")
                    if accuracy is not None and fmt is not None:
                        _add_score(bucket["success"], float(abs(accuracy) < 1e-9 and fmt == 0))

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICRL reward ablations</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18202b;background:#f8fafc}
h1{font-size:1.6rem} .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;margin:1rem 0}
select{width:100%;padding:.65rem;font-size:1rem}canvas{width:100%;height:330px}table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:.55rem;border-bottom:1px solid #e2e8f0;vertical-align:top}td{word-break:break-word}
.muted{color:#64748b}.meta{display:flex;gap:2rem;flex-wrap:wrap}.scroll{max-height:420px;overflow:auto}
.step-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}
.step-grid canvas{height:240px}
</style></head><body>
<h1>ICRL reward ablations</h1><p class="muted">Local runs · refreshes every 2 seconds · curves update after each evaluated puzzle</p>
<div class="card"><label for="run">Run</label><select id="run"></select><div id="status" class="meta"></div></div>
<div class="card"><h2 id="chartTitle">Performance by trial</h2><p id="chartNote" class="muted"></p>
<canvas id="chart" width="1000" height="330"></canvas></div>
<div class="card" id="game24RewardCard" style="display:none"><h2>Game24 mean judge reward by trial</h2>
<p class="muted">Mean total of the four judge scores (0–19). This is separate from the success rate above.</p>
<canvas id="game24RewardChart" width="1000" height="330"></canvas></div>
<div class="card" id="stepCard" style="display:none"><h2>Game24 judge rewards by trial</h2>
<p class="muted">Four separate scores per puzzle: three intermediate steps (0-3), then the final answer (0-10). Each plot shows the mean across puzzles, not an extra arithmetic step.</p>
<div class="step-grid">
<canvas id="step1" width="480" height="240"></canvas><canvas id="step2" width="480" height="240"></canvas>
<canvas id="step3" width="480" height="240"></canvas><canvas id="step4" width="480" height="240"></canvas>
</div></div>
<div class="card"><h2>Latest evaluations and attempts</h2><div class="scroll"><table><thead><tr>
<th>Time</th><th>Sample</th><th>Round</th><th>Step</th><th>Numeric reward</th><th>Shown to model</th><th>Event</th>
</tr></thead><tbody id="events"></tbody></table></div></div>
<script>
const picker=document.getElementById('run'), statusBox=document.getElementById('status'), tbody=document.getElementById('events');
let offset=0, records=[], selected='', metrics=null;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function refreshRuns(){
  const res=await fetch('/api/runs'), runs=await res.json();
  const previous=selected;
  picker.innerHTML=runs.map(r=>`<option value="${esc(r.id)}">${esc(r.task)} / ${esc(r.model_alias)} / ${esc(r.feedback)} · ${esc(new Date(r.started_at).toLocaleString())} · ${esc(r.status)}</option>`).join('');
  if(runs.some(r=>r.id===previous)) picker.value=previous;
  else if(runs.length) picker.value=runs[0].id;
  if(picker.value!==selected){selected=picker.value;offset=0;records=[];metrics=null;}
  const current=runs.find(r=>r.id===selected);
  statusBox.innerHTML=current?`<p><b>Task:</b> ${esc(current.task)}</p><p><b>Model:</b> ${esc(current.model)}</p><p><b>Feedback:</b> ${esc(current.feedback)}</p><p><b>Status:</b> ${current.finished?'finished':'running/partial'}</p>`:'<p>No runs yet.</p>';
}
async function refreshEvents(){
  if(!selected)return;
  const res=await fetch(`/api/events?run=${encodeURIComponent(selected)}&offset=${offset}`);
  if(!res.ok)return;
  const payload=await res.json();offset=payload.offset;records.push(...payload.events);
  if(records.length>20000)records=records.slice(-20000);
  const rows=records.filter(x=>x.event==='eval_step'||x.event==='attempt_end').slice(-60).reverse();
  tbody.innerHTML=rows.map(x=>`<tr><td>${esc(x.time?.slice(11,19))}</td><td>${esc(x.sample_id)}</td><td>${esc(x.round)}</td><td>${esc(x.step??'—')}</td><td>${esc(x.reward)}</td><td>${esc(x.display_reward)}</td><td>${esc(x.event)}${x.status?' · '+esc(x.status):''}</td></tr>`).join('');
}
function plotSeries(canvas, points, label, color, fixedMax){
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
  ctx.font='14px system-ui';ctx.fillStyle='#334155';ctx.fillText(label,55,19);
  const valid=points.filter(p=>Number.isFinite(p.mean));
  if(!valid.length){ctx.fillText('Waiting for evaluations…',55,52);return;}
  const left=58,right=w-20,top=31,bottom=h-42;
  const xmin=valid[0].round,xmax=Math.max(xmin+1,valid.at(-1).round);
  const ymax=fixedMax??Math.max(1,...valid.map(p=>p.mean+(p.sem??0)))*1.08;
  const x=r=>left+(r-xmin)/(xmax-xmin)*(right-left);
  const y=v=>bottom-Math.max(0,Math.min(ymax,v))/ymax*(bottom-top);
  ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const value=ymax*i/4,py=y(value);ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();
    ctx.fillStyle='#64748b';ctx.fillText(value.toFixed(ymax<=1?2:1),4,py+4);
  }
  ctx.fillText(String(xmin),left,bottom+19);ctx.fillText(String(xmax),right-20,bottom+19);
  ctx.fillText('Trial / round',Math.max(left,(left+right)/2-40),h-5);
  if(valid.length>1){
    ctx.fillStyle=color;ctx.globalAlpha=.16;ctx.beginPath();
    valid.forEach((p,i)=>i?ctx.lineTo(x(p.round),y(p.mean+(p.sem??0))):ctx.moveTo(x(p.round),y(p.mean+(p.sem??0))));
    [...valid].reverse().forEach(p=>ctx.lineTo(x(p.round),y(p.mean-(p.sem??0))));ctx.closePath();ctx.fill();ctx.globalAlpha=1;
  }
  ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.beginPath();
  valid.forEach((p,i)=>i?ctx.lineTo(x(p.round),y(p.mean)):ctx.moveTo(x(p.round),y(p.mean)));ctx.stroke();
  for(const p of valid){
    ctx.beginPath();ctx.arc(x(p.round),y(p.mean),4,0,Math.PI*2);
    ctx.fillStyle=p.complete?color:'#fff';ctx.fill();ctx.strokeStyle=color;ctx.stroke();
  }
}
async function refreshMetrics(){
  if(!selected)return;
  const res=await fetch(`/api/metrics?run=${encodeURIComponent(selected)}`);
  if(!res.ok)throw new Error('Metrics request failed');
  metrics=await res.json();
  const game24=metrics.task==='game24',rounds=metrics.rounds;
  document.getElementById('chartTitle').textContent=game24?'Game24 success rate by trial':'Mean numeric reward by trial';
  document.getElementById('game24RewardCard').style.display=game24?'block':'none';
  document.getElementById('stepCard').style.display=game24?'block':'none';
  const latest=rounds.at(-1);
  document.getElementById('chartNote').textContent=latest?
    `Trial ${latest.round}: ${latest.reward.count}/${metrics.expected_samples} puzzles evaluated${latest.complete?' (complete)':' (partial; updating live)'}. ${rounds.length<2?'The first point becomes a curve when trial 1 begins.':''} Filled dots are complete trials; open dots are partial. Shading is ±1 standard error.`:
    'Waiting for evaluated puzzles. Each trial contains one attempt per puzzle.';
  const primary=rounds.map(r=>({round:r.round,complete:r.complete,...(game24?r.success:r.reward)}));
  plotSeries(document.getElementById('chart'),primary,game24?'Mean success rate (0-1)':'Mean numeric reward','#2563eb',game24?1:null);
  if(game24){
    const rewardPoints=rounds.map(r=>({round:r.round,complete:r.complete,...r.reward}));
    plotSeries(document.getElementById('game24RewardChart'),rewardPoints,
      'Mean total judge reward (0-19)','#0f766e',19);
    const colors=['#7c3aed','#059669','#d97706','#dc2626'];
    for(let index=0;index<4;index++){
      const points=rounds.map(r=>({round:r.round,complete:r.complete,...r.steps[index]}));
      plotSeries(document.getElementById(`step${index+1}`),points,
        index<3?`Step ${index+1} judge score (0-3)`:'Final answer judge score (0-10)',
        colors[index],index<3?3:10);
    }
  }
}
picker.addEventListener('change',()=>{selected=picker.value;offset=0;records=[];metrics=null;refreshEvents();refreshMetrics();});
async function tick(){try{await refreshRuns();await Promise.all([refreshEvents(),refreshMetrics()]);}catch(e){statusBox.textContent='Dashboard read failed: '+e.message;}}
tick();setInterval(tick,2000);
</script></body></html>"""


def _run_dir(run_id: str) -> Path:
    path = (OUTPUT_ROOT / run_id).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()) or not (path / "config.json").is_file():
        abort(404)
    return path


def _last_event_kind(path: Path) -> str | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        tail = b""
        while position:
            chunk_size = min(position, 4096)
            position -= chunk_size
            stream.seek(position)
            tail = stream.read(chunk_size) + tail
            lines = tail.split(b"\n")
            # The first line may be cut off when we have not reached byte zero.
            for line in reversed(lines[1:] if position else lines):
                if line:
                    try:
                        return json.loads(line).get("event")
                    except (ValueError, UnicodeDecodeError):
                        continue
    return None


def _run_status(run_dir: Path) -> str:
    if (run_dir / "summary.json").is_file():
        return "succeeded"
    event_path = run_dir / "events.jsonl"
    if event_path.is_file():
        with event_path.open("rb") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return "running"
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        if _last_event_kind(event_path) == "run_error":
            return "failed"
    return "incomplete"


def _started_at(run_dir: Path, config_file: Path) -> str:
    try:
        stamp = run_dir.name.split("-", 1)[0]
        return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return datetime.fromtimestamp(config_file.stat().st_mtime, timezone.utc).isoformat()


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/runs")
def runs():
    items = []
    if OUTPUT_ROOT.exists():
        for config_file in OUTPUT_ROOT.rglob("config.json"):
            config = json.loads(config_file.read_text(encoding="utf-8"))
            run_dir = config_file.parent
            items.append({
                "id": str(run_dir.relative_to(OUTPUT_ROOT)),
                "task": config["task"], "model": config["model"],
                "model_alias": config.get("model_alias", config["model"]),
                "feedback": config["feedback"], "started_at": _started_at(run_dir, config_file),
                "status": _run_status(run_dir),
                "finished": (run_dir / "summary.json").exists(),
            })
    items.sort(key=lambda item: (item["started_at"], item["id"]), reverse=True)
    return jsonify(items)


@app.get("/api/events")
def events():
    path = _run_dir(request.args.get("run", "")) / "events.jsonl"
    try:
        offset = max(0, int(request.args.get("offset", "0")))
    except ValueError:
        abort(400)
    if not path.exists():
        return jsonify({"offset": 0, "events": []})
    size = path.stat().st_size
    if offset > size:
        offset = 0
    result = []
    with path.open("rb") as stream:
        stream.seek(offset)
        for _ in range(500):
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.seek(-len(line), 1)
                break
            result.append(json.loads(line))
        offset = stream.tell()
    return jsonify({"offset": offset, "events": result})


@app.get("/api/metrics")
def metrics():
    run_dir = _run_dir(request.args.get("run", ""))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    path = run_dir / "events.jsonl"
    expected_samples = config["samples"]
    if not path.exists():
        return jsonify({"task": config["task"], "expected_samples": expected_samples, "rounds": []})
    stat = path.stat()
    key = str(path)
    with _metric_lock:
        state = _metric_cache.get(key)
        if state is None or state["inode"] != stat.st_ino or stat.st_size < state["offset"]:
            state = {"inode": stat.st_ino, "offset": 0, "task": config["task"], "rounds": {}}
            _metric_cache[key] = state
        _update_metrics(path, state)
        rounds = []
        for round_idx, bucket in sorted(state["rounds"].items()):
            reward = _mean_sem(bucket["reward"])
            rounds.append({
                "round": round_idx,
                "complete": reward["count"] >= expected_samples,
                "reward": reward,
                "success": _mean_sem(bucket["success"]),
                "steps": [_mean_sem(score) for score in bucket["steps"]],
            })
    return jsonify({"task": config["task"], "expected_samples": expected_samples, "rounds": rounds})


def main():
    parser = argparse.ArgumentParser(description="View local ablation rewards as they arrive")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
