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
        return {"count": 0, "mean": None, "sem": None, "std": None}
    mean = stats["sum"] / count
    variance = max(0.0, (stats["sum_sq"] - count * mean * mean) / (count - 1)) if count > 1 else 0.0
    return {"count": count, "mean": mean, "sem": math.sqrt(variance / count),
            "std": math.sqrt(variance) if count > 1 else None}


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

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICRL reward ablations</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18202b;background:#f8fafc}
h1{font-size:1.6rem} .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;margin:1rem 0}
select{width:100%;padding:.65rem;font-size:1rem}canvas{width:100%;height:330px;cursor:crosshair}table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:.55rem;border-bottom:1px solid #e2e8f0;vertical-align:top}td{word-break:break-word}
.muted{color:#64748b}.meta{display:flex;gap:2rem;flex-wrap:wrap}.scroll{max-height:420px;overflow:auto}
.step-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}
.step-grid canvas{height:240px}
.controls{display:flex;gap:1rem;flex-wrap:wrap;align-items:end}
.controls label{display:grid;gap:.3rem;flex:1;min-width:150px}
.run-list{max-height:260px;overflow:auto;margin-top:1rem}.run-list label{display:flex;gap:.6rem;padding:.5rem;border-bottom:1px solid #e2e8f0;align-items:center}
.legend{display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem}.swatch{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:.4rem}
.run-filter{display:flex;align-items:center;gap:.4rem;margin-top:.75rem}
</style></head><body>
<h1>ICRL reward ablations</h1><p class="muted">Local runs · refreshes every 2 seconds · curves update after each evaluated puzzle</p>
<div class="card"><div class="controls">
<label>Task<select id="task"><option value="game24">Game24</option><option value="creative_writing">Creative Writing</option><option value="sciworld">ScienceWorld</option></select></label>
<label>Uncertainty band<select id="band"><option value="sem">Standard error</option><option value="std">Standard deviation</option><option value="none">None</option></select></label>
<label>Band multiplier<select id="bandScale"><option value="1">1×</option><option value="2">2×</option><option value="3">3×</option></select></label>
<label>Curve<select id="smoothing"><option value="raw">Raw means</option><option value="ema">EMA · all iterations</option><option value="odd">EMA · odd iterations (1, 3, 5, …)</option></select></label>
<label>EMA α<input id="alpha" type="range" min="0.05" max="1" step="0.05" value="0.3"><output id="alphaValue">0.30</output></label>
</div>
<p class="muted">Select runs to overlay within the selected task.</p>
<div id="run" class="run-list"></div>
<label class="run-filter"><input id="completedOnly" type="checkbox">Completed runs only (iteration 50 reached)</label>
<div id="legend" class="legend"></div><div id="status" class="meta"></div></div>
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
<div class="card"><h2>Latest evaluations and attempts</h2><label>Inspect run<select id="eventRun"></select></label><div class="scroll"><table><thead><tr>
<th>Time</th><th>Sample</th><th>Round</th><th>Step</th><th>Numeric reward</th><th>Shown to model</th><th>Event</th>
</tr></thead><tbody id="events"></tbody></table></div></div>
<script>
const picker=document.getElementById('run'), statusBox=document.getElementById('status'), tbody=document.getElementById('events');
const completedOnly=document.getElementById('completedOnly'),taskPicker=document.getElementById('task');
const band=document.getElementById('band'),bandScale=document.getElementById('bandScale');
const smoothing=document.getElementById('smoothing'),alpha=document.getElementById('alpha');
const eventPicker=document.getElementById('eventRun');
const palette=['#2563eb','#dc2626','#059669','#7c3aed','#d97706','#0891b2','#db2777','#475569'];
let runs=[],selected=new Set(),initialized=false,metrics=new Map(),eventRun='',offset=0,records=[],refreshing=false;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function runLabel(r){return `${r.model_alias} / ${r.feedback} / ${r.reward_polarity} · ${new Date(r.started_at).toLocaleString()}`;}
function selectedRuns(){return runs.filter(r=>selected.has(r.id));}
function colorFor(r){return palette[runs.findIndex(item=>item.id===r.id)%palette.length];}
function syncSelection(){
  const chosen=selectedRuns();
  document.getElementById('legend').innerHTML=chosen.map(r=>`<span><i class="swatch" style="background:${colorFor(r)}"></i>${esc(runLabel(r))}</span>`).join('');
  statusBox.innerHTML=chosen.map(r=>`<p>${esc(r.model)} · ${esc(r.reward_polarity)} · ${esc(r.status)}</p>`).join('');
  if(!runs.length)statusBox.textContent=completedOnly.checked?'No runs have completed iteration 50 yet.':'No runs yet for this task.';
  const previous=eventRun;
  eventPicker.innerHTML=chosen.map(r=>`<option value="${esc(r.id)}">${esc(runLabel(r))}</option>`).join('');
  eventRun=selected.has(previous)?previous:(chosen[0]?.id??'');
  eventPicker.value=eventRun;eventPicker.disabled=!chosen.length;
  if(previous!==eventRun){offset=0;records=[];tbody.innerHTML='';}
}
async function refreshRuns(){
  const res=await fetch('/api/runs');
  if(!res.ok)throw new Error('Run list request failed');
  const allRuns=await res.json();
  runs=allRuns.filter(r=>r.task===taskPicker.value&&(!completedOnly.checked||r.reached_iteration_50));
  selected=new Set([...selected].filter(id=>runs.some(r=>r.id===id)));
  if(!initialized&&runs.length){selected.add(runs[0].id);initialized=true;}
  const markup=runs.map(r=>`<label><input type="checkbox" value="${esc(r.id)}"${selected.has(r.id)?' checked':''}><span>${esc(r.task)} / ${esc(runLabel(r))} · ${esc(r.status)}</span></label>`).join('');
  if(picker.innerHTML!==markup)picker.innerHTML=markup;
  syncSelection();
}
async function refreshEvents(){
  const id=eventRun,requestOffset=offset;
  if(!id)return;
  const res=await fetch(`/api/events?run=${encodeURIComponent(id)}&offset=${offset}`);
  if(!res.ok)return;
  const payload=await res.json();
  if(id!==eventRun||offset!==requestOffset)return;
  if(payload.offset<offset)records=[];
  offset=payload.offset;records.push(...payload.events);
  if(records.length>20000)records=records.slice(-20000);
  const rows=records.filter(x=>x.event==='eval_step'||x.event==='attempt_end').slice(-60).reverse();
  tbody.innerHTML=rows.map(x=>`<tr><td>${esc(x.time?.slice(11,19))}</td><td>${esc(x.sample_id)}</td><td>${esc(x.round)}</td><td>${esc(x.step??'—')}</td><td>${esc(x.reward)}</td><td>${esc(x.display_reward)}</td><td>${esc(x.event)}${x.status?' · '+esc(x.status):''}</td></tr>`).join('');
}
function preparePoints(points){
  const alphaValue=Number(alpha.value);
  let mean=null;
  return points.filter(p=>Number.isFinite(p.mean)&&(smoothing.value!=='odd'||p.round%2===1)).map(p=>{
    const width=band.value==='none'?0:(p[band.value]??0)*Number(bandScale.value);
    if(smoothing.value==='raw')return {...p,width};
    mean=mean===null?p.mean:alphaValue*p.mean+(1-alphaValue)*mean;
    // Keep the empirical across-puzzle spread visible around the smoothed mean.
    return {...p,mean,width};
  });
}
const chartStates=new WeakMap();
function formatValue(value){return Number(value).toFixed(3).replace(/\.?0+$/,'');}
function drawSeries(state){
  const {canvas,series,label,fixedMax}=state;
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
  ctx.font='14px system-ui';ctx.fillStyle='#334155';ctx.fillText(label,58,19);
  const valid=series.flatMap(s=>s.points);
  if(!valid.length){state.layout=null;ctx.fillText('Select runs with evaluated puzzles.',58,52);return;}
  const left=58,right=w-20,top=31,bottom=h-42;
  const xmin=Math.min(...valid.map(p=>p.round)),xmax=Math.max(xmin+1,...valid.map(p=>p.round));
  const ymin=fixedMax===undefined?Math.min(0,...valid.map(p=>p.mean-p.width)):0;
  const ymax=fixedMax??Math.max(1,...valid.map(p=>p.mean+p.width))*1.08;
  const x=r=>left+(r-xmin)/(xmax-xmin)*(right-left);
  const y=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top);
  state.layout={valid,left,right,top,bottom,x,y};
  ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const value=ymin+(ymax-ymin)*i/4,py=y(value);ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();
    ctx.fillStyle='#64748b';ctx.fillText(value.toFixed(ymax<=1?2:1),4,py+4);
  }
  ctx.fillText(String(xmin),left,bottom+19);ctx.fillText(String(xmax),right-20,bottom+19);
  ctx.fillText('Trial / round',Math.max(left,(left+right)/2-40),h-5);
  ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();
  for(const {points,color} of series){
    if(!points.length)continue;
    if(points.length>1&&band.value!=='none'){
      ctx.fillStyle=color;ctx.globalAlpha=.12;ctx.beginPath();
      points.forEach((p,i)=>i?ctx.lineTo(x(p.round),y(p.mean+p.width)):ctx.moveTo(x(p.round),y(p.mean+p.width)));
      [...points].reverse().forEach(p=>ctx.lineTo(x(p.round),y(p.mean-p.width)));ctx.closePath();ctx.fill();ctx.globalAlpha=1;
    }
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
    points.forEach((p,i)=>i?ctx.lineTo(x(p.round),y(p.mean)):ctx.moveTo(x(p.round),y(p.mean)));ctx.stroke();
    for(const p of points){
      ctx.beginPath();ctx.arc(x(p.round),y(p.mean),3,0,Math.PI*2);
      ctx.fillStyle=p.complete?color:'#fff';ctx.fill();ctx.strokeStyle=color;ctx.stroke();
    }
  }
  ctx.restore();
  const hovered=series.map(s=>({...s,point:s.points.find(p=>p.round===state.hoverRound)})).filter(s=>s.point);
  if(!hovered.length)return;
  const hx=x(state.hoverRound);
  ctx.save();ctx.strokeStyle='#475569';ctx.lineWidth=1;ctx.setLineDash([5,4]);
  ctx.beginPath();ctx.moveTo(hx,top);ctx.lineTo(hx,bottom);ctx.stroke();ctx.setLineDash([]);
  ctx.font='12px system-ui';
  const tick=`Iter ${state.hoverRound}`,tw=ctx.measureText(tick).width+12;
  const tx=Math.max(left,Math.min(right-tw,hx-tw/2));
  ctx.fillStyle='#fff';ctx.fillRect(tx,bottom+5,tw,19);
  ctx.fillStyle='#0f172a';ctx.fillText(tick,tx+6,bottom+19);
  const lines=hovered.map(s=>({color:s.color,text:`${s.name} · y = ${formatValue(s.point.mean)} · n = ${s.point.count}`}));
  const boxWidth=Math.min(right-left,Math.max(...lines.map(l=>ctx.measureText(l.text).width))+16);
  const boxX=Math.max(left,Math.min(right-boxWidth,hx+10)),boxY=top+5;
  ctx.fillStyle='rgba(255,255,255,.95)';ctx.fillRect(boxX,boxY,boxWidth,lines.length*19+8);
  lines.forEach((line,i)=>{ctx.fillStyle=line.color;ctx.fillText(line.text,boxX+8,boxY+18+i*19,boxWidth-16);});
  ctx.restore();
}
function plotSeries(canvas,series,label,fixedMax){
  const bounds=canvas.getBoundingClientRect(),width=Math.round(bounds.width),height=Math.round(bounds.height);
  if(width>0&&height>0&&(canvas.width!==width||canvas.height!==height)){canvas.width=width;canvas.height=height;}
  let state=chartStates.get(canvas);
  if(!state){
    state={canvas,hoverRound:null,layout:null};chartStates.set(canvas,state);
    canvas.addEventListener('mousemove',event=>{
      const plot=state.layout;if(!plot)return;
      const rect=canvas.getBoundingClientRect();
      const mx=(event.clientX-rect.left)*canvas.width/rect.width,my=(event.clientY-rect.top)*canvas.height/rect.height;
      if(mx<plot.left||mx>plot.right||my<plot.top||my>plot.bottom){
        if(state.hoverRound!==null){state.hoverRound=null;drawSeries(state);}return;
      }
      const nearest=plot.valid.reduce((best,p)=>Math.abs(plot.x(p.round)-mx)<Math.abs(plot.x(best.round)-mx)?p:best);
      if(state.hoverRound!==nearest.round){state.hoverRound=nearest.round;drawSeries(state);}
    });
    canvas.addEventListener('mouseleave',()=>{state.hoverRound=null;drawSeries(state);});
  }
  Object.assign(state,{series,label,fixedMax});drawSeries(state);
}
function renderCharts(){
  const game24=taskPicker.value==='game24',chosen=selectedRuns();
  document.getElementById('chartTitle').textContent=game24?'Game24 success rate by trial':'Mean numeric reward by trial';
  document.getElementById('game24RewardCard').style.display=game24?'block':'none';
  document.getElementById('stepCard').style.display=game24?'block':'none';
  function seriesFor(metric){
    return chosen.map(r=>({
      name:runLabel(r),color:colorFor(r),
      points:preparePoints((metrics.get(r.id)?.rounds??[]).map(p=>({round:p.round,complete:p.complete,...metric(p)}))),
    }));
  }
  const bandNote=band.value==='none'?'Bands hidden.':`Bands show ±${bandScale.value} ${band.value==='std'?'standard deviation':'standard error'} across puzzles; clipped at score limits.`;
  const emaNote=smoothing.value==='raw'?'':` EMA α=${Number(alpha.value).toFixed(2)}${smoothing.value==='odd'?' uses iterations 1, 3, 5, … only':''}. Bands retain the raw per-iteration spread.`;
  document.getElementById('chartNote').textContent=`Comparing ${chosen.length} run(s). Filled dots are complete trials; open dots are partial. ${bandNote}${emaNote}`;
  plotSeries(document.getElementById('chart'),seriesFor(p=>game24?p.success:p.reward),
    game24?'Mean success rate (0-1)':'Mean numeric reward',game24?1:undefined);
  if(game24){
    plotSeries(document.getElementById('game24RewardChart'),seriesFor(p=>p.reward),'Mean total judge reward (0-19)',19);
    for(let i=0;i<4;i++)plotSeries(document.getElementById(`step${i+1}`),seriesFor(p=>p.steps[i]),
      i<3?`Step ${i+1} judge score (0-3)`:'Final answer judge score (0-10)',i<3?3:10);
  }
}
async function refreshMetrics(){
  const results=await Promise.all(selectedRuns().map(async r=>{
    const res=await fetch(`/api/metrics?run=${encodeURIComponent(r.id)}`);
    if(!res.ok)throw new Error('Metrics request failed');
    return [r.id,await res.json()];
  }));
  for(const [id,data] of results)if(selected.has(id))metrics.set(id,data);
  renderCharts();
}
picker.addEventListener('change',event=>{
  const input=event.target;if(input.type!=='checkbox')return;
  if(input.checked)selected.add(input.value);else selected.delete(input.value);
  syncSelection();renderCharts();tick();
});
eventPicker.addEventListener('change',()=>{eventRun=eventPicker.value;offset=0;records=[];tbody.innerHTML='';refreshEvents().catch(e=>{statusBox.textContent=e.message;});});
for(const control of [band,bandScale,smoothing,alpha])control.addEventListener('input',()=>{
  document.getElementById('alphaValue').textContent=Number(alpha.value).toFixed(2);
  renderCharts();
});
taskPicker.addEventListener('change',()=>{
  selected.clear();initialized=false;metrics.clear();eventRun='';offset=0;records=[];tbody.innerHTML='';
  renderCharts();tick();
});
completedOnly.addEventListener('change',()=>{initialized=false;tick();});
window.addEventListener('resize',renderCharts);
async function tick(){
  if(refreshing)return;
  refreshing=true;
  try{await refreshRuns();await Promise.all([refreshEvents(),refreshMetrics()]);}
  catch(e){statusBox.textContent='Dashboard read failed: '+e.message;}
  finally{refreshing=false;}
}
tick();setInterval(tick,2000);
</script></body></html>"""


def _run_dir(run_id: str) -> Path:
    path = (OUTPUT_ROOT / run_id).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()) or not (path / "config.json").is_file():
        abort(404)
    return path


def _last_event(path: Path, kind: str | None = None) -> dict | None:
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
            tail = lines.pop(0) if position else b""
            for line in reversed(lines):
                if line:
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if kind is None or event.get("event") == kind:
                        return event
    return None


def _reached_iteration_50(run_dir: Path) -> bool:
    summary = run_dir / "summary.json"
    if summary.is_file():
        # Rounds are zero-based: a completed iteration 50 means 51 rounds.
        return json.loads(summary.read_text(encoding="utf-8")).get("rounds_completed", 0) > 50
    event = _last_event(run_dir / "events.jsonl", "round_end")
    return event is not None and event.get("round", -1) >= 50


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
        if (_last_event(event_path) or {}).get("event") == "run_error":
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
                "feedback": config["feedback"], "reward_polarity": config.get("reward_polarity", "normal"),
                "started_at": _started_at(run_dir, config_file),
                "status": _run_status(run_dir),
                "finished": (run_dir / "summary.json").exists(),
                "reached_iteration_50": _reached_iteration_50(run_dir),
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
