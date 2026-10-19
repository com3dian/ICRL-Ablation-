"""OpenAI-compatible SURF and Cliproxy transports."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

ROOT = Path(__file__).resolve().parents[1]
SURF_BASE_URL = "https://willma.surf.nl/api/v0"
CLIPROXY_BASE_URL = "https://cliproxy.sub2api-best.com/v1"
SURF_MODELS = {
    "gemma4": "RedHatAI/gemma-4-31B-it-NVFP4",
    "qwen35b": "Qwen/Qwen3.6-35B-A3B-FP8",
    "oss120b": "openai/gpt-oss-120b",
}
CLIPROXY_MODELS = {"gpt6luna": "gpt-6-luna", "gpt56luna": "gpt-5.6-luna"}
MODELS = SURF_MODELS | CLIPROXY_MODELS


def generation_options(model: str) -> dict:
    """Return model-specific OpenAI-compatible request options."""
    if model == MODELS["qwen35b"]:
        # Qwen3.6 otherwise spends the output budget in its reasoning field.
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if model in CLIPROXY_MODELS.values():
        # Low-cost smoke tests should not spend their output budget on reasoning.
        return {"reasoning_effort": "none"}
    return {}


def judge_options(model: str) -> dict:
    """Keep the GPT-OSS judge's reasoning within a small scoring budget."""
    if model == MODELS["oss120b"]:
        return {"reasoning_effort": "low"}
    return {}


class EmptyResponseError(RuntimeError):
    """A provider returned no visible answer after its completion budget."""


class SurfClient:
    def __init__(self, provider: str = "surf"):
        load_dotenv(ROOT / ".env", override=False)
        if provider not in ("surf", "cliproxy"):
            raise ValueError(f"Unknown provider: {provider}")
        self.provider = provider
        env_name = "SURF_API_KEY" if provider == "surf" else "CLIPROXY_API_KEY"
        key = os.getenv(env_name)
        if not key:
            raise RuntimeError(f"{env_name} is missing from {ROOT / '.env'}")
        base_url = (os.getenv("SURF_API_BASE", SURF_BASE_URL) if provider == "surf"
                    else os.getenv("CLIPROXY_API_BASE", CLIPROXY_BASE_URL))
        options = {"default_headers": {"X-API-KEY": key}} if provider == "surf" else {}
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=120.0,
                             max_retries=0, **options)

    def generate(
        self, model: str, prompt: str | list[dict], *, temperature: float,
        max_tokens: int, reasoning_effort: str | None = None,
        retry_empty_budgets: tuple[int, ...] = (), on_response=None,
    ) -> str:
        delays = (0, 5, 15, 30)
        budgets = (max_tokens, *retry_empty_budgets)
        if any(budget < 1 for budget in budgets):
            raise ValueError("Completion budgets must be positive")
        for budget_index, budget in enumerate(budgets):
            for attempt, delay in enumerate(delays):
                if delay:
                    time.sleep(delay)
                try:
                    options = generation_options(model)
                    if reasoning_effort is not None:
                        options["reasoning_effort"] = reasoning_effort
                    limit_name = "max_completion_tokens" if getattr(self, "provider", "surf") == "cliproxy" else "max_tokens"
                    result = self.client.chat.completions.create(
                        model=model,
                        messages=prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
                        temperature=temperature,
                        **{limit_name: budget},
                        **options,
                    )
                    choice = result.choices[0]
                    content = choice.message.content
                    extra = choice.message.model_extra or {}
                    details = getattr(result.usage, "completion_tokens_details", None) if result.usage else None
                    report = {
                        "model": model, "budget": budget, "finish_reason": choice.finish_reason,
                        "completion_tokens": result.usage.completion_tokens if result.usage else None,
                        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                        "reasoning_present": bool(extra.get("reasoning") or extra.get("reasoning_content")),
                        "content_present": bool(content),
                    }
                    if on_response is not None:
                        on_response(report)
                    if content:
                        return content
                    if choice.finish_reason == "length" and budget_index < len(budgets) - 1:
                        break
                    raise EmptyResponseError(
                        f"Empty response from {model}; finish_reason={choice.finish_reason}, "
                        f"completion_tokens={report['completion_tokens']}, "
                        f"reasoning_present={report['reasoning_present']}, budget={budget}"
                    )
                except (APIConnectionError, APITimeoutError):
                    if attempt == len(delays) - 1:
                        raise
                except APIStatusError as exc:
                    if exc.status_code not in (408, 429, 500, 502, 503, 504) or attempt == len(delays) - 1:
                        raise
            else:
                raise AssertionError("Retry loop ended unexpectedly")
        raise AssertionError("Completion budgets ended unexpectedly")
