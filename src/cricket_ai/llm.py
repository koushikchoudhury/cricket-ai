"""One switch for the language model used by RAG (module 4) and the agent (module 5).

Pick a provider with ``--llm`` on the command line or the ``CRICKET_AI_LLM`` environment
variable, and optionally a model with ``--model`` / ``CRICKET_AI_LLM_MODEL``:

=========== ========================== ==============================================
provider    default model              needs
=========== ========================== ==============================================
none        -                          nothing (retrieval only, no generated answers)
ollama      qwen2.5:3b                 Ollama running locally (fits a 4 GB GPU)
groq        openai/gpt-oss-120b        GROQ_API_KEY (free tier, no card), pip install langchain-groq
anthropic   claude-haiku-5-5           ANTHROPIC_API_KEY, pip install langchain-anthropic
openai      gpt-4o-mini                OPENAI_API_KEY, pip install langchain-openai
=========== ========================== ==============================================
"""

from __future__ import annotations

import importlib.util
import os

PROVIDERS = ("none", "ollama", "groq", "anthropic", "openai")
DEFAULT_MODELS = {
    "ollama": "qwen2.5:3b",
    # Groq's free tier retired Llama 3.3 70B on 2026-08-16; gpt-oss-120b is its replacement.
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
}
# Keep the local model in memory between questions: reloading it costs many seconds.
OLLAMA_KEEP_ALIVE = "30m"
# Answers here are a few sentences or one SQL query; a cap bounds the worst case.
MAX_OUTPUT_TOKENS = 512


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_models(timeout: float = 1.0) -> list[str]:
    """Names of the models installed in a running Ollama; [] if Ollama is not reachable."""
    import requests

    try:
        reply = requests.get(f"{ollama_host()}/api/tags", timeout=timeout)
        reply.raise_for_status()
        return sorted(m["name"] for m in reply.json().get("models", []))
    except Exception:  # noqa: BLE001 - not running, wrong port, bad reply: simply unavailable
        return []


def warm_up_ollama(model: str, timeout: float = 120.0) -> bool:
    """Load ``model`` into Ollama's memory now, so the first question does not wait for it.

    An empty prompt makes Ollama load the model and return without generating anything.
    """
    import requests

    try:
        reply = requests.post(
            f"{ollama_host()}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": OLLAMA_KEEP_ALIVE},
            timeout=timeout,
        )
        return reply.ok
    except Exception:  # noqa: BLE001 - warming up is best effort
        return False


def available_providers() -> dict[str, list[str]]:
    """Providers that can be used right now, each with the models to offer.

    * ollama    - Ollama is running and has at least one model pulled
    * anthropic - ANTHROPIC_API_KEY is set and langchain-anthropic is installed
    * openai    - OPENAI_API_KEY is set and langchain-openai is installed
    """
    found: dict[str, list[str]] = {"none": []}
    if importlib.util.find_spec("langchain_ollama"):
        models = ollama_models()
        if models:
            found["ollama"] = models
    for provider, key, package in (
        ("groq", "GROQ_API_KEY", "langchain_groq"),
        ("anthropic", "ANTHROPIC_API_KEY", "langchain_anthropic"),
        ("openai", "OPENAI_API_KEY", "langchain_openai"),
    ):
        if os.environ.get(key) and importlib.util.find_spec(package):
            found[provider] = [DEFAULT_MODELS[provider]]
    return found


def llm_settings(provider: str | None = None, model: str | None = None) -> tuple[str, str | None]:
    provider = (provider or os.environ.get("CRICKET_AI_LLM") or "none").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {provider!r}; choose from {PROVIDERS}")
    model = model or os.environ.get("CRICKET_AI_LLM_MODEL") or DEFAULT_MODELS.get(provider)
    return provider, model


def get_chat_model(provider: str | None = None, model: str | None = None, temperature: float = 0.0):
    """A LangChain chat model, or ``None`` for provider ``none``."""
    provider, model = llm_settings(provider, model)
    if provider == "none":
        return None
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=ollama_host(),
            keep_alive=OLLAMA_KEEP_ALIVE,
            num_predict=MAX_OUTPUT_TOKENS,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        extra = {}
        if model.startswith("openai/gpt-oss"):
            # A reasoning model: low effort keeps answers fast and within the free tier's
            # tokens-per-minute limit. Reasoning tokens count towards max_tokens.
            extra["reasoning_effort"] = "low"
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=2 * MAX_OUTPUT_TOKENS,
            max_retries=1,
            **extra,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, max_tokens=1024)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature)


def message_text(message) -> str:
    """Plain text of a chat message (some providers return a list of content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def call_stats(message, seconds: float) -> dict:
    """Where one model call spent its time.

    Ollama reports loading the model, reading the prompt and writing the reply separately
    (in nanoseconds); other providers only report token counts.
    """
    meta = getattr(message, "response_metadata", None) or {}
    usage = getattr(message, "usage_metadata", None) or {}

    def secs(key: str) -> float | None:
        value = meta.get(key)
        return round(value / 1e9, 2) if isinstance(value, (int, float)) else None

    return {
        "seconds": round(seconds, 2),
        "load_s": secs("load_duration"),
        "prompt_s": secs("prompt_eval_duration"),
        "output_s": secs("eval_duration"),
        "prompt_tokens": meta.get("prompt_eval_count") or usage.get("input_tokens"),
        "output_tokens": meta.get("eval_count") or usage.get("output_tokens"),
    }


def describe_timing(llm_calls: list[dict], tool_seconds: float = 0.0) -> str:
    """One line: 'model 6.1s in 2 call(s) (loading 0.0s, reading ...) · tools 0.02s'."""
    if not llm_calls:
        return ""

    def total(key: str) -> float | None:
        values = [c[key] for c in llm_calls if c.get(key) is not None]
        return round(sum(values), 2) if values else None

    model = total("seconds")
    parts = []
    if total("load_s") is not None:
        parts.append(f"loading {total('load_s')}s")
    if total("prompt_tokens") is not None:
        took = f" {total('prompt_s')}s" if total("prompt_s") is not None else ""
        parts.append(f"reading {int(total('prompt_tokens')):,} tokens{took}")
    if total("output_tokens") is not None:
        took = f" {total('output_s')}s" if total("output_s") is not None else ""
        parts.append(f"writing {int(total('output_tokens')):,} tokens{took}")
    detail = f" ({', '.join(parts)})" if parts else ""
    return f"model {model}s in {len(llm_calls)} call(s){detail} · tools {round(tool_seconds, 2)}s"


def is_rate_limited(exc: BaseException) -> bool:
    """True for 'too many requests' errors (HTTP 429) from any provider."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError":
            return True
        if "429" in str(exc) and "rate" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False
