"""Wraps the configured LLM provider so every call is (1) looked up in the
exact-response cache first and (2) counted against the current KT's usage
tracker. Prompts, parameters and the returned text are passed through
unchanged; on a miss the real provider is called exactly as before.
"""
import sys
import time
from typing import Any, Optional

from llm import cache
from llm.usage import current_tracker

# Function names on the call stack -> pipeline stage. Lets the wrapper label
# each call without changing any call site's signature.
_CALL_SITES = {
    "_verify_classification_with_llm": "classification_check",
    "_try_llm_repair": "sentence_repair",
    "_polish_one": "section_polish",
    "_extract_structured_section": "structured_extraction",
    "_populate_fields_recursive": "field_fill",
}


def infer_call_site(max_depth: int = 12) -> str:
    frame = sys._getframe(2)
    depth = 0
    while frame is not None and depth < max_depth:
        site = _CALL_SITES.get(frame.f_code.co_name)
        if site:
            return site
        frame = frame.f_back
        depth += 1
    return "other"


def _provider_name(provider: Any) -> str:
    return type(provider).__name__.replace("Provider", "").lower() or "llm"


class CachedLLMProvider:
    def __init__(self, inner: Any):
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        # model, api_key, client, ... stay reachable as before.
        return getattr(self.inner, name)

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        from llm_provider import consume_call_stats, reset_call_stats, estimate_tokens

        call_site = infer_call_site()
        tracker = current_tracker()
        provider = _provider_name(self.inner)
        model = str(getattr(self.inner, "model", "") or "")
        params = {
            "args": list(args),
            "temperature": kwargs.get("temperature"),
            "max_output_tokens": kwargs.get("max_output_tokens", kwargs.get("max_tokens")),
            "stop_sequences": kwargs.get("stop_sequences"),
            "system_prompt": kwargs.get("system_prompt"),
        }
        key = cache.make_key(provider=provider, model=model, prompt=prompt, params=params)

        hit = cache.get(key)
        if hit is not None:
            if tracker is not None:
                tracker.record_cache_hit(call_site, input_tokens=hit["input_tokens"], output_tokens=hit["output_tokens"])
            return hit["response"]

        reset_call_stats()
        started = time.time()
        try:
            response = self.inner.generate(prompt, *args, **kwargs)
        except Exception:
            stats = consume_call_stats()
            if tracker is not None:
                tracker.record_call(
                    call_site,
                    input_tokens=stats.get("input_tokens") or estimate_tokens(params["system_prompt"], prompt),
                    output_tokens=stats.get("output_tokens") or 0,
                    estimated=stats.get("input_tokens") is None,
                    seconds=time.time() - started,
                    retries=stats.get("retries", 0),
                    failed=True,
                    provider=provider,
                    model=model,
                )
            raise

        stats = consume_call_stats()
        input_tokens = stats.get("input_tokens")
        output_tokens = stats.get("output_tokens")
        estimated = input_tokens is None or output_tokens is None
        if input_tokens is None:
            input_tokens = estimate_tokens(params["system_prompt"], prompt)
        if output_tokens is None:
            output_tokens = estimate_tokens(response if isinstance(response, str) else "")
        if tracker is not None:
            tracker.record_call(
                call_site,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated=estimated,
                seconds=time.time() - started,
                retries=stats.get("retries", 0),
                provider=provider,
                model=model,
            )
        cache.put(
            key,
            response=response,
            call_site=call_site,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return response
