import os
import json
import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)

# Groq/OpenAI-compatible SDK support
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Gemini support via Google GenAI
try:
    from google import genai
except Exception:
    genai = None

import requests

LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class LLMProvider:
    def generate(self, prompt: str, *args, **kwargs) -> str:
        raise NotImplementedError("LLMProvider subclasses must implement generate()")


class GeminiProvider(LLMProvider):
    def __init__(self):
        self.model = GEMINI_MODEL
        self.api_key = GEMINI_API_KEY
        self.client = None
        if genai is not None and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                LOGGER.warning("Gemini client initialization failed: %s", exc)
                self.client = None

    def generate(self, prompt: str, *args, **kwargs) -> str:
        if not self.api_key and not self.client:
            raise RuntimeError("Gemini provider is not configured. Set GEMINI_API_KEY or install google-genai.")

        if self.client:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=kwargs.get("temperature", 0.2),
                    max_output_tokens=kwargs.get("max_output_tokens", 1024),
                    stop_sequences=kwargs.get("stop_sequences", ["\n\n"])
                )
            )
            text = getattr(response, "text", None) or str(response)
            return text.strip()

        url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "prompt": {"text": prompt},
            "temperature": kwargs.get("temperature", 0.2),
            "maxOutputTokens": kwargs.get("max_output_tokens", 1024),
            "stop_sequences": kwargs.get("stop_sequences", ["\n\n"])
        }
        r = requests.post(url, json=payload, timeout=(10, 120))
        r.raise_for_status()
        data = r.json()
        text = ""
        if isinstance(data, dict):
            if "candidates" in data and data["candidates"]:
                candidates = data["candidates"]
                if isinstance(candidates, list):
                    parts = []
                    for c in candidates:
                        if isinstance(c, dict):
                            parts.append(c.get("content") or c.get("output") or c.get("text", ""))
                        else:
                            parts.append(str(c))
                    text = "\n".join([p for p in parts if p])
            elif "output" in data:
                text = data.get("output")
            elif "response" in data and isinstance(data.get("response"), dict):
                text = data.get("response", {}).get("output", "") or json.dumps(data.get("response", {}))
            else:
                text = data.get("text") or json.dumps(data)
        else:
            text = str(data)
        return str(text).strip()


class GroqProvider(LLMProvider):
    def __init__(self):
        self.model = GROQ_MODEL
        self.api_key = GROQ_API_KEY
        self.base_url = GROQ_BASE_URL
        self.client = None
        if OpenAI is not None and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as exc:
                LOGGER.warning("Groq OpenAI client initialization failed: %s", exc)
                self.client = None

    def generate(self, prompt: str, *args, **kwargs) -> str:
        if not self.api_key:
            raise RuntimeError("Groq provider is not configured. Set GROQ_API_KEY.")
        if self.client is None:
            raise RuntimeError("OpenAI SDK is not available. Install openai.")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[
                {"role": "system", "content": kwargs.get("system_prompt", "You are an expert technical writer.")},
                {"role": "user", "content": prompt}
            ]
        )
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = getattr(choice, "message", None) or choice.get("message", {})
            if isinstance(message, dict):
                return str(message.get("content", "")).strip()
            return str(getattr(message, "content", "")).strip()
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return str(message.get("content", "")).strip()
        raise RuntimeError("Groq response did not contain a valid completion.")


class FallbackLLMProvider(LLMProvider):
    def __init__(self, providers):
        self.providers = providers

    def generate(self, prompt: str, **kwargs) -> str:
        last_error = None
        for provider in self.providers:
            try:
                return provider.generate(prompt, **kwargs)
            except Exception as exc:
                last_error = exc
                LOGGER.warning("LLM provider %s failed: %s", provider.__class__.__name__, exc)
                continue
        raise RuntimeError("All LLM providers failed.") from last_error


def create_llm_provider() -> LLMProvider:
    provider_name = LLM_PROVIDER_NAME
    if provider_name == "groq":
        return GroqProvider()
    if provider_name in {"gemini", "google", "genai"}:
        return GeminiProvider()
    if provider_name == "fallback":
        return FallbackLLMProvider([GeminiProvider(), GroqProvider()])
    raise ValueError(f"Unsupported LLM_PROVIDER '{provider_name}'. Use 'gemini', 'groq', or 'fallback'.")


def get_llm_provider() -> Optional[LLMProvider]:
    try:
        return create_llm_provider()
    except Exception as e:
        LOGGER.warning("Failed to initialize LLM provider: %s", e)
        return None
