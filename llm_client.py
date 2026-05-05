"""Minimal LLM client wrapper supporting Ollama (cloud/local) and Google Generative AI.

The `generate(prompt, model=...)` function will prefer Ollama when `OLLAMA_HOST` is set
or `USE_OLLAMA=1` is present in the environment. Otherwise it falls back to Google
Generative AI (the existing `google.generativeai` usage).

This keeps embedding calls working with the current Gemini embedding setup while
allowing text generation to run on Ollama's `gpt-oss:20b-cloud` model.
"""

import os
import json
import requests


def _ollama_generate(prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    api_key = os.getenv("OLLAMA_API_KEY")
    model = model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gpt-oss:20b-cloud"

    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError:
        return resp.text

    # Common response shapes returned by various Ollama deployments
    if isinstance(data, dict):
        if "text" in data:
            return data["text"]
        if "output" in data:
            return data["output"]
        if "result" in data:
            return data["result"]
        if "choices" in data and len(data["choices"]) > 0:
            c = data["choices"][0]
            if isinstance(c, dict):
                if "message" in c and isinstance(c["message"], dict) and "content" in c["message"]:
                    return c["message"]["content"]
                if "content" in c:
                    return c["content"]

    return json.dumps(data)


def _genai_generate(prompt: str, model: str | None = None) -> str:
    import google.generativeai as genai

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not key:
        raise RuntimeError("No API key found for Google Generative AI (set GEMINI_API_KEY or GOOGLE_API_KEY)")

    genai.configure(api_key=key)
    model_name = model or os.getenv("GEMINI_MODEL")
    model_obj = genai.GenerativeModel(model_name)
    return model_obj.generate_content(prompt).text


def generate(prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    """Generate text using Ollama if configured, otherwise use Google Generative AI.

    - Set `OLLAMA_HOST` (e.g. https://cloud.ollama.com or http://localhost:11434) to use Ollama.
    - Or set `USE_OLLAMA=1` to force Ollama usage.
    - Otherwise the function falls back to Google GenAI and requires the usual API key env vars.
    """
    if os.getenv("OLLAMA_HOST") or os.getenv("USE_OLLAMA") == "1":
        return _ollama_generate(prompt, model=model, temperature=temperature)

    return _genai_generate(prompt, model=model)
