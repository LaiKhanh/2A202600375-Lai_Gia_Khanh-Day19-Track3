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


def _ollama_generate(prompt: str, model: str | None = None, temperature: float | None = None):
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
        text = resp.text
        return text, {}

    # Common response shapes returned by various Ollama deployments
    text = None
    if isinstance(data, dict):
        if "text" in data:
            text = data["text"]
        elif "output" in data:
            text = data["output"]
        elif "result" in data:
            text = data["result"]
        elif "choices" in data and len(data["choices"]) > 0:
            c = data["choices"][0]
            if isinstance(c, dict):
                if "message" in c and isinstance(c["message"], dict) and "content" in c["message"]:
                    text = c["message"]["content"]
                elif "content" in c:
                    text = c["content"]

    if text is None:
        # fallback: return full JSON as string
        return json.dumps(data), data

    return text, data


def _genai_generate(prompt: str, model: str | None = None):
    import google.generativeai as genai

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not key:
        raise RuntimeError("No API key found for Google Generative AI (set GEMINI_API_KEY or GOOGLE_API_KEY)")

    genai.configure(api_key=key)
    model_name = model or os.getenv("GEMINI_MODEL")
    model_obj = genai.GenerativeModel(model_name)
    resp = model_obj.generate_content(prompt)

    # Try to return both text and the raw response for metadata inspection
    text = getattr(resp, "text", None)
    try:
        raw = resp
    except Exception:
        raw = None

    if text is None:
        # Try common dict shapes
        try:
            d = resp.to_dict()
        except Exception:
            d = None
        if isinstance(d, dict):
            if "candidates" in d and len(d["candidates"])>0:
                cand = d["candidates"][0]
                text = cand.get("content") or cand.get("text")
        if text is None:
            text = str(resp)

    return text, raw


def generate(prompt: str, model: str | None = None, temperature: float | None = None, return_meta: bool = False) -> str:
    """Generate text using Ollama if configured, otherwise use Google Generative AI.

    - Set `OLLAMA_HOST` (e.g. https://cloud.ollama.com or http://localhost:11434) to use Ollama.
    - Or set `USE_OLLAMA=1` to force Ollama usage.
    - Otherwise the function falls back to Google GenAI and requires the usual API key env vars.
    """
    import time

    start = time.perf_counter()

    if os.getenv("OLLAMA_HOST") or os.getenv("USE_OLLAMA") == "1":
        text, raw = _ollama_generate(prompt, model=model, temperature=temperature)
    else:
        text, raw = _genai_generate(prompt, model=model)

    elapsed = time.perf_counter() - start

    # token estimation heuristic: 1 token ~= 4 characters
    prompt_chars = len(prompt or "")
    resp_chars = len(text or "")
    prompt_tokens_est = int(round(prompt_chars / 4))
    response_tokens_est = int(round(resp_chars / 4))
    total_tokens_est = prompt_tokens_est + response_tokens_est

    meta = {
        "model": model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL"),
        "duration_s": elapsed,
        "prompt_chars": prompt_chars,
        "response_chars": resp_chars,
        "prompt_tokens_est": prompt_tokens_est,
        "response_tokens_est": response_tokens_est,
        "total_tokens_est": total_tokens_est,
        "raw": raw,
    }

    if return_meta:
        return text, meta

    return text
