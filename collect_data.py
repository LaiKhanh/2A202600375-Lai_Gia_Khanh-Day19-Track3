import re
import time
import wikipedia
import os
import json
from wikipedia.exceptions import PageError, DisambiguationError, HTTPTimeoutError

companies = [
    "Anthropic", "DeepMind", "Cohere",
    "Mistral AI", "Stability AI", "Hugging Face",
    "Character.ai", "Scale AI"
]
os.makedirs("data/raw_articles", exist_ok=True)

LIMIT = 5

def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^A-Za-z0-9._-]', '_', name)
    return name

wikipedia.set_lang('en')

for company in companies[:LIMIT]:
    try:
        page = None
        # Try exact title first (no auto-suggest)
        try:
            page = wikipedia.page(company, auto_suggest=False)
        except (PageError, DisambiguationError, HTTPTimeoutError):
            # Fallback: search for likely titles and try them
            results = wikipedia.search(company, results=5)
            if not results:
                print(company, "no search results")
                continue
            for title in results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    break
                except Exception:
                    continue

        if not page:
            print(company, "no usable Wikipedia page found")
            continue

        content = page.content
        filename = safe_filename(company) + ".txt"
        outpath = os.path.join("data", "raw_articles", filename)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(content)

        print("saved", company, "->", outpath)
        time.sleep(1)

    except Exception as e:
        print(company, "error:", e)