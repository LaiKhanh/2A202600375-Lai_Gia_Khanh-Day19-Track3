import os
import json
import re
from pathlib import Path
import sys
import time

import dotenv
dotenv.load_dotenv()

import llm_client

PROMPT = """
You are an information extraction system.

Extract knowledge graph triples from the given text.

RULES:
1. Return ONLY valid JSON.
2. Output must be a JSON array.

Each item:
{
  "subject": "entity1",
  "subject_type": "COMPANY|PERSON|PRODUCT|LOCATION|YEAR|TECHNOLOGY|ORGANIZATION",
  "relation": "RELATION_NAME",
  "object": "entity2",
  "object_type": "COMPANY|PERSON|PRODUCT|LOCATION|YEAR|TECHNOLOGY|ORGANIZATION"
}

ALLOWED RELATIONS:
- FOUNDED_BY
- FOUNDED_IN
- WORKED_AT
- ACQUIRED_BY
- INVESTED_BY
- LOCATED_IN
- DEVELOPED
- PARTNERS_WITH
- COMPETES_WITH
- CEO_OF

IMPORTANT:
- Keep entity names canonical.
- Remove duplicate facts.
- Do not invent facts.
- Maximum 30 triples.

Return JSON only.
"""


def clean_json(text):
    text = text.strip()

    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    return text.strip()


def extract_from_file(filepath):
    text = Path(filepath).read_text(encoding="utf-8")[:15000]

    prompt_text = PROMPT + "\n\nTEXT:\n" + text

    try:
        response_text, meta = llm_client.generate(
            prompt_text,
            model=os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gpt-oss:20b-cloud",
            return_meta=True,
        )
    except Exception as e:
        print("LLM call failed for", filepath, e)
        return [], {"duration_s": 0, "total_tokens_est": 0}

    cleaned = clean_json(response_text)

    try:
        triples = json.loads(cleaned)
        return triples, meta
    except Exception as e:
        print("JSON parse error:", filepath, e)
        return [], meta


def main():
    input_dir = Path("data/raw_articles")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    all_triples = []
    metrics = []

    files = list(input_dir.glob("*.txt"))

    for file in files:
        print("processing:", file.name)
        triples, meta = extract_from_file(file)
        all_triples.extend(triples)

        metrics.append({
            "file": file.name,
            "n_triples": len(triples),
            "duration_s": meta.get("duration_s") if isinstance(meta, dict) else None,
            "prompt_tokens_est": meta.get("prompt_tokens_est") if isinstance(meta, dict) else None,
            "response_tokens_est": meta.get("response_tokens_est") if isinstance(meta, dict) else None,
            "total_tokens_est": meta.get("total_tokens_est") if isinstance(meta, dict) else None,
            "model": meta.get("model") if isinstance(meta, dict) else None,
        })

    with open(output_dir / "triples.json", "w", encoding="utf-8") as f:
        json.dump(all_triples, f, indent=2, ensure_ascii=False)

    # write extraction metrics
    with open(output_dir / "graph_extraction_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("saved outputs/triples.json and outputs/graph_extraction_metrics.json")


if __name__ == "__main__":
    main()