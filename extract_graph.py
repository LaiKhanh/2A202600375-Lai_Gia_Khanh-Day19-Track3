import os
import json
import re
from pathlib import Path
import sys
import google.generativeai as genai

import dotenv
dotenv.load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
if not API_KEY:
    sys.exit(
        "No API key found. Set the GEMINI_API_KEY (or GOOGLE_API_KEY / GENAI_API_KEY) environment variable."
    )

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

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

    response = model.generate_content(
        PROMPT + "\n\nTEXT:\n" + text
    )

    cleaned = clean_json(response.text)

    try:
        triples = json.loads(cleaned)
        return triples
    except Exception as e:
        print("JSON parse error:", filepath, e)
        return []


def main():
    input_dir = Path("data/raw_articles")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    all_triples = []

    files = list(input_dir.glob("*.txt"))

    for file in files:
        print("processing:", file.name)
        triples = extract_from_file(file)
        all_triples.extend(triples)

    with open(output_dir / "triples.json", "w", encoding="utf-8") as f:
        json.dump(all_triples, f, indent=2, ensure_ascii=False)

    print("saved outputs/triples.json")


if __name__ == "__main__":
    main()