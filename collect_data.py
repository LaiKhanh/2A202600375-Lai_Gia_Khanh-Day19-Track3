import wikipedia
import os
import json

companies = [
    "OpenAI", "Anthropic", "DeepMind", "Cohere",
    "Mistral AI", "Stability AI", "Hugging Face",
    "Character.ai", "Scale AI", "Perplexity AI"
]

os.makedirs("data/raw_articles", exist_ok=True)

for company in companies:
    try:
        page = wikipedia.page(company)
        content = page.content

        with open(f"data/raw_articles/{company}.txt", "w", encoding="utf-8") as f:
            f.write(content)

        print("saved", company)

    except Exception as e:
        print(company, e)