import os
import sys
from neo4j import GraphDatabase
import google.generativeai as genai

import dotenv

from extract_graph import API_KEY
dotenv.load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def get_subgraph(entity):
    query = """
    MATCH (a {name:$entity})-[r*1..2]-(b)
    RETURN a,r,b
    LIMIT 50
    """
    with driver.session() as session:
        result = session.run(query, entity=entity)
        return [str(r) for r in result]

def answer(question, entity):
    context = "\n".join(get_subgraph(entity))

    prompt = f"""
    Context:
    {context}

    Question:
    {question}
    """
    API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not API_KEY:
        sys.exit(
            "No API key found. Set the GEMINI_API_KEY (or GOOGLE_API_KEY / GENAI_API_KEY) environment variable."
        )

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL"))
    return model.generate_content(prompt).text

print(answer(
    "Who is the founder of OPENAI?",
    "OPENAI"
))