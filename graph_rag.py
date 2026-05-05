import os
import sys
from neo4j import GraphDatabase
import dotenv
import llm_client

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
    return llm_client.generate(
        prompt,
        model=os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gpt-oss:20b-cloud",
    )

if __name__ == "__main__":
    print(answer(
        "Who is the founder of OPENAI?",
        "OPENAI"
    ))