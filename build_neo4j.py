import os
import json
import re
from neo4j import GraphDatabase

import dotenv
dotenv.load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    URI,
    auth=(USER, PASSWORD)
)


def normalize_entity(name):
    name = name.lower().strip()

    suffixes = [
        " inc", " llc", " ltd", " corp",
        " corporation", " company"
    ]

    name = re.sub(r"[^\w\s]", "", name)

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    return name.strip()


def canonical_entity(name):
    normalized = normalize_entity(name)

    mapping = {
        "google llc": "Google",
        "google": "Google",
        "openai inc": "OpenAI",
        "openai": "OpenAI",
        "microsoft corporation": "Microsoft",
        "microsoft": "Microsoft",
        "deepmind technologies": "DeepMind",
        "deepmind": "DeepMind"
    }

    return mapping.get(normalized, name.strip())


def deduplicate_triples(triples):
    seen = set()
    cleaned = []

    for t in triples:
        subject = canonical_entity(t["subject"])
        obj = canonical_entity(t["object"])
        relation = t["relation"]

        key = (subject, relation, obj)

        if key not in seen:
            seen.add(key)

            t["subject"] = subject
            t["object"] = obj

            cleaned.append(t)

    return cleaned


def create_constraints():
    query = """
    CREATE CONSTRAINT entity_name IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE e.name IS UNIQUE
    """

    with driver.session() as session:
        session.run(query)


def push_triple(tx, triple):
    ALLOWED_RELATIONS = {
        "FOUNDED_BY",
        "FOUNDED_IN",
        "WORKED_AT",
        "ACQUIRED_BY",
        "INVESTED_BY",
        "LOCATED_IN",
        "DEVELOPED",
        "PARTNERS_WITH",
        "COMPETES_WITH",
        "CEO_OF"
    }

    relation = triple["relation"]

    if relation not in ALLOWED_RELATIONS:
        return

    query = f"""
    MERGE (a:Entity {{name:$subject}})
    ON CREATE SET a.type = $subject_type

    MERGE (b:Entity {{name:$object}})
    ON CREATE SET b.type = $object_type

    MERGE (a)-[:{relation}]->(b)
    """

    tx.run(
        query,
        subject=triple["subject"],
        subject_type=triple["subject_type"],
        object=triple["object"],
        object_type=triple["object_type"]
    )


def main():
    create_constraints()

    with open("outputs/triples.json", "r", encoding="utf-8") as f:
        triples = json.load(f)

    triples = deduplicate_triples(triples)

    print("triples after dedup:", len(triples))

    with driver.session() as session:
        for triple in triples:
            session.execute_write(push_triple, triple)

    print("graph inserted into neo4j")


if __name__ == "__main__":
    main()