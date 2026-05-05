import os
import json
import re
from pathlib import Path
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

    # measure insertion time
    import time
    insertion_times = []

    insertion_start = time.perf_counter()
    with driver.session() as session:
        for triple in triples:
            t0 = time.perf_counter()
            session.execute_write(push_triple, triple)
            t1 = time.perf_counter()
            insertion_times.append(t1 - t0)
    insertion_total = time.perf_counter() - insertion_start

    # save insertion metrics
    insertion_metrics = {
        "n_triples": len(triples),
        "total_insertion_time_s": insertion_total,
        "avg_time_per_triple_s": (sum(insertion_times) / len(insertion_times)) if insertion_times else 0,
        "per_triple_times": insertion_times[:1000]
    }

    with open("outputs/graph_insertion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(insertion_metrics, f, indent=2, ensure_ascii=False)

    print("graph inserted into neo4j")
    print("saved outputs/graph_insertion_metrics.json")

    # try to produce a combined markdown analysis if extraction metrics exist
    extraction_metrics_path = Path("outputs/graph_extraction_metrics.json")
    md_path = Path("outputs/graph_cost_analysis.md")

    total_prompt_tokens = 0
    total_response_tokens = 0
    total_token_est = 0
    total_extract_time = 0
    n_files = 0

    if extraction_metrics_path.exists():
        with open(extraction_metrics_path, "r", encoding="utf-8") as f:
            em = json.load(f)
        n_files = len(em)
        for entry in em:
            total_prompt_tokens += entry.get("prompt_tokens_est") or 0
            total_response_tokens += entry.get("response_tokens_est") or 0
            total_token_est += entry.get("total_tokens_est") or 0
            total_extract_time += entry.get("duration_s") or 0

    # write markdown summary
    lines = []
    lines.append("# Graph Build Cost Analysis")
    lines.append("")
    lines.append("## Extraction (LLM) metrics")
    lines.append("")
    lines.append(f"- Files processed: {n_files}")
    lines.append(f"- Total estimated prompt tokens: {total_prompt_tokens}")
    lines.append(f"- Total estimated response tokens: {total_response_tokens}")
    lines.append(f"- Total estimated tokens (prompt+response): {total_token_est}")
    lines.append(f"- Total extraction time (s): {total_extract_time:.2f}")
    lines.append("")
    lines.append("## Insertion (Neo4j) metrics")
    lines.append("")
    lines.append(f"- Triples inserted: {insertion_metrics['n_triples']}")
    lines.append(f"- Total insertion time (s): {insertion_metrics['total_insertion_time_s']:.2f}")
    lines.append(f"- Average time per triple (s): {insertion_metrics['avg_time_per_triple_s']:.4f}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Token counts are heuristic estimates (1 token ≈ 4 characters). If your LLM provider returns exact token usage, prefer those values.")
    lines.append("- Extraction time includes the LLM call duration for each file; insertion time measures the Python call to create nodes/relationships in Neo4j.")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved markdown graph cost analysis to:", md_path)


if __name__ == "__main__":
    main()