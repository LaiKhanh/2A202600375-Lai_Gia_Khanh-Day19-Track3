import json
import re
import traceback
from pathlib import Path


BENCHMARK_FILE = Path("benchmark_rag_questions.json")
TRIPLES_FILE = Path("outputs/triples.json")
RESULTS_JSON = Path("outputs/benchmark_results.json")
RESULTS_MD = Path("outputs/benchmark_comparison.md")


def load_benchmark():
    return json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))


def load_entities():
    if not TRIPLES_FILE.exists():
        return []

    triples = json.loads(TRIPLES_FILE.read_text(encoding="utf-8"))
    ents = set()
    for t in triples:
        subj = t.get("subject")
        obj = t.get("object")
        if subj:
            ents.add(subj)
        if obj:
            ents.add(obj)

    # sort by length desc so longest matches tried first
    return sorted(list(ents), key=lambda s: -len(s))


def find_entity_for_question(question, entities):
    q = question.lower()
    for e in entities:
        if e and e.lower() in q:
            return e

    # try capitalized tokens heuristic
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9\.\-&]+\b", question)
    for t in tokens:
        for e in entities:
            if t.lower() in e.lower() or e.lower() in t.lower():
                return e

    return None


def escape_cell(s: str):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("|", "\\|")
    s = s.replace("\n", "<br>")
    return s


def main():
    data = load_benchmark()

    # prepare modules
    flat_mod = None
    graph_mod = None

    try:
        import flat_rag as flat_mod
    except Exception as e:
        print("Warning: could not import flat_rag:", e)

    try:
        import graph_rag as graph_mod
    except Exception as e:
        print("Warning: could not import graph_rag:", e)

    entities = load_entities()

    results = []

    # If flat module loaded, ensure index exists
    if flat_mod is not None:
        try:
            existing = flat_mod.collection.count()
            if existing == 0:
                print("Flat RAG index empty; building index (this may take time)...")
                flat_mod.build_index()
        except Exception as e:
            print("Warning while checking/building flat index:", e)

    for idx, item in enumerate(data, start=1):
        q = item.get("question")
        print(f"[{idx}] {q}")

        # Flat RAG answer
        flat_ans = None
        if flat_mod is not None:
            try:
                flat_ans = flat_mod.answer_query(q)
            except Exception as e:
                flat_ans = f"ERROR: {e.__class__.__name__}: {e}"
                traceback.print_exc()
        else:
            flat_ans = "FLAT_RAG_NOT_AVAILABLE"

        # Graph RAG answer
        graph_ans = None
        if graph_mod is not None:
            try:
                entity = find_entity_for_question(q, entities)
                if entity is None:
                    graph_ans = "NO_ENTITY_FOUND"
                else:
                    graph_ans = graph_mod.answer(q, entity)
            except Exception as e:
                graph_ans = f"ERROR: {e.__class__.__name__}: {e}"
                traceback.print_exc()
        else:
            graph_ans = "GRAPH_RAG_NOT_AVAILABLE"

        results.append({
            "id": idx,
            "question": q,
            "flat_rag_answer": flat_ans,
            "graph_rag_answer": graph_ans
        })

    # save json
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # write markdown table
    md_lines = [
        "| id | question | flat_rag answer | graph_rag answer |",
        "|---|---|---|---|"
    ]

    for r in results:
        md_lines.append(
            f"| {r['id']} | {escape_cell(r['question'])} | {escape_cell(r['flat_rag_answer'])} | {escape_cell(r['graph_rag_answer'])} |"
        )

    RESULTS_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("Saved JSON results to:", RESULTS_JSON)
    print("Saved Markdown comparison to:", RESULTS_MD)


if __name__ == "__main__":
    main()
