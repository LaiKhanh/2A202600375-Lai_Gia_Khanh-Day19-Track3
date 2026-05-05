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


def extract_response(answer_str):
    """Try to extract a concise 'response' text from an LLM output.

    Handles:
    - plain JSON strings with a top-level 'response' or 'text' key
    - JSON-like substrings inside larger text
    - simple 'Response: ...' patterns
    Fallback: return the original string.
    """
    if answer_str is None:
        return ""

    if not isinstance(answer_str, str):
        return str(answer_str)

    s = answer_str.strip()

    # Try parse full string as JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            for key in ("response", "text", "output", "result", "answer"):
                if key in obj:
                    return obj[key]
            # common nested shapes
            if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                first = obj["choices"][0]
                if isinstance(first, dict):
                    for k in ("text", "message", "content"):
                        if k in first:
                            v = first[k]
                            if isinstance(v, dict) and "content" in v:
                                return v["content"]
                            return v
        else:
            return str(obj)
    except Exception:
        pass

    # Try to find a JSON substring and parse it
    start = s.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if ch == '"' and not escape:
                in_string = not in_string
            if ch == '\\' and not escape:
                escape = True
                continue
            else:
                escape = False
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i+1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                for key in ("response", "text", "output", "result", "answer"):
                                    if key in obj:
                                        return obj[key]
                                return json.dumps(obj)
                        except Exception:
                            break

    # Regex fallback for explicit response fields
    m = re.search(r'"response"\s*:\s*"([^"]+)"', s)
    if m:
        return m.group(1)

    m = re.search(r'(?:Response|response)\s*[:\-]\s*(.+)$', s, re.IGNORECASE)
    if m:
        return m.group(1).strip()

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

        # extract concise response text for markdown reporting
        flat_resp = extract_response(flat_ans)
        graph_resp = extract_response(graph_ans)

        results.append({
            "id": idx,
            "question": q,
            "flat_rag_answer": flat_ans,
            "flat_rag_response": flat_resp,
            "graph_rag_answer": graph_ans,
            "graph_rag_response": graph_resp
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
            f"| {r['id']} | {escape_cell(r['question'])} | {escape_cell(r.get('flat_rag_response',''))} | {escape_cell(r.get('graph_rag_response',''))} |"
        )

    RESULTS_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("Saved JSON results to:", RESULTS_JSON)
    print("Saved Markdown comparison to:", RESULTS_MD)


if __name__ == "__main__":
    main()
