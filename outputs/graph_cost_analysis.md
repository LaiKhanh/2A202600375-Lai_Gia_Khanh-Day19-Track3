# Graph Build Cost Analysis

## Extraction (LLM) metrics

- Files processed: 5
- Total estimated prompt tokens: 14871
- Total estimated response tokens: 18099
- Total estimated tokens (prompt+response): 32970
- Total extraction time (s): 85.97

## Insertion (Neo4j) metrics

- Triples inserted: 147
- Total insertion time (s): 5.74
- Average time per triple (s): 0.0390

## Notes

- Token counts are heuristic estimates (1 token ≈ 4 characters). If your LLM provider returns exact token usage, prefer those values.
- Extraction time includes the LLM call duration for each file; insertion time measures the Python call to create nodes/relationships in Neo4j.