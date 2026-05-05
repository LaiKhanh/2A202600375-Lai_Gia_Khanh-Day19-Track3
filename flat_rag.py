import os
from pathlib import Path
import chromadb
import google.generativeai as genai

import dotenv
dotenv.load_dotenv()

# =========================
# CONFIG
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")
CHAT_MODEL = genai.GenerativeModel(os.getenv("GEMINI_MODEL"))

DATA_DIR = Path("data/raw_articles")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "wiki_articles"


# =========================
# CHROMADB INIT
# =========================
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)


# =========================
# EMBEDDING
# =========================
def embed_text(text: str):
    """
    Generate embedding using Gemini embedding model.
    """
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_document"
    )

    return response["embedding"]


# =========================
# TEXT CHUNKING
# =========================
def chunk_text(text, chunk_size=1200, overlap=200):
    """
    Split text into overlapping chunks.
    """
    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================
# INDEXING
# =========================
def build_index():
    """
    Build ChromaDB index from raw articles.
    """
    txt_files = list(DATA_DIR.glob("*.txt"))

    all_ids = []
    all_docs = []
    all_embeddings = []

    idx = 0

    for file in txt_files:
        print(f"Processing {file.name}")

        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for chunk in chunks:
            embedding = embed_text(chunk)

            all_ids.append(str(idx))
            all_docs.append(chunk)
            all_embeddings.append(embedding)

            idx += 1

    collection.add(
        ids=all_ids,
        documents=all_docs,
        embeddings=all_embeddings
    )

    print(f"Indexed {len(all_docs)} chunks.")


# =========================
# RETRIEVAL
# =========================
def retrieve(query, top_k=3):
    """
    Retrieve top-k relevant chunks.
    """
    query_embedding = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# =========================
# GENERATION
# =========================
def answer_query(query):
    """
    Retrieve context then answer with Gemini.
    """
    docs = retrieve(query)

    context = "\n\n".join(docs)

    prompt = f"""
        You are a QA assistant.

        Answer ONLY based on the provided context.

        CONTEXT:
        {context}

        QUESTION:
        {query}
    """

    response = CHAT_MODEL.generate_content(prompt)

    return response.text


# =========================
# MAIN FLOW
# =========================
def main():
    print("==== FLAT RAG PIPELINE ====")

    # check if collection empty
    existing = collection.count()

    if existing == 0:
        print("No index found. Building index...")
        build_index()
    else:
        print(f"Existing index found: {existing} chunks")

    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            break

        answer = answer_query(query)

        print("\n===== ANSWER =====")
        print(answer)
        print("==================\n")


if __name__ == "__main__":
    main()