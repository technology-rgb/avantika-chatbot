"""
Build the FAISS knowledge base from documents in knowledge_base/
Run once before starting the app: python ingest.py
"""
import os
import json
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = "knowledge_base"
VECTOR_STORE_DIR = "vector_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 550
CHUNK_OVERLAP = 80


def load_documents():
    docs = []
    for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(KNOWLEDGE_DIR, fname), encoding="utf-8") as f:
            content = f.read()
        source = fname.replace(".txt", "").replace("_", " ").title()
        docs.append({"content": content, "source": source})
        print(f"  Loaded: {fname} ({len(content):,} chars)")
    return docs


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text on section boundaries, then by size."""
    chunks = []
    # First split on section headers (===)
    sections = [s.strip() for s in text.split("===") if s.strip()]

    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        # Split large sections by paragraph
        paragraphs = section.split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                # If single paragraph exceeds chunk size, split by sentence
                if len(para) > chunk_size:
                    sentences = para.replace("\n", " ").split(". ")
                    current = ""
                    for sent in sentences:
                        if len(current) + len(sent) + 2 <= chunk_size:
                            current = (current + ". " + sent).strip()
                        else:
                            if current:
                                chunks.append(current)
                            current = sent
                    if current:
                        chunks.append(current)
                    current = ""
                else:
                    current = para
        if current:
            chunks.append(current)

    return [c for c in chunks if len(c.strip()) > 50]


def build_knowledge_base():
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    print("\nLoading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s)")

    print("\nSplitting into chunks...")
    all_chunks = []
    for doc in docs:
        splits = split_text(doc["content"])
        for chunk in splits:
            all_chunks.append({"text": chunk.strip(), "source": doc["source"]})
    print(f"Created {len(all_chunks)} chunks")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    print("\nBuilding FAISS index (cosine similarity)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "index.faiss"))

    metadata = [{"text": c["text"], "source": c["source"]} for c in all_chunks]
    with open(os.path.join(VECTOR_STORE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Write manifest so app.py can detect when KB changes
    manifest = {}
    for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
        if fname.endswith(".txt"):
            fpath = os.path.join(KNOWLEDGE_DIR, fname)
            with open(fpath, "rb") as f:
                manifest[fname] = hashlib.md5(f.read()).hexdigest()
    with open(os.path.join(VECTOR_STORE_DIR, "kb_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Knowledge base saved to '{VECTOR_STORE_DIR}/'")
    print(f"Total chunks indexed: {len(all_chunks)}")
    print("Run 'streamlit run app.py' to start the chatbot.")


if __name__ == "__main__":
    build_knowledge_base()
