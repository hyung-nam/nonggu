import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE = Path(__file__).resolve().parent
CHUNKS = BASE / "data" / "chunks.jsonl"

DB_DIR = BASE / "chroma_db_local"
COLLECTION_NAME = "tosoha1_chunks_local"

def main():
    if not CHUNKS.exists():
        raise FileNotFoundError(f"Missing: {CHUNKS}")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"source": "blog.naver.com/tosoha1", "embed": "local:all-MiniLM-L6-v2"},
    )

    if col.count() > 0:
        print(f"Collection already has {col.count()} chunks. If you want rebuild, delete {DB_DIR} folder.")
        return

    ids, docs, metas = [], [], []
    batch = 256
    added = 0

    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            cid = rec.get("id")
            if not cid:
                continue

            text = (rec.get("content") or "").strip()
            if not text:
                continue

            meta = {
                "doc_id": rec.get("doc_id"),
                "title": rec.get("title"),
                "date": rec.get("date"),
                "url": rec.get("url"),
                "category": rec.get("category", ""),
                "heading_path": rec.get("heading_path", ""),
                "chunk_index": rec.get("chunk_index"),
                "total_chunks": rec.get("total_chunks"),
                "source": rec.get("source"),
            }

            ids.append(cid)
            docs.append(text)
            metas.append(meta)

            if len(ids) >= batch:
                col.add(ids=ids, documents=docs, metadatas=metas)
                added += len(ids)
                print(f"Added {added} chunks...")
                ids, docs, metas = [], [], []

    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)
        added += len(ids)

    print(f"Done. Total chunks in collection: {col.count()}")

if __name__ == "__main__":
    main()
