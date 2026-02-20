import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

BASE = Path(__file__).resolve().parent
CHUNKS = BASE / "data" / "chunks.jsonl"

DB_DIR = BASE / "chroma_db_openai"
COLLECTION_NAME = "tosoha1_chunks_openai"

def main():
    if not CHUNKS.exists():
        raise FileNotFoundError(f"Missing: {CHUNKS}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    embed_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )

    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"source": "blog.naver.com/tosoha1", "embed": "openai:text-embedding-3-small"},
    )

    # 이미 채워졌으면 종료
    if col.count() > 0:
        print(f"Collection already has {col.count()} chunks. If you want rebuild, delete {DB_DIR} folder.")
        return

    ids, docs, metas = [], [], []
    batch = 128
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
