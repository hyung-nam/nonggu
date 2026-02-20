from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE = Path(__file__).resolve().parent
DB_DIR = BASE / "chroma_db_local"
COLLECTION_NAME = "tosoha1_chunks_local"

def main():
    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    print("검색 테스트 — 종료: /exit\n")
    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in ["/exit", "exit", "quit", "q"]:
            break

        res = col.query(query_texts=[q], n_results=5)
        for i in range(len(res["ids"][0])):
            m = res["metadatas"][0][i]
            print(f"\n[{i+1}] {m.get('title','')} | {m.get('date','')} | {m.get('url','')}")
            print(f"heading: {m.get('heading_path','')}")
            print(res["documents"][0][i][:500].replace("\n"," ") + "...")
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()
