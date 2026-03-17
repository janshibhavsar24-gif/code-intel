"""
embed.py — parse a repo and index every CodeNode into ChromaDB.

Usage:
  python embed.py /path/to/repo
  python embed.py /path/to/repo --db ./chroma_db --reset
"""
import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from extractor import extract_repo
from resolve import resolve

COLLECTION = "code_nodes"
DEFAULT_DB  = "./chroma_db"
BATCH_SIZE  = 100  # ChromaDB has per-call limits


def index(repo_path: str, db_path: str = DEFAULT_DB, reset: bool = False) -> None:
    print(f"Parsing {repo_path} ...")
    files = extract_repo(repo_path)
    nodes, edges = resolve(files)
    print(f"  {len(nodes)} nodes  |  {len(edges)} edges")

    client = chromadb.PersistentClient(path=db_path)

    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"  Cleared existing collection '{COLLECTION}'")
        except Exception:
            pass

    ef = embedding_functions.DefaultEmbeddingFunction()

    try:
        collection = client.get_collection(COLLECTION, embedding_function=ef)
        print(f"  Using existing collection (add --reset to rebuild from scratch)")
    except Exception:
        collection = client.create_collection(COLLECTION, embedding_function=ef)
        print(f"  Created new collection '{COLLECTION}'")

    # Build text to embed: name + docstring + source (gives richer semantics than source alone)
    def doc_text(n) -> str:
        parts = [f"{n.kind.value} {n.name}"]
        if n.docstring:
            parts.append(n.docstring)
        parts.append(n.source)
        return "\n".join(parts)

    # Skip nodes already in the collection
    existing = set(collection.get()["ids"])
    new_nodes = [n for n in nodes if n.id not in existing]

    if not new_nodes:
        print("  Nothing new to index.")
        return

    print(f"  Indexing {len(new_nodes)} nodes ...")
    for i in range(0, len(new_nodes), BATCH_SIZE):
        batch = new_nodes[i : i + BATCH_SIZE]
        collection.add(
            ids       = [n.id for n in batch],
            documents = [doc_text(n) for n in batch],
            metadatas = [
                {
                    "name"       : n.name,
                    "kind"       : n.kind.value,
                    "file"       : n.file,
                    "start_line" : n.start_line,
                    "end_line"   : n.end_line,
                }
                for n in batch
            ],
        )
        done = min(i + BATCH_SIZE, len(new_nodes))
        print(f"  {done}/{len(new_nodes)}")

    total = collection.count()
    print(f"\nDone. {total} nodes in '{COLLECTION}' at {db_path}")
    print(f"Run: python query.py {repo_path} \"your question here\"")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Index a Python repo into ChromaDB")
    p.add_argument("repo",    help="Path to the repo to index")
    p.add_argument("--db",    default=DEFAULT_DB, help="ChromaDB storage directory")
    p.add_argument("--reset", action="store_true", help="Wipe and rebuild the collection")
    args = p.parse_args()

    index(args.repo, args.db, args.reset)
