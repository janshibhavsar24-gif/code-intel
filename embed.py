"""
embed.py — parse a repo and index every CodeNode into ChromaDB.

Supports incremental indexing: only changed or new files are re-embedded,
and nodes from deleted files are removed automatically.
File hashes are stored in {db_path}/file_hashes.json.

Usage:
  python embed.py /path/to/repo
  python embed.py /path/to/repo --db ./chroma_db --reset
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from extractor import extract_repo
from resolve import resolve

COLLECTION = "code_nodes"
DEFAULT_DB  = "./chroma_db"
BATCH_SIZE  = 100
HASH_FILE   = "file_hashes.json"


# ── file hash helpers ─────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_hashes(db_path: str) -> dict[str, str]:
    p = Path(db_path) / HASH_FILE
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_hashes(db_path: str, hashes: dict[str, str]) -> None:
    Path(db_path).mkdir(parents=True, exist_ok=True)
    (Path(db_path) / HASH_FILE).write_text(json.dumps(hashes, indent=2))


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _ids_for_file(collection, rel_path: str) -> list[str]:
    """Return all node IDs currently indexed for a given relative file path."""
    result = collection.get(where={"file": rel_path})
    return result["ids"]


def _remove_file_nodes(collection, rel_path: str) -> int:
    ids = _ids_for_file(collection, rel_path)
    if ids:
        collection.delete(ids=ids)
    return len(ids)


# ── main indexer ──────────────────────────────────────────────────────────────

def index(repo_path: str, db_path: str = DEFAULT_DB, reset: bool = False) -> None:
    root = Path(repo_path).resolve()

    print(f"Parsing {repo_path} ...")
    files = extract_repo(repo_path)
    nodes, edges = resolve(files)
    print(f"  {len(nodes)} nodes  |  {len(edges)} edges")

    # Build a map from relative file path → nodes in that file
    file_to_nodes: dict[str, list] = {}
    for n in nodes:
        file_to_nodes.setdefault(n.file, []).append(n)

    # Compute current hashes for every parsed source file
    current_hashes: dict[str, str] = {}
    for pf in files:
        src_path = root / pf.path
        if src_path.exists():
            current_hashes[pf.path] = _file_hash(src_path)

    # Set up ChromaDB
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
        print(f"  Using existing collection")
    except Exception:
        collection = client.create_collection(COLLECTION, embedding_function=ef)
        print(f"  Created new collection '{COLLECTION}'")

    # Load stored hashes (empty if reset or first run)
    stored_hashes = {} if reset else _load_hashes(db_path)

    # Classify files
    changed  = [f for f, h in current_hashes.items() if stored_hashes.get(f) != h]
    deleted  = [f for f in stored_hashes if f not in current_hashes]
    skipped  = len(current_hashes) - len(changed)

    print(f"  {len(changed)} changed/new  |  {len(deleted)} deleted  |  {skipped} unchanged")

    # Remove nodes for deleted files
    removed_total = 0
    for rel_path in deleted:
        removed_total += _remove_file_nodes(collection, rel_path)
    if removed_total:
        print(f"  Removed {removed_total} nodes from {len(deleted)} deleted file(s)")

    # Re-index changed/new files
    nodes_to_index = []
    for rel_path in changed:
        # Remove stale nodes for this file (if it was previously indexed)
        if rel_path in stored_hashes:
            _remove_file_nodes(collection, rel_path)
        nodes_to_index.extend(file_to_nodes.get(rel_path, []))

    if not nodes_to_index:
        print("  Nothing to index — all files up to date.")
        _save_hashes(db_path, current_hashes)
        total = collection.count()
        print(f"\nDone. {total} nodes in '{COLLECTION}' at {db_path}")
        return

    def doc_text(n) -> str:
        parts = [f"{n.kind.value} {n.name}"]
        if n.docstring:
            parts.append(n.docstring)
        parts.append(n.source)
        return "\n".join(parts)

    print(f"  Indexing {len(nodes_to_index)} nodes ...")
    for i in range(0, len(nodes_to_index), BATCH_SIZE):
        batch = nodes_to_index[i : i + BATCH_SIZE]
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
        done = min(i + BATCH_SIZE, len(nodes_to_index))
        print(f"  {done}/{len(nodes_to_index)}")

    # Persist updated hashes
    _save_hashes(db_path, current_hashes)

    total = collection.count()
    print(f"\nDone. {total} nodes in '{COLLECTION}' at {db_path}")
    print(f"Run: python query.py {repo_path} \"your question here\"")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Incrementally index a repo into ChromaDB")
    p.add_argument("repo",    help="Path to the repo to index")
    p.add_argument("--db",    default=DEFAULT_DB, help="ChromaDB storage directory")
    p.add_argument("--reset", action="store_true", help="Wipe and rebuild from scratch")
    args = p.parse_args()

    index(args.repo, args.db, args.reset)
