"""
query.py — hybrid vector+graph retrieval answered by Claude.

Pipeline:
  question → embed → ChromaDB top-K → 1-hop graph BFS → rank → Claude

Usage:
  python query.py /path/to/repo "how does auth work?"
  python query.py /path/to/repo                        # interactive mode
  python query.py /path/to/repo --analyze              # coupling + dead code report
"""
import argparse
import os
import sys

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from extractor import extract_repo
from resolve import resolve, build_graph, coupling_scores, dead_code, change_impact

COLLECTION      = "code_nodes"
DEFAULT_DB      = "./chroma_db"
TOP_K           = 5
MAX_CONTEXT     = 20_000   # chars sent to Claude


# ── graph + collection loader ─────────────────────────────────────────────────

def _load(repo_path: str, db_path: str):
    """Parse repo → build graph → open ChromaDB collection."""
    files      = extract_repo(repo_path)
    nodes, edges = resolve(files)
    G          = build_graph(nodes, edges)
    node_map   = {n.id: n for n in nodes}

    client = chromadb.PersistentClient(path=db_path)
    ef     = embedding_functions.DefaultEmbeddingFunction()
    try:
        collection = client.get_collection(COLLECTION, embedding_function=ef)
    except Exception:
        rprint(f"[red]Collection not found. Run embed.py first:[/red]")
        rprint(f"  python embed.py {repo_path} --db {db_path}")
        sys.exit(1)

    return G, node_map, collection


# ── hybrid retrieval ──────────────────────────────────────────────────────────

def retrieve(question: str, collection, G, node_map, top_k: int = TOP_K):
    """
    1. Vector search  → seed nodes semantically close to the question
    2. 1-hop BFS      → callers + callees of each seed
    3. Rank by PageRank so load-bearing nodes surface first
    """
    results  = collection.query(query_texts=[question], n_results=top_k)
    seed_ids = results["ids"][0]

    expanded = set(seed_ids)
    for nid in seed_ids:
        if nid in G:
            expanded.update(G.predecessors(nid))   # things that call this
            expanded.update(G.successors(nid))      # things this calls

    scores = coupling_scores(G)
    ranked = sorted(
        [nid for nid in expanded if nid in node_map],
        key=lambda nid: scores.get(nid, 0),
        reverse=True,
    )
    return [node_map[nid] for nid in ranked], seed_ids


def _build_context(nodes, max_chars: int = MAX_CONTEXT) -> str:
    parts, total = [], 0
    for n in nodes:
        chunk = (
            f"### {n.kind.value.upper()}: {n.name}\n"
            f"# file: {n.file}  lines: {n.start_line}-{n.end_line}\n"
            f"{n.source}\n"
        )
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


# ── Claude answer ─────────────────────────────────────────────────────────────

def ask_claude(question: str, context: str) -> str:
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are a code intelligence assistant. You receive source code snippets "
            "retrieved by semantic + structural search from a real codebase. "
            "Answer the user's question using only the provided code context. "
            "Be specific: cite function names, file paths, and line numbers. "
            "If the context doesn't contain enough information, say so clearly."
        ),
        messages=[{
            "role": "user",
            "content": f"Code context:\n\n{context}\n\n---\n\nQuestion: {question}",
        }],
    )
    return msg.content[0].text


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_ask(question: str, G, node_map, collection, verbose: bool = False):
    nodes, seeds = retrieve(question, collection, G, node_map)
    context      = _build_context(nodes)

    if verbose:
        rprint(f"\n[dim]Seeds:[/dim] {seeds}")
        rprint(f"[dim]Expanded to {len(nodes)} nodes, {len(context)} chars[/dim]\n")

    answer = ask_claude(question, context)
    rprint(Panel(answer, title=f"[bold cyan]{question}[/bold cyan]", expand=False))


def cmd_analyze(G, node_map):
    """Print coupling scores + dead code — no LLM needed."""
    scores = coupling_scores(G)
    top15  = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]

    t = Table(title="Top 15 load-bearing nodes (PageRank)", show_lines=True)
    t.add_column("Rank",  style="dim",    width=5)
    t.add_column("Score", style="yellow", width=8)
    t.add_column("Kind",  style="cyan",   width=10)
    t.add_column("Node",  style="white")
    for i, (nid, score) in enumerate(top15, 1):
        kind = G.nodes[nid].get("kind", "?")
        t.add_row(str(i), f"{score:.4f}", kind, nid)
    rprint(t)

    dead = dead_code(G)
    rprint(f"\n[bold red]Dead code candidates[/bold red] ({len(dead)} total):")
    for d in dead[:20]:
        n = node_map[d]
        rprint(f"  [dim]{n.file}:{n.start_line}[/dim]  {n.kind.value}  [white]{n.name}[/white]")


def cmd_impact(node_id: str, G):
    """Show what breaks if you change a given node."""
    if node_id not in G:
        rprint(f"[red]Node not found:[/red] {node_id}")
        return
    affected = change_impact(G, node_id)
    rprint(f"\n[bold yellow]Change impact for[/bold yellow] {node_id}")
    rprint(f"  {len(affected)} nodes would be affected:\n")
    for nid in affected:
        rprint(f"  → {nid}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _check_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        rprint("[red]ANTHROPIC_API_KEY not set.[/red]")
        rprint("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hybrid code-intel Q&A")
    p.add_argument("repo",       help="Repo path (must have been indexed with embed.py)")
    p.add_argument("question",   nargs="?", help="Question (omit for interactive mode)")
    p.add_argument("--db",       default=DEFAULT_DB)
    p.add_argument("--analyze",  action="store_true", help="Show coupling + dead code report")
    p.add_argument("--impact",   metavar="NODE_ID",   help="Show change impact for a node")
    p.add_argument("--verbose",  action="store_true")
    args = p.parse_args()

    rprint("[dim]Loading graph + collection...[/dim]")
    G, node_map, collection = _load(args.repo, args.db)
    rprint(f"[dim]{G.number_of_nodes()} nodes  |  {G.number_of_edges()} edges[/dim]\n")

    if args.analyze:
        cmd_analyze(G, node_map)

    elif args.impact:
        cmd_impact(args.impact, G)

    elif args.question:
        _check_api_key()
        cmd_ask(args.question, G, node_map, collection, verbose=args.verbose)

    else:
        # interactive mode
        _check_api_key()
        rprint("[bold green]Code-Intel Q&A[/bold green]  (type [dim]quit[/dim] to exit, [dim]analyze[/dim] for report)\n")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            if q.lower() == "analyze":
                cmd_analyze(G, node_map)
                continue
            cmd_ask(q, G, node_map, collection, verbose=args.verbose)
