"""
server.py — MCP server exposing code-intel as Claude-callable tools.

Tools:
  ask_question   — hybrid vector+graph Q&A answered by Claude (single or multi-repo)
  analyze_repo   — coupling scores + dead code report (single or multi-repo)
  change_impact  — blast radius for a given node ID (single or multi-repo)
  find_usages    — all call sites of a symbol across one or more repos

Transport: stdio (works with Claude Desktop, Cursor, VS Code + MCP extension)

Usage:
  python server.py

Configure in claude_desktop_config.json:
  {
    "mcpServers": {
      "code-intel": {
        "command": "python",
        "args": ["/absolute/path/to/code-intel/server.py"],
        "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
      }
    }
  }

Or with uv (no venv activation needed):
  "command": "uv",
  "args": ["run", "--project", "/path/to/code-intel", "python", "server.py"]
"""
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from extractor import extract_repo
from resolve import (
    resolve,
    build_graph,
    coupling_scores,
    dead_code,
    change_impact as _graph_change_impact,
)

server = Server("code-intel")

COLLECTION   = "code_nodes"
DEFAULT_DB   = "./chroma_db"
TOP_K        = 5
MAX_CONTEXT  = 20_000

# ── in-memory caches ──────────────────────────────────────────────────────────

_graph_cache: dict[str, tuple]      = {}   # repo_path → (G, node_map)
_coll_cache:  dict[tuple, Any]      = {}   # (repo_path, db_path) → collection


def _load_graph(repo_path: str) -> tuple:
    key = str(Path(repo_path).resolve())
    if key not in _graph_cache:
        files        = extract_repo(repo_path)
        nodes, edges = resolve(files)
        G            = build_graph(nodes, edges)
        node_map     = {n.id: n for n in nodes}
        _graph_cache[key] = (G, node_map)
    return _graph_cache[key]


def _load_collection(repo_path: str, db_path: str) -> Any:
    key = (str(Path(repo_path).resolve()), str(Path(db_path).resolve()))
    if key not in _coll_cache:
        client = chromadb.PersistentClient(path=db_path)
        ef     = embedding_functions.DefaultEmbeddingFunction()
        _coll_cache[key] = client.get_collection(COLLECTION, embedding_function=ef)
    return _coll_cache[key]


# ── retrieval helpers (same logic as query.py) ────────────────────────────────

def _retrieve(question: str, collection, G, node_map, top_k: int = TOP_K) -> list:
    results  = collection.query(query_texts=[question], n_results=top_k)
    seed_ids = results["ids"][0]

    expanded = set(seed_ids)
    for nid in seed_ids:
        if nid in G:
            expanded.update(G.predecessors(nid))
            expanded.update(G.successors(nid))

    scores = coupling_scores(G)
    ranked = sorted(
        [nid for nid in expanded if nid in node_map],
        key=lambda nid: scores.get(nid, 0),
        reverse=True,
    )
    return [node_map[nid] for nid in ranked]


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


def _ask_claude(question: str, context: str) -> str:
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


# ── multi-repo helpers ────────────────────────────────────────────────────────

def _normalize_repo_paths(arguments: dict) -> list[str]:
    """Accept either repo_path (str) or repo_paths (list) from tool arguments."""
    if "repo_paths" in arguments:
        paths = arguments["repo_paths"]
        return [paths] if isinstance(paths, str) else paths
    return [arguments["repo_path"]]


def _normalize_db_paths(arguments: dict, repo_paths: list[str]) -> list[str]:
    """Accept either db_path (str) or db_paths (list), defaulting per repo."""
    if "db_paths" in arguments:
        paths = arguments["db_paths"]
        return [paths] if isinstance(paths, str) else paths
    if "db_path" in arguments:
        return [arguments["db_path"]] * len(repo_paths)
    return [DEFAULT_DB] * len(repo_paths)


# ── tool implementations (sync, run in thread) ────────────────────────────────

def _tool_ask_question(repo_paths: list[str], question: str, db_paths: list[str]) -> str:
    per_repo_budget = MAX_CONTEXT // len(repo_paths)
    contexts = []

    for repo_path, db_path in zip(repo_paths, db_paths):
        label = Path(repo_path).name
        try:
            G, node_map = _load_graph(repo_path)
            collection  = _load_collection(repo_path, db_path)
        except Exception as exc:
            contexts.append(
                f"### REPO: {label}\n"
                f"[Error: {exc} — run `python embed.py {repo_path} --db {db_path}`]\n"
            )
            continue
        nodes   = _retrieve(question, collection, G, node_map)
        context = _build_context(nodes, max_chars=per_repo_budget)
        contexts.append(f"### REPO: {label}\n{context}")

    combined = "\n\n".join(contexts)
    return _ask_claude(question, combined)


def _tool_analyze_repo(repo_paths: list[str]) -> str:
    sections = []

    for repo_path in repo_paths:
        label = Path(repo_path).name
        try:
            G, node_map = _load_graph(repo_path)
        except Exception as exc:
            sections.append(f"## {label}\nFailed to load repo: {exc}\n")
            continue

        scores = coupling_scores(G)
        top15  = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]

        lines = [f"## {label}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)\n"]
        lines.append(f"### Top 15 load-bearing nodes (PageRank)\n")
        lines.append(f"{'Rank':<5} {'Score':<8} {'Kind':<10} Node")
        lines.append("-" * 60)
        for i, (nid, score) in enumerate(top15, 1):
            kind = G.nodes[nid].get("kind", "?")
            lines.append(f"{i:<5} {score:.4f}   {kind:<10} {nid}")

        dead = dead_code(G)
        lines.append(f"\n### Dead code candidates ({len(dead)} total)\n")
        for d in dead[:20]:
            n = node_map[d]
            lines.append(f"  {n.file}:{n.start_line}  {n.kind.value}  {n.name}")
        if len(dead) > 20:
            lines.append(f"  … and {len(dead) - 20} more")

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


def _tool_change_impact(repo_paths: list[str], node_id: str) -> str:
    sections = []

    for repo_path in repo_paths:
        label = Path(repo_path).name
        try:
            G, node_map = _load_graph(repo_path)
        except Exception as exc:
            sections.append(f"## {label}\nFailed to load repo: {exc}\n")
            continue

        # resolve node_id — exact match first, then fuzzy
        resolved = node_id
        if resolved not in G:
            matches = [nid for nid in G.nodes() if node_id in nid]
            if not matches:
                sections.append(f"## {label}\nNode `{node_id}` not found — no partial matches.\n")
                continue
            if len(matches) == 1:
                resolved = matches[0]
            else:
                suggestions = "\n".join(f"  - {m}" for m in matches[:10])
                sections.append(
                    f"## {label}\n"
                    f"Node `{node_id}` ambiguous. Did you mean:\n{suggestions}\n"
                )
                continue

        affected = _graph_change_impact(G, resolved)
        if not affected:
            sections.append(
                f"## {label}\n"
                f"No callers found for `{resolved}` — appears to be a leaf node.\n"
            )
            continue

        lines = [f"## {label} — change impact for `{resolved}`\n"]
        lines.append(f"{len(affected)} node(s) transitively depend on this:\n")
        for nid in affected:
            n = node_map.get(nid)
            if n:
                lines.append(f"  → {n.file}:{n.start_line}  {n.kind.value}  {n.name}")
            else:
                lines.append(f"  → {nid}")
        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


def _usages_in_node(node, symbol: str) -> list[tuple[int, str]]:
    """
    Scan a node's source for lines containing `symbol` as a whole word.
    Returns (absolute_line_number, line_text) pairs.
    """
    pattern = re.compile(rf'\b{re.escape(symbol)}\b')
    hits = []
    for offset, line in enumerate(node.source.splitlines()):
        if pattern.search(line):
            hits.append((node.start_line + offset, line.rstrip()))
    return hits


def _tool_find_usages(repo_paths: list[str], symbol: str) -> str:
    sections = []

    for repo_path in repo_paths:
        label = Path(repo_path).name
        try:
            G, node_map = _load_graph(repo_path)
        except Exception as exc:
            sections.append(f"## {label}\nFailed to load repo: {exc}\n")
            continue

        # 1. Graph pass — find nodes that have a CALLS/IMPORTS/INHERITS edge to any
        #    node named `symbol`. These are the most reliable hits.
        target_nids = {nid for nid, d in G.nodes(data=True) if d.get("name") == symbol}
        graph_caller_nids: set[str] = set()
        for target in target_nids:
            graph_caller_nids.update(G.predecessors(target))

        # 2. Text scan every node's source (whole-word match).
        #    Graph callers are scanned first; remaining nodes fill in anything missed
        #    (dynamic calls, import aliases, string references, etc.).
        all_hits: set[tuple[str, int, str]] = set()
        scanned: set[str] = set()

        def scan(nid: str) -> None:
            if nid not in node_map or nid in scanned:
                return
            scanned.add(nid)
            node = node_map[nid]
            for line_no, line_text in _usages_in_node(node, symbol):
                all_hits.add((node.file, line_no, line_text))

        for nid in graph_caller_nids:
            scan(nid)
        for nid in node_map:
            scan(nid)

        if not all_hits:
            sections.append(f"## {label}\nNo usages of `{symbol}` found.\n")
            continue

        sorted_hits = sorted(all_hits, key=lambda x: (x[0], x[1]))

        lines = [f"## {label} — {len(sorted_hits)} usage(s) of `{symbol}`\n"]
        current_file = None
        for file, line_no, line_text in sorted_hits:
            if file != current_file:
                lines.append(f"\n**{file}**")
                current_file = file
            lines.append(f"  {line_no:>5}  {line_text}")

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


# ── MCP tool registry ─────────────────────────────────────────────────────────

_REPO_PATHS_SCHEMA = {
    "oneOf": [
        {"type": "string", "description": "Absolute path to a single repo root."},
        {
            "type": "array",
            "items": {"type": "string"},
            "description": "Absolute paths to multiple repo roots.",
            "minItems": 1,
        },
    ]
}

_DB_PATHS_SCHEMA = {
    "oneOf": [
        {"type": "string", "description": "ChromaDB directory for a single repo (default: ./chroma_db)."},
        {
            "type": "array",
            "items": {"type": "string"},
            "description": "ChromaDB directories, one per repo in repo_paths order.",
        },
    ]
}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="ask_question",
            description=(
                "Ask a natural-language question about one or more codebases. "
                "Uses hybrid vector+graph retrieval to find relevant code, "
                "then answers with Claude citing file paths and line numbers. "
                "Pass repo_paths as a list to query multiple repos at once. "
                "Requires each repo to have been indexed with embed.py first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_paths": _REPO_PATHS_SCHEMA,
                    "repo_path":  {"type": "string", "description": "Alias for repo_paths (single repo)."},
                    "question":   {"type": "string", "description": "The question to answer."},
                    "db_paths":   _DB_PATHS_SCHEMA,
                    "db_path":    {"type": "string", "description": "Alias for db_paths (single repo)."},
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="analyze_repo",
            description=(
                "Analyze one or more codebases for structural health: "
                "shows the top load-bearing nodes by PageRank and lists "
                "dead code candidates (functions/methods with no callers). "
                "Pass repo_paths as a list to analyze multiple repos at once. "
                "Does not require embed.py — works from source alone."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_paths": _REPO_PATHS_SCHEMA,
                    "repo_path":  {"type": "string", "description": "Alias for repo_paths (single repo)."},
                },
            },
        ),
        types.Tool(
            name="change_impact",
            description=(
                "Show the blast radius of changing a specific function, method, or class "
                "across one or more repos. Returns all nodes that transitively call the given "
                "node (up to 3 hops). Pass repo_paths as a list to search across multiple repos. "
                "Accepts a full node ID like 'src/utils.py::MyClass::my_method' "
                "or a partial name for fuzzy matching."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_paths": _REPO_PATHS_SCHEMA,
                    "repo_path":  {"type": "string", "description": "Alias for repo_paths (single repo)."},
                    "node_id":    {"type": "string", "description": "Full or partial node ID, e.g. 'utils.py::parse' or just 'parse'."},
                },
                "required": ["node_id"],
            },
        ),
        types.Tool(
            name="find_usages",
            description=(
                "Find every call site of a symbol (function, class, or method name) "
                "across one or more repos. Returns file paths and exact line numbers grouped by file. "
                "Uses a two-pass approach: graph edges identify likely callers first, "
                "then a whole-word text scan catches anything the graph missed (imports, aliases, etc.). "
                "Pass repo_paths as a list to search across multiple repos."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_paths": _REPO_PATHS_SCHEMA,
                    "repo_path":  {"type": "string", "description": "Alias for repo_paths (single repo)."},
                    "symbol":     {"type": "string", "description": "The symbol name to search for, e.g. 'resolve', 'CodeNode', 'extract_repo'."},
                },
                "required": ["symbol"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "ask_question":
        repo_paths = _normalize_repo_paths(arguments)
        db_paths   = _normalize_db_paths(arguments, repo_paths)
        question   = arguments["question"]
        result     = await asyncio.to_thread(_tool_ask_question, repo_paths, question, db_paths)

    elif name == "analyze_repo":
        repo_paths = _normalize_repo_paths(arguments)
        result     = await asyncio.to_thread(_tool_analyze_repo, repo_paths)

    elif name == "change_impact":
        repo_paths = _normalize_repo_paths(arguments)
        node_id    = arguments["node_id"]
        result     = await asyncio.to_thread(_tool_change_impact, repo_paths, node_id)

    elif name == "find_usages":
        repo_paths = _normalize_repo_paths(arguments)
        symbol     = arguments["symbol"]
        result     = await asyncio.to_thread(_tool_find_usages, repo_paths, symbol)

    else:
        result = f"Unknown tool: {name}"

    return [types.TextContent(type="text", text=result)]


# ── entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Warning: ANTHROPIC_API_KEY not set — ask_question tool will fail.",
            file=sys.stderr,
        )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
