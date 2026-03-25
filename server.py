"""
server.py — MCP server exposing code-intel as Claude-callable tools.

Tools:
  ask_question   — hybrid vector+graph Q&A answered by Claude (single or multi-repo)
  analyze_repo   — coupling scores + dead code report (single or multi-repo)
  change_impact  — blast radius for a given node ID (single or multi-repo)
  find_usages    — all call sites of a symbol across one or more repos
  explain_code   — plain-English walkthrough of a file or line range
  coverage_hints — identify untested files and functions
  find_similar      — find semantically similar code via vector embeddings
  complexity_report — cyclomatic complexity ranked by risk
  generate_tests    — generate pytest/Jest test skeleton for a function or class
  dependency_map    — external package usage report (single or multi-repo)

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
_files_cache: dict[str, list]       = {}   # repo_path → list[ParsedFile]


def _load_graph(repo_path: str) -> tuple:
    key = str(Path(repo_path).resolve())
    if key not in _graph_cache:
        files        = extract_repo(repo_path)
        nodes, edges = resolve(files)
        G            = build_graph(nodes, edges)
        node_map     = {n.id: n for n in nodes}
        _graph_cache[key]  = (G, node_map)
        _files_cache[key]  = files
    return _graph_cache[key]


def _load_files(repo_path: str) -> list:
    key = str(Path(repo_path).resolve())
    if key not in _files_cache:
        _load_graph(repo_path)   # populates both caches
    return _files_cache[key]


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


def _tool_explain_code(repo_path: str, file_path: str,
                       start_line: int | None, end_line: int | None) -> str:
    root     = Path(repo_path).resolve()
    abs_path = (root / file_path).resolve()

    if not abs_path.exists():
        # try interpreting file_path as already absolute
        abs_path = Path(file_path).resolve()
    if not abs_path.exists():
        return f"File not found: `{file_path}` (looked in {root})"

    all_lines = abs_path.read_text(errors="replace").splitlines()
    total     = len(all_lines)

    # normalise line range (1-indexed, inclusive)
    s = max(1, start_line or 1)
    e = min(total, end_line or total)
    snippet   = "\n".join(f"{s + i:>5}  {ln}" for i, ln in enumerate(all_lines[s - 1 : e]))

    # relative path for graph lookup
    try:
        rel = str(abs_path.relative_to(root))
    except ValueError:
        rel = abs_path.name

    # pull graph neighbours that overlap the requested range for extra context
    graph_context = ""
    try:
        G, node_map = _load_graph(repo_path)
        overlapping = [
            n for n in node_map.values()
            if n.file == rel and n.start_line <= e and n.end_line >= s
        ]
        if overlapping:
            neighbor_ids: set[str] = set()
            for n in overlapping:
                neighbor_ids.update(G.predecessors(n.id))
                neighbor_ids.update(G.successors(n.id))
            neighbor_ids -= {n.id for n in overlapping}

            parts = []
            for nid in list(neighbor_ids)[:8]:   # cap at 8 neighbours
                nb = node_map.get(nid)
                if nb:
                    parts.append(
                        f"// {nb.kind.value} {nb.name}  ({nb.file}:{nb.start_line})\n"
                        f"{nb.source[:800]}"
                    )
            if parts:
                graph_context = "\n\n---\nRelated code (callers / callees):\n\n" + "\n\n".join(parts)
    except Exception:
        pass   # graph context is best-effort

    range_desc = f"lines {s}–{e}" if (start_line or end_line) else "entire file"
    label      = rel

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1536,
        system=(
            "You are a code explanation assistant. Given a code snippet and optional "
            "context about its callers and callees, produce a clear plain-English walkthrough. "
            "Cover: what it does, how it works step by step, key design decisions, "
            "and how it connects to the surrounding code. Be concrete — use names and line numbers."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"File: {label}  ({range_desc})\n\n"
                f"```\n{snippet}\n```"
                f"{graph_context}"
            ),
        }],
    )
    return msg.content[0].text


def _tool_find_similar(repo_paths: list[str], db_paths: list[str],
                       symbol: str | None, snippet: str | None,
                       top_k: int) -> str:
    if not symbol and not snippet:
        return "Provide either `symbol` (a function/class name) or `snippet` (raw code)."

    sections = []

    for repo_path, db_path in zip(repo_paths, db_paths):
        label = Path(repo_path).name
        try:
            _, node_map  = _load_graph(repo_path)
            collection   = _load_collection(repo_path, db_path)
        except Exception as exc:
            sections.append(
                f"## {label}\n"
                f"Failed to load repo or collection: {exc}\n"
                f"Make sure you have run: python embed.py {repo_path} --db {db_path}\n"
            )
            continue

        # resolve query text
        query_node_id: str | None = None
        if symbol:
            matches = [n for n in node_map.values() if n.name == symbol]
            if not matches:
                # fuzzy fallback
                matches = [n for n in node_map.values() if symbol in n.name]
            if not matches:
                sections.append(f"## {label}\nSymbol `{symbol}` not found.\n")
                continue
            # prefer functions/methods over modules
            matches.sort(key=lambda n: 0 if n.kind.value in ("function", "method") else 1)
            query_node    = matches[0]
            query_node_id = query_node.id
            query_text    = f"{query_node.kind.value} {query_node.name}\n{query_node.source}"
        else:
            query_text = snippet  # type: ignore[assignment]

        # query ChromaDB — fetch top_k + 1 to account for the query node itself
        results = collection.query(
            query_texts=[query_text],
            n_results=min(top_k + 1, collection.count()),
            include=["metadatas", "distances"],
        )

        ids       = results["ids"][0]
        distances = results["distances"][0]
        metas     = results["metadatas"][0]

        lines = [f"## {label} — top {top_k} similar to `{symbol or 'snippet'}`\n"]
        lines.append(f"{'#':<4} {'Score':<7} {'Kind':<10} {'Name':<30} Location")
        lines.append("-" * 75)

        rank = 0
        for nid, dist, meta in zip(ids, distances, metas):
            if nid == query_node_id:
                continue   # skip the query node itself
            # convert L2 distance → 0-1 similarity (1 = identical)
            similarity = 1.0 / (1.0 + dist)
            rank += 1
            name     = meta.get("name", nid)
            kind     = meta.get("kind", "?")
            file     = meta.get("file", "?")
            line     = meta.get("start_line", "?")
            lines.append(f"{rank:<4} {similarity:.3f}   {kind:<10} {name:<30} {file}:{line}")
            if rank >= top_k:
                break

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


# ── cyclomatic complexity ─────────────────────────────────────────────────────

# Decision-point keywords / tokens that add a branch, per language
_PY_BRANCH  = re.compile(r'\b(if|elif|for|while|except|and|or|case)\b')
_TS_BRANCH  = re.compile(r'\b(if|else\s+if|for|while|catch|case|switch)\b|&&|\|\||\?(?![\?:])')


def _complexity(node) -> int:
    """
    Cyclomatic complexity ≈ 1 + number of branch-inducing tokens in source.
    Uses a Python pattern for .py files, TypeScript pattern for .ts/.tsx.
    """
    is_ts = node.file.endswith((".ts", ".tsx"))
    pat   = _TS_BRANCH if is_ts else _PY_BRANCH
    return 1 + len(pat.findall(node.source))


_RISK_HIGH   = 10   # CC > 10 → high risk, hard to test
_RISK_MEDIUM = 5    # CC > 5  → moderate, worth reviewing


def _tool_complexity_report(repo_paths: list[str], top_n: int) -> str:
    sections = []

    for repo_path in repo_paths:
        label = Path(repo_path).name
        try:
            _, node_map = _load_graph(repo_path)
        except Exception as exc:
            sections.append(f"## {label}\nFailed to load repo: {exc}\n")
            continue

        # compute complexity for every function / method
        scored = [
            (n, _complexity(n))
            for n in node_map.values()
            if n.kind.value in ("function", "method")
            and not _TEST_PAT.search(n.file)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        high   = [(n, c) for n, c in scored if c > _RISK_HIGH]
        medium = [(n, c) for n, c in scored if _RISK_MEDIUM < c <= _RISK_HIGH]
        avg    = sum(c for _, c in scored) / len(scored) if scored else 0

        lines = [
            f"## {label}\n",
            f"Functions/methods analysed : {len(scored)}",
            f"Average complexity          : {avg:.1f}",
            f"High risk  (CC > {_RISK_HIGH})     : {len(high)}",
            f"Medium risk (CC > {_RISK_MEDIUM})      : {len(medium)}",
        ]

        lines.append(f"\n### Top {top_n} most complex functions\n")
        lines.append(f"{'CC':<5} {'Risk':<8} {'Kind':<8} {'Name':<35} Location")
        lines.append("-" * 80)

        for n, cc in scored[:top_n]:
            risk = "🔴 HIGH  " if cc > _RISK_HIGH else ("🟡 MEDIUM" if cc > _RISK_MEDIUM else "🟢 OK    ")
            lines.append(f"{cc:<5} {risk}  {n.kind.value:<8} {n.name:<35} {n.file}:{n.start_line}")

        if high:
            lines.append(f"\n### All high-risk functions (CC > {_RISK_HIGH})\n")
            for n, cc in high:
                lines.append(f"  CC={cc:<4} {n.file}:{n.start_line:<6} {n.kind.value}  {n.name}")

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


def _tool_generate_tests(repo_path: str, symbol: str, file_path: str | None) -> str:
    try:
        G, node_map = _load_graph(repo_path)
    except Exception as exc:
        return f"Failed to load repo: {exc}"

    # Find matching nodes — prefer exact name match, optionally scoped to file
    candidates = [
        n for n in node_map.values()
        if n.name == symbol
        and n.kind.value in ("function", "method", "class")
        and (file_path is None or file_path in n.file)
    ]
    if not candidates:
        # fuzzy fallback
        candidates = [
            n for n in node_map.values()
            if symbol in n.name
            and n.kind.value in ("function", "method", "class")
            and (file_path is None or file_path in n.file)
        ]
    if not candidates:
        return f"Symbol `{symbol}` not found in repo."
    if len(candidates) > 1 and file_path is None:
        locs = "\n".join(f"  - {n.file}:{n.start_line}  {n.kind.value}  {n.name}" for n in candidates[:10])
        return f"Multiple matches for `{symbol}`. Narrow with file_path:\n{locs}"

    target = candidates[0]
    is_ts  = target.file.endswith((".ts", ".tsx"))
    framework = "Jest/Vitest" if is_ts else "pytest"
    lang      = "TypeScript" if is_ts else "Python"

    # Gather callers and callees for context
    callers  = [node_map[nid] for nid in G.predecessors(target.id) if nid in node_map][:5]
    callees  = [node_map[nid] for nid in G.successors(target.id)  if nid in node_map][:5]

    context_parts = []
    if callers:
        context_parts.append("### Callers (how it is used)\n" +
            "\n".join(f"// {n.file}:{n.start_line}\n{n.source[:400]}" for n in callers))
    if callees:
        context_parts.append("### Callees (what it calls)\n" +
            "\n".join(f"// {n.file}:{n.start_line}\n{n.source[:400]}" for n in callees))

    context_block = ("\n\n" + "\n\n".join(context_parts)) if context_parts else ""

    cc = _complexity(target)

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=(
            f"You are an expert {lang} test writer. "
            f"Generate a complete, runnable {framework} test file for the given code. "
            "Requirements:\n"
            "- Cover the happy path, edge cases, and error/boundary conditions\n"
            "- Mock external dependencies (DB calls, HTTP, filesystem) where needed\n"
            "- Use descriptive test names that explain what is being tested\n"
            "- Include setup/teardown if needed\n"
            "- Add a brief comment above each test explaining its purpose\n"
            "- Output ONLY the test file code, no explanation outside the code"
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Generate tests for this {lang} {target.kind.value}.\n"
                f"File: {target.file}  lines: {target.start_line}-{target.end_line}  "
                f"cyclomatic complexity: {cc}\n\n"
                f"```{('typescript' if is_ts else 'python')}\n{target.source}\n```"
                f"{context_block}"
            ),
        }],
    )
    return msg.content[0].text


# test file pattern — matches common conventions across Python and TS/TSX
_TEST_PAT = re.compile(
    r"(^|/)test_|_test\.(py|ts|tsx)$"
    r"|\.spec\.(ts|tsx)$|\.test\.(ts|tsx)$"
    r"|/tests/|/test/|/e2e/"
)


def _tool_coverage_hints(repo_paths: list[str]) -> str:
    sections = []

    for repo_path in repo_paths:
        label = Path(repo_path).name
        try:
            G, node_map = _load_graph(repo_path)
        except Exception as exc:
            sections.append(f"## {label}\nFailed to load repo: {exc}\n")
            continue

        src_files  = sorted({
            n.file for n in node_map.values()
            if not _TEST_PAT.search(n.file)
            and n.file.endswith((".py", ".ts", ".tsx"))
        })
        test_files = sorted({
            n.file for n in node_map.values()
            if _TEST_PAT.search(n.file)
        })

        # ── file-level coverage ───────────────────────────────────────────────
        # A source file is "covered" if any test file contains its stem in its name
        def _is_file_covered(src: str) -> bool:
            stem = Path(src).stem.lower().lstrip("_")
            return any(stem in Path(tf).name.lower() for tf in test_files)

        uncovered_files = [f for f in src_files if not _is_file_covered(f)]

        # ── symbol-level coverage ─────────────────────────────────────────────
        # BFS from all test nodes following CALLS edges — everything reachable = tested
        test_node_ids = {nid for nid, d in G.nodes(data=True) if _TEST_PAT.search(d.get("file", ""))}

        tested: set[str] = set(test_node_ids)
        frontier = list(test_node_ids)
        while frontier:
            nxt = []
            for nid in frontier:
                for successor in G.successors(nid):
                    if successor not in tested:
                        tested.add(successor)
                        nxt.append(successor)
            frontier = nxt

        # untested = source functions/methods never reached from any test
        scores = coupling_scores(G)
        untested_symbols = sorted(
            [
                n for n in node_map.values()
                if n.id not in tested
                and not _TEST_PAT.search(n.file)
                and n.kind.value in ("function", "method")
            ],
            key=lambda n: scores.get(n.id, 0),
            reverse=True,   # most load-bearing first
        )

        # ── format output ─────────────────────────────────────────────────────
        total_src  = len(src_files)
        total_syms = sum(
            1 for n in node_map.values()
            if not _TEST_PAT.search(n.file) and n.kind.value in ("function", "method")
        )
        tested_syms = total_syms - len(untested_symbols)

        lines = [
            f"## {label}\n",
            f"Test files detected: {len(test_files)}",
            f"Source files: {total_src}  |  without a test file: {len(uncovered_files)}",
            f"Functions/methods: {total_syms}  |  unreachable from tests: {len(untested_symbols)}",
        ]

        if uncovered_files:
            lines.append(f"\n### Source files with no corresponding test file ({len(uncovered_files)})\n")
            for f in uncovered_files[:30]:
                lines.append(f"  {f}")
            if len(uncovered_files) > 30:
                lines.append(f"  … and {len(uncovered_files) - 30} more")

        if untested_symbols:
            lines.append(f"\n### Top untested functions/methods (by PageRank)\n")
            for n in untested_symbols[:25]:
                lines.append(f"  {n.file}:{n.start_line:<5}  {n.kind.value:<8}  {n.name}")
            if len(untested_symbols) > 25:
                lines.append(f"  … and {len(untested_symbols) - 25} more")

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


# ── dependency map ────────────────────────────────────────────────────────────

import sys as _sys
_STDLIB = getattr(_sys, "stdlib_module_names", frozenset())


def _is_external_py(dotted: str, repo_names: set[str]) -> bool:
    """True if a Python import is an external (third-party) package."""
    if dotted.startswith("."):
        return False
    top = dotted.split(".")[0]
    if top in _STDLIB:
        return False
    return top not in repo_names


def _is_external_ts(source: str) -> bool:
    """True if a TypeScript import path is an external package."""
    return not source.startswith((".", "/", "@/", "~/", "#"))


def _pkg_name_ts(source: str) -> str:
    """Extract the npm package name from a TS import path."""
    if source.startswith("@"):
        parts = source.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else source
    return source.split("/")[0]


def _tool_dependency_map(repo_paths: list[str]) -> str:
    from collections import defaultdict

    sections = []

    for repo_path in repo_paths:
        label = Path(repo_path).name
        try:
            files = _load_files(repo_path)
        except Exception as exc:
            sections.append(f"## {label}\nFailed to load repo: {exc}\n")
            continue

        # Build set of internal module/package names for Python classification
        repo_names: set[str] = set()
        for pf in files:
            p = Path(pf.path)
            repo_names.add(p.stem)
            for part in p.parts[:-1]:   # add every parent dir as a potential package name
                repo_names.add(part)

        # package → { file → [imported symbols] }
        deps: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

        for pf in files:
            is_ts = pf.path.endswith((".ts", ".tsx"))
            for alias, source in pf.imports:
                if is_ts:
                    if not _is_external_ts(source):
                        continue
                    pkg = _pkg_name_ts(source)
                else:
                    if not _is_external_py(source, repo_names):
                        continue
                    pkg = source.split(".")[0]

                deps[pkg][pf.path].append(alias)

        if not deps:
            sections.append(f"## {label}\nNo external dependencies detected.\n")
            continue

        sorted_deps = sorted(deps.items(), key=lambda x: len(x[1]), reverse=True)

        lines = [f"## {label} — {len(sorted_deps)} external package(s)\n"]
        for pkg, file_map in sorted_deps:
            file_count = len(file_map)
            all_symbols = sorted({s for syms in file_map.values() for s in syms})
            sym_preview = ", ".join(all_symbols[:6])
            if len(all_symbols) > 6:
                sym_preview += f", +{len(all_symbols) - 6} more"
            lines.append(f"\n**{pkg}** — {file_count} file(s)  [{sym_preview}]")
            for filepath in sorted(file_map):
                syms = sorted(set(file_map[filepath]))
                lines.append(f"  {filepath}  ({', '.join(syms[:4])}{'...' if len(syms) > 4 else ''})")

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
        types.Tool(
            name="explain_code",
            description=(
                "Produce a plain-English walkthrough of a file or specific line range. "
                "Automatically enriches the explanation with graph context: "
                "what the code calls and what calls it. "
                "Omit start_line/end_line to explain the entire file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file within the repo, e.g. 'backend/routes/watchlist.py'.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to explain (1-indexed, inclusive). Omit for start of file.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to explain (1-indexed, inclusive). Omit for end of file.",
                    },
                },
                "required": ["repo_path", "file_path"],
            },
        ),
        types.Tool(
            name="coverage_hints",
            description=(
                "Identify gaps in test coverage across one or more repos. "
                "Reports: (1) source files with no corresponding test file, "
                "(2) functions and methods never reachable from any test via BFS on the call graph, "
                "ranked by PageRank so the most critical untested code surfaces first. "
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
            name="find_similar",
            description=(
                "Find semantically similar functions or classes using vector embeddings. "
                "Given a symbol name or a raw code snippet, returns the most similar "
                "code in the repo ranked by cosine similarity. "
                "Useful for spotting duplication, finding prior art, or discovering "
                "related implementations before writing new code. "
                "Requires the repo to have been indexed with embed.py first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_paths": _REPO_PATHS_SCHEMA,
                    "repo_path":  {"type": "string", "description": "Alias for repo_paths (single repo)."},
                    "symbol": {
                        "type": "string",
                        "description": "Name of a function or class to use as the similarity query.",
                    },
                    "snippet": {
                        "type": "string",
                        "description": "Raw code snippet to use as the similarity query (alternative to symbol).",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar results to return (default: 8).",
                    },
                    "db_paths": _DB_PATHS_SCHEMA,
                    "db_path":  {"type": "string", "description": "Alias for db_paths (single repo)."},
                },
            },
        ),
        types.Tool(
            name="complexity_report",
            description=(
                "Compute cyclomatic complexity for every function and method in one or more repos. "
                "Ranks by complexity and flags high-risk (CC > 10) and medium-risk (CC > 5) code. "
                "High complexity + no tests (from coverage_hints) = highest refactor priority. "
                "Works on Python and TypeScript/TSX. Does not require embed.py."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_paths": _REPO_PATHS_SCHEMA,
                    "repo_path":  {"type": "string", "description": "Alias for repo_paths (single repo)."},
                    "top_n": {
                        "type": "integer",
                        "description": "How many functions to show in the ranked table (default: 20).",
                    },
                },
            },
        ),
        types.Tool(
            name="generate_tests",
            description=(
                "Generate a complete, runnable pytest (Python) or Jest/Vitest (TypeScript) "
                "test file for a given function or class. "
                "Covers happy path, edge cases, and error conditions. "
                "Automatically includes caller/callee context so mocks are accurate. "
                "If multiple matches exist for the symbol, pass file_path to disambiguate."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Absolute path to the repo root.",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Function, method, or class name to generate tests for.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional relative file path to disambiguate when symbol appears in multiple files.",
                    },
                },
                "required": ["repo_path", "symbol"],
            },
        ),
        types.Tool(
            name="dependency_map",
            description=(
                "Show every external (third-party) package imported across one or more repos, "
                "how many files use each package, and which symbols are imported. "
                "Useful before upgrades, audits, or licence reviews. "
                "Works on Python and TypeScript/TSX. Does not require embed.py."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_paths": _REPO_PATHS_SCHEMA,
                    "repo_path":  {"type": "string", "description": "Alias for repo_paths (single repo)."},
                },
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

    elif name == "explain_code":
        repo_path  = arguments["repo_path"]
        file_path  = arguments["file_path"]
        start_line = arguments.get("start_line")
        end_line   = arguments.get("end_line")
        result     = await asyncio.to_thread(_tool_explain_code, repo_path, file_path, start_line, end_line)

    elif name == "coverage_hints":
        repo_paths = _normalize_repo_paths(arguments)
        result     = await asyncio.to_thread(_tool_coverage_hints, repo_paths)

    elif name == "find_similar":
        repo_paths = _normalize_repo_paths(arguments)
        db_paths   = _normalize_db_paths(arguments, repo_paths)
        symbol     = arguments.get("symbol")
        snippet    = arguments.get("snippet")
        top_k      = int(arguments.get("top_k", 8))
        result     = await asyncio.to_thread(_tool_find_similar, repo_paths, db_paths, symbol, snippet, top_k)

    elif name == "complexity_report":
        repo_paths = _normalize_repo_paths(arguments)
        top_n      = int(arguments.get("top_n", 20))
        result     = await asyncio.to_thread(_tool_complexity_report, repo_paths, top_n)

    elif name == "generate_tests":
        repo_path = arguments["repo_path"]
        symbol    = arguments["symbol"]
        file_path = arguments.get("file_path")
        result    = await asyncio.to_thread(_tool_generate_tests, repo_path, symbol, file_path)

    elif name == "dependency_map":
        repo_paths = _normalize_repo_paths(arguments)
        result     = await asyncio.to_thread(_tool_dependency_map, repo_paths)

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
