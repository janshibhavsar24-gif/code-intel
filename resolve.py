"""
resolve.py — name resolution pass + NetworkX graph assembly.

After extractor.py gives us raw call names like "foo" or "Bar.baz",
this module figures out which node IDs those names actually refer to,
then builds a directed NetworkX graph you can run PageRank / BFS on.
"""
from pathlib import Path
from typing import Optional
import networkx as nx

from models import CodeNode, CodeEdge, EdgeKind, ParsedFile


# ── name resolution ───────────────────────────────────────────────────────────

def _build_name_index(all_files: list[ParsedFile]) -> dict[str, str]:
    """
    Build a reverse map: short_name → node_id (best guess).
    When multiple nodes share a name, last writer wins — good enough
    for a first pass; you can refine with import context later.
    """
    index: dict[str, str] = {}
    for pf in all_files:
        for node in pf.nodes:
            index[node.name] = node.id          # "foo"        → id
            # also index by "module.name" dotted form
            stem = Path(pf.path).stem
            index[f"{stem}.{node.name}"] = node.id  # "utils.foo" → id
    return index


def _module_path_to_file_key(dotted: str, all_files: list[ParsedFile]) -> Optional[str]:
    """
    Try to find a module node id matching a dotted import path.
    e.g. "myapp.utils" → "myapp/utils.py"
    """
    # convert dots to path separators and try suffix match
    as_path = dotted.replace(".", "/")
    for pf in all_files:
        stem = pf.path.removesuffix(".py").replace("\\", "/")
        if stem == as_path or stem.endswith("/" + as_path):
            return pf.path  # module node id == rel path
    return None


def resolve(all_files: list[ParsedFile]) -> tuple[list[CodeNode], list[CodeEdge]]:
    """
    Two-pass resolution:
      1. Build a name → node_id index across all files.
      2. For each raw_call, look up the callee name; emit an edge if found.

    Returns the flat lists of all nodes and all resolved edges.
    """
    all_nodes: list[CodeNode] = []
    node_ids:  set[str]       = set()

    for pf in all_files:
        for n in pf.nodes:
            if n.id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.id)

    name_index = _build_name_index(all_files)

    edges: set[CodeEdge] = set()

    for pf in all_files:
        # Build a local import map for this file: alias → node_id (or dotted path)
        local_imports: dict[str, str] = {}
        for alias, dotted in pf.imports:
            # try to resolve to a node directly
            if dotted in name_index:
                local_imports[alias] = name_index[dotted]
            else:
                # store dotted path; we'll try again during call resolution
                local_imports[alias] = dotted

        for (caller_id, raw_callee) in pf.raw_calls:
            # ── inheritance edges ──────────────────────────────────────────
            if raw_callee.startswith("__inherits__:"):
                base_name = raw_callee.removeprefix("__inherits__:")
                target = (
                    local_imports.get(base_name)
                    or name_index.get(base_name)
                )
                if target and target in node_ids:
                    edges.add(CodeEdge(src=caller_id, dst=target, kind=EdgeKind.INHERITS))
                continue

            # ── call edges ────────────────────────────────────────────────
            # Resolution order:
            #   1. local import alias (e.g. "Path" → "pathlib.Path" node)
            #   2. global name index (e.g. "foo" → "utils.py::foo")
            #   3. dotted form "obj.method" split
            resolved = (
                local_imports.get(raw_callee)
                or name_index.get(raw_callee)
            )

            # try dotted: "self.foo" → look up "foo"
            if resolved is None and "." in raw_callee:
                method_part = raw_callee.split(".")[-1]
                resolved = name_index.get(method_part)

            if resolved and resolved in node_ids and resolved != caller_id:
                edges.add(CodeEdge(src=caller_id, dst=resolved, kind=EdgeKind.CALLS))

        # ── import edges (module → module) ────────────────────────────────
        mod_id = pf.path   # module node id == rel path
        for alias, dotted in pf.imports:
            target_file = _module_path_to_file_key(dotted, all_files)
            if target_file and target_file != pf.path:
                edges.add(CodeEdge(src=mod_id, dst=target_file, kind=EdgeKind.IMPORTS))

    return all_nodes, list(edges)


# ── graph assembly ────────────────────────────────────────────────────────────

def build_graph(nodes: list[CodeNode], edges: list[CodeEdge]) -> nx.DiGraph:
    """
    Assemble a directed NetworkX graph.
    Each node carries its CodeNode as the 'data' attribute.
    """
    G = nx.DiGraph()

    for n in nodes:
        G.add_node(n.id, data=n, kind=n.kind.value, name=n.name, file=n.file)

    for e in edges:
        if e.src in G and e.dst in G:
            G.add_edge(e.src, e.dst, kind=e.kind.value)

    return G


# ── graph analytics ───────────────────────────────────────────────────────────

def coupling_scores(G: nx.DiGraph, alpha: float = 0.85, iterations: int = 50) -> dict[str, float]:
    """
    PageRank over the call graph — high score = load-bearing node.
    Pure-Python power iteration; no scipy needed.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    rank = {v: 1.0 / n for v in nodes}
    for _ in range(iterations):
        new_rank: dict[str, float] = {}
        for v in nodes:
            incoming = sum(rank[u] / max(G.out_degree(u), 1) for u in G.predecessors(v))
            new_rank[v] = (1 - alpha) / n + alpha * incoming
        rank = new_rank
    return rank


def dead_code(G: nx.DiGraph, entry_points: Optional[set] = None) -> list[str]:
    """
    Nodes with zero in-degree that are NOT declared entry points.
    These are candidates for dead code (nothing calls them).
    """
    entries = entry_points or set()
    return [
        nid for nid, deg in G.in_degree()
        if deg == 0
        and nid not in entries
        and G.nodes[nid].get("kind") in ("function", "method")
    ]


def change_impact(G: nx.DiGraph, node_id: str, max_hops: int = 3) -> list[str]:
    """
    Reverse BFS from node_id — returns everything that transitively calls it.
    These are the things that would break if you changed node_id.
    """
    reverse = G.reverse(copy=False)
    reached = nx.single_source_shortest_path_length(reverse, node_id, cutoff=max_hops)
    return [n for n in reached if n != node_id]


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rich import print as rprint
    from rich.table import Table
    from extractor import extract_repo

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files  = extract_repo(target)

    nodes, edges = resolve(files)
    G = build_graph(nodes, edges)

    rprint(f"\n[bold green]Graph built[/bold green]")
    rprint(f"  nodes : {G.number_of_nodes()}")
    rprint(f"  edges : {G.number_of_edges()}")

    # top 10 by PageRank
    scores = coupling_scores(G)
    top10  = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

    t = Table(title="Top 10 load-bearing nodes (PageRank)", show_lines=True)
    t.add_column("Score", style="yellow", width=8)
    t.add_column("Kind",  style="cyan",   width=10)
    t.add_column("ID",    style="white")
    for nid, score in top10:
        kind = G.nodes[nid].get("kind", "?")
        t.add_row(f"{score:.4f}", kind, nid)
    rprint(t)

    # dead code
    dead = dead_code(G)
    rprint(f"\n[bold red]Dead code candidates[/bold red]: {len(dead)}")
    for d in dead[:10]:
        rprint(f"  {d}")
