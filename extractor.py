"""
extractor.py — parse Python files with tree-sitter into CodeNodes + raw call edges.

Does NOT resolve call targets to node IDs — that happens in resolve.py.
"""
from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from models import CodeNode, CodeEdge, NodeKind, EdgeKind, ParsedFile

PY_LANGUAGE = Language(tspython.language())


def _make_parser() -> Parser:
    return Parser(PY_LANGUAGE)


# ── helpers ──────────────────────────────────────────────────────────────────

def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_id(rel_path: str, *parts: str) -> str:
    """Stable node ID: 'rel/path.py' or 'rel/path.py::Class::method'."""
    segments = [rel_path] + [p for p in parts if p]
    return "::".join(segments)


def _docstring(body: Optional[Node], src: bytes) -> str:
    """Pull the first string literal from a function/class body as docstring."""
    if body is None:
        return ""
    for child in body.children:
        if child.type == "expression_statement" and child.children:
            expr = child.children[0]
            if expr.type == "string":
                raw = _text(expr, src)
                return raw.strip("'\" \n\t")
    return ""


def _collect_calls(fn_node: Node, src: bytes) -> list[str]:
    """
    Walk the entire function AST and return every call target name we see.
    Returns raw names only — resolution happens later.
    """
    calls: list[str] = []

    def walk(node: Node) -> None:
        if node.type == "call":
            func = node.child_by_field_name("function")
            if func:
                if func.type == "identifier":
                    calls.append(_text(func, src))
                elif func.type == "attribute":
                    obj  = func.child_by_field_name("object")
                    attr = func.child_by_field_name("attribute")
                    if attr:
                        calls.append(_text(attr, src))         # bare method name
                    if obj and attr:
                        calls.append(f"{_text(obj, src)}.{_text(attr, src)}")  # dotted
        for child in node.children:
            walk(child)

    walk(fn_node)
    return calls


def _collect_imports(root: Node, src: bytes) -> list[tuple[str, str]]:
    """
    Return (local_alias, dotted_path) for every import in the file.
      import os            →  ("os",   "os")
      from pathlib import Path → ("Path", "pathlib.Path")
      from . import utils  →  ("utils", ".utils")
    """
    pairs: list[tuple[str, str]] = []

    def walk(node: Node) -> None:
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    full = _text(child, src)
                    pairs.append((full.split(".")[-1], full))
                elif child.type == "aliased_import":
                    alias    = child.child_by_field_name("alias")
                    original = child.child_by_field_name("name")
                    if alias and original:
                        pairs.append((_text(alias, src), _text(original, src)))

        elif node.type == "import_from_statement":
            mod_node = node.child_by_field_name("module_name")
            module   = _text(mod_node, src) if mod_node else ""
            for child in node.children:
                if child.type == "dotted_name" and child is not mod_node:
                    name = _text(child, src)
                    pairs.append((name, f"{module}.{name}" if module else name))
                elif child.type == "aliased_import":
                    alias    = child.child_by_field_name("alias")
                    original = child.child_by_field_name("name")
                    if alias and original:
                        orig = _text(original, src)
                        pairs.append((_text(alias, src), f"{module}.{orig}" if module else orig))

        for child in node.children:
            walk(child)

    walk(root)
    return pairs


# ── per-file extraction ───────────────────────────────────────────────────────

def extract_file(path: Path, root: Path) -> ParsedFile:
    """Parse one .py file and return a ParsedFile (no resolution yet)."""
    src   = path.read_bytes()
    tree  = _make_parser().parse(src)
    rel   = str(path.relative_to(root))

    result = ParsedFile(path=rel)

    # Module-level node
    mod_id = _node_id(rel)
    result.nodes.append(CodeNode(
        id=mod_id,
        name=path.stem,
        kind=NodeKind.MODULE,
        file=rel,
        start_line=1,
        end_line=src.count(b"\n") + 1,
        source=src.decode("utf-8", errors="replace")[:3000],  # cap for embedding
    ))

    result.imports = _collect_imports(tree.root_node, src)

    # ── walk top-level declarations ──────────────────────────────────────────

    def process_function(fn: Node, class_name: Optional[str] = None) -> None:
        name_node = fn.child_by_field_name("name")
        body      = fn.child_by_field_name("body")
        if not name_node:
            return
        name = _text(name_node, src)
        nid  = _node_id(rel, class_name, name) if class_name else _node_id(rel, name)
        kind = NodeKind.METHOD if class_name else NodeKind.FUNCTION

        result.nodes.append(CodeNode(
            id=nid,
            name=name,
            kind=kind,
            file=rel,
            start_line=fn.start_point[0] + 1,
            end_line=fn.end_point[0] + 1,
            source=_text(fn, src),
            docstring=_docstring(body, src),
        ))

        for callee in _collect_calls(fn, src):
            result.raw_calls.append((nid, callee))

    def process_class(cls: Node) -> None:
        name_node = cls.child_by_field_name("name")
        body      = cls.child_by_field_name("body")
        if not name_node:
            return
        cls_name = _text(name_node, src)
        cid = _node_id(rel, cls_name)

        result.nodes.append(CodeNode(
            id=cid,
            name=cls_name,
            kind=NodeKind.CLASS,
            file=rel,
            start_line=cls.start_point[0] + 1,
            end_line=cls.end_point[0] + 1,
            source=_text(cls, src),
            docstring=_docstring(body, src),
        ))

        if body:
            for child in body.children:
                fn = _unwrap_decorated(child)
                if fn and fn.type == "function_definition":
                    process_function(fn, class_name=cls_name)

        # inheritance edges stored as raw_calls with special prefix
        for base in cls.children:
            if base.type == "argument_list":
                for arg in base.children:
                    if arg.type in ("identifier", "dotted_name"):
                        result.raw_calls.append((cid, f"__inherits__:{_text(arg, src)}"))

    def _unwrap_decorated(node: Node) -> Optional[Node]:
        if node.type == "function_definition":
            return node
        if node.type == "class_definition":
            return node
        if node.type == "decorated_definition":
            return node.child_by_field_name("definition")
        return None

    for child in tree.root_node.children:
        unwrapped = _unwrap_decorated(child)
        if unwrapped is None:
            continue
        if unwrapped.type == "function_definition":
            process_function(unwrapped)
        elif unwrapped.type == "class_definition":
            process_class(unwrapped)

    return result


# ── directory walk ────────────────────────────────────────────────────────────

def extract_repo(repo_root: str, exclude: Optional[set] = None, typescript: bool = True) -> list[ParsedFile]:
    """
    Walk repo_root, parse every .py (and optionally .ts/.tsx) file.
    Skips hidden dirs, __pycache__, and anything in `exclude`.
    """
    from extractor_ts import extract_ts_repo

    root = Path(repo_root).resolve()
    skip = (exclude or set()) | {"__pycache__", ".git", ".venv", "venv", "node_modules"}
    results = []

    for py_file in sorted(root.rglob("*.py")):
        if any(part.startswith(".") or part in skip for part in py_file.parts):
            continue
        try:
            results.append(extract_file(py_file, root))
        except Exception as exc:
            print(f"  [warn] skipping {py_file}: {exc}")

    if typescript:
        results.extend(extract_ts_repo(repo_root, exclude=exclude))

    return results


# ── quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rich import print as rprint

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files  = extract_repo(target)

    total_nodes = sum(len(f.nodes) for f in files)
    total_calls = sum(len(f.raw_calls) for f in files)

    rprint(f"\n[bold green]Parsed {len(files)} files[/bold green]")
    rprint(f"  nodes      : {total_nodes}")
    rprint(f"  raw calls  : {total_calls}")
    rprint()

    for pf in files[:5]:
        rprint(f"  [cyan]{pf.path}[/cyan]")
        for n in pf.nodes:
            rprint(f"    {n.kind.value:8s}  {n.name}  ({n.start_line}-{n.end_line})")
