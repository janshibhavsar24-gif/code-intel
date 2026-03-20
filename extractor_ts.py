"""
extractor_ts.py — parse TypeScript / TSX files with tree-sitter.

Mirrors the interface of extractor.py: returns ParsedFile objects
with CodeNodes and raw call edges (unresolved names).
"""
from pathlib import Path
from typing import Optional

import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser, Node

from models import CodeNode, NodeKind, ParsedFile

TS_LANGUAGE  = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())

# Which tree-sitter language to use per file extension
_LANG_FOR_EXT = {
    ".ts":  TS_LANGUAGE,
    ".tsx": TSX_LANGUAGE,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_id(rel_path: str, *parts: str) -> str:
    return "::".join([rel_path] + [p for p in parts if p])


def _collect_calls(fn_node: Node, src: bytes) -> list:
    """Walk the AST and return every call target name we see."""
    calls = []

    def walk(node: Node) -> None:
        if node.type == "call_expression":
            func = node.child_by_field_name("function")
            if func:
                if func.type == "identifier":
                    calls.append(_text(func, src))
                elif func.type == "member_expression":
                    obj  = func.child_by_field_name("object")
                    prop = func.child_by_field_name("property")
                    if prop:
                        calls.append(_text(prop, src))          # bare name
                    if obj and prop:
                        calls.append(f"{_text(obj, src)}.{_text(prop, src)}")
        for child in node.children:
            walk(child)

    walk(fn_node)
    return calls


def _collect_imports(root: Node, src: bytes) -> list:
    """
    Return (local_alias, source_module) pairs.
    e.g.  import { useWatchlists } from '../hooks/useWatchlists'
          → ("useWatchlists", "../hooks/useWatchlists")
    """
    pairs = []

    def walk(node: Node) -> None:
        if node.type == "import_statement":
            # find the quoted source path
            source = ""
            for child in node.children:
                if child.type == "string":
                    source = _text(child, src).strip("'\"")

            # find imported names
            for child in node.children:
                if child.type == "import_clause":
                    for sub in child.children:
                        if sub.type == "identifier":
                            # import Foo from "..."  (default import)
                            pairs.append((_text(sub, src), source))
                        elif sub.type == "named_imports":
                            for spec in sub.children:
                                if spec.type == "import_specifier":
                                    alias = spec.child_by_field_name("alias")
                                    name  = spec.child_by_field_name("name")
                                    local = alias or name
                                    if local:
                                        pairs.append((_text(local, src), source))
                        elif sub.type == "namespace_import":
                            # import * as Foo from "..."
                            for ns_child in sub.children:
                                if ns_child.type == "identifier":
                                    pairs.append((_text(ns_child, src), source))

        for child in node.children:
            walk(child)

    walk(root)
    return pairs


# ── per-file extraction ───────────────────────────────────────────────────────

def extract_ts_file(path: Path, root: Path) -> ParsedFile:
    """Parse one .ts/.tsx file and return a ParsedFile."""
    src  = path.read_bytes()
    lang = _LANG_FOR_EXT.get(path.suffix.lower(), TS_LANGUAGE)
    tree = Parser(lang).parse(src)
    rel  = str(path.relative_to(root))

    result = ParsedFile(path=rel)

    # Module-level node
    result.nodes.append(CodeNode(
        id=_node_id(rel),
        name=path.stem,
        kind=NodeKind.MODULE,
        file=rel,
        start_line=1,
        end_line=src.count(b"\n") + 1,
        source=src.decode("utf-8", errors="replace")[:3000],
    ))

    result.imports = _collect_imports(tree.root_node, src)

    # ── inner helpers (closures share `result`, `src`, `rel`) ────────────────

    def add_function(fn_node: Node, name: str, class_name: Optional[str] = None) -> None:
        nid  = _node_id(rel, class_name, name) if class_name else _node_id(rel, name)
        kind = NodeKind.METHOD if class_name else NodeKind.FUNCTION
        result.nodes.append(CodeNode(
            id=nid,
            name=name,
            kind=kind,
            file=rel,
            start_line=fn_node.start_point[0] + 1,
            end_line=fn_node.end_point[0] + 1,
            source=_text(fn_node, src),
        ))
        for callee in _collect_calls(fn_node, src):
            result.raw_calls.append((nid, callee))

    def add_class(cls_node: Node) -> None:
        name_node = cls_node.child_by_field_name("name")
        body      = cls_node.child_by_field_name("body")
        if not name_node:
            return
        cls_name = _text(name_node, src)
        result.nodes.append(CodeNode(
            id=_node_id(rel, cls_name),
            name=cls_name,
            kind=NodeKind.CLASS,
            file=rel,
            start_line=cls_node.start_point[0] + 1,
            end_line=cls_node.end_point[0] + 1,
            source=_text(cls_node, src),
        ))
        if body:
            for child in body.children:
                visit(child, class_name=cls_name)

    def visit(node: Node, class_name: Optional[str] = None) -> None:
        """Recursively handle a declaration node."""
        t = node.type

        if t in ("function_declaration", "generator_function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                add_function(node, _text(name_node, src), class_name)

        elif t == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                add_function(node, _text(name_node, src), class_name)

        elif t == "class_declaration" and class_name is None:
            add_class(node)

        elif t in ("lexical_declaration", "variable_declaration"):
            # const foo = () => {}  or  const foo = function() {}
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    val       = child.child_by_field_name("value")
                    if name_node and val and val.type in ("arrow_function", "function_expression"):
                        add_function(val, _text(name_node, src), class_name)

        elif t == "export_statement":
            # export function foo() {}
            # export const foo = () => {}
            # export default function foo() {}
            decl = node.child_by_field_name("declaration")
            if decl:
                visit(decl, class_name)
            else:
                for child in node.children:
                    if child.type in (
                        "function_declaration", "class_declaration",
                        "arrow_function", "function_expression",
                        "lexical_declaration", "variable_declaration",
                    ):
                        visit(child, class_name)

    for child in tree.root_node.children:
        visit(child)

    return result


# ── directory walk ────────────────────────────────────────────────────────────

def extract_ts_repo(repo_root: str, exclude: Optional[set] = None) -> list:
    """Walk repo_root for .ts/.tsx files and return list of ParsedFile."""
    root = Path(repo_root).resolve()
    skip = (exclude or set()) | {"__pycache__", ".git", ".venv", "venv", "node_modules", "dist", "build", ".next"}
    results = []

    for ext in ("*.ts", "*.tsx"):
        for f in sorted(root.rglob(ext)):
            if any(part.startswith(".") or part in skip for part in f.parts):
                continue
            # skip generated files
            if f.name.endswith(".d.ts"):
                continue
            try:
                results.append(extract_ts_file(f, root))
            except Exception as exc:
                print(f"  [warn] skipping {f}: {exc}")

    return results


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rich import print as rprint

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    files  = extract_ts_repo(target)

    total_nodes = sum(len(f.nodes) for f in files)
    total_calls = sum(len(f.raw_calls) for f in files)

    rprint(f"\n[bold green]Parsed {len(files)} TS/TSX files[/bold green]")
    rprint(f"  nodes     : {total_nodes}")
    rprint(f"  raw calls : {total_calls}")
    rprint()

    for pf in files[:5]:
        rprint(f"  [cyan]{pf.path}[/cyan]")
        for n in pf.nodes:
            rprint(f"    {n.kind.value:8s}  {n.name}  ({n.start_line}-{n.end_line})")
