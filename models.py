from dataclasses import dataclass, field
from enum import Enum


class NodeKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class EdgeKind(str, Enum):
    CALLS = "calls"       # function A calls function B
    IMPORTS = "imports"   # module A imports from module B
    DEFINES = "defines"   # class A defines method B
    INHERITS = "inherits" # class A inherits from class B


@dataclass
class CodeNode:
    # Stable ID format: "rel/path.py::ClassName::method_name"
    id: str
    name: str
    kind: NodeKind
    file: str         # relative path from repo root
    start_line: int
    end_line: int
    source: str       # raw text (used for embedding)
    docstring: str = ""

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class CodeEdge:
    src: str      # source node id
    dst: str      # destination node id
    kind: EdgeKind

    def __hash__(self):
        return hash((self.src, self.dst, self.kind))

    def __eq__(self, other):
        return (self.src, self.dst, self.kind) == (other.src, other.dst, other.kind)


@dataclass
class ParsedFile:
    """Intermediate result from parsing one file before name resolution."""
    path: str                              # relative file path
    nodes: list[CodeNode] = field(default_factory=list)
    # (caller_id, raw_callee_name) — callee not yet resolved to a node id
    raw_calls: list[tuple[str, str]] = field(default_factory=list)
    # (local_alias, dotted_import_path) e.g. ("Path", "pathlib.Path")
    imports: list[tuple[str, str]] = field(default_factory=list)
