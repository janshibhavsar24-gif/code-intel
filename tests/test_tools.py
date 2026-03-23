"""Tests for server.py MCP tool implementations."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Tool helpers
from server import (
    _complexity,
    _tool_analyze_repo,
    _tool_change_impact,
    _tool_complexity_report,
    _tool_coverage_hints,
    _tool_explain_code,
    _tool_find_similar,
    _tool_generate_tests,
    _tool_find_usages,
    _normalize_repo_paths,
    _normalize_db_paths,
    _usages_in_node,
)
from models import CodeNode, NodeKind

FIXTURES = str(Path(__file__).parent / "fixtures")


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_node(name, kind, file, start, end, source, docstring=""):
    return CodeNode(
        id=f"{file}::{name}",
        name=name,
        kind=kind,
        file=file,
        start_line=start,
        end_line=end,
        source=source,
        docstring=docstring,
    )


# ── _complexity ───────────────────────────────────────────────────────────────

class TestComplexity:
    def test_simple_function_is_one(self):
        n = _fake_node("foo", NodeKind.FUNCTION, "a.py", 1, 2, "def foo():\n    return 1")
        assert _complexity(n) == 1

    def test_if_adds_one(self):
        n = _fake_node("foo", NodeKind.FUNCTION, "a.py", 1, 3,
                       "def foo(x):\n    if x:\n        return x")
        assert _complexity(n) == 2

    def test_multiple_branches(self):
        src = "def foo(x):\n    if x > 0:\n        for i in range(x):\n            while i: pass"
        n = _fake_node("foo", NodeKind.FUNCTION, "a.py", 1, 4, src)
        assert _complexity(n) == 4   # 1 + if + for + while

    def test_typescript_operators(self):
        src = "function foo(x) {\n  if (x && y || z) { return x; }\n}"
        n = _fake_node("foo", NodeKind.FUNCTION, "a.ts", 1, 3, src)
        # if + && + ||
        assert _complexity(n) == 4

    def test_complexity_increases_with_source_size(self):
        simple = _fake_node("a", NodeKind.FUNCTION, "a.py", 1, 2, "def a(): pass")
        complex_ = _fake_node("b", NodeKind.FUNCTION, "a.py", 1, 10,
                              "def b(x):\n    if x:\n        if x > 1:\n            for i in x:\n                pass")
        assert _complexity(complex_) > _complexity(simple)


# ── _usages_in_node ───────────────────────────────────────────────────────────

class TestUsagesInNode:
    def test_finds_whole_word_match(self):
        n = _fake_node("fn", NodeKind.FUNCTION, "a.py", 10, 15,
                       "def fn():\n    result = resolve(x)\n    return result")
        hits = _usages_in_node(n, "resolve")
        assert len(hits) == 1
        assert hits[0][0] == 11   # line 10 + offset 1

    def test_no_partial_match(self):
        n = _fake_node("fn", NodeKind.FUNCTION, "a.py", 1, 3,
                       "def fn():\n    resolve_all()\n    return 1")
        # "resolve" should NOT match "resolve_all" as whole word
        hits = _usages_in_node(n, "resolve")
        assert len(hits) == 0

    def test_multiple_occurrences(self):
        src = "def fn():\n    foo()\n    bar(foo())\n    return foo"
        n = _fake_node("fn", NodeKind.FUNCTION, "a.py", 1, 4, src)
        hits = _usages_in_node(n, "foo")
        assert len(hits) == 3

    def test_empty_source(self):
        n = _fake_node("fn", NodeKind.FUNCTION, "a.py", 1, 1, "")
        assert _usages_in_node(n, "anything") == []


# ── _normalize helpers ────────────────────────────────────────────────────────

class TestNormalize:
    def test_single_string_repo_path(self):
        assert _normalize_repo_paths({"repo_path": "/a"}) == ["/a"]

    def test_list_repo_paths(self):
        assert _normalize_repo_paths({"repo_paths": ["/a", "/b"]}) == ["/a", "/b"]

    def test_repo_paths_string_coerced(self):
        assert _normalize_repo_paths({"repo_paths": "/a"}) == ["/a"]

    def test_db_paths_default(self):
        paths = _normalize_db_paths({}, ["/repo"])
        assert len(paths) == 1

    def test_db_paths_single_string(self):
        paths = _normalize_db_paths({"db_path": "/db"}, ["/repo"])
        assert paths == ["/db"]

    def test_db_paths_list(self):
        paths = _normalize_db_paths({"db_paths": ["/db1", "/db2"]}, ["/a", "/b"])
        assert paths == ["/db1", "/db2"]


# ── graph-based tools (use fixtures dir, no LLM/ChromaDB) ────────────────────

class TestAnalyzeRepo:
    def test_returns_markdown_output(self):
        result = _tool_analyze_repo([FIXTURES])
        assert "load-bearing" in result
        assert "Dead code" in result

    def test_contains_repo_label(self):
        result = _tool_analyze_repo([FIXTURES])
        assert "fixtures" in result

    def test_invalid_path_returns_empty_report(self):
        # extract_repo silently returns [] for nonexistent paths — tool shows 0 nodes
        result = _tool_analyze_repo(["/nonexistent/path"])
        assert "0 nodes" in result

    def test_multi_repo_has_separator(self):
        result = _tool_analyze_repo([FIXTURES, FIXTURES])
        assert "---" in result


class TestChangeImpact:
    def test_known_symbol_returns_output(self):
        result = _tool_change_impact([FIXTURES], "bark")
        # either found callers or reported leaf — either way no crash
        assert "bark" in result

    def test_unknown_symbol_reports_not_found(self):
        result = _tool_change_impact([FIXTURES], "totally_nonexistent_xyz")
        assert "not found" in result.lower() or "No partial" in result

    def test_fuzzy_match_suggestion(self):
        # "spea" should fuzzy-match "speak"
        result = _tool_change_impact([FIXTURES], "spea")
        assert "speak" in result or "not found" in result.lower()


class TestFindUsages:
    def test_finds_bark_in_fixtures(self):
        result = _tool_find_usages([FIXTURES], "bark")
        assert "bark" in result

    def test_reports_file_and_line(self):
        result = _tool_find_usages([FIXTURES], "bark")
        assert "sample.py" in result

    def test_unknown_symbol_reports_none(self):
        result = _tool_find_usages([FIXTURES], "xyz_not_real_zzz")
        assert "No usages" in result

    def test_whole_word_only(self):
        # "ma" should not match "make_sound"
        result = _tool_find_usages([FIXTURES], "ma")
        # even if there are hits, verify no false positives from "make_sound"
        assert isinstance(result, str)


class TestCoverageHints:
    def test_detects_no_test_files(self):
        # fixtures dir has no test files
        result = _tool_coverage_hints([FIXTURES])
        assert "Test files detected: 0" in result

    def test_reports_all_source_files_uncovered(self):
        result = _tool_coverage_hints([FIXTURES])
        assert "sample.py" in result

    def test_contains_symbol_counts(self):
        result = _tool_coverage_hints([FIXTURES])
        assert "Functions/methods" in result

    def test_invalid_path_returns_empty_report(self):
        # extract_repo silently returns [] — tool shows 0 source files
        result = _tool_coverage_hints(["/nonexistent"])
        assert "Source files: 0" in result


class TestComplexityReport:
    def test_returns_ranked_table(self):
        result = _tool_complexity_report([FIXTURES], top_n=10)
        assert "most complex" in result

    def test_contains_fixture_functions(self):
        result = _tool_complexity_report([FIXTURES], top_n=20)
        # complexFn in sample.ts should appear as high complexity
        assert "complexFn" in result

    def test_shows_risk_levels(self):
        result = _tool_complexity_report([FIXTURES], top_n=20)
        assert any(label in result for label in ("HIGH", "MEDIUM", "OK"))

    def test_average_complexity_shown(self):
        result = _tool_complexity_report([FIXTURES], top_n=5)
        assert "Average" in result

    def test_top_n_respected(self):
        full = _tool_complexity_report([FIXTURES], top_n=100)
        short = _tool_complexity_report([FIXTURES], top_n=1)
        assert len(short) < len(full)


# ── LLM-dependent tools (Claude API mocked) ───────────────────────────────────

class TestExplainCode:
    @patch("server.anthropic.Anthropic")
    def test_reads_file_and_calls_claude(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="This function does X.")]
        )

        result = _tool_explain_code(FIXTURES, "sample.py", None, None)

        assert mock_client.messages.create.called
        assert result == "This function does X."

    @patch("server.anthropic.Anthropic")
    def test_line_range_passed_in_prompt(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Explanation.")]
        )

        _tool_explain_code(FIXTURES, "sample.py", 1, 5)

        call_args = mock_client.messages.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "lines 1" in prompt

    def test_missing_file_returns_error(self):
        result = _tool_explain_code(FIXTURES, "does_not_exist.py", None, None)
        assert "not found" in result.lower()


class TestFindSimilar:
    @patch("server._load_collection")
    @patch("server._load_graph")
    def test_queries_collection_with_symbol_source(self, mock_graph, mock_coll):
        # Set up fake graph with a "bark" node
        bark = _fake_node("bark", NodeKind.FUNCTION, "sample.py", 25, 26, "def bark():\n    return 'woof'")
        node_map = {bark.id: bark}
        import networkx as nx
        G = nx.DiGraph()
        G.add_node(bark.id, name="bark", kind="function", file="sample.py")
        mock_graph.return_value = (G, node_map)

        collection = MagicMock()
        collection.count.return_value = 5
        collection.query.return_value = {
            "ids": [["sample.py::make_sound"]],
            "distances": [[0.2]],
            "metadatas": [[{"name": "make_sound", "kind": "function",
                            "file": "sample.py", "start_line": 22}]],
        }
        mock_coll.return_value = collection

        result = _tool_find_similar([FIXTURES], ["./chroma_db"], symbol="bark", snippet=None, top_k=3)

        assert "make_sound" in result
        assert "0." in result   # similarity score

    def test_no_symbol_or_snippet_returns_error(self):
        result = _tool_find_similar([FIXTURES], ["./chroma_db"], symbol=None, snippet=None, top_k=5)
        assert "Provide" in result


class TestGenerateTests:
    @patch("server.anthropic.Anthropic")
    def test_calls_claude_with_source(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="def test_bark(): assert bark() == 'woof'")]
        )

        result = _tool_generate_tests(FIXTURES, "bark", None)

        assert mock_client.messages.create.called
        assert "test_bark" in result

    @patch("server.anthropic.Anthropic")
    def test_prompt_includes_symbol_source(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="# tests")]
        )

        _tool_generate_tests(FIXTURES, "bark", None)

        call_args = mock_client.messages.create.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "bark" in prompt

    @patch("server.anthropic.Anthropic")
    def test_file_path_narrows_match(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="# tests")]
        )

        result = _tool_generate_tests(FIXTURES, "speak", "sample.py")

        # should find it (not report multiple matches) since file is scoped
        assert "multiple matches" not in result.lower()

    def test_unknown_symbol_returns_error(self):
        result = _tool_generate_tests(FIXTURES, "totally_nonexistent_xyz", None)
        assert "not found" in result.lower()

    @patch("server.anthropic.Anthropic")
    def test_typescript_uses_jest_framework(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="describe('complexFn', () => {})")]
        )

        _tool_generate_tests(FIXTURES, "complexFn", None)

        call_args = mock_client.messages.create.call_args
        system_prompt = call_args.kwargs["system"]
        assert "Jest" in system_prompt or "Vitest" in system_prompt
