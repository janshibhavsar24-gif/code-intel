"""Tests for extractor.py — Python AST extraction."""
from pathlib import Path

import pytest

from extractor import extract_file, extract_repo
from models import NodeKind


class TestExtractFile:
    def test_module_node_created(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        kinds = [n.kind for n in pf.nodes]
        assert NodeKind.MODULE in kinds

    def test_classes_extracted(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        class_names = {n.name for n in pf.nodes if n.kind == NodeKind.CLASS}
        assert "Animal" in class_names
        assert "Dog" in class_names

    def test_methods_extracted(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        method_names = {n.name for n in pf.nodes if n.kind == NodeKind.METHOD}
        assert "speak" in method_names
        assert "breathe" in method_names

    def test_functions_extracted(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        fn_names = {n.name for n in pf.nodes if n.kind == NodeKind.FUNCTION}
        assert "make_sound" in fn_names
        assert "bark" in fn_names

    def test_docstrings_extracted(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        animal = next(n for n in pf.nodes if n.name == "Animal")
        assert "Base animal class" in animal.docstring

    def test_raw_calls_collected(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        all_callees = {callee for _, callee in pf.raw_calls}
        assert "bark" in all_callees
        assert "make_sound" in all_callees

    def test_inheritance_recorded(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        inherit_calls = [callee for _, callee in pf.raw_calls if callee.startswith("__inherits__")]
        assert any("Animal" in c for c in inherit_calls)

    def test_line_numbers_correct(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        bark = next(n for n in pf.nodes if n.name == "bark")
        assert bark.start_line > 0
        assert bark.end_line >= bark.start_line

    def test_relative_path_stored(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        assert pf.path == "sample.py"
        assert all(n.file == "sample.py" for n in pf.nodes)

    def test_node_ids_are_stable(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        bark = next(n for n in pf.nodes if n.name == "bark")
        assert bark.id == "sample.py::bark"

    def test_method_id_includes_class(self, sample_py, fixtures_dir):
        pf = extract_file(sample_py, fixtures_dir)
        speak_methods = [n for n in pf.nodes if n.name == "speak" and n.kind == NodeKind.METHOD]
        ids = {n.id for n in speak_methods}
        assert any("Animal" in nid for nid in ids)
        assert any("Dog" in nid for nid in ids)


class TestExtractRepo:
    def test_extracts_multiple_files(self, fixtures_dir):
        results = extract_repo(str(fixtures_dir), typescript=False)
        assert len(results) >= 1

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("def secret(): pass")
        results = extract_repo(str(tmp_path), typescript=False)
        paths = [pf.path for pf in results]
        assert not any(".hidden" in p for p in paths)

    def test_skips_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "compiled.py").write_text("x = 1")
        results = extract_repo(str(tmp_path), typescript=False)
        paths = [pf.path for pf in results]
        assert not any("__pycache__" in p for p in paths)

    def test_includes_typescript_when_enabled(self, fixtures_dir):
        results = extract_repo(str(fixtures_dir), typescript=True)
        ts_files = [pf for pf in results if pf.path.endswith(".ts")]
        assert len(ts_files) >= 1
