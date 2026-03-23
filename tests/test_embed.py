"""Tests for embed.py — file hash cache and incremental indexing logic."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from embed import _file_hash, _ids_for_file, _load_hashes, _remove_file_nodes, _save_hashes, index


class TestFileHash:
    def test_consistent_hash(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo(): pass")
        assert _file_hash(f) == _file_hash(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("def foo(): pass")
        f2.write_text("def bar(): pass")
        assert _file_hash(f1) != _file_hash(f2)

    def test_hash_is_string(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        assert isinstance(_file_hash(f), str)


class TestHashPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        hashes = {"a.py": "abc123", "b.py": "def456"}
        _save_hashes(str(tmp_path), hashes)
        loaded = _load_hashes(str(tmp_path))
        assert loaded == hashes

    def test_load_missing_returns_empty(self, tmp_path):
        result = _load_hashes(str(tmp_path / "nonexistent"))
        assert result == {}

    def test_save_creates_directory(self, tmp_path):
        db = tmp_path / "new_db"
        _save_hashes(str(db), {"x.py": "hash"})
        assert (db / "file_hashes.json").exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        _save_hashes(str(tmp_path), {"f.py": "123"})
        raw = (tmp_path / "file_hashes.json").read_text()
        parsed = json.loads(raw)
        assert parsed == {"f.py": "123"}


class TestCollectionHelpers:
    def test_ids_for_file_queries_by_metadata(self):
        collection = MagicMock()
        collection.get.return_value = {"ids": ["a::foo", "a::bar"]}
        result = _ids_for_file(collection, "a.py")
        collection.get.assert_called_once_with(where={"file": "a.py"})
        assert result == ["a::foo", "a::bar"]

    def test_remove_file_nodes_deletes_ids(self):
        collection = MagicMock()
        collection.get.return_value = {"ids": ["a::foo"]}
        count = _remove_file_nodes(collection, "a.py")
        collection.delete.assert_called_once_with(ids=["a::foo"])
        assert count == 1

    def test_remove_file_nodes_skips_delete_when_empty(self):
        collection = MagicMock()
        collection.get.return_value = {"ids": []}
        count = _remove_file_nodes(collection, "missing.py")
        collection.delete.assert_not_called()
        assert count == 0


class TestIncrementalIndex:
    """Integration-style tests for the index() function with mocked ChromaDB."""

    def _make_collection(self, existing_ids=None):
        col = MagicMock()
        col.count.return_value = len(existing_ids or [])
        col.get.return_value = {"ids": existing_ids or []}
        return col

    def _make_client(self, collection):
        client = MagicMock()
        client.get_collection.return_value = collection
        return client

    @patch("embed.chromadb.PersistentClient")
    def test_first_run_indexes_all_files(self, mock_chroma, fixtures_dir, tmp_path):
        col = self._make_collection()
        mock_chroma.return_value = self._make_client(col)

        index(str(fixtures_dir), db_path=str(tmp_path), reset=False)

        assert col.add.called
        total_added = sum(len(call.kwargs["ids"]) for call in col.add.call_args_list)
        assert total_added > 0

    @patch("embed.chromadb.PersistentClient")
    def test_second_run_skips_unchanged_files(self, mock_chroma, fixtures_dir, tmp_path):
        col = self._make_collection()
        mock_chroma.return_value = self._make_client(col)

        # First run — writes hashes
        index(str(fixtures_dir), db_path=str(tmp_path), reset=False)
        col.add.reset_mock()

        # Second run — all files unchanged
        index(str(fixtures_dir), db_path=str(tmp_path), reset=False)

        col.add.assert_not_called()

    @patch("embed.chromadb.PersistentClient")
    def test_reset_wipes_and_reindexes(self, mock_chroma, fixtures_dir, tmp_path):
        col = self._make_collection()
        client = self._make_client(col)
        mock_chroma.return_value = client

        # First run
        index(str(fixtures_dir), db_path=str(tmp_path), reset=False)
        col.add.reset_mock()

        # Reset run
        index(str(fixtures_dir), db_path=str(tmp_path), reset=True)

        client.delete_collection.assert_called()
        assert col.add.called

    @patch("embed.chromadb.PersistentClient")
    def test_changed_file_removes_old_nodes_then_reindexes(self, mock_chroma, tmp_path):
        # Set up a repo with one Python file
        repo = tmp_path / "repo"
        repo.mkdir()
        src = repo / "foo.py"
        src.write_text("def alpha(): pass")

        db = tmp_path / "db"
        col = self._make_collection()
        mock_chroma.return_value = self._make_client(col)

        # First index
        index(str(repo), db_path=str(db), reset=False)
        col.add.reset_mock()
        col.get.reset_mock()

        # Modify the file
        src.write_text("def alpha(): pass\ndef beta(): return alpha()")

        # Second index — should detect change
        index(str(repo), db_path=str(db), reset=False)

        # Old nodes removed, new nodes added
        col.get.assert_called()
        assert col.add.called
