"""Shared fixtures for the code-intel test suite."""
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def sample_py(fixtures_dir) -> Path:
    return fixtures_dir / "sample.py"


@pytest.fixture
def sample_ts(fixtures_dir) -> Path:
    return fixtures_dir / "sample.ts"
