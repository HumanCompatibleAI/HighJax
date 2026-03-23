'''Pytest configuration and fixtures for ozette tests.'''
from __future__ import annotations

import pathlib
import tempfile

import pytest


@pytest.fixture
def temp_base_dir():
    '''Provide a temporary directory for ozette logs.'''
    with tempfile.TemporaryDirectory(prefix='ozette_test_') as tmpdir:
        yield pathlib.Path(tmpdir)
