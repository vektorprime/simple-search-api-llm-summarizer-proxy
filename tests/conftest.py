"""Shared fixtures for the whole suite."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_config_file(tmp_path, monkeypatch):
    """Keep tests independent of ./config.json / ./.env in the working directory.

    The config file beats the environment, so a real ./config.json would
    override the env vars tests set. Every test gets a fresh, nonexistent
    CONFIG_FILE; monkeypatch also undoes any CONFIG_FILE a test sets itself.
    """
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.json"))
