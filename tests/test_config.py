"""Tests for inference.config.load_runtime_config precedence."""

from __future__ import annotations

from unittest.mock import patch

from inference.config import load_runtime_config


def test_tau_env_wins(monkeypatch):
    monkeypatch.setenv("TAU", "1.25")
    monkeypatch.setenv("TAU_ARTIFACT_PATH", __file__)  # ignored when TAU set
    rc = load_runtime_config()
    assert rc.tau == 1.25
    assert rc.source == "env"


def test_artifact_file_when_no_tau_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TAU", raising=False)
    path = tmp_path / "tau.json"
    path.write_text('{"tau": 2.5, "version": 1}', encoding="utf-8")
    monkeypatch.setenv("TAU_ARTIFACT_PATH", str(path))
    rc = load_runtime_config()
    assert rc.tau == 2.5
    assert str(path) in rc.source


def test_default_when_missing_file_and_no_firestore(monkeypatch, tmp_path):
    monkeypatch.delenv("TAU", raising=False)
    monkeypatch.setenv("USE_FIRESTORE_CONFIG", "false")
    monkeypatch.setenv("TAU_ARTIFACT_PATH", str(tmp_path / "does_not_exist.json"))
    rc = load_runtime_config()
    assert rc.tau is None
    assert rc.source == "default"


def test_firestore_when_no_env_or_file(monkeypatch, tmp_path):
    monkeypatch.delenv("TAU", raising=False)
    monkeypatch.setenv("USE_FIRESTORE_CONFIG", "true")
    monkeypatch.setenv("TAU_ARTIFACT_PATH", str(tmp_path / "missing.json"))
    with patch("inference.config._load_tau_from_firestore", return_value=3.14):
        rc = load_runtime_config()
    assert rc.tau == 3.14
    assert rc.source == "firestore"
