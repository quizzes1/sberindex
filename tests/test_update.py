"""Обновление данных с сайта (src/update.py, этап 7 доработки)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pytest

from src import update


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "STATE", tmp_path)
    monkeypatch.setattr(update, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(update, "LOG", tmp_path / "run.log")
    return tmp_path


class FakePopen:
    calls = 0

    def __init__(self, *a, **kw):
        FakePopen.calls += 1
        self.pid = os.getpid()  # «живой» процесс — текущий


def test_idle_without_status(state):
    assert update.read_status() == {"state": "idle"}


def test_start_once_then_locked(state, monkeypatch):
    FakePopen.calls = 0
    monkeypatch.setattr(update.subprocess, "Popen", FakePopen)
    st = update.start("fast")
    assert st["state"] == "running" and st["mode"] == "fast" and FakePopen.calls == 1
    st2 = update.start("download")  # второй запуск, пока первый жив, — не запускается
    assert st2["mode"] == "fast" and FakePopen.calls == 1


def test_dead_process_marked_failed(state):
    old = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="seconds")
    update.write_status({"state": "running", "mode": "fast", "pid": 999_999_999, "started": old})
    st = update.read_status()
    assert st["state"] == "failed" and st["note"] == "процесс прерван"
    assert json.loads((state / "status.json").read_text(encoding="utf-8"))["state"] == "failed"


def test_just_started_without_pid_is_running(state):
    update.write_status({"state": "running", "mode": "fast", "pid": None, "started": datetime.now().isoformat()})
    assert update.read_status()["state"] == "running"


def test_unknown_mode(state):
    with pytest.raises(ValueError):
        update.start("code")  # обновление кода через сайт не предусмотрено


def test_log_parsing(state):
    (state / "run.log").write_text(
        "Запуск\n\n=== build_panel.py \n…\n=== build_panel.py: 463 с\n=== build_gmp.py \n", encoding="utf-8"
    )
    assert update.steps_done() == ["build_panel.py: 463 с"]
    assert update.tail(1).strip() == "=== build_gmp.py"


def test_modes_never_skip_tls_or_touch_code():
    for m in update.MODES.values():
        assert all(a.startswith("--") for a in m["args"])
        assert not any("verify" in a or "git" in a for a in m["args"])


def test_clear_caches_resets_lru():
    from src import network

    network._registry()
    assert network._registry.cache_info().currsize == 1
    update.clear_caches()
    assert network._registry.cache_info().currsize == 0
