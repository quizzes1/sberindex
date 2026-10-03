"""Клонирует внешние репозитории (KEFRiN, CANUS, Pattern) на закреплённых коммитах в external/.

Запуск:  python scripts/fetch_external.py [--only KEFRiN,CANUS]

Код внешних репозиториев НЕ копируется в наш репозиторий (external/ в .gitignore): у KEFRiN и
CANUS нет файла лицензии, Pattern распространяется под GPL-3.0. Мы только вызываем их функции
из локальной копии (адаптеры в src/clustering.py) и используем Pattern для сверки индексов в тестах.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT, load_yaml  # noqa: E402

EXTERNAL = ROOT / "external"


def fetch(name: str, url: str, commit: str) -> None:
    """Неглубокий клон ровно одного коммита (без истории): быстро и мало места, в т.ч. в Docker-образе."""
    dest = EXTERNAL / name
    marker = dest / ".commit"  # метка остаётся и после удаления .git (например, в Docker-образе)
    if marker.exists() and marker.read_text().strip() == commit:
        print(f"{name}: {commit[:10]} уже на месте → {dest}")
        return
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(dest), "init", "--quiet"], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.stdout.strip() != commit:
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "FETCH_HEAD"], check=True)
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    assert head.stdout.strip() == commit, f"{name}: HEAD {head.stdout.strip()} != {commit}"
    marker.write_text(commit + "\n")
    print(f"{name}: {commit[:10]} → {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    ext = load_yaml("clustering.yaml")["external"]
    names = a.only.split(",") if a.only else list(ext)
    EXTERNAL.mkdir(exist_ok=True)
    for n in names:
        e = ext[n]
        print(f"{n}: лицензия — {e['license']}")
        fetch(n, e["url"], e["commit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
