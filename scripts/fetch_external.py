"""Загружает нужные файлы внешних репозиториев (KEFRiN, CANUS, Pattern) на закреплённых коммитах в external/.

Запуск:  python scripts/fetch_external.py [--only KEFRiN,CANUS] [--soft]

Код внешних репозиториев НЕ копируется в наш репозиторий (external/ в .gitignore): у KEFRiN и
CANUS нет файла лицензии, Pattern распространяется под GPL-3.0. Мы только вызываем их функции
из локальной копии (адаптеры в src/clustering.py) и используем Pattern для сверки индексов в тестах.

Скачиваются только файлы, которые нужны коду (configs/clustering.yaml → external.<имя>.files), — несколько
десятков КБ вместо сотен МБ демонстрационных данных: напрямую по адресу raw.githubusercontent.com/<репо>/<коммит>/
<файл> с повторными попытками (TLS проверяется); если не вышло — неглубокий git-клон без содержимого файлов
(--filter=blob:none) с выборочной выгрузкой только этих файлов.

--soft — не завершаться с ошибкой, если репозиторий недоступен (для сборки Docker-образа: сайт работает и без
KEFRiN/CANUS, метод в интерфейсе помечается недоступным).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import ROOT, load_yaml, session  # noqa: E402

EXTERNAL = ROOT / "external"


def _raw_url(url: str, commit: str, path: str) -> str:
    owner_repo = url.removeprefix("https://github.com/").removesuffix(".git").strip("/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{path}"


def fetch_raw(name: str, url: str, commit: str, files: list[str], retries: int = 4) -> None:
    """Файлы по одному через raw.githubusercontent.com (закреплённый коммит), с повторами."""
    s = session()
    dest = EXTERNAL / name
    for f in files:
        out = dest / f
        out.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(1, retries + 1):
            try:
                r = s.get(_raw_url(url, commit, f), timeout=60)
                if r.status_code == 404 and f in ("LICENSE", "README.md"):
                    last = None  # необязательный файл
                    break
                r.raise_for_status()
                out.write_bytes(r.content)
                last = None
                break
            except Exception as e:  # noqa: BLE001 — сеть: повторяем
                last = e
                time.sleep(5 * attempt)
        if last is not None:
            raise RuntimeError(f"{name}/{f}: {last}")


def fetch_git(name: str, url: str, commit: str, files: list[str]) -> None:
    """Запасной путь: git без содержимого файлов (--filter=blob:none) + выборочная выгрузка только files."""
    dest = EXTERNAL / name
    tmp = EXTERNAL / f".{name}.git-tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    def git(*a):
        subprocess.run(["git", "-C", str(tmp), *a], check=True)

    git("init", "--quiet")
    git("remote", "add", "origin", url)
    git("config", "http.lowSpeedLimit", "1000")
    git("config", "http.lowSpeedTime", "60")
    git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("sparse-checkout", "set", "--no-cone", *files)
    git("checkout", "--quiet", "FETCH_HEAD")
    for f in files:
        if (tmp / f).exists():
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp / f, dest / f)
    shutil.rmtree(tmp, ignore_errors=True)


def fetch(name: str, url: str, commit: str, files: list[str]) -> None:
    dest = EXTERNAL / name
    marker = dest / ".commit"  # метка: какой коммит лежит в external/<имя>
    if marker.exists() and marker.read_text().strip() == commit and all((dest / f).exists() for f in files[:1]):
        print(f"{name}: {commit[:10]} уже на месте → {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    try:
        fetch_raw(name, url, commit, files)
        how = "файлы напрямую"
    except Exception as e:  # noqa: BLE001
        print(f"{name}: напрямую не скачалось ({e}) — пробую git без лишних файлов", flush=True)
        fetch_git(name, url, commit, files)
        how = "git (только нужные файлы)"
    missing = [f for f in files[:1] if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"{name}: нет файлов {missing}")
    marker.write_text(commit + "\n")
    print(f"{name}: {commit[:10]} → {dest} ({how}: {', '.join(files)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--soft", action="store_true", help="не падать, если репозиторий недоступен")
    a = ap.parse_args()
    ext = load_yaml("clustering.yaml")["external"]
    names = a.only.split(",") if a.only else list(ext)
    EXTERNAL.mkdir(exist_ok=True)
    failed = []
    for n in names:
        e = ext[n]
        print(f"{n}: лицензия — {e['license']}", flush=True)
        try:
            fetch(n, e["url"], e["commit"], e["files"])
        except Exception as err:  # noqa: BLE001
            failed.append(n)
            print(f"⚠ {n} не загружен: {err}", flush=True)
    if failed:
        print(
            f"⚠ Не загружены: {', '.join(failed)}. Интерфейс работает без них (метод помечен недоступным); "
            "повторить: python scripts/fetch_external.py --only " + ",".join(failed),
            flush=True,
        )
        return 0 if a.soft else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
