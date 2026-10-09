"""Весь конвейер с нуля одной командой.

Запуск:  python scripts/run_all.py [--skip-download] [--quick] [--fast]

  --skip-download  не обращаться к сети (данные уже в data/raw/)
  --quick          пропустить отчёт об инвентаризации, тесты и медленный метод CANUS
  --refresh-data   перекачать и уже скачанные исходники (при ошибке остаётся прежняя версия)
  --fast           сетка кластеров без KEFRiN и CANUS (минуты вместо получаса на всей России);
                   интерфейс считает любые методы сам — сетка нужна для отчёта reports/CLUSTERS.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str, check: bool = True) -> None:
    """Запустить шаг конвейера и вывести его длительность."""
    t = time.time()
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=check, cwd=ROOT)
    if r.returncode:
        print(f"=== {script}: ошибка (код {r.returncode}) — продолжаю", flush=True)
    print(f"=== {script}: {time.time() - t:.0f} с", flush=True)


def main() -> int:
    """Точка входа: весь конвейер с нуля одной командой."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--refresh-data", action="store_true", help="перекачать и уже скачанные исходники")
    a = ap.parse_args()
    if not a.skip_download:
        run("download_data.py", *(["--refresh"] if a.refresh_data else []))
    if not a.quick:
        run("build_inventory.py")
    run("build_panel.py")
    run("build_gmp.py")
    run("build_indicators.py")
    run("build_data_gaps.py")
    run("build_networks.py")
    if not a.skip_download:
        run("fetch_external.py", check=False)  # нет доступа к GitHub — внешние методы из прежней копии
    run("build_clusters.py", *(["--quick"] if a.quick else []), *(["--skip", "kefrin", "canus"] if a.fast else []))
    run("build_dynamics.py")
    run("build_convergence.py")
    run("build_summary.py")
    if not a.quick:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
