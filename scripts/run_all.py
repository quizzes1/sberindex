"""Весь конвейер с нуля одной командой.

Запуск:  python scripts/run_all.py [--skip-download] [--quick]

  --skip-download  не обращаться к сети (данные уже в data/raw/)
  --quick          пропустить отчёт об инвентаризации, тесты и медленный метод CANUS
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    t = time.time()
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True, cwd=ROOT)
    print(f"=== {script}: {time.time() - t:.0f} с", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not a.skip_download:
        run("download_data.py")
    if not a.quick:
        run("build_inventory.py")
    run("build_panel.py")
    run("build_gmp.py")
    run("build_indicators.py")
    run("build_networks.py")
    if not a.skip_download:
        run("fetch_external.py")
    run("build_clusters.py", *(["--quick"] if a.quick else []))
    run("build_dynamics.py")
    run("build_convergence.py")
    if not a.quick:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
