"""Обёртка пересчёта, запускаемая со страницы «Обновление данных» (src/update.py).

Запуск вручную:  python scripts/update_runner.py fast|full|download
Пишет состояние в data/update/status.json и журнал в data/update/run.log.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import update  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(update.run(sys.argv[1] if len(sys.argv) > 1 else "fast"))
