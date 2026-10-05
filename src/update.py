"""Обновление данных и пересчёт с сайта (этап 7 доработки).

Страница «Обновление данных» запускает конвейер scripts/run_all.py отдельным процессом через
scripts/update_runner.py; обёртка сама пишет состояние в data/update/status.json и журнал в
data/update/run.log, поэтому процесс не зависит от вкладки браузера и переживает перезагрузку страницы.
Одновременно идёт не больше одного пересчёта. Код через сайт не обновляется — только данные и результаты.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime

from src.io import DATA, ROOT

STATE = DATA / "update"
STATUS = STATE / "status.json"
LOG = STATE / "run.log"

MODES = {
    "fast": {
        "name": "Пересчитать (быстро)",
        "args": ["--skip-download", "--quick", "--fast"],
        "about": "Без скачивания: панель, ВМП, показатели, сети, сетка кластеров без KEFRiN/CANUS, динамика, "
        "конвергенция, сводные таблицы. Нужен после правки configs/ или ручной замены файлов в data/raw. "
        "На всей России — около 15 минут.",
    },
    "full": {
        "name": "Пересчитать полностью",
        "args": ["--skip-download", "--quick"],
        "about": "То же, но в сетке кластеров и KEFRiN (для отчёта reports/CLUSTERS.md). Около 45 минут.",
    },
    "download": {
        "name": "Скачать свежие данные и пересчитать",
        "args": ["--quick", "--fast", "--refresh-data"],
        "about": "Заново скачивает все источники (СберИндекс, БДПМО, Росстат; ~1,5 ГБ; если файл не скачался — "
        "остаётся прежняя версия) и пересчитывает всё. От 30 минут до 1,5 часа; сайты СберИндекса и Росстата "
        "иногда ограничивают запросы — тогда повторите позже.",
    },
}


def read_status() -> dict:
    """Состояние последнего запуска: state = idle | running | done | failed (+ mode, pid, started, finished, code)."""
    if not STATUS.exists():
        return {"state": "idle"}
    try:
        st = json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "idle"}
    fresh = st.get("pid") is None and _age(st.get("started")) < 60  # запуск только что записан, pid ещё нет
    if st.get("state") == "running" and not fresh and not _alive(st.get("pid")):
        # процесс исчез, не записав итог (перезапуск контейнера, нехватка памяти)
        st.update(state="failed", code=None, finished=st.get("finished") or _now(), note="процесс прерван")
        write_status(st)
    return st


def write_status(st: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATUS)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _age(ts: str | None) -> float:
    try:
        return (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        import psutil

        p = psutil.Process(int(pid))
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def start(mode: str) -> dict:
    """Запустить пересчёт в фоне. Если уже идёт — вернуть текущее состояние, второй не запускать."""
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим: {mode}")
    st = read_status()
    if st.get("state") == "running":
        return st
    STATE.mkdir(parents=True, exist_ok=True)
    # состояние «running» пишется до запуска: второй щелчок в эту же секунду увидит занятость
    write_status({"state": "running", "mode": mode, "pid": None, "started": _now()})
    kw: dict = {"cwd": ROOT, "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    cmd = [sys.executable, str(ROOT / "scripts" / "update_runner.py"), mode]
    if os.name == "nt":
        # Windows: вне группы процессов (и, если разрешено, вне job-объекта) родителя — иначе пересчёт
        # завершится вместе с процессом, который его запустил
        base = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        try:
            p = subprocess.Popen(cmd, creationflags=base | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
        except OSError:
            p = subprocess.Popen(cmd, creationflags=base, **kw)
    else:
        p = subprocess.Popen(cmd, start_new_session=True, **kw)  # не завершится вместе с сеансом Streamlit
    st = {"state": "running", "mode": mode, "pid": p.pid, "started": _now()}
    write_status(st)
    return st


def tail(n: int = 60) -> str:
    """Последние n строк журнала."""
    if not LOG.exists():
        return ""
    with open(LOG, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


def steps_done() -> list[str]:
    """Завершённые шаги конвейера по журналу («=== build_x.py: 12 с»)."""
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("=== ") and line.rstrip().endswith(" с") and ":" in line:
            out.append(line[4:].strip())
    return out


def clear_caches() -> None:
    """Сбросить кэши модулей src (lru_cache) — после пересчёта интерфейс читает новые файлы."""
    for name, mod in list(sys.modules.items()):
        if not name.startswith("src.") or mod is None:
            continue
        for obj in list(vars(mod).values()):
            if callable(obj) and hasattr(obj, "cache_clear") and getattr(obj, "__module__", "") == name:
                obj.cache_clear()


def run(mode: str) -> int:
    """Тело scripts/update_runner.py: run_all с параметрами режима, журнал и итоговое состояние."""
    started = _now()
    write_status({"state": "running", "mode": mode, "pid": os.getpid(), "started": started})
    t = time.time()
    with open(LOG, "w", encoding="utf-8") as log:
        log.write(f"Запуск {started}: {MODES[mode]['name']} (run_all.py {' '.join(MODES[mode]['args'])})\n")
        log.flush()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        code = subprocess.call(
            [sys.executable, str(ROOT / "scripts" / "run_all.py"), *MODES[mode]["args"]],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        log.write(f"\nГотово за {(time.time() - t) / 60:.1f} мин, код выхода {code}\n")
    write_status(
        {
            "state": "done" if code == 0 else "failed",
            "mode": mode,
            "pid": None,
            "started": started,
            "finished": _now(),
            "code": code,
            "minutes": round((time.time() - t) / 60, 1),
        }
    )
    return code
