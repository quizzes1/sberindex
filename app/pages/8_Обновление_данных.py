"""Обновление данных: скачать свежие исходники и/или пересчитать результаты с сайта (без обновления кода)."""

from __future__ import annotations

import hmac
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from common import sidebar

from src import update
from src.io import DATA, PROCESSED, RAW

st.set_page_config(page_title="Обновление данных", layout="wide")
sidebar()
st.title("Обновление данных")
st.markdown(
    "Здесь можно скачать свежие данные и пересчитать все результаты. Код и настройки при этом не меняются. "
    "Пересчёт идёт на сервере, страницу можно закрыть; одновременно идёт только один. Когда он закончится, "
    "все страницы покажут новые данные."
)


def fmt_age(ts: str | None) -> str:
    """Дата и время в виде «09.10.2026 12:30 (15 мин назад)»."""
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    mins = (datetime.now() - dt).total_seconds() / 60
    return f"{dt:%d.%m.%Y %H:%M} ({mins:.0f} мин назад)" if mins < 120 else f"{dt:%d.%m.%Y %H:%M}"


@st.fragment(run_every=10)
def status_block() -> None:
    """Блок состояния пересчёта; обновляется сам каждые 10 секунд."""
    stt = update.read_status()
    state = stt.get("state", "idle")
    mode = update.MODES.get(stt.get("mode"), {}).get("name", "—")
    if state == "running":
        st.info(f"Идёт: **{mode}**, начато {fmt_age(stt.get('started'))}. Статус обновляется каждые 10 секунд.")
        done = update.steps_done()
        if done:
            st.caption("Готовые шаги: " + " · ".join(done))
        st.code(update.tail(25) or "журнал пока пуст", language=None)
    elif state == "done":
        st.success(
            f"Последний пересчёт ({mode}) закончился {fmt_age(stt.get('finished'))}, "
            f"занял {stt.get('minutes', '—')} мин."
        )
    elif state == "failed":
        st.error(
            f"Последний пересчёт ({mode}) прервался: {stt.get('note') or 'код ' + str(stt.get('code'))}, "
            f"{fmt_age(stt.get('finished'))}. Часть данных уже новая — запустите пересчёт ещё раз. Подробности в журнале."
        )
    else:
        st.caption("Пересчёт с сайта ещё не запускали.")


status_block()

stt = update.read_status()
running = stt.get("state") == "running"
has_raw = (RAW / "sber").exists() and (RAW / "tochno").exists()

st.subheader("Запуск")
mode = st.radio(
    "Что сделать",
    list(update.MODES),
    format_func=lambda m: update.MODES[m]["name"],
    disabled=running,
)
st.caption(update.MODES[mode]["about"])
need_raw = mode != "download" and not has_raw
if need_raw:
    st.warning("На сервере ещё нет исходных данных. Начните с «Скачать свежие данные и пересчитать».")
ok = st.checkbox(
    "Понимаю: пока идёт пересчёт, страницы показывают частично обновлённые результаты, а сервер загружен.",
    disabled=running,
)
# необязательный пароль только на запуск пересчёта (переменная окружения UPDATE_PASSWORD; пусто — без пароля)
need_pw = bool(os.environ.get("UPDATE_PASSWORD"))
pw_ok = True
if need_pw:
    pw = st.text_input("Пароль для запуска пересчёта", type="password", disabled=running)
    pw_ok = hmac.compare_digest(pw.encode(), os.environ["UPDATE_PASSWORD"].encode())
    if pw and not pw_ok:
        st.error("Неверный пароль.")
if st.button("▶ Запустить", type="primary", disabled=running or not ok or need_raw or not pw_ok):
    res = update.start(mode)
    if res.get("mode") != mode and res.get("state") == "running":
        st.warning("Пересчёт уже запущен кем-то другим, второй не начнётся.")
    st.rerun()
if running:
    st.caption("Кнопка недоступна, пока идёт пересчёт.")

st.subheader("Журнал")
if update.LOG.exists():
    with st.expander("Последние 200 строк"):
        st.code(update.tail(200), language=None)
    st.download_button("Скачать журнал", update.LOG.read_bytes(), "run.log", "text/plain", key="dl_log")
else:
    st.caption("Журнала пока нет.")

st.subheader("Когда обновлялись данные")
rows = []
for name, p in [
    ("показатели (indicators_wide)", PROCESSED / "indicators_wide.parquet"),
    ("ВМП", PROCESSED / "gmp.parquet"),
    ("индексы цен", PROCESSED / "price_levels.parquet"),
    ("сводные таблицы", PROCESSED / "summary_table2.xlsx"),
    ("сети (список)", DATA / "networks" / "index.csv"),
]:
    rows.append(
        {
            "что": name,
            "обновлено": datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
            if p.exists()
            else "нет файла",
        }
    )
raw_files = [f for f in RAW.rglob("*") if f.is_file()] if RAW.exists() else []
if raw_files:
    last = max(f.stat().st_mtime for f in raw_files)
    rows.append(
        {
            "что": f"исходники в data/raw ({len(raw_files)} файлов)",
            "обновлено": f"{datetime.fromtimestamp(last):%d.%m.%Y %H:%M}",
        }
    )
else:
    rows.append({"что": "исходники в data/raw", "обновлено": "нет — для пересчёта сначала «Скачать свежие данные»"})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
failed = RAW / "FAILED.txt"
if failed.exists() and failed.read_text(encoding="utf-8").strip():
    with st.expander("Источники, которые не скачались в последний раз"):
        st.code(failed.read_text(encoding="utf-8"), language=None)
st.caption(
    "«Пересчитать» работает с уже скачанными данными. Если сайты СберИндекса или Росстата недоступны, "
    "скачивание не пройдёт, но прежние файлы останутся."
)
