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
    "Здесь любой посетитель сайта может **скачать свежие данные и пересчитать результаты**: панель, ВМП, "
    "показатели, сети, кластеры, динамику, конвергенцию и сводные таблицы. Код и настройки по умолчанию через сайт не "
    "меняются. Пересчёт идёт на сервере в фоне — страницу можно закрыть; одновременно идёт не больше одного пересчёта. "
    "Когда он закончится, все страницы сами перечитают новые данные."
)


def fmt_age(ts: str | None) -> str:
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
    stt = update.read_status()
    state = stt.get("state", "idle")
    mode = update.MODES.get(stt.get("mode"), {}).get("name", "—")
    if state == "running":
        st.info(f"⏳ Идёт: **{mode}**, запущено {fmt_age(stt.get('started'))}. Состояние обновляется каждые 10 с.")
        done = update.steps_done()
        if done:
            st.caption("Готовые шаги: " + " · ".join(done))
        st.code(update.tail(25) or "журнал пока пуст", language=None)
    elif state == "done":
        st.success(
            f"✅ Последний пересчёт — **{mode}** — завершён {fmt_age(stt.get('finished'))} за {stt.get('minutes', '—')} мин."
        )
    elif state == "failed":
        st.error(
            f"❌ Последний пересчёт — **{mode}** — завершился с ошибкой ({stt.get('note') or 'код ' + str(stt.get('code'))}), "
            f"{fmt_age(stt.get('finished'))}. Данные на сайте — частично новые: повторите пересчёт; журнал — ниже."
        )
    else:
        st.caption("С сайта пересчёт ещё не запускался.")


status_block()

stt = update.read_status()
running = stt.get("state") == "running"
has_raw = (RAW / "sber").exists() and (RAW / "tochno").exists()

st.subheader("Запустить")
mode = st.radio(
    "Что сделать",
    list(update.MODES),
    format_func=lambda m: update.MODES[m]["name"],
    disabled=running,
)
st.caption(update.MODES[mode]["about"])
need_raw = mode != "download" and not has_raw
if need_raw:
    st.warning(
        "Исходников в data/raw нет (на сервер из репозитория попадают только готовые результаты) — выберите «Скачать свежие данные и пересчитать»."
    )
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
        st.warning("Пересчёт уже идёт (его запустил кто-то другой) — второй не запускается.")
    st.rerun()
if running:
    st.caption("Кнопка недоступна, пока идёт пересчёт.")

st.subheader("Журнал последнего пересчёта")
if update.LOG.exists():
    with st.expander("Последние 200 строк"):
        st.code(update.tail(200), language=None)
    st.download_button("⬇ Журнал целиком", update.LOG.read_bytes(), "run.log", "text/plain", key="dl_log")
else:
    st.caption("Журнала нет.")

st.subheader("Свежесть данных")
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
    "Режимы «Пересчитать» работают с уже скачанными исходниками в data/raw. На свежем сервере их нет (в репозитории "
    "только готовые результаты) — начните со «Скачать свежие данные». Сайты СберИндекса и Росстата бывают недоступны "
    "из-за рубежа: тогда скачивание не пройдёт, а прежние файлы останутся."
)
