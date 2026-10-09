"""Стартовая страница «Проект»: презентация и методологический отчёт (просмотр на сайте и скачивание), GitHub.

Файлы — app/static/docs/: исходные PDF и заранее отрисованные страницы (JPEG). Раздаются как статические
файлы Streamlit (server.enableStaticServing), поэтому браузер кэширует их, а сервер не пересчитывает.
Обновить документы: заменить PDF и запустить python scripts/build_docs.py.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError

DOCS = Path(__file__).resolve().parent / "static" / "docs"
URL = "app/static/docs"  # путь статических файлов Streamlit (относительно корня сайта)
GITHUB = "https://github.com/quizzes1/sberindex"

PRESENTATION_SECTIONS = {
    "Титул": 1,
    "Команда": 2,
    "Резюме и пайплайн": 3,
    "Составляющие проекта": 4,
    "Данные": 5,
    "Качество данных": 6,
    "Отбор показателей": 7,
    "Итоговые показатели": 8,
    "Сеть": 9,
    "Кластеры: выбор алгоритма": 12,
    "Кластеры: число кластеров": 13,
    "Кластеры: интерпретация": 15,
    "Динамика": 17,
    "Таймлайн событий": 19,
    "Траектории": 23,
    "Итоги проекта": 27,
}
REPORT_SECTIONS = {
    "Титул и оглавление": 1,
    "1. Резюме проекта": 3,
    "2–4. Источники, составляющие, параметры": 4,
    "5. Данные и качество": 5,
    "6. Показатели": 9,
    "7. Сеть": 11,
    "8. Кластеры": 14,
    "9. Динамика": 20,
    "10. Сводные таблицы": 22,
    "11. Итоги проекта": 25,
}


def n_pages(name: str) -> int:
    """Число отрисованных страниц документа (файлы NN.jpg)."""
    return len([p for p in (DOCS / name).glob("[0-9][0-9].jpg")])


N_SLIDES, N_PAGES = n_pages("presentation"), n_pages("report")
# метка версии в адресе картинок: после пересборки (scripts/build_docs.py) браузер не покажет старые из кэша
VER = int(max((p.stat().st_mtime for p in DOCS.rglob("*.jpg")), default=0))

st.set_page_config(page_title="Кластерный анализ муниципалитетов России", page_icon="🗺️", layout="wide")

st.markdown(
    f"""
<style>
.hub-hero {{
  background: radial-gradient(120% 140% at 0% 100%, #21a038 0%, #0b6b3a 28%, #063d23 55%, #04140d 100%);
  border-radius: 22px; padding: 44px 48px 38px; color: #fff; margin-bottom: 18px;
  box-shadow: 0 10px 40px rgba(0,0,0,.25);
}}
.hub-hero .kicker {{ font-size: .85rem; letter-spacing: .14em; text-transform: uppercase; opacity: .75; }}
.hub-hero h1 {{ color: #fff; font-size: 2.6rem; line-height: 1.12; margin: .35rem 0 .5rem; font-weight: 800; }}
.hub-hero .years {{ font-size: 1.25rem; opacity: .9; margin-bottom: 1.1rem; }}
.hub-hero .team {{ font-size: .95rem; opacity: .82; }}
.hub-hero .team b {{ color: #6be59a; font-weight: 600; }}
.hub-btns {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }}
.hub-btn {{
  display: inline-flex; align-items: center; gap: 8px; padding: 10px 18px; border-radius: 999px;
  font-weight: 600; font-size: .95rem; text-decoration: none !important; transition: transform .12s, background .12s;
}}
.hub-btn:hover {{ transform: translateY(-1px); }}
.hub-btn.primary {{ background: #21a038; color: #fff !important; }}
.hub-btn.primary:hover {{ background: #1c8c31; }}
.hub-btn.ghost {{ background: rgba(255,255,255,.10); color: #fff !important; border: 1px solid rgba(255,255,255,.28); }}
.hub-btn.ghost:hover {{ background: rgba(255,255,255,.18); }}
.hub-card {{ min-height: 146px; }}
.hub-card-title {{ font-size: 1.15rem; font-weight: 700; margin-bottom: .15rem; }}
.hub-card-sub {{ opacity: .7; font-size: .9rem; margin-bottom: .6rem; }}
.hub-links {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.hub-links a {{
  display: inline-block; margin: 0; padding: 6px 14px; border-radius: 999px; font-size: .88rem;
  font-weight: 600; text-decoration: none !important; border: 1px solid rgba(33,160,56,.55); color: #21a038 !important;
}}
.hub-links a:hover {{ background: rgba(33,160,56,.12); }}
.hub-stats {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin: 22px 0 18px; }}
.hub-stat-box {{ display: flex; flex-direction: column; }}
.hub-stat {{
  font-size: 2rem; font-weight: 800; color: #21a038; line-height: 1; white-space: nowrap;
  height: 2.6rem; display: flex; align-items: flex-end;
}}
.hub-stat.small {{ font-size: 1.5rem; }}
.hub-stat-l {{ opacity: .72; font-size: .88rem; margin-top: .4rem; }}
@media (max-width: 900px) {{ .hub-stats {{ grid-template-columns: repeat(2, 1fr); }} }}
.hub-thumb img {{ border-radius: 8px; }}
.hub-page {{ width: 100%; border-radius: 10px; box-shadow: 0 4px 24px rgba(0,0,0,.18); }}
</style>
<div class="hub-hero">
  <div class="kicker">СберИндекс · конкурс 2026 · трек «Кластеризация»</div>
  <h1>Кластерный анализ муниципалитетов<br>на территории России</h1>
  <div class="years">2014–2024 · ~2 600 муниципальных образований</div>
  <div class="team">Команда Политеха Петра Великого: разработчик <b>Павел Путинцев</b>,
    аналитики <b>Александр Панов</b>, <b>Иван Напрюшкин</b> и <b>Дмитрий Быков</b></div>
  <div class="hub-btns">
    <a class="hub-btn primary" href="#presentation-view">Смотреть презентацию</a>
    <a class="hub-btn ghost" href="#report-view">Читать отчёт</a>
    <a class="hub-btn ghost" href="{GITHUB}" target="_blank" rel="noopener">Код на GitHub</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- карточки материалов
c1, c2, c3 = st.columns(3, gap="medium")
cards = [
    (
        c1,
        "Презентация",
        f"{N_SLIDES} слайдов · PDF, {(DOCS / 'presentation.pdf').stat().st_size / 2**20:.1f} МБ",
        "presentation.pdf",
        "СберИндекс_кластерный_анализ_презентация.pdf",
        "#presentation-view",
    ),
    (
        c2,
        "Методологический отчёт",
        f"{N_PAGES} страниц · PDF, {(DOCS / 'report.pdf').stat().st_size / 2**20:.1f} МБ",
        "report.pdf",
        "СберИндекс_методологический_отчёт.pdf",
        "#report-view",
    ),
]
for col, title, sub, fname, dl_name, anchor in cards:
    with col.container(border=True):
        st.markdown(
            f"""<div class="hub-card"><div class="hub-card-title">{title}</div><div class="hub-card-sub">{sub}</div>
<div class="hub-links"><a href="{anchor}">Смотреть на сайте</a>
<a href="{URL}/{fname}" download="{dl_name}">Скачать PDF</a>
<a href="{URL}/{fname}" target="_blank" rel="noopener">Открыть в браузере</a></div></div>""",
            unsafe_allow_html=True,
        )
with c3.container(border=True):
    st.markdown(
        f"""<div class="hub-card"><div class="hub-card-title">Код и данные</div>
<div class="hub-card-sub">Расчёты, интерфейс и инструкция по запуску</div>
<div class="hub-links"><a href="{GITHUB}" target="_blank" rel="noopener">GitHub</a>
<a href="{GITHUB}#readme" target="_blank" rel="noopener">Как запустить</a>
<a href="{GITHUB}/blob/main/reports/METHODS.md" target="_blank" rel="noopener">Методика</a></div></div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- ключевые цифры
stats = [
    ("~2 600", "муниципальных образований"),
    ("2014–2024", "11 лет наблюдений"),
    ("5", "показателей экономики"),
    ("5", "типов муниципалитетов"),
    ("Уорд + k-means", "метод кластеризации"),
]
st.markdown(
    '<div class="hub-stats">'
    + "".join(
        f'<div class="hub-stat-box"><div class="hub-stat{" small" if len(v) > 10 else ""}">{v}</div>'
        f'<div class="hub-stat-l">{lab}</div></div>'
        for v, lab in stats
    )
    + "</div>",
    unsafe_allow_html=True,
)

links = [
    ("overview.py", "Обзор", ":material/dashboard:"),
    ("pages/4_Кластеры.py", "Кластеры", ":material/scatter_plot:"),
    ("pages/5_Динамика.py", "Динамика", ":material/timeline:"),
    ("pages/7_Сводные_таблицы.py", "Сводные таблицы", ":material/table_chart:"),
]
with st.container(border=True):
    st.markdown('<div class="hub-card-title">Перейти к анализу</div>', unsafe_allow_html=True)
    for col, (page, label, icon) in zip(st.columns(len(links)), links):
        try:
            col.page_link(page, label=label, icon=icon, use_container_width=True)
        except StreamlitPageNotFoundError:  # страница открыта напрямую, без меню Home.py
            pass

st.divider()


# ---------------------------------------------------------------- просмотр
def pager(key: str, total: int, sections: dict[str, int]) -> int:
    """Навигация по страницам: раздел, ◀ ▶, ползунок. Возвращает номер страницы (1…total)."""
    s = st.session_state
    s.setdefault(key, 1)

    def go(delta: int) -> None:
        s[key] = min(total, max(1, s[key] + delta))

    def to_section() -> None:
        s[key] = sections[s[f"{key}_sec"]]

    a, b, c, d = st.columns([2.2, 0.7, 4, 0.7])
    a.selectbox("Раздел", list(sections), key=f"{key}_sec", on_change=to_section, label_visibility="collapsed")
    b.button("◀", key=f"{key}_prev", on_click=go, args=(-1,), use_container_width=True, disabled=s[key] <= 1)
    c.slider("Страница", 1, total, key=key, label_visibility="collapsed")
    d.button("▶", key=f"{key}_next", on_click=go, args=(1,), use_container_width=True, disabled=s[key] >= total)
    return int(s[key])


st.markdown('<div id="presentation-view"></div>', unsafe_allow_html=True)
st.subheader("Презентация")
slide = pager("hub_slide", N_SLIDES, PRESENTATION_SECTIONS)
st.markdown(
    f'<img class="hub-page" src="{URL}/presentation/{slide:02d}.jpg?v={VER}" alt="Слайд {slide}">',
    unsafe_allow_html=True,
)
st.caption(f"Слайд {slide} из {N_SLIDES}")
with st.expander("Все слайды", expanded=False):
    per_row = 7
    for start in range(1, N_SLIDES + 1, per_row):
        cols = st.columns(per_row)
        for i, col in zip(range(start, min(start + per_row, N_SLIDES + 1)), cols):
            col.markdown(
                f'<div class="hub-thumb"><img src="{URL}/presentation/t{i:02d}.jpg?v={VER}" style="width:100%" '
                f'alt="Слайд {i}"></div>',
                unsafe_allow_html=True,
            )
            col.button(
                f"{i}",
                key=f"hub_thumb_{i}",
                on_click=lambda i=i: st.session_state.update(hub_slide=i),
                use_container_width=True,
                type="primary" if i == slide else "secondary",
            )

st.divider()
st.markdown('<div id="report-view"></div>', unsafe_allow_html=True)
st.subheader("Методологический отчёт")
mode = st.segmented_control(
    "Вид",
    ["По страницам", "Весь документ"],
    default="По страницам",
    key="hub_report_mode",
    label_visibility="collapsed",
)
left, mid, right = st.columns([1, 3.2, 1])
with mid:
    if mode == "Весь документ":
        for p in range(1, N_PAGES + 1):
            st.markdown(
                f'<img class="hub-page" src="{URL}/report/{p:02d}.jpg?v={VER}" alt="Страница {p}" loading="lazy" '
                'style="margin-bottom:18px">',
                unsafe_allow_html=True,
            )
    else:
        page = pager("hub_page", N_PAGES, REPORT_SECTIONS)
        st.markdown(
            f'<img class="hub-page" src="{URL}/report/{page:02d}.jpg?v={VER}" alt="Страница {page}">',
            unsafe_allow_html=True,
        )
        st.caption(f"Страница {page} из {N_PAGES}")

st.caption(
    "Отчёт и презентация — версия на дату сдачи. Цифры на страницах анализа пересчитываются при обновлении данных."
)
