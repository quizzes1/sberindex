"""Точка входа интерфейса: меню страниц. Запуск: streamlit run app/Home.py

Первая (стартовая) страница — «Проект»: презентация и методологический отчёт (просмотр и скачивание), GitHub.
Дальше — страницы анализа; их адреса (url) прежние: /Кластеры, /Динамика и т.д.
"""

from __future__ import annotations

import streamlit as st

pages = {
    "Проект": [
        st.Page("hub.py", title="Отчёт и презентация", icon=":material/auto_stories:", default=True),
    ],
    "Анализ": [
        st.Page("overview.py", title="Обзор", icon=":material/dashboard:", url_path="Обзор"),
        st.Page("pages/1_Данные_и_качество.py", title="Данные и качество", icon=":material/fact_check:"),
        st.Page("pages/2_Показатели.py", title="Показатели", icon=":material/query_stats:"),
        st.Page("pages/3_Сеть.py", title="Сеть", icon=":material/hub:"),
        st.Page("pages/4_Кластеры.py", title="Кластеры", icon=":material/scatter_plot:"),
        st.Page("pages/5_Динамика.py", title="Динамика", icon=":material/timeline:"),
        st.Page("pages/6_Конвергенция.py", title="Конвергенция", icon=":material/trending_down:"),
        st.Page("pages/7_Сводные_таблицы.py", title="Сводные таблицы", icon=":material/table_chart:"),
    ],
    "Сервис": [
        st.Page("pages/8_Обновление_данных.py", title="Обновление данных", icon=":material/sync:"),
    ],
}
st.navigation(pages).run()
