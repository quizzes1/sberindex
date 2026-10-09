"""Помечает страницы Streamlit как русскоязычные: <html lang="en"> → <html lang="ru"> в index.html пакета streamlit.

Без этого браузер видит русский текст на «английской» странице и предлагает перевести её, а автоперевод ломает
надписи и шрифты. Запускается при сборке Docker-образа; локально — один раз после установки зависимостей.
Повторный запуск ничего не меняет.
"""

from __future__ import annotations

from pathlib import Path

import streamlit

index = Path(streamlit.__file__).parent / "static" / "index.html"
html = index.read_text(encoding="utf-8")
if '<html lang="ru">' in html:
    print(f"уже ru: {index}")
else:
    index.write_text(html.replace('<html lang="en">', '<html lang="ru">', 1), encoding="utf-8")
    print(f"lang=ru: {index}")
