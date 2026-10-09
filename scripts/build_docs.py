"""Страницы презентации и отчёта для стартовой страницы «Проект» (app/hub.py).

Запуск:  python scripts/build_docs.py [--presentation путь.pdf] [--report путь.pdf]

Копирует PDF в app/static/docs/{presentation,report}.pdf (если пути указаны) и отрисовывает страницы в JPEG:
app/static/docs/presentation/NN.jpg (+ миниатюры tNN.jpg), app/static/docs/report/NN.jpg. Картинки лежат в
репозитории, поэтому серверу библиотека отрисовки не нужна; локально: uv pip install pypdfium2.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "app" / "static" / "docs"
# ширина в точках — с запасом для экранов высокой плотности (масштаб 150–200 %): слайд во всю ширину страницы
# занимает ~1400 CSS-пикселей → нужно ~2800 точек
WIDTH = {"presentation": 2880, "report": 2000}
# JPEG без понижения разрешения цвета (subsampling=0, 4:4:4): иначе мелкий цветной текст на тёмном фоне «мылится»
JPEG = {"quality": 90, "subsampling": 0, "optimize": True, "progressive": True}


def render(name: str) -> int:
    """Отрисовать страницы PDF в JPEG (и миниатюры для презентации)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(DOCS / f"{name}.pdf"))
    out = DOCS / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    for i in range(len(pdf)):
        page = pdf[i]
        img = page.render(scale=WIDTH[name] / page.get_width()).to_pil().convert("RGB")
        img.save(out / f"{i + 1:02d}.jpg", **JPEG)
        if name == "presentation":
            t = img.copy()
            t.thumbnail((480, 270))
            t.save(out / f"t{i + 1:02d}.jpg", quality=85, subsampling=0)
    return len(pdf)


def main() -> int:
    """Точка входа: страницы презентации и отчёта для стартовой страницы «Проект» (app/hub.py)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--presentation", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    DOCS.mkdir(parents=True, exist_ok=True)
    for name in ("presentation", "report"):
        src = getattr(a, name)
        if src:
            shutil.copyfile(src, DOCS / f"{name}.pdf")
        print(f"{name}: {render(name)} стр. → {DOCS / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
