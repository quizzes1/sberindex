"""Скачивает все исходные данные в data/raw/ (этап 1).

Запуск:  python scripts/download_data.py [--with-optional] [--only sber,rosstat,...] [--verify]

- Повторный запуск докачивает только недостающее (файл есть и записан в SHA256SUMS — пропуск).
- Контрольные суммы: data/raw/SHA256SUMS (формат sha256sum: «хэш  путь»).
- Неудачи: data/raw/FAILED.txt (перезаписывается при каждом запуске).
- Архивы СберИндекса распаковываются рядом (data/raw/sber/...).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import RAW, build_ca_bundle, download, load_yaml, session, sha256_file  # noqa: E402

SUMS = RAW / "SHA256SUMS"
FAILED = RAW / "FAILED.txt"


def read_sums() -> dict[str, str]:
    if not SUMS.exists():
        return {}
    out = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split("  ", 1)
            out[p] = h
    return out


def write_sums(sums: dict[str, str]) -> None:
    SUMS.write_text("".join(f"{h}  {p}\n" for p, h in sorted(sums.items())), encoding="utf-8")


def collect_jobs(cfg: dict, with_optional: bool) -> list[dict]:
    """Список заданий на загрузку из configs/sources.yaml."""
    jobs = []
    for group in ("sber", "rosstat", "tochno_regions"):
        for it in cfg[group]:
            jobs.append({"group": group, **it})
    b = cfg["tochno_bdmo"]
    d = b["dest_dir"]
    jobs.append(
        {
            "group": "tochno_bdmo",
            "id": "bdmo_description",
            "url": b["description_url"],
            "dest": f"{d}/description_bdpmo_{b['version']}.pdf",
        }
    )
    jobs.append(
        {
            "group": "tochno_bdmo",
            "id": "bdmo_indicator_list",
            "url": b["indicator_list_url"],
            "dest": f"{d}/indicator_list_{b['version']}.xlsx",
        }
    )
    for ind in b["indicators"]:
        if ind.get("optional") and not with_optional:
            continue
        url = b["indicator_url"].format(section=ind["section"], code=ind["code"])
        jobs.append(
            {"group": "tochno_bdmo", "id": f"bdmo_{ind['code']}", "url": url, "dest": f"{d}/{url.rsplit('/', 1)[-1]}"}
        )
    return jobs


def unpack(path: Path) -> None:
    """Распаковывает архивы СберИндекса (zip и rar) в папку рядом с архивом."""
    out = path.with_suffix("")
    if out.exists() and any(out.iterdir()):
        return
    out.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                # имена в архиве хакатона записаны в cp866/utf-8 без флага — берём только базовое имя
                name = info.filename
                if not info.flag_bits & 0x800:
                    try:
                        name = name.encode("cp437").decode("utf-8")
                    except UnicodeError:
                        pass
                if name.endswith("/"):
                    continue
                target = out / Path(name).name
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        return
    # RAR: libarchive (bsdtar; в Windows 10+ это встроенный tar.exe), иначе unrar/7z
    candidates = [
        ["C:/Windows/System32/tar.exe", "-xf", str(path), "-C", str(out)],
        ["bsdtar", "-xf", str(path), "-C", str(out)],
        ["unrar", "x", "-o+", str(path), str(out) + "/"],
        ["7z", "x", "-y", f"-o{out}", str(path)],
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]) or Path(cmd[0]).exists():
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                return
    raise RuntimeError(f"Не удалось распаковать {path}: нужен bsdtar, unrar или 7z")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-optional", action="store_true", help="качать и опциональные (большие) показатели")
    ap.add_argument("--only", default="", help="группы через запятую: sber,rosstat,tochno_regions,tochno_bdmo")
    ap.add_argument("--verify", action="store_true", help="пересчитать SHA-256 уже скачанных файлов")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="перекачать и уже скачанные файлы (атомарно: при ошибке остаётся прежняя версия, это не считается неудачей)",
    )
    args = ap.parse_args()

    cfg = load_yaml("sources.yaml")
    build_ca_bundle()
    s = session()
    jobs = collect_jobs(cfg, args.with_optional)
    if args.only:
        groups = set(args.only.split(","))
        jobs = [j for j in jobs if j["group"] in groups]

    RAW.mkdir(parents=True, exist_ok=True)
    sums = read_sums()
    failed = []
    kept = []  # --refresh: не скачалось, но прежняя версия есть — работаем с ней
    changed = 0
    for i, j in enumerate(jobs, 1):
        dest = RAW / j["dest"]
        rel = dest.relative_to(RAW).as_posix()
        if dest.exists() and rel in sums and not args.refresh:
            if args.verify and sha256_file(dest) != sums[rel]:
                print(f"[{i}/{len(jobs)}] ХЭШ НЕ СОВПАЛ, перекачиваю: {rel}")
            else:
                print(f"[{i}/{len(jobs)}] есть: {rel}")
                if j["group"] == "sber" and dest.suffix in (".zip", ".rar"):
                    unpack(dest)
                continue
        print(f"[{i}/{len(jobs)}] качаю: {rel}", flush=True)
        try:
            is_sber = "sberindex.ru" in j["url"] or "sberbank" in j["url"]
            download(s, j["url"], dest, api=j.get("api", False), pause=15 if is_sber else 5)
            new = sha256_file(dest)
            changed += sums.get(rel) != new
            sums[rel] = new
            write_sums(sums)
            if j["group"] == "sber" and dest.suffix in (".zip", ".rar"):
                unpack(dest)
            if is_sber:
                time.sleep(5)  # WAF СберИндекса ограничивает частоту запросов
        except Exception as e:  # noqa: BLE001 — любую ошибку пишем в FAILED.txt и идём дальше
            print(f"   ОШИБКА: {e}")
            line = f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{j['id']}\t{j['url']}\t{e}"
            if args.refresh and dest.exists():
                print("   оставлена прежняя версия файла")
                kept.append(line)
            else:
                failed.append(line)

    write_sums(sums)
    FAILED.write_text("".join(f + "\n" for f in failed + kept), encoding="utf-8")
    print(f"Готово: {len(jobs) - len(failed)} из {len(jobs)}; неудач: {len(failed)} (см. {FAILED})")
    if args.refresh:
        print(f"Изменилось файлов: {changed}; не скачалось, оставлена прежняя версия: {len(kept)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
