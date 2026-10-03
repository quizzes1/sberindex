"""Ввод-вывод: пути проекта, загрузка файлов с проверкой TLS, контрольные суммы."""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

import certifi
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
GEO = DATA / "geo"
NETWORKS = DATA / "networks"
REPORTS = ROOT / "reports"
CERTS = ROOT / "certs"
CA_BUNDLE = CERTS / "ca-bundle.pem"

USER_AGENT = "Mozilla/5.0 (municipal-clustering research; +https://tochno.st/datasets/bdmo)"


def load_yaml(name: str) -> dict:
    """Читает YAML из папки configs/."""
    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cert_fingerprint(pem_path: Path) -> str:
    """SHA-256 отпечаток сертификата в формате AA:BB:..."""
    import ssl

    der = ssl.PEM_cert_to_DER_cert(pem_path.read_text(encoding="ascii"))
    h = hashlib.sha256(der).hexdigest().upper()
    return ":".join(h[i : i + 2] for i in range(0, len(h), 2))


def build_ca_bundle() -> Path:
    """Собирает certs/ca-bundle.pem = certifi + дополнительные сертификаты из sources.yaml.

    Каждый дополнительный сертификат сверяется с закреплённым отпечатком SHA-256;
    при несовпадении — ошибка. Проверка TLS при этом остаётся включённой.
    """
    cfg = load_yaml("sources.yaml")["tls"]["extra_certs"]
    parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    for c in cfg:
        p = ROOT / c["path"]
        fp = _cert_fingerprint(p)
        if fp != c["sha256"]:
            raise RuntimeError(f"Отпечаток {p.name} не совпадает: {fp} != {c['sha256']}")
        parts.append(p.read_text(encoding="ascii"))
    CA_BUNDLE.write_text("\n".join(parts), encoding="utf-8")
    return CA_BUNDLE


def session() -> requests.Session:
    """HTTP-сессия с отдельным CA-бандлом (TLS проверяется всегда)."""
    if not CA_BUNDLE.exists():
        build_ca_bundle()
    s = requests.Session()
    s.verify = str(CA_BUNDLE)
    s.headers["User-Agent"] = USER_AGENT
    return s


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 файла."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(
    s: requests.Session,
    url: str,
    dest: Path,
    api: bool = False,
    retries: int = 6,
    pause: float = 10.0,
    timeout: float = 120.0,
) -> Path:
    """Скачивает url в dest атомарно (через .part) с повторами и паузой.

    api=True добавляет заголовок RqUID, который требует API sberindex.ru.
    Сервер СберИндекса иногда отвечает 504/403 или рвёт соединение —
    тогда ждём pause * номер попытки и пробуем снова.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        headers = {"RqUID": uuid.uuid4().hex, "X-Language": "ru"} if api else {}
        try:
            with s.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                ctype = r.headers.get("Content-Type", "")
                if "text/html" in ctype and not url.endswith((".html", ".htm")):
                    raise requests.HTTPError(f"вместо файла пришёл HTML ({ctype})")
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            part.replace(dest)
            return dest
        except (requests.RequestException, OSError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(pause * attempt)
    if part.exists():
        part.unlink()
    raise RuntimeError(f"{url}: {last_err}")
