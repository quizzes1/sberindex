# Интерфейс аналитиков (Streamlit) + весь конвейер данных.
# Сборка:  docker compose build        Запуск: docker compose up -d     (см. DEPLOY.md)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl libarchive-tools \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# зависимости — отдельным слоем (кэшируется, пока не меняется requirements.txt)
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r requirements.txt

# CANUS требует PyTorch (+~700 МБ к образу) — по умолчанию выключен
ARG WITH_CANUS=0
RUN if [ "$WITH_CANUS" = "1" ]; then pip install -r requirements-optional.txt; fi

COPY . .

# Внешние методы (KEFRiN, при WITH_CANUS=1 — и CANUS) на закреплённых коммитах из configs/clustering.yaml.
# Их код не входит в наш репозиторий (нет файла лицензии) — клонируется при сборке образа;
# демонстрационные датасеты и история git удаляются.
RUN EXT="KEFRiN"; if [ "$WITH_CANUS" = "1" ]; then EXT="KEFRiN,CANUS"; fi \
 && python scripts/fetch_external.py --only "$EXT" \
 && rm -rf external/*/.git external/KEFRiN/data external/CANUS/Datasets

# непривилегированный пользователь (uid 1000 — обычно совпадает с пользователем на сервере,
# чтобы контейнер мог писать в смонтированную папку data/)
RUN useradd --uid 1000 --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/Home.py"]
