# Кластеризация муниципалитетов: данные и интерфейс

Конкурс СберИндекса, трек «Кластеризация». Анализ — по всем МО России (~2 600 территорий);
ДФО и другие федеральные округа — готовые фильтры (пресеты).
Узел сети — МО верхнего уровня в неизменных границах, атрибуты — экономические и социальные
показатели, рёбра — экономическая близость МО; сеть строится на каждый год, окно по умолчанию —
**2014–2024** (одно на весь сайт; меняется в боковой панели).

```bash
python scripts/run_all.py             # весь конвейер с нуля (флаги --skip-download, --quick, --fast, --refresh-data)
streamlit run app/Home.py             # интерфейс аналитиков
```

Развёртывание на сервере (Docker + HTTPS): **[DEPLOY.md](DEPLOY.md)** —
`cp .env.example .env && docker compose up -d --build`.

## Где что лежит

```
configs/    sources.yaml (источники), panel.yaml, indicators.yaml (реестр показателей и цены), gmp.yaml,
            network.yaml, clustering.yaml, dynamics.yaml, regions.yaml, summary.yaml (сводные таблицы)
scripts/    download_data, build_inventory, build_panel, build_gmp, build_indicators, build_networks,
            build_data_gaps, fetch_external, build_clusters, build_dynamics, build_convergence,
            build_summary, run_all, update_runner
src/        io, dictionary, bdmo, regional, panel, geo, gmp, indicators, normalize, network,
            prices, clustering, icvi, dynamics, convergence, summary, update
app/        Home.py (меню), hub.py (стартовая: отчёт и презентация), overview.py + pages/ (8 страниц), common.py,
            static/docs/ (PDF и страницы для просмотра; пересобрать: scripts/build_docs.py)
notebooks/  01_quickstart.ipynb — панель, сеть, кластеры, динамика и конвергенция в коде
tests/      pytest (данные, ВМП, нормировка, сеть, индексы, методы, динамика, конвергенция, интерфейс)
data/       raw/ (не в git), processed/, geo/, networks/, clusters/
reports/    INVENTORY, DATA, METHODS, GMP_CHECKS, INDICATORS, NETWORKS, CLUSTERS, DYNAMICS, CONVERGENCE
```

## Интерфейс

`streamlit run app/Home.py`. Стартовая страница «Отчёт и презентация» — презентация и методологический отчёт с просмотром на сайте и скачиванием PDF, ссылка на GitHub; дальше — страницы анализа. Боковая панель (общая для всех страниц): выборка (вся Россия по умолчанию / пресет федерального округа / свой
список субъектов), годы, исключение МО со сменой границ, способ и область нормировки. Страницы:
1. **Данные и качество** — покрытие показателей (тепловая карта) по годам и субъектам, МО с пропусками,
   непривязанные строки, найденные/не найденные источники;
2. **Показатели** — реестр, распределения до и после нормировки, корреляции, картограмма, таблица;
3. **Сеть** — признаки и веса, география, функция «расстояние → вес», прореживание; граф на карте и
   силовая раскладка, статистика, матрица расстояний; сохранение сети по хэшу;
4. **Кластеры** — режим (каждый год отдельно / одна модель на все годы — pooled), метод (в т.ч. «Уорд +
   k-means» с дендрограммой и выбором k по скачку расстояния слияния), k, год; карта, граф, паспорта типов,
   индексы SW/CH/S_Dbw/AVI/AVU/MQ, сравнение методов, перебор k, названия типов (сохраняются в
   `data/cluster_names.yaml`). Кластеры пронумерованы K1…Kn по убыванию ВМП на душу в ценах базового года;
5. **Динамика** — диаграмма Санки переходов, устойчивость по годам, матрица переходов, мигранты,
   карта «кто куда перешёл»;
6. **Конвергенция** — σ по годам, «начальный уровень — рост» с β-регрессией, коэффициенты, по типам;
7. **Сводные таблицы** — таблица 1 «Характерные признаки кластеров»: число МО, автоматическое описание
   (можно исправить вручную — правки сохраняются в `data/cluster_descriptions.yaml`), примеры МО, матрица
   «кластер × признак»; таблица 2 «Результаты кластеризации по периодам»: кластер МО в выбранных годах и
   траектория (стабильный / рост / снижение / колебание) с поиском, фильтрами и постраничным выводом; версия по
   субъектам (доминирующий кластер и доля — по числу МО и по населению); сводка переходов по России и округам;
   выгрузка в Excel с цветами, CSV и страница для печати. Для разбиения по умолчанию обе таблицы — в
   `reports/SUMMARY.md` (`scripts/build_summary.py`);
8. **Обновление данных** — скачать свежие исходники и/или пересчитать все результаты прямо с сайта (без
   обновления кода): пересчёт идёт в фоне, одновременно не больше одного, состояние и журнал — на странице
   (`src/update.py`, `data/update/`).

**Цены.** Денежные показатели хранятся в текущих ценах и по умолчанию используются **в ценах 2023 г.**
(дефлятор — свой для каждого показателя: ИПЦ, дефлятор ВРП, индекс цен инвестиционной продукции, индексы цен
производителей; `src/prices.py`, `reports/METHODS.md`, раздел 3.1). В боковой панели можно выбрать базовый год,
индексы по России или по субъектам, межрегиональную поправку на стоимость фиксированного набора и текущие цены.

На каждой странице — выгрузка текущей таблицы (CSV) и параметров (YAML). Сеть, построенная на странице
«Сеть», используется страницами «Кластеры» и «Динамика»; типы со страницы «Динамика» — страницей
«Конвергенция». Карты — в азимутальной равновеликой проекции с центром на выборке: Чукотка не рвётся на
180-м меридиане (полигоны хранятся в долготах 0…360).

## Установка

```bash
uv venv --python 3.11 .venv          # или python -m venv .venv
uv pip install -r requirements.txt   # или .venv/Scripts/pip install -r requirements.txt
```

## Этап 1

```bash
python scripts/download_data.py       # всё в data/raw/, SHA256SUMS, FAILED.txt; повторный запуск докачивает недостающее
python scripts/build_inventory.py     # reports/INVENTORY.md
```

Источники перечислены в `configs/sources.yaml`. Проверка TLS не отключается: сайты СберИндекса и
Росстата отдают неполные цепочки сертификатов, поэтому недостающие промежуточные сертификаты
(TrustAsia LiteSSL RSA CA 2025, Russian Trusted Sub CA) и корень Минцифры лежат в `certs/` с
закреплёнными отпечатками SHA-256, и скрипт собирает из них и `certifi` отдельный бандл `certs/ca-bundle.pem`.

## Этап 2 — годовая панель

```bash
python scripts/build_panel.py         # data/processed/panel_*.parquet, mo.parquet, data/geo/*
python -m pytest -q                   # тесты
```

Ключ панели — `territory_id × year` (МО в неизменных границах из справочника СберИндекса).
Все решения по склейке рядов, флаги и словарь данных — в `reports/DATA.md`.

Главное правило работы с панелью: для **среза по году** (сеть, кластеры) брать строки
`valid_in_year == True`; строки `derived == True` — достроенные значения прошлых лет для
преемников объединений, они нужны только для динамики. МО со сменой границ отмечены в
`mo.parquet` (`boundary_change`).

## Этап 3 — ВМП и показатели

```bash
python scripts/build_gmp.py           # data/processed/gmp*.parquet, reports/GMP_CHECKS.md
python scripts/build_indicators.py    # data/processed/indicators*.parquet, reports/INDICATORS.md
```

ВМП (валовой муниципальный продукт) — **расчётная оценка команды**: ВДС субъекта по отраслям
распределяется между МО по отгрузке, продукции сельского хозяйства и отраслевому ФОТ
(`configs/gmp.yaml`, методика — `reports/METHODS.md`).

### Как добавить показатель

1. Запись в `configs/indicators.yaml` (код, название, формула, источник, единица, блок, `log`,
   `direction`; для денежного — `monetary: true` и `deflator` — индекс цен из `src/prices.py`:
   cpi | grp | invest | ppi_industry | ppi_manufacturing | ppi_agri; без дефлятора тест не пройдёт).
2. Функция с тем же именем в `src/indicators.py`: принимает `Context`, возвращает `Series`
   с индексом `(territory_id, year)`. Нужен новый исходный ряд — добавьте его в
   `configs/sources.yaml` и `configs/panel.yaml`.
3. `python scripts/build_indicators.py`. Тест `test_registry_has_functions` проверит, что функция есть.

### Нормировка

`src/normalize.py`: логарифм (по реестру) → опционально винзоризация → `minmax` или `zscore`.
**По умолчанию шкала общая для всей панели** (`scope="panel"`): если нормировать каждый год
отдельно, общий рост показателей исчезает, значения разных лет становятся несопоставимы, и
переходы МО между кластерами во времени окажутся артефактом нормировки.

## Этап 4 — расстояния и сеть

```bash
python scripts/build_networks.py      # сети по умолчанию (Россия, Россия+дороги, ДФО) → data/networks/{hash}/
```

```python
from src import network
p = network.merge_params(network.default_params(), {"features": {"gmp_pc": 2, "density": 1}, "geo": {"alpha": 0.8}})
nets = network.build(p)               # {год: YearNetwork}; nets[2022].edges(), .graph(), .stats(), .D
h = network.save(nets, p)             # рёбра: data/networks/{h}/edges_{год}.parquet (source, target, weight, distance)
```

**Методическое замечание.** Если ребро строится по одному показателю, кластеризация сети
сводится к нарезке МО на интервалы значений этого показателя. Поэтому в сравнение методов
включаются комбинации из нескольких показателей.

## Этап 5 — кластеризация и индексы качества

```bash
python scripts/fetch_external.py      # KEFRiN, CANUS (и Pattern для сверки индексов) на закреплённых коммитах → external/
python scripts/build_clusters.py      # k = 2…10 × методы × годы → data/clusters/{hash}/, reports/CLUSTERS.md (--quick — без CANUS)
```

```python
from src import clustering, icvi
lab = clustering.fit(X, W, {"method": "leiden", "k": 5})   # единый интерфейс для всех методов
icvi.compute_all(X, W, lab)                                 # SW, CH, DBI, S_Dbw, AVI, AVU, ANUI, MQ
```

Внешние методы KEFRiN и CANUS (Шалилех, Миркин) не копируются в репозиторий: у их репозиториев
нет файла лицензии, поэтому они клонируются отдельно в `external/` (вне git). CANUS требует
PyTorch: `pip install -r requirements-optional.txt`.

### Как добавить метод кластеризации

1. Функция `_fit_<имя>(X, W, p) -> labels` в `src/clustering.py` (X — нормированные признаки,
   W — веса рёбер разреженной сети, p — параметры, в т.ч. `k` и `seed`).
2. Запись `<имя>: {family: attributes | graph | attributed_network, ...}` в `configs/clustering.yaml`
   и строка в `FAMILIES`.
3. Тест `tests/test_clustering.py` автоматически проверит метод на трёх разделимых облаках, если
   добавить имя в `METHODS`.

## Этап 6 — динамика и конвергенция

```bash
python scripts/build_dynamics.py                      # сквозные типы, переходы, ARI, бутстрэп → reports/DYNAMICS.md
python scripts/build_dynamics.py --mode pooled --method kmeans   # одна модель на всю панель
python scripts/build_convergence.py                   # σ, β (абсолютная и панельная), клубы → reports/CONVERGENCE.md
```

Параметры — `configs/dynamics.yaml`. Для длинного окна 2013–2024 ВМП берётся по базовому методу
(единому для всех лет), отраслевой — отдельно на 2017–2024.

## Источники и цитирование

- Данные о границах и преобразованиях муниципальных образований. СберИндекс. https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities (CC BY-SA 4.0).
- Потребительские безналичные расходы на уровне муниципальных образований; Индекс доступности рынков; Автодорожные и железнодорожные связи между муниципальными образованиями. СберИндекс. https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim (CC BY-SA 4.0).
- Железнодорожные расстояния между муниципалитетами России. СберИндекс. https://sberindex.ru/ru/research/jeleznodorojnye-rasstoania-mejdu-munizipalitetami-rossii (CC BY-SA 4.0).
- «Муниципальная статистика России с 2005 года» // Росстат; обработка: «Если быть точным», 2025. https://tochno.st/datasets/bdmo (CC BY 4.0).
- Социально-экономические показатели регионов России // Росстат; обработка «Если быть точным». https://tochno.st/datasets/regions_collection (CC BY 4.0).
- Shalileh S., Mirkin B. Community partitioning over feature-rich networks using an extended k-means method. Entropy 2022, 24(5), 626 (KEFRiN). Shalileh S. A filtered gradient descent clustering method to recover communities in attributed networks. IEEE Access, 2025 (CANUS).
- Росстат, раздел «Национальные счета»: ВРП, ВДС по ОКВЭД2, валовой городской продукт ДФО. https://rosstat.gov.ru/statistics/accounts
