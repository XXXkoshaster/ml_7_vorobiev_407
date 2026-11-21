# Лабораторные работы по машинному обучению

Сравниваем модели sklearn и собственные реализации пяти алгоритмов на двух задачах. Для классификации используем **Spambase** (спам или обычное письмо), для регрессии — **SML2010** (температура в столовой через 15 минут). Разбиения одинаковы во всех лабораторных. Признаки, метрики и источники описаны в [протоколе оценки](docs/data-protocol.md).

## Установка и запуск

Команды для macOS и Linux выполняйте из корня репозитория. Нужны **Python 3.12** и `curl`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p data/raw
curl -fL https://archive.ics.uci.edu/static/public/94/spambase.zip -o data/raw/spambase.zip
curl -fL https://archive.ics.uci.edu/static/public/274/sml2010.zip -o data/raw/sml2010.zip

jupyter lab
```

Архивы оставьте в `data/raw/` без распаковки. Если они уже там, скачивать повторно не нужно. Чтобы проверить загрузку без Jupyter, из корня выполните:

```bash
python -c 'from data_setup import load_spambase, load_sml2010; print([part.shape for part in load_spambase()[:2]]); print([part.shape for part in load_sml2010()[:2]])'
```

Ожидаемые размеры признаков (обучение / тест): Spambase — `(3363, 57)` / `(841, 57)`, SML2010 — `(2763, 18)` / `(1370, 18)`.

## Порядок работы

После добавления ноутбуков откройте их в JupyterLab и выполните сверху вниз: сначала `01_knn.ipynb` — `05_gradient_boosting.ipynb` по порядку, затем `06_summary.ipynb`. Лабораторные сохраняют метрики в `results/`, а итоговый ноутбук читает эти CSV и не переобучает модели.

| Путь | Содержимое |
|---|---|
| `notebooks/` | Пять лабораторных и итоговое сравнение |
| `data_setup.py` | Загрузка датасетов и общие разбиения |
| `data/raw/` | Локальные архивы UCI, не входят в Git |
| `results/` | CSV с результатами лабораторных |
| [`docs/`](docs/README.md) | ТЗ, структура ноутбуков и рабочий процесс |

- [ЛР1: KNN](notebooks/01_knn.ipynb) — [метрики](results/lab01.csv)
- [ЛР2: логистическая и линейная регрессии](notebooks/02_regressions.ipynb) — [метрики](results/lab02.csv)
- [ЛР3: решающее дерево](notebooks/03_decision_tree.ipynb) — [метрики](results/lab03.csv)
- [ЛР4: случайный лес](notebooks/04_random_forest.ipynb) — [метрики](results/lab04.csv)
- [ЛР5: градиентный бустинг](notebooks/05_gradient_boosting.ipynb) — [метрики](results/lab05.csv)
