# Лабораторные работы

## Установка и запуск

Выполните команды из корня репозитория. Нужны Python 3.12 и `curl`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p data/raw
curl -fL https://archive.ics.uci.edu/static/public/94/spambase.zip -o data/raw/spambase.zip
curl -fL https://archive.ics.uci.edu/static/public/274/sml2010.zip -o data/raw/sml2010.zip
jupyter lab
```

Ожидаемые размеры признаков (обучение / тест): Spambase — `(3363, 57)` / `(841, 57)`, SML2010 — `(2763, 18)` / `(1370, 18)`.

## Порядок работы

Выполните ноутбуки ЛР1–5 по порядку, затем итоговый. Лабораторные сохраняют метрики в `results/`, а итоговый ноутбук читает эти CSV.

| Путь | Содержимое |
|---|---|
| `notebooks/` | Пять лабораторных и итоговое сравнение |
| `data_setup.py` | Загрузка датасетов и общие разбиения |
| `data/raw/` | Локальные архивы UCI |
| `results/` | CSV с результатами лабораторных |

- [ЛР1: KNN](notebooks/01_knn.ipynb) — [метрики](results/lab01.csv)
- [ЛР2: логистическая и линейная регрессии](notebooks/02_regressions.ipynb) — [метрики](results/lab02.csv)
- [ЛР3: решающее дерево](notebooks/03_decision_tree.ipynb) — [метрики](results/lab03.csv)
- [ЛР4: случайный лес](notebooks/04_random_forest.ipynb) — [метрики](results/lab04.csv)
- [ЛР5: градиентный бустинг](notebooks/05_gradient_boosting.ipynb) — [метрики](results/lab05.csv)
- [Итоговое сравнение ЛР1–5](notebooks/06_summary.ipynb)
