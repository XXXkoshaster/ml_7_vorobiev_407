from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split


RAW_DATA = Path(__file__).resolve().parent / "data" / "raw"
RANDOM_STATE = 42
FORECAST_HORIZON = pd.Timedelta(minutes=15)
DINING_TEMPERATURE = "Temperature_Comedor_Sensor"


def load_spambase():
    """Вернуть X_train, X_test, y_train, y_test для классификации."""
    with ZipFile(RAW_DATA / "spambase.zip") as archive:
        lines = archive.read("spambase.names").decode("ascii").splitlines()
        features = [line.split(":", 1)[0] for line in lines if line.endswith("continuous.")]
        with archive.open("spambase.data") as source:
            data = pd.read_csv(source, header=None, names=[*features, "spam"])

    # Убираем повторы; оставшиеся совпадения по X имеют противоречивые метки.
    data = data.drop_duplicates()
    data = data.loc[~data.duplicated(subset=features, keep=False)]
    return train_test_split(
        data[features], data["spam"], test_size=0.2,
        stratify=data["spam"], random_state=RANDOM_STATE,
    )


def _read_sml2010_period(archive, filename):
    with TextIOWrapper(archive.open(filename), encoding="ascii") as source:
        columns = [field.split(":", 1)[1] for field in source.readline().split()[1:]]
        data = pd.read_csv(source, sep=r"\s+", header=None, names=columns)

    data.index = pd.to_datetime(
        data.pop("Date") + " " + data.pop("Time"), format="%d/%m/%Y %H:%M", utc=True,
    ).rename("timestamp_utc")
    data = data.sort_index()

    # Ищем цель строго в t + 15 минут, а не в следующей строке (в ряду есть разрывы).
    future_temperature = data[DINING_TEMPERATURE].reindex(data.index + FORECAST_HORIZON)
    future_temperature.index = data.index
    future_temperature = future_temperature.dropna().rename("temperature_at_t_plus_15m")
    features = data.drop(columns=[
        "Weather_Temperature", "Exterior_Entalpic_1", "Exterior_Entalpic_2", "Exterior_Entalpic_turbo",
    ])
    return features.loc[future_temperature.index], future_temperature


def load_sml2010():
    """Вернуть X_train, X_test, y_train, y_test в хронологическом порядке."""
    with ZipFile(RAW_DATA / "sml2010.zip") as archive:
        X_train, y_train = _read_sml2010_period(archive, "NEW-DATA-1.T15.txt")
        X_test, y_test = _read_sml2010_period(archive, "NEW-DATA-2.T15.txt")
    return X_train, X_test, y_train, y_test


def spambase_cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def sml2010_cv():
    # На обучающем ряду шаг 15 минут. Строка зазора исключает общую границу целей и признаков.
    return TimeSeriesSplit(n_splits=3, gap=1)
