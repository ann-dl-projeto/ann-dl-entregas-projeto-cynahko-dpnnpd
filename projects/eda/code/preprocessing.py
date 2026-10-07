"""Carga, limpeza, split e pipeline de pré-processamento do CarDekho (Car details v3).

Uso:
    from preprocessing import load_clean, split, build_pipeline
    df = load_clean()
    X_tr, X_te, y_tr, y_te = split(df)
    prep = build_pipeline().fit(X_tr)       # estatísticas aprendidas só no treino
    Z_tr, Z_te = prep.transform(X_tr), prep.transform(X_te)

O alvo é modelado em escala log: y = log(selling_price).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Car details v3.csv"

TARGET = "selling_price"
# numéricas com cauda longa à direita: recebem log1p antes de padronizar
NUM_LOG = ["km_driven", "engine", "max_power", "torque_nm"]
NUM_LIN = ["year", "mileage", "seats"]
NUM = NUM_LIN + NUM_LOG
CAT = ["fuel", "seller_type", "transmission", "owner", "brand"]
KGM_TO_NM = 9.80665


def _first_number(s):
    """Primeiro número de uma string ('23.4 kmpl' -> 23.4); None/'' -> NaN."""
    if not isinstance(s, str):
        return np.nan
    m = re.search(r"\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group()) if m else np.nan


def parse_torque(s):
    """Torque em Nm. Converte as linhas registradas em kgm (1 kgm = 9,80665 Nm)."""
    if not isinstance(s, str):
        return np.nan
    t = s.lower().replace(",", "")
    v = _first_number(t)
    if np.isnan(v):
        return np.nan
    # Se 'kgm' aparece sem 'nm' antes, o valor está em kgm ('12.7@ 2700(kgm@ rpm)',
    # '22.4 kgm at 1750'). Acima de 60 o rótulo está errado e o valor já é Nm:
    # '115@ 2500(kgm@ rpm)' daria 1128 Nm num Tata Sumo, e o maior torque plausível
    # do dataset é 640 Nm.
    if "kgm" in t and "nm" not in t.split("kgm")[0] and v <= 60:
        v *= KGM_TO_NM
    return v


def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    """Converte texto com unidade em número, extrai a marca e descarta colunas/linhas.

    - mileage/engine/max_power/torque: texto com unidade -> float
    - max_power == 0 (6 linhas), ' bhp' (1 linha) e mileage == 0 (17 linhas) -> NaN (impossível)
    - name (~2000 modelos, quase um ID) -> brand (primeira palavra)
    - duplicatas exatas saem antes do split, para a mesma linha não cair no treino e no teste
    """
    df = df.copy()
    df["mileage"] = df["mileage"].map(_first_number).replace(0, np.nan)
    df["engine"] = df["engine"].map(_first_number)
    df["max_power"] = df["max_power"].map(_first_number).replace(0, np.nan)
    df["torque_nm"] = df["torque"].map(parse_torque)
    df["brand"] = df["name"].str.split().str[0]
    df = df.drop(columns=["name", "torque"])
    df = df.drop_duplicates().reset_index(drop=True)
    return df[NUM + CAT + [TARGET]]


def load_clean(path=DATA_PATH):
    return clean(load_raw(path))


def split(df, test_size=0.2, seed=SEED):
    """Split 80/20 estratificado por decis de log(preço); y já em escala log."""
    X = df.drop(columns=[TARGET])
    y = np.log(df[TARGET]).rename("log_price")
    bins = pd.qcut(y, 10, labels=False)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=bins)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Corta cada coluna nos quantis [lower, upper] aprendidos no fit (só treino)."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_pipeline():
    """ColumnTransformer: winsoriza -> imputa -> (log) -> padroniza; one-hot nas categóricas."""
    num_lin = Pipeline([
        ("win", Winsorizer()),
        ("imp", SimpleImputer(strategy="median")),
        ("esc", StandardScaler()),
    ])
    num_log = Pipeline([
        ("win", Winsorizer()),
        ("imp", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("esc", StandardScaler()),
    ])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        # categorias com < 20 linhas no treino viram "infrequent", e uma categoria que
        # só aparece no teste cai nessa mesma coluna (sem isso, o transform quebraria)
        ("cod", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_lin, NUM_LIN),
        ("log", num_log, NUM_LOG),
        ("cat", cat, CAT),
        # 1 = linha sem ficha técnica; as specs faltam juntas e a ausência diz algo sobre o carro
        ("miss", MissingIndicator(features="all"), ["engine"]),
    ], verbose_feature_names_out=True)


if __name__ == "__main__":
    df = load_clean()
    X_tr, X_te, y_tr, y_te = split(df)
    prep = build_pipeline().fit(X_tr)
    Z_tr, Z_te = prep.transform(X_tr), prep.transform(X_te)
    print("treino", Z_tr.shape, "teste", Z_te.shape)
    print("NaN:", np.isnan(Z_tr).sum(), np.isnan(Z_te).sum())
    print(list(prep.get_feature_names_out()))
