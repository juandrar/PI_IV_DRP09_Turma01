"""Pipeline de ML: prever incidência DDA (casos/10k hab) com split temporal."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.analysis import load_processed

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "modelo_incidencia.joblib"
METRICS_PATH = MODELS_DIR / "metricas.json"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "predicoes_teste.parquet"

FEATURE_NUM = [
    "taxa_ecoli",
    "taxa_coliformes",
    "log_habitantes",
    "mes",
    "lag1_incidencia",
    "lag2_incidencia",
    "lag3_incidencia",
    "lag1_taxa_ecoli",
    "lag1_taxa_coliformes",
]
FEATURE_CAT = ["uf", "regiao"]
TARGET = "total_casos_por_10k_hab"


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["codigo_ibge", "ano", "mes"]).copy()
    g = out.groupby("codigo_ibge", sort=False)
    out["lag1_incidencia"] = g[TARGET].shift(1)
    out["lag2_incidencia"] = g[TARGET].shift(2)
    out["lag3_incidencia"] = g[TARGET].shift(3)
    out["lag1_taxa_ecoli"] = g["taxa_ecoli"].shift(1)
    out["lag1_taxa_coliformes"] = g["taxa_coliformes"].shift(1)
    return out


def prepare_ml_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_lags(df)
    out = out.dropna(subset=[TARGET])
    # remover extremos absurdos (proteção)
    q99 = out[TARGET].quantile(0.995)
    out = out[out[TARGET] <= q99]
    return out


def split_temporal(df: pd.DataFrame):
    train = df[df["ano"] <= 2023]
    test = df[df["ano"].isin([2024, 2025])]
    holdout = df[df["ano"] == 2026]
    return train, test, holdout


def build_pipeline(model_name: str = "gbr") -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, FEATURE_NUM),
            ("cat", categorical, FEATURE_CAT),
        ]
    )
    if model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
    elif model_name == "linear":
        model = LinearRegression()
    else:
        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
        )
    return Pipeline([("pre", pre), ("model", model)])


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def baseline_mean_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    mean_val = float(train[TARGET].mean())
    pred = np.full(len(test), mean_val)
    return {"mean_train": mean_val, **_metrics(test[TARGET], pred)}


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    try:
        cat_names = list(
            pre.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(FEATURE_CAT)
        )
    except Exception:
        cat_names = FEATURE_CAT
    names = FEATURE_NUM + cat_names
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    # alinhar tamanhos
    n = min(len(names), len(imp))
    return (
        pd.DataFrame({"feature": names[:n], "importance": imp[:n]})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run(model_name: str = "gbr") -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Carregando dados processados...")
    df = load_processed()
    ml = prepare_ml_frame(df)
    train, test, holdout = split_temporal(ml)
    print(f"Train ≤2023: {len(train):,} | Test 2024–25: {len(test):,} | Holdout 2026: {len(holdout):,}")

    X_train = train[FEATURE_NUM + FEATURE_CAT]
    y_train = train[TARGET]
    X_test = test[FEATURE_NUM + FEATURE_CAT]
    y_test = test[TARGET]

    base = baseline_mean_predict(train, test)
    print("Baseline (média):", base)

    # linear baseline
    lin = build_pipeline("linear")
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_m = _metrics(y_test, lin_pred)

    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    test_m = _metrics(y_test, pred)

    holdout_m = None
    if len(holdout) > 0:
        X_h = holdout[FEATURE_NUM + FEATURE_CAT]
        y_h = holdout[TARGET]
        holdout_m = _metrics(y_h, pipe.predict(X_h))

    imp = feature_importance(pipe)
    imp.to_csv(IMPORTANCE_PATH, index=False)

    joblib.dump(
        {
            "pipeline": pipe,
            "features_num": FEATURE_NUM,
            "features_cat": FEATURE_CAT,
            "target": TARGET,
            "model_name": model_name,
        },
        MODEL_PATH,
    )

    pred_df = test[
        ["codigo_ibge", "municipio", "uf", "ano", "mes", TARGET]
    ].copy()
    pred_df["y_pred"] = pred
    pred_df["erro"] = pred_df[TARGET] - pred_df["y_pred"]
    pred_df.to_parquet(PREDICTIONS_PATH, index=False)

    report = {
        "target": TARGET,
        "model_name": model_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_holdout_2026": int(len(holdout)),
        "baseline_mean": base,
        "linear": lin_m,
        "model": test_m,
        "holdout_2026": holdout_m,
        "limitacoes": [
            "Associação estatística não implica causalidade.",
            "Lags de incidência capturam persistência temporal, não só qualidade da água.",
            "Cobertura e qualidade das amostras variam entre municípios.",
            "2026 pode estar com dados parciais.",
        ],
        "artifacts": {
            "model": str(MODEL_PATH),
            "metrics": str(METRICS_PATH),
            "importance": str(IMPORTANCE_PATH),
            "predictions": str(PREDICTIONS_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
