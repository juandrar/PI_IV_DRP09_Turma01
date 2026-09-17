"""Agregações e indicadores para EDA e dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT / "data" / "processed" / "agua_dda_mensal.parquet"


def load_processed(path: Path = DEFAULT_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo processado não encontrado: {path}. Execute: python -m src.data_prep"
        )
    return pd.read_parquet(path)


def filter_data(
    df: pd.DataFrame,
    ano_min: int | None = None,
    ano_max: int | None = None,
    ufs: list[str] | None = None,
    codigo_ibge: int | None = None,
) -> pd.DataFrame:
    out = df
    if ano_min is not None:
        out = out[out["ano"] >= ano_min]
    if ano_max is not None:
        out = out[out["ano"] <= ano_max]
    if ufs:
        out = out[out["uf"].isin(ufs)]
    if codigo_ibge is not None:
        out = out[out["codigo_ibge"] == codigo_ibge]
    return out.copy()


def kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n_municipios": 0,
            "n_registros": 0,
            "casos_total": 0,
            "incidencia_media_10k": 0.0,
            "taxa_ecoli_media": 0.0,
            "taxa_coliformes_media": 0.0,
            "total_amostras_ecoli": 0,
            "total_amostras_coliformes": 0,
        }
    return {
        "n_municipios": int(df["codigo_ibge"].nunique()),
        "n_registros": int(len(df)),
        "casos_total": int(df["casos_total"].fillna(0).sum()),
        "incidencia_media_10k": float(df["total_casos_por_10k_hab"].mean(skipna=True) or 0),
        "taxa_ecoli_media": float(df["taxa_ecoli"].mean(skipna=True) or 0),
        "taxa_coliformes_media": float(df["taxa_coliformes"].mean(skipna=True) or 0),
        "total_amostras_ecoli": int(df["total_amostras_para_ecoli"].fillna(0).sum()),
        "total_amostras_coliformes": int(df["total_amostras_para_coliformes"].fillna(0).sum()),
    }


def serie_temporal_nacional(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega indicadores nacionais por ano-mês."""
    g = (
        df.groupby(["ano", "mes"], as_index=False)
        .agg(
            casos_total=("casos_total", "sum"),
            habitantes=("habitantes", "sum"),
            amostras_ecoli=("total_amostras_para_ecoli", "sum"),
            ecoli_pos=("amostras_com_ecoli", "sum"),
            amostras_coliformes=("total_amostras_para_coliformes", "sum"),
            coliformes_pos=("amostras_com_coliformes", "sum"),
            n_municipios=("codigo_ibge", "nunique"),
        )
    )
    g["incidencia_10k"] = np.where(
        g["habitantes"] > 0,
        g["casos_total"] / g["habitantes"] * 10000,
        np.nan,
    )
    g["taxa_ecoli"] = np.where(
        g["amostras_ecoli"] > 0,
        g["ecoli_pos"] / g["amostras_ecoli"],
        np.nan,
    )
    g["taxa_coliformes"] = np.where(
        g["amostras_coliformes"] > 0,
        g["coliformes_pos"] / g["amostras_coliformes"],
        np.nan,
    )
    g["data_ref"] = pd.to_datetime(dict(year=g["ano"], month=g["mes"], day=1))
    return g.sort_values("data_ref")


def serie_temporal_uf(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["uf", "ano", "mes"], as_index=False)
        .agg(
            casos_total=("casos_total", "sum"),
            habitantes=("habitantes", "sum"),
            amostras_ecoli=("total_amostras_para_ecoli", "sum"),
            ecoli_pos=("amostras_com_ecoli", "sum"),
            amostras_coliformes=("total_amostras_para_coliformes", "sum"),
            coliformes_pos=("amostras_com_coliformes", "sum"),
        )
    )
    g["incidencia_10k"] = np.where(
        g["habitantes"] > 0,
        g["casos_total"] / g["habitantes"] * 10000,
        np.nan,
    )
    g["taxa_ecoli"] = np.where(
        g["amostras_ecoli"] > 0,
        g["ecoli_pos"] / g["amostras_ecoli"],
        np.nan,
    )
    g["taxa_coliformes"] = np.where(
        g["amostras_coliformes"] > 0,
        g["coliformes_pos"] / g["amostras_coliformes"],
        np.nan,
    )
    g["data_ref"] = pd.to_datetime(dict(year=g["ano"], month=g["mes"], day=1))
    return g.sort_values(["uf", "data_ref"])


def ranking_municipios(
    df: pd.DataFrame,
    metric: str = "total_casos_por_10k_hab",
    top_n: int = 20,
    ascending: bool = False,
) -> pd.DataFrame:
    """Ranking médio por município no período filtrado."""
    # Evitar colunas duplicadas quando metric já é taxa_ecoli / taxa_coliformes
    named: dict[str, tuple[str, str]] = {
        metric: (metric, "mean"),
        "casos_total": ("casos_total", "sum"),
        "habitantes": ("habitantes", "mean"),
        "n_meses": ("mes", "count"),
    }
    if metric != "taxa_ecoli":
        named["taxa_ecoli"] = ("taxa_ecoli", "mean")
    if metric != "taxa_coliformes":
        named["taxa_coliformes"] = ("taxa_coliformes", "mean")

    agg = df.groupby(["codigo_ibge", "municipio", "uf"], as_index=False).agg(**named)
    return agg.sort_values(metric, ascending=ascending).head(top_n)


def agregacao_uf(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["uf", "regiao"], as_index=False)
        .agg(
            casos_total=("casos_total", "sum"),
            habitantes=("habitantes", "mean"),
            incidencia_10k=("total_casos_por_10k_hab", "mean"),
            taxa_ecoli=("taxa_ecoli", "mean"),
            taxa_coliformes=("taxa_coliformes", "mean"),
            n_municipios=("codigo_ibge", "nunique"),
        )
    )
    return g.sort_values("incidencia_10k", ascending=False)


def casos_por_faixa_etaria(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "casos_menos_de_um_ano": "< 1 ano",
        "casos_um_a_quatro_anos": "1–4 anos",
        "casos_cinco_a_nove_anos": "5–9 anos",
        "casos_dez_anos_e_mais": "10+ anos",
        "casos_fx_ignorada": "Ignorada",
    }
    rows = []
    for col, label in cols.items():
        rows.append({"faixa": label, "casos": int(df[col].fillna(0).sum())})
    return pd.DataFrame(rows)


def correlacao_agua_saude(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "taxa_ecoli",
        "taxa_coliformes",
        "total_casos_por_10k_hab",
        "casos_total",
        "log_habitantes",
        "mes",
    ]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        return pd.DataFrame()
    return sub.corr()


def amostra_dispersao(df: pd.DataFrame, max_points: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Amostra para scatter (performance no Plotly)."""
    cols = [
        "codigo_ibge",
        "municipio",
        "uf",
        "ano",
        "mes",
        "taxa_ecoli",
        "taxa_coliformes",
        "total_casos_por_10k_hab",
        "casos_total",
        "habitantes",
    ]
    sub = df[cols].dropna(subset=["taxa_ecoli", "total_casos_por_10k_hab"])
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=seed)
    return sub


def heatmap_uf_mes(df: pd.DataFrame, value: str = "incidencia_10k") -> pd.DataFrame:
    serie = serie_temporal_uf(df)
    if serie.empty:
        return pd.DataFrame()
    pivot = serie.pivot_table(index="uf", columns="mes", values=value, aggfunc="mean")
    return pivot
