"""ETL: limpeza do CSV água/DDA, join com população IBGE e export Parquet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "arq" / "extracao_dados_agua_dda-2018-2026-com_habitantes-v01.csv"
RAW_XLSX = ROOT / "arq" / "pop_ibge_2018_2026-v01.xlsx"
OUT_DIR = ROOT / "data" / "processed"
OUT_PARQUET = OUT_DIR / "agua_dda_mensal.parquet"
OUT_QC = OUT_DIR / "qc_report.json"


def parse_br_year(series: pd.Series) -> pd.Series:
    """Converte anos no formato BR (ex.: '2.018') ou float para int."""
    s = series.astype(str).str.strip()

    def _one(x: str) -> int | float:
        if x.count(".") == 1 and len(x.split(".")[1]) == 3:
            return int(x.replace(".", ""))
        return int(float(x.replace(",", ".")))

    return s.map(_one).astype(int)


def parse_br_float(series: pd.Series) -> pd.Series:
    """Converte números com vírgula decimal (ex.: '31,29') para float."""
    s = series.astype(str).str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_raw_csv(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    return df


def load_ibge_pop(path: Path = RAW_XLSX) -> pd.DataFrame:
    pop = pd.read_excel(path, sheet_name="Planilha1")
    pop = pop.rename(
        columns={
            "UF": "uf_ibge",
            "COD. UF": "cod_uf",
            "COD. MUNIC": "cod_munic",
            "NOME DO MUNICÍPIO": "nome_mun_ibge",
            "POPULAÇÃO ESTIMADA": "pop_ibge",
            "Ano": "ano",
        }
    )
    pop["cod_uf"] = pop["cod_uf"].astype(str).str.zfill(2)
    pop["cod_munic"] = pop["cod_munic"].astype(str).str.zfill(5)
    # código IBGE oficial (7 dígitos, com dígito verificador)
    pop["codigo_ibge_7"] = (pop["cod_uf"] + pop["cod_munic"]).astype(int)
    # CSV do projeto usa 6 dígitos (sem DV) → chave de join
    pop["codigo_ibge"] = (pop["codigo_ibge_7"] // 10).astype(int)
    pop["ano"] = pop["ano"].astype(int)
    pop["pop_ibge"] = pd.to_numeric(pop["pop_ibge"], errors="coerce")
    return pop[["codigo_ibge", "codigo_ibge_7", "ano", "pop_ibge", "nome_mun_ibge", "uf_ibge"]]


def clean_agua_dda(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["regiao"] = df["regiao"].str.strip().str.upper()
    out["uf"] = df["UF"].str.strip().str.upper()
    out["codigo_ibge"] = pd.to_numeric(df["Codigo_IBGE"], errors="coerce").astype("Int64")
    out["municipio"] = df["Municipio"].str.strip()
    out["ano"] = parse_br_year(df["Ano_de_referencia"])
    out["mes"] = pd.to_numeric(df["Mes_de_referencia"], errors="coerce").astype("Int64")

    int_cols = [
        "total_amostras_para_coliformes",
        "amostras_sem_coliformes",
        "amostras_com_coliformes",
        "total_amostras_para_ecoli",
        "amostras_sem_ecoli",
        "amostras_com_ecoli",
        "casos_um_a_quatro_anos",
        "casos_dez_anos_e_mais",
        "casos_cinco_a_nove_anos",
        "casos_menos_de_um_ano",
        "casos_fx_ignorada",
        "casos_total",
        "habitantes",
    ]
    for c in int_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    # taxa original (vírgula BR); será recalculada com pop IBGE quando disponível
    out["taxa_casos_10k_original"] = parse_br_float(df["total_casos_por_10k_hab"])
    out["nome_mun_fonte"] = df["Nm_Mun"].str.strip()

    out = out.dropna(subset=["codigo_ibge", "ano", "mes"])
    out["codigo_ibge"] = out["codigo_ibge"].astype(int)
    out["mes"] = out["mes"].astype(int)
    out["ano"] = out["ano"].astype(int)

    # deduplicar município-mês
    out = out.drop_duplicates(subset=["codigo_ibge", "ano", "mes"], keep="first")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    tot_col = out["total_amostras_para_coliformes"].replace(0, np.nan)
    tot_eco = out["total_amostras_para_ecoli"].replace(0, np.nan)
    out["taxa_coliformes"] = out["amostras_com_coliformes"] / tot_col
    out["taxa_ecoli"] = out["amostras_com_ecoli"] / tot_eco
    out["sem_amostra_coliformes"] = (out["total_amostras_para_coliformes"].fillna(0) == 0).astype(int)
    out["sem_amostra_ecoli"] = (out["total_amostras_para_ecoli"].fillna(0) == 0).astype(int)

    pop = out["habitantes"].replace(0, np.nan)
    out["total_casos_por_10k_hab"] = (out["casos_total"] / pop) * 10000

    out["data_ref"] = pd.to_datetime(
        dict(year=out["ano"], month=out["mes"], day=1),
        errors="coerce",
    )
    out["log_habitantes"] = np.log1p(out["habitantes"].clip(lower=0))
    return out


def merge_ibge(df: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(pop, on=["codigo_ibge", "ano"], how="left", suffixes=("", "_ibge"))
    merged["habitantes_csv"] = merged["habitantes"]
    # preferir população IBGE quando disponível
    merged["habitantes"] = merged["pop_ibge"].fillna(merged["habitantes_csv"])
    pop_ref = merged["habitantes"].replace(0, np.nan)
    merged["total_casos_por_10k_hab"] = (merged["casos_total"] / pop_ref) * 10000
    merged["log_habitantes"] = np.log1p(merged["habitantes"].clip(lower=0))
    merged["match_ibge"] = merged["pop_ibge"].notna().astype(int)
    return merged


def quality_report(df: pd.DataFrame) -> dict:
    key = ["codigo_ibge", "ano", "mes"]
    dup = int(df.duplicated(subset=key).sum())
    years = sorted(df["ano"].unique().tolist())
    by_year = {str(y): int((df["ano"] == y).sum()) for y in years}

    # gaps: municípios com meses faltantes no período completo
    expected_months = set(range(1, 13))
    gaps = 0
    for (_, _), g in df.groupby(["codigo_ibge", "ano"]):
        missing = expected_months - set(g["mes"].tolist())
        # 2026 pode ser parcial
        if g["ano"].iloc[0] == 2026:
            continue
        gaps += len(missing)

    return {
        "n_rows": int(len(df)),
        "n_municipios": int(df["codigo_ibge"].nunique()),
        "n_ufs": int(df["uf"].nunique()),
        "anos": years,
        "rows_por_ano": by_year,
        "duplicatas_mun_mes": dup,
        "gaps_mes_exceto_2026": gaps,
        "pct_match_ibge": float(df["match_ibge"].mean() * 100),
        "sem_amostra_coliformes": int(df["sem_amostra_coliformes"].sum()),
        "sem_amostra_ecoli": int(df["sem_amostra_ecoli"].sum()),
        "casos_total_zero": int((df["casos_total"].fillna(0) == 0).sum()),
        "habitantes_nulos": int(df["habitantes"].isna().sum()),
    }


def run(
    csv_path: Path = RAW_CSV,
    xlsx_path: Path = RAW_XLSX,
    out_parquet: Path = OUT_PARQUET,
    out_qc: Path = OUT_QC,
) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Carregando CSV...")
    raw = load_raw_csv(csv_path)
    print(f"  {len(raw):,} linhas brutas")
    print("Limpando...")
    clean = clean_agua_dda(raw)
    print("Carregando IBGE...")
    pop = load_ibge_pop(xlsx_path)
    print("Cruzando e gerando features...")
    merged = merge_ibge(clean, pop)
    featured = add_features(merged)

    # ordenar colunas principais
    cols_front = [
        "regiao",
        "uf",
        "codigo_ibge",
        "codigo_ibge_7",
        "municipio",
        "nome_mun_fonte",
        "nome_mun_ibge",
        "ano",
        "mes",
        "data_ref",
        "habitantes",
        "habitantes_csv",
        "pop_ibge",
        "match_ibge",
        "total_amostras_para_coliformes",
        "amostras_sem_coliformes",
        "amostras_com_coliformes",
        "taxa_coliformes",
        "sem_amostra_coliformes",
        "total_amostras_para_ecoli",
        "amostras_sem_ecoli",
        "amostras_com_ecoli",
        "taxa_ecoli",
        "sem_amostra_ecoli",
        "casos_menos_de_um_ano",
        "casos_um_a_quatro_anos",
        "casos_cinco_a_nove_anos",
        "casos_dez_anos_e_mais",
        "casos_fx_ignorada",
        "casos_total",
        "total_casos_por_10k_hab",
        "taxa_casos_10k_original",
        "log_habitantes",
    ]
    other = [c for c in featured.columns if c not in cols_front]
    featured = featured[cols_front + other]

    featured.to_parquet(out_parquet, index=False)
    print(f"Salvo: {out_parquet} ({len(featured):,} linhas)")

    qc = quality_report(featured)
    out_qc.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"QC: {out_qc}")
    print(json.dumps(qc, indent=2, ensure_ascii=False))
    return featured


if __name__ == "__main__":
    run()
