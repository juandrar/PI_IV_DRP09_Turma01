"""Validação de cobertura, taxas e artefatos do pipeline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "processed" / "agua_dda_mensal.parquet"
QC = ROOT / "data" / "processed" / "qc_report.json"
METRICS = ROOT / "models" / "metricas.json"


def main() -> int:
    errors: list[str] = []

    if not PARQUET.exists():
        errors.append(f"Parquet ausente: {PARQUET}")
        _report(errors)
        return 1

    t0 = time.perf_counter()
    df = pd.read_parquet(PARQUET)
    load_s = time.perf_counter() - t0

    if load_s > 5:
        errors.append(f"Leitura Parquet lenta: {load_s:.2f}s (esperado < 5s)")

    years = [int(y) for y in sorted(df["ano"].unique())]
    if years[0] > 2018 or years[-1] < 2024:
        errors.append(f"Cobertura temporal inesperada: {years[0]}–{years[-1]}")

    recent = df[(df["ano"] >= 2024) & (df["ano"] <= 2026)]
    if len(recent) == 0:
        errors.append("Sem registros no recorte padrão 2024–2026")

    # taxa recalculada vs casos/habitantes
    sample = df.dropna(subset=["casos_total", "habitantes"]).head(5000)
    expected = (sample["casos_total"] / sample["habitantes"].replace(0, pd.NA)) * 10000
    diff = (sample["total_casos_por_10k_hab"] - expected).abs()
    if float(diff.max(skipna=True) or 0) > 0.01:
        errors.append(
            f"Inconsistência taxa/10k (max diff {float(diff.max(skipna=True)):.4f})"
        )

    if QC.exists():
        qc = json.loads(QC.read_text(encoding="utf-8"))
        if qc.get("duplicatas_mun_mes", 1) > 0:
            errors.append(f"Duplicatas município-mês: {qc['duplicatas_mun_mes']}")
        if qc.get("pct_match_ibge", 0) < 90:
            errors.append(f"Match IBGE baixo: {qc.get('pct_match_ibge')}%")
    else:
        errors.append(f"QC ausente: {QC}")

    if not METRICS.exists():
        errors.append(f"Métricas ML ausentes: {METRICS} (rode ml_pipeline)")

    # smoke test analysis imports
    from src.analysis import filter_data, kpis, serie_temporal_nacional

    f = filter_data(df, ano_min=2024, ano_max=2026)
    k = kpis(f)
    if k["n_municipios"] < 100:
        errors.append(f"Poucos municípios no filtro 2024–2026: {k['n_municipios']}")
    st = serie_temporal_nacional(f)
    if st.empty:
        errors.append("Série temporal nacional vazia")

    summary = {
        "n_rows": int(len(df)),
        "anos": years,
        "n_municipios": int(df["codigo_ibge"].nunique()),
        "registros_2024_2026": int(len(recent)),
        "parquet_load_seconds": round(load_s, 3),
        "kpis_2024_2026": k,
        "errors": errors,
        "ok": len(errors) == 0,
    }
    out = ROOT / "data" / "processed" / "validation_report.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok"] else 1


def _report(errors: list[str]) -> None:
    print(json.dumps({"ok": False, "errors": errors}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
