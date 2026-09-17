"""Dashboard Streamlit — Qualidade da água × DDA (PI IV)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import (
    agregacao_uf,
    amostra_dispersao,
    casos_por_faixa_etaria,
    correlacao_agua_saude,
    filter_data,
    heatmap_uf_mes,
    kpis,
    load_processed,
    ranking_municipios,
    serie_temporal_nacional,
    serie_temporal_uf,
)

PARQUET = ROOT / "data" / "processed" / "agua_dda_mensal.parquet"
METRICS = ROOT / "models" / "metricas.json"
IMPORTANCE = ROOT / "models" / "feature_importance.csv"
QC = ROOT / "data" / "processed" / "qc_report.json"

st.set_page_config(
    page_title="Água × DDA | PI IV",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Carregando dados processados...")
def get_data() -> pd.DataFrame:
    return load_processed(PARQUET)


@st.cache_data
def get_metrics() -> dict | None:
    if not METRICS.exists():
        return None
    return json.loads(METRICS.read_text(encoding="utf-8"))


@st.cache_data
def get_importance() -> pd.DataFrame | None:
    if not IMPORTANCE.exists():
        return None
    return pd.read_csv(IMPORTANCE)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")
    ano_min_data = int(df["ano"].min())
    ano_max_data = int(df["ano"].max())
    # padrão 2024–2026, limitado ao que existir
    default_min = max(2024, ano_min_data)
    default_max = min(2026, ano_max_data)
    anos = st.sidebar.slider(
        "Período (ano)",
        min_value=ano_min_data,
        max_value=ano_max_data,
        value=(default_min, default_max),
    )
    ufs = sorted(df["uf"].dropna().unique().tolist())
    uf_sel = st.sidebar.multiselect("UF", options=ufs, default=[])
    mun_opts = (
        df if not uf_sel else df[df["uf"].isin(uf_sel)]
    )[["codigo_ibge", "municipio", "uf"]].drop_duplicates()
    mun_opts = mun_opts.sort_values(["uf", "municipio"])
    mun_labels = {
        int(r.codigo_ibge): f"{r.municipio} ({r.uf})"
        for r in mun_opts.itertuples()
    }
    mun_sel = st.sidebar.selectbox(
        "Município (opcional)",
        options=[None] + list(mun_labels.keys()),
        format_func=lambda x: "Todos" if x is None else mun_labels.get(x, str(x)),
    )
    filtered = filter_data(
        df,
        ano_min=anos[0],
        ano_max=anos[1],
        ufs=uf_sel or None,
        codigo_ibge=mun_sel,
    )
    st.sidebar.caption(f"{len(filtered):,} registros após filtro")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Associação nos dados não implica causalidade. "
        "Fonte: SISAGUA / SIVEP-DDA / IBGE (bases públicas)."
    )
    return filtered


def page_visao(df: pd.DataFrame) -> None:
    st.subheader("Visão geral")
    m = kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Municípios", f"{m['n_municipios']:,}".replace(",", "."))
    c2.metric("Casos DDA", f"{m['casos_total']:,}".replace(",", "."))
    c3.metric("Incidência média /10k", f"{m['incidencia_media_10k']:.2f}")
    c4.metric("Taxa média E. coli", f"{m['taxa_ecoli_media']*100:.1f}%")

    serie = serie_temporal_nacional(df)
    if not serie.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=serie["data_ref"],
                y=serie["incidencia_10k"],
                name="Incidência DDA /10k",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=serie["data_ref"],
                y=serie["taxa_ecoli"] * 100,
                name="% amostras E. coli+",
                yaxis="y2",
                mode="lines",
            )
        )
        fig.update_layout(
            title="Incidência DDA e contaminação por E. coli ao longo do tempo",
            yaxis=dict(title="Casos / 10 mil hab."),
            yaxis2=dict(title="% E. coli positivas", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            height=420,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    uf = agregacao_uf(df)
    if not uf.empty:
        fig2 = px.bar(
            uf.head(15),
            x="uf",
            y="incidencia_10k",
            color="taxa_ecoli",
            labels={
                "incidencia_10k": "Incidência média /10k",
                "taxa_ecoli": "Taxa E. coli",
                "uf": "UF",
            },
            title="Top UF por incidência média de DDA",
        )
        st.plotly_chart(fig2, use_container_width=True)


def page_agua(df: pd.DataFrame) -> None:
    st.subheader("Qualidade da água")
    serie = serie_temporal_nacional(df)
    if not serie.empty:
        long = serie.melt(
            id_vars=["data_ref"],
            value_vars=["taxa_ecoli", "taxa_coliformes"],
            var_name="indicador",
            value_name="taxa",
        )
        long["indicador"] = long["indicador"].map(
            {"taxa_ecoli": "E. coli", "taxa_coliformes": "Coliformes"}
        )
        long["taxa_pct"] = long["taxa"] * 100
        fig = px.line(
            long,
            x="data_ref",
            y="taxa_pct",
            color="indicador",
            labels={"data_ref": "Mês", "taxa_pct": "% amostras positivas"},
            title="Percentual de amostras positivas",
        )
        st.plotly_chart(fig, use_container_width=True)

    rank = ranking_municipios(df, metric="taxa_ecoli", top_n=20)
    if not rank.empty:
        fig2 = px.bar(
            rank.sort_values("taxa_ecoli"),
            x="taxa_ecoli",
            y="municipio",
            color="uf",
            orientation="h",
            labels={"taxa_ecoli": "Taxa média E. coli", "municipio": "Município"},
            title="Municípios com maior taxa média de E. coli",
        )
        fig2.update_layout(height=560, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric(
        "Registros sem amostra coliformes",
        f"{int(df['sem_amostra_coliformes'].sum()):,}".replace(",", "."),
    )
    c2.metric(
        "Registros sem amostra E. coli",
        f"{int(df['sem_amostra_ecoli'].sum()):,}".replace(",", "."),
    )


def page_dda(df: pd.DataFrame) -> None:
    st.subheader("Casos de DDA")
    serie = serie_temporal_nacional(df)
    if not serie.empty:
        fig = px.area(
            serie,
            x="data_ref",
            y="casos_total",
            labels={"data_ref": "Mês", "casos_total": "Casos"},
            title="Casos totais de DDA no período",
        )
        st.plotly_chart(fig, use_container_width=True)

    faixa = casos_por_faixa_etaria(df)
    fig2 = px.pie(faixa, names="faixa", values="casos", title="Casos por faixa etária")
    st.plotly_chart(fig2, use_container_width=True)

    rank = ranking_municipios(df, metric="total_casos_por_10k_hab", top_n=20)
    if not rank.empty:
        fig3 = px.bar(
            rank.sort_values("total_casos_por_10k_hab"),
            x="total_casos_por_10k_hab",
            y="municipio",
            color="uf",
            orientation="h",
            labels={
                "total_casos_por_10k_hab": "Incidência média /10k",
                "municipio": "Município",
            },
            title="Municípios com maior incidência média de DDA",
        )
        fig3.update_layout(height=560, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig3, use_container_width=True)


def page_relacao(df: pd.DataFrame) -> None:
    st.subheader("Relação água × saúde")
    corr = correlacao_agua_saude(df)
    if not corr.empty:
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Matriz de correlação",
        )
        st.plotly_chart(fig, use_container_width=True)

    scatter = amostra_dispersao(df)
    if not scatter.empty:
        fig2 = px.scatter(
            scatter,
            x="taxa_ecoli",
            y="total_casos_por_10k_hab",
            color="uf",
            hover_data=["municipio", "ano", "mes"],
            labels={
                "taxa_ecoli": "Taxa E. coli",
                "total_casos_por_10k_hab": "Casos / 10k hab.",
            },
            title="Dispersão: E. coli × incidência DDA (amostra)",
            opacity=0.5,
        )
        st.plotly_chart(fig2, use_container_width=True)

    heat = heatmap_uf_mes(df, value="incidencia_10k")
    if not heat.empty:
        fig3 = px.imshow(
            heat,
            aspect="auto",
            labels=dict(x="Mês", y="UF", color="Incidência /10k"),
            title="Heatmap: incidência média por UF e mês",
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "Interpretação cautelosa: correlação e padrões visuais não estabelecem relação causal. "
        "Fatores como saneamento, clima, densidades e subnotificação podem confundir."
    )


def page_ml(df: pd.DataFrame) -> None:
    st.subheader("Modelo de ML")
    metrics = get_metrics()
    imp = get_importance()
    if metrics is None:
        st.warning(
            "Artefatos de ML não encontrados. Execute: `python -m src.ml_pipeline`"
        )
        return

    st.markdown(
        f"**Alvo:** `{metrics['target']}` · **Modelo:** `{metrics['model_name']}` · "
        f"Treino: {metrics['n_train']:,} · Teste 2024–25: {metrics['n_test']:,}"
    )
    c1, c2, c3 = st.columns(3)
    m = metrics["model"]
    c1.metric("RMSE (teste)", f"{m['rmse']:.2f}")
    c2.metric("MAE (teste)", f"{m['mae']:.2f}")
    c3.metric("R² (teste)", f"{m['r2']:.3f}")

    b1, b2, b3 = st.columns(3)
    base = metrics["baseline_mean"]
    lin = metrics["linear"]
    b1.metric("Baseline RMSE", f"{base['rmse']:.2f}")
    b2.metric("Linear RMSE", f"{lin['rmse']:.2f}")
    if metrics.get("holdout_2026"):
        b3.metric("Holdout 2026 RMSE", f"{metrics['holdout_2026']['rmse']:.2f}")

    if imp is not None and not imp.empty:
        top = imp.head(15)
        fig = px.bar(
            top.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Importância das variáveis (top 15)",
        )
        fig.update_layout(height=480, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Limitações")
    for lim in metrics.get("limitacoes", []):
        st.markdown(f"- {lim}")


def page_dados() -> None:
    st.subheader("Qualidade dos dados (QC)")
    if QC.exists():
        qc = json.loads(QC.read_text(encoding="utf-8"))
        st.json(qc)
    else:
        st.warning("Relatório QC não encontrado. Execute: `python -m src.data_prep`")


def main() -> None:
    st.title("Qualidade da água e DDA")
    st.caption(
        "Projeto Integrador IV — integração SISAGUA × SIVEP-DDA × IBGE · "
        "filtro temporal padrão 2024–2026"
    )

    if not PARQUET.exists():
        st.error(
            "Dados processados não encontrados. No terminal, execute:\n\n"
            "`python -m src.data_prep`\n\n"
            "depois `python -m src.ml_pipeline` e reinicie o app."
        )
        return

    df = get_data()
    filtered = sidebar_filters(df)

    tabs = st.tabs(
        [
            "Visão geral",
            "Qualidade da água",
            "Casos DDA",
            "Água × Saúde",
            "Modelo ML",
            "QC",
        ]
    )
    with tabs[0]:
        page_visao(filtered)
    with tabs[1]:
        page_agua(filtered)
    with tabs[2]:
        page_dda(filtered)
    with tabs[3]:
        page_relacao(filtered)
    with tabs[4]:
        page_ml(filtered)
    with tabs[5]:
        page_dados()


if __name__ == "__main__":
    main()
