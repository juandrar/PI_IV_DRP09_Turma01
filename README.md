# Qualidade da água e DDA — Projeto Integrador IV

Análise da relação entre qualidade da água (coliformes / *E. coli*) e doenças de veiculação hídrica (DDA), com pipeline de dados, modelo de ML e dashboard Streamlit.

## Dados

| Arquivo | Descrição |
|---------|-----------|
| `arq/extracao_dados_agua_dda-2018-2026-com_habitantes-v01.csv` | Água + casos DDA + habitantes (mensal, 2018–2026) |
| `arq/pop_ibge_2018_2026-v01.xlsx` | População estimada IBGE por município/ano |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# EDA com Jupyter (opcional):
# pip install -r requirements-dev.txt
```

## Pipeline

```bash
# 1. Limpeza, join IBGE, features → Parquet
python -m src.data_prep

# 2. Treino do modelo (exige passo 1)
python -m src.ml_pipeline

# 3. Dashboard (filtro temporal padrão: 2024–2026)
streamlit run app/streamlit_app.py

# 4. Validação
python -m src.validate
```

## Estrutura

```
src/data_prep.py      # ETL
src/analysis.py       # Agregações e indicadores
src/ml_pipeline.py    # Treino e métricas
app/streamlit_app.py  # Dashboard
data/processed/       # Parquet processado
models/               # Artefatos de ML
notebooks/            # EDA
```

## Observações

- Associação nos dados **não implica causalidade**.
- Cobertura de 2026 pode estar parcial.
- População IBGE é usada para validar/recalcular a taxa de casos por 10 mil habitantes.
# PI_IV_DRP09_TURMA01
# PI_IV_DRP09_TURMA01
