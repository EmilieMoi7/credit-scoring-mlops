import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Monitoring Crédit Scoring", layout="wide")

st.title("Dashboard de monitoring - Crédit scoring")

logs_path = Path("logs/production_logs.jsonl")


@st.cache_data
def load_logs(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


df = load_logs(logs_path)

st.subheader("Aperçu des logs de production")

if df.empty:
    st.warning("Aucun log disponible. Vérifie que logs/production_logs.jsonl existe.")
    st.stop()

st.dataframe(df)

col1, col2, col3 = st.columns(3)

col1.metric("Nombre de requêtes", len(df))
col2.metric("Latence moyenne", f"{df['latency'].mean():.3f} sec")

error_rate = (df["status"] == "error").mean() * 100
col3.metric("Taux d'erreur", f"{error_rate:.2f}%")

st.subheader("Analyse des statuts et prédictions")

col1, col2 = st.columns(2)

with col1:
    st.write("Distribution des statuts")
    status_counts = df["status"].value_counts()

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(status_counts.index, status_counts.values)
    ax.set_xlabel("Statut")
    ax.set_ylabel("Nombre")
    st.pyplot(fig, use_container_width=False)

with col2:
    st.write("Distribution des prédictions")

    fig, ax = plt.subplots(figsize=(4, 3))
    df["prediction"].dropna().hist(ax=ax)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig, use_container_width=False)


def extract_inputs(df):
    inputs = []

    if "input" not in df.columns:
        return pd.DataFrame()

    for val in df["input"].dropna():
        try:
            if isinstance(val, str):
                inputs.append(json.loads(val))
            elif isinstance(val, dict):
                inputs.append(val)
        except Exception:
            pass

    return pd.DataFrame(inputs)


df_inputs = extract_inputs(df)

cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]

st.subheader("Distribution des variables d'entrée")

if df_inputs.empty:
    st.warning("Aucune donnée d'entrée exploitable.")
    st.stop()

df_inputs_clean = df_inputs[cols].dropna()

if df_inputs_clean.empty:
    st.warning("Pas assez de données pour afficher les distributions.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Revenus")
    fig, ax = plt.subplots()
    ax.hist(df_inputs_clean["AMT_INCOME_TOTAL"], bins=10)
    ax.set_xlabel("Revenu")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

with col2:
    st.write("Montants de crédit")
    fig, ax = plt.subplots()
    ax.hist(df_inputs_clean["AMT_CREDIT"], bins=10)
    ax.set_xlabel("Montant crédit")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

with col3:
    st.write("Âges")
    ages = -df_inputs_clean["DAYS_BIRTH"] / 365
    fig, ax = plt.subplots()
    ax.hist(ages, bins=10)
    ax.set_xlabel("Âge")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)