---
title: Credit Scoring App
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Credit Scoring MLOps

## Description
Ce projet met en place un pipeline de credit scoring avec une approche MLOps complète :
- Feature engineering
- Entraînement et tracking des modèles avec MLflow
- API de prédiction avec Gradio
- Déploiement automatisé via CI/CD sur Hugging Face Spaces

---

## Structure du projet

```bash
├── data/
│   ├── raw/                # Données récupérées depuis Hugging Face
│   └── processed/          # Features pré-calculées
├── notebooks/
│   └── 01_credit_scoring_pipeline.ipynb
├── src/
│   ├── features.py
│   └── predict.py
├── tests/
│   └── test_predict.py
├── models/
│   └── credit_scoring_pipeline.joblib
├── app.py                  # Interface Gradio
├── dashboard.py            # Dashboard Streamlit de monitoring
├── requirements.txt
├── Dockerfile
├── .github/workflows/
│   └── ci-cd.yml
├── benchmarks/
│   ├── benchmark_baseline.py
│   └── benchmark_optimized.py
├── reports/
│   ├── baseline_metrics.json
│   ├── optimized_metrics_v1.json
│   └── optimisation_report.md
└── README.md
```

## Données

Les données ne sont pas incluses dans ce dépôt.
Elles sont téléchargées dynamiquement depuis Hugging Face :
https://huggingface.co/datasets/Emilie7/credit-scoring-data

## API de prédiction 

L’application permet de prédire le risque de crédit à partir de variables utilisateur :
- âge
- revenu
- montant du crédit
- durée du prêt
- etc.

Le modèle est chargé une seule fois au démarrage pour optimiser les performances.

## CI/CD

Le pipeline GitHub Actions automatise :

- exécution des tests (pytest)
- build de l’image Docker
- déploiement automatique vers Hugging Face Space

Chaque git push sur la branche develop déclenche le pipeline.

## Lancer le projet

1. Créer un environnement :
```bash
conda create -n credit_scoring python=3.11
conda activate credit_scoring
```

2. Installer les dépendances 
```bash
pip install -r requirements.txt
```

3. Lancer l'application

```bash
python app.py
```

## Lancer les tests

```bash
PYTHONPATH=. pytest
```

## Analyse de couverture des tests

Le coverage des tests peut être exécuté avec :

```bash
PYTHONPATH=. pytest --cov=src
```

Les tests couvrent principalement le pipeline d'inférence et les cas critiques de prédiction dans `predict.py`.

## Benchmarks de performance

Benchmark baseline :

```bash
python benchmarks/benchmark_baseline.py
```

Benchmark optimisé :

```bash
python benchmarks/benchmark_optimized.py
```

## Technologies utilisées 

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Gradio
- Docker
- GitHub Actions
- PyArrow

## Monitoring et optimisation

Le projet inclut un système de monitoring permettant de collecter :
- les temps d'inférence ;
- l'utilisation CPU ;
- l'utilisation mémoire ;
- les inputs / outputs des prédictions ;
- les erreurs d'exécution.

Les données sont stockées dans des logs JSON et visualisées avec un dashboard Streamlit.

Lancer le dashboard :

```bash
streamlit run dashboard.py
```

Une phase d'analyse de performance a été réalisée avec :
- benchmark de baseline ;
- profiling avec cProfile ;
- optimisation du preprocessing.

L'optimisation par cache de features pré-calculées a permis de réduire la latence moyenne :
- de ~3 sec à ~0.006 sec ;
- et d'augmenter fortement le throughput.

Le profiling a montré que le principal bottleneck provenait du recalcul des features avec `build_features()` à chaque requête.

Une optimisation basée sur un cache de features pré-calculées (`features_cache.parquet`) a ensuite été mise en place.

## Résultats de l'optimisation

| Métrique | Baseline | Optimized v1 |
|---|---|---|
| Mean latency | 3.01 sec | 0.006 sec |
| Throughput | 0.33 req/sec | 159 req/sec |
| Memory usage | 2106 MB | 357 MB |
| Error rate | 0 % | 0 % |


