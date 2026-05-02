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
│   └── raw/                # Données récupérées depuis Hugging Face
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
├── requirements.txt
├── Dockerfile
├── .github/workflows/
│   └── ci-cd.yml
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

## Technologies utilisées 

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Gradio
- Docker
- GitHub Actions

