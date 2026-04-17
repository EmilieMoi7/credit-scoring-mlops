# Credit Scoring MLOps

## Description
Ce projet met en place un pipeline de credit scoring avec suivi des expérimentations via MLflow.

Le notebook principal contient :
- Analyse exploratoire des données (EDA)
- Feature engineering
- Pipeline de modélisation
- Suivi des expérimentations avec MLflow

## Structure du projet

```bash
├── data/
│ └── raw/ # Données brutes (non versionnées)
├── notebooks/
│ └── 01_credit_scoring_pipeline.ipynb
├── mlruns/ # Tracking MLflow (ignoré par Git)
├── .gitignore
└── README.md
```

## Données

Les données ne sont pas incluses dans ce dépôt.

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

3. Lancer le notebook 

Ouvrir le projet dans VS Code et exécuter le notebook avec l'extension Jupyter.

## Technologies utilisées 

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Matplotlib / Seaborn

