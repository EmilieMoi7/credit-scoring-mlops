import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_scoring_pipeline.joblib")
FEATURES_CACHE_PATH = os.path.join(BASE_DIR, "data", "processed", "features_cache.parquet")

model = joblib.load(MODEL_PATH)

features_cache = pd.read_parquet(FEATURES_CACHE_PATH)


def get_reference_features() -> pd.DataFrame:
    """
    Récupère une ligne de référence depuis le cache de features pré-calculées.
    """
    return features_cache.iloc[[0]].copy()


def prepare_input(data: dict) -> pd.DataFrame:
    """
    Prépare les features d'entrée à partir du cache pré-calculé.
    Les valeurs saisies par l'utilisateur remplacent celles de la ligne de référence.
    """
    sk_id_curr = data.get("SK_ID_CURR")

    if sk_id_curr is not None and "SK_ID_CURR" in features_cache.columns:
        X = features_cache[features_cache["SK_ID_CURR"] == sk_id_curr].copy()

        if X.empty:
            raise ValueError(f"Aucun client trouvé avec SK_ID_CURR={sk_id_curr}")
    else:
        X = get_reference_features()

    for key, value in data.items():
        if key in X.columns:
            X.loc[:, key] = value

    missing_cols = [col for col in model.feature_names_in_ if col not in X.columns]

    for col in missing_cols:
        X[col] = 0

    X = X[list(model.feature_names_in_)]

    return X


def predict_credit_score(data: dict):
    X = prepare_input(data)
    prediction = model.predict(X)[0]
    return int(prediction)