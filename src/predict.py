import os
import joblib
import pandas as pd

from src.features import build_features

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_scoring_pipeline.joblib")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

PRECOMPUTED_AGGS_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "precomputed_aggs.joblib"
)

DATASET_BASE_URL = "https://huggingface.co/datasets/Emilie7/credit-scoring-data/resolve/main"

REQUIRED_FILES = [
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
]


def download_data_if_missing():
    os.makedirs(DATA_DIR, exist_ok=True)

    for filename in REQUIRED_FILES:
        file_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(file_path):
            url = f"{DATASET_BASE_URL}/{filename}"
            print(f"Téléchargement de {filename}...")
            df = pd.read_csv(url)
            df.to_csv(file_path, index=False)


download_data_if_missing()

model = joblib.load(MODEL_PATH)

application = pd.read_csv(os.path.join(DATA_DIR, "application_test.csv"))
bureau = pd.read_csv(os.path.join(DATA_DIR, "bureau.csv"))
bb = pd.read_csv(os.path.join(DATA_DIR, "bureau_balance.csv"))
prev = pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv"))
inst = pd.read_csv(os.path.join(DATA_DIR, "installments_payments.csv"))
pos = pd.read_csv(os.path.join(DATA_DIR, "POS_CASH_balance.csv"))
cc = pd.read_csv(os.path.join(DATA_DIR, "credit_card_balance.csv"))

precomputed_aggs = joblib.load(PRECOMPUTED_AGGS_PATH)


def get_reference_client() -> pd.DataFrame:
    """
    Crée un client de référence à partir de application_test.
    Numériques : médiane
    Catégorielles : modalité la plus fréquente
    """
    ref = {}

    for col in application.columns:
        if col == "SK_ID_CURR":
            ref[col] = application[col].iloc[0]
        elif pd.api.types.is_numeric_dtype(application[col]):
            ref[col] = application[col].median()
        else:
            mode_value = application[col].mode(dropna=True)
            ref[col] = mode_value.iloc[0] if not mode_value.empty else None

    return pd.DataFrame([ref])


def prepare_input(data: dict) -> pd.DataFrame:
    sk_id_curr = data.get("SK_ID_CURR")

    if sk_id_curr is not None:
        app_client = application[application["SK_ID_CURR"] == sk_id_curr].copy()

        if app_client.empty:
            raise ValueError(f"Aucun client trouvé avec SK_ID_CURR={sk_id_curr}")
    else:
        app_client = get_reference_client()

    # Remplacer les valeurs applicatives par celles saisies
    for key, value in data.items():
        if key in app_client.columns:
            app_client.loc[:, key] = value

    X, _ = build_features(
        app=app_client,
        bureau=bureau,
        bb=bb,
        prev=prev,
        inst=inst,
        pos=pos,
        cc=cc,
        precomputed_aggs=precomputed_aggs,
    )

    missing_cols = [col for col in model.feature_names_in_ if col not in X.columns]

    for col in missing_cols:
        X[col] = 0

    X = X[list(model.feature_names_in_)]

    return X


def predict_credit_score(data: dict):
    X = prepare_input(data)
    prediction = model.predict(X)[0]
    return int(prediction)