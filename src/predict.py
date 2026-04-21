import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_scoring_pipeline.joblib")

model = joblib.load(MODEL_PATH)

def prepare_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    missing_cols = [col for col in model.feature_names_in_ if col not in df.columns]

    if missing_cols:
        df_missing = pd.DataFrame([{col: 0 for col in missing_cols}])
        df = pd.concat([df, df_missing], axis=1)

    df = df[list(model.feature_names_in_)]

    return df

def predict_credit_score(data: dict):
    try:
        df = prepare_input(data)
        prediction = model.predict(df)[0]
        return int(prediction)
    except Exception as e:
        return {"error": str(e)}