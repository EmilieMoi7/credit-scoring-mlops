import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_scoring_pipeline.joblib")

model = joblib.load(MODEL_PATH)

def predict_credit_score(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return int(prediction)
    except Exception as e:
        return {"error": str(e)}