import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import os
import pandas as pd

from src.features import build_features

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print("Chargement des datasets...")

application = pd.read_csv(os.path.join(RAW_DATA_DIR, "application_test.csv"))
bureau = pd.read_csv(os.path.join(RAW_DATA_DIR, "bureau.csv"))
bb = pd.read_csv(os.path.join(RAW_DATA_DIR, "bureau_balance.csv"))
prev = pd.read_csv(os.path.join(RAW_DATA_DIR, "previous_application.csv"))
inst = pd.read_csv(os.path.join(RAW_DATA_DIR, "installments_payments.csv"))
pos = pd.read_csv(os.path.join(RAW_DATA_DIR, "POS_CASH_balance.csv"))
cc = pd.read_csv(os.path.join(RAW_DATA_DIR, "credit_card_balance.csv"))

print("Construction des features...")

X, _ = build_features(
    app=application,
    bureau=bureau,
    bb=bb,
    prev=prev,
    inst=inst,
    pos=pos,
    cc=cc,
)

output_path = os.path.join(PROCESSED_DATA_DIR, "features_cache.parquet")

print("Sauvegarde des features...")

X.to_parquet(output_path, index=False)

print(f"Features sauvegardées dans : {output_path}")