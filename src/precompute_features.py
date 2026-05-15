import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import os
import joblib
import pandas as pd

from src.features import (
    aggregate_previous,
    aggregate_bureau,
    aggregate_installments,
    aggregate_pos,
    aggregate_credit_card,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print("Chargement des datasets...")

bureau = pd.read_csv(os.path.join(RAW_DATA_DIR, "bureau.csv"))
bb = pd.read_csv(os.path.join(RAW_DATA_DIR, "bureau_balance.csv"))
prev = pd.read_csv(os.path.join(RAW_DATA_DIR, "previous_application.csv"))
inst = pd.read_csv(os.path.join(RAW_DATA_DIR, "installments_payments.csv"))
pos = pd.read_csv(os.path.join(RAW_DATA_DIR, "POS_CASH_balance.csv"))
cc = pd.read_csv(os.path.join(RAW_DATA_DIR, "credit_card_balance.csv"))

print("Pré-calcul des agrégations...")

precomputed_aggs = {
    "prev_agg": aggregate_previous(prev),
    "bureau_agg": aggregate_bureau(bureau, bb),
    "inst_agg": aggregate_installments(inst),
    "pos_agg": aggregate_pos(pos),
    "cc_agg": aggregate_credit_card(cc),
}

output_path = os.path.join(PROCESSED_DATA_DIR, "precomputed_aggs.joblib")

joblib.dump(precomputed_aggs, output_path)

print(f"Agrégations sauvegardées dans : {output_path}")