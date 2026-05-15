import pandas as pd

def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    prev_agg = prev.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
    ).reset_index()
    return prev_agg


def aggregate_bureau(bureau: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BB_MONTHS_MAX=("MONTHS_BALANCE", "max"),
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
    ).reset_index()

    bureau2 = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    bureau_agg = bureau2.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_CREDIT_SUM_DEBT_MEAN=("AMT_CREDIT_SUM_DEBT", "mean"),
        BUREAU_BB_MONTHS_COUNT_MEAN=("BB_MONTHS_COUNT", "mean"),
    ).reset_index()

    return bureau_agg


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    inst2 = inst.copy()
    inst2["DAYS_LATE"] = inst2["DAYS_ENTRY_PAYMENT"] - inst2["DAYS_INSTALMENT"]
    inst2["LATE_FLAG"] = (inst2["DAYS_LATE"] > 0).astype(int)
    inst2["AMT_DIFF"] = inst2["AMT_INSTALMENT"] - inst2["AMT_PAYMENT"]
    inst2["UNDERPAY_FLAG"] = (inst2["AMT_DIFF"] > 0).astype(int)

    inst_agg = inst2.groupby("SK_ID_CURR").agg(
        INST_COUNT=("SK_ID_CURR", "count"),
        INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INST_LATE_RATE=("LATE_FLAG", "mean"),
        INST_UNDERPAY_RATE=("UNDERPAY_FLAG", "mean"),
        INST_AMT_DIFF_MEAN=("AMT_DIFF", "mean"),
    ).reset_index()

    return inst_agg


def aggregate_pos(pos: pd.DataFrame) -> pd.DataFrame:
    return pos.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_CURR", "count"),
        POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    ).reset_index()


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    return cc.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_CURR", "count"),
        CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_DRAWINGS_SUM=("AMT_DRAWINGS_ATM_CURRENT", "sum"),
        CC_DPD_MEAN=("SK_DPD", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
    ).reset_index()


def build_features(
    app,
    bureau=None,
    bb=None,
    prev=None,
    inst=None,
    pos=None,
    cc=None,
    high_missing_cols=None,
    target_col="TARGET",
    precomputed_aggs=None
):
    df = app.copy()

    if precomputed_aggs is not None:
        prev_agg = precomputed_aggs["prev_agg"]
        bureau_agg = precomputed_aggs["bureau_agg"]
        inst_agg = precomputed_aggs["inst_agg"]
        pos_agg = precomputed_aggs["pos_agg"]
        cc_agg = precomputed_aggs["cc_agg"]
    else:
        prev_agg = aggregate_previous(prev)
        bureau_agg = aggregate_bureau(bureau, bb)
        inst_agg = aggregate_installments(inst)
        pos_agg = aggregate_pos(pos)
        cc_agg = aggregate_credit_card(cc)

    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
    df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    df = df.merge(inst_agg, on="SK_ID_CURR", how="left")
    df = df.merge(pos_agg, on="SK_ID_CURR", how="left")
    df = df.merge(cc_agg, on="SK_ID_CURR", how="left")

    df["HAS_CC"] = (df["CC_COUNT"] > 0).astype(int)
    df["HAS_BUREAU"] = (df["BUREAU_COUNT"] > 0).astype(int)
    df["HAS_POS"] = (df["POS_COUNT"] > 0).astype(int)
    df["HAS_INST"] = (df["INST_COUNT"] > 0).astype(int)

    for prefix in ["CC_", "BUREAU_", "POS_", "INST_"]:
        cols = [col for col in df.columns if col.startswith(prefix)]
        df[cols] = df[cols].fillna(0)

    if high_missing_cols is not None:
        for col in high_missing_cols:
            if col in df.columns:
                df[col + "_IS_MISSING"] = df[col].isna().astype(int)

    if target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    if "SK_ID_CURR" in X.columns:
        X = X.drop(columns=["SK_ID_CURR"])

    return X, y