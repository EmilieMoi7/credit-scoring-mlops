from src.predict import predict_credit_score

# Test sans SK_ID_CURR (nouveau client)
data_new = {
    "AMT_INCOME_TOTAL": 120000,
    "AMT_CREDIT": 300000,
    "AMT_ANNUITY": 20000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2000,
    "CNT_CHILDREN": 1
}

print("Test nouveau client :", predict_credit_score(data_new))


# Test avec SK_ID_CURR (ancien fonctionnement)
data_existing = {
    "SK_ID_CURR": 100001
}

print("Test client existant :", predict_credit_score(data_existing))