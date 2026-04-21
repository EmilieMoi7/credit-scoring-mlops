from src.predict import predict_credit_score

test_input = {
    "AMT_INCOME_TOTAL": 200000,
    "AMT_CREDIT": 500000,
    "CNT_CHILDREN": 0,
    "FLAG_OWN_CAR": 1,
    "FLAG_OWN_REALTY": 1
}

result = predict_credit_score(test_input)
print("Résultat final :", result)