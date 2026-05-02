import pytest

from src.predict import predict_credit_score


# Vérifie que le modèle renvoie bien une prédiction valide (0 ou 1)
def test_prediction_returns_valid_output():
    data = {
        "AMT_INCOME_TOTAL": 200000,
        "AMT_CREDIT": 500000
    }

    result = predict_credit_score(data)

    assert result in [0, 1]


# Vérifie que la prédiction fonctionne SANS SK_ID_CURR (nouveau comportement)
def test_prediction_without_sk_id_curr_works():
    data = {
        "AMT_INCOME_TOTAL": 120000,
        "AMT_CREDIT": 300000,
        "AMT_ANNUITY": 20000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "CNT_CHILDREN": 1
    }

    result = predict_credit_score(data)

    assert result in [0, 1]


# Vérifie que des données invalides provoquent bien une erreur
def test_prediction_with_invalid_input_raises_error():
    data = {
        "AMT_INCOME_TOTAL": "abc"
    }

    with pytest.raises(Exception):
        predict_credit_score(data)