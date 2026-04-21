#Test pour vérifier le bon fonctionnement du modèle 
from src.predict import predict_credit_score

def test_prediction_returns_valid_output():
    data = {
        "AMT_INCOME_TOTAL": 200000,
        "AMT_CREDIT": 500000
    }

    result = predict_credit_score(data)

    assert result in [0, 1]

#Test pour vérifier la gestion des erreurs 
def test_prediction_with_invalid_input():
    data = {
        "AMT_INCOME_TOTAL": "abc"
    }

    result = predict_credit_score(data)

    assert isinstance(result, dict)
    assert "error" in result