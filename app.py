import gradio as gr
from src.predict import predict_credit_score

def predict_interface(amt_income_total, amt_credit, cnt_children, flag_own_car, flag_own_realty):
    data = {
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "CNT_CHILDREN": cnt_children,
        "FLAG_OWN_CAR": flag_own_car,
        "FLAG_OWN_REALTY": flag_own_realty
    }
    result = predict_credit_score(data)
    return result

demo = gr.Interface(
    fn=predict_interface,
    inputs=[
        gr.Number(label="Revenu total"),
        gr.Number(label="Montant du crédit"),
        gr.Number(label="Nombre d'enfants"),
        gr.Number(label="Possède une voiture ? (0 ou 1)"),
        gr.Number(label="Possède un bien immobilier ? (0 ou 1)")
    ],
    outputs=gr.Number(label="Prédiction (0 = non défaut, 1 = défaut)"),
    title="API de Credit Scoring",
    description="Prédiction du risque de défaut à partir des données saisies."
)

if __name__ == "__main__":
    demo.launch()