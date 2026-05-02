import gradio as gr
from src.predict import predict_credit_score


def predict_interface(
    age,
    amt_income_total,
    amt_credit,
    loan_duration,
    cnt_children,
    flag_own_car,
    flag_own_realty
):
    try:
        # Validations métier
        if age <= 0 or age > 100:
            return "Erreur : l'âge doit être compris entre 1 et 100 ans."

        if amt_income_total <= 0:
            return "Erreur : le revenu total doit être positif."

        if amt_credit <= 0:
            return "Erreur : le montant du crédit doit être positif."

        if loan_duration <= 0 or loan_duration > 40:
            return "Erreur : la durée du prêt doit être comprise entre 1 et 40 ans."

        if cnt_children < 0:
            return "Erreur : le nombre d'enfants ne peut pas être négatif."

        if flag_own_car not in [0, 1]:
            return "Erreur : voiture doit valoir 0 ou 1."

        if flag_own_realty not in [0, 1]:
            return "Erreur : bien immobilier doit valoir 0 ou 1."

        # Conversion métier vers colonnes attendues
        days_birth = -int(age * 365)
        amt_annuity = amt_credit / loan_duration

        data = {
            "DAYS_BIRTH": days_birth,
            "AMT_INCOME_TOTAL": amt_income_total,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "CNT_CHILDREN": cnt_children,
            "FLAG_OWN_CAR": int(flag_own_car),
            "FLAG_OWN_REALTY": int(flag_own_realty),
        }

        result = predict_credit_score(data)

        if result == 0:
            return "✔ Crédit accordé (risque faible)"
        else:
            return "❌ Crédit refusé (risque élevé)"

    except Exception as e:
        return f"Erreur : {str(e)}"


demo = gr.Interface(
    fn=predict_interface,
    inputs=[
        gr.Number(label="Âge"),
        gr.Number(label="Revenu total"),
        gr.Number(label="Montant du crédit"),
        gr.Number(label="Durée du prêt (en années)"),
        gr.Number(label="Nombre d'enfants"),
        gr.Number(label="Possède une voiture ? (0 ou 1)"),
        gr.Number(label="Possède un bien immobilier ? (0 ou 1)")
    ],
    outputs=gr.Textbox(label="Décision de crédit"),
    title="Simulation de crédit",
    description="Simulation d'une décision d'octroi de crédit à partir des données saisies."
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)