import gradio as gr
import json
import time
import os
import psutil

from datetime import datetime
from src.predict import predict_credit_score


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_prediction(
    data,
    prediction,
    latency,
    inference_time,
    cpu_percent,
    memory_mb,
    status="success",
    error=None
):
    os.makedirs("logs", exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": data,
        "prediction": prediction,
        "latency": latency,
        "inference_time": inference_time,
        "cpu_percent": cpu_percent,
        "memory_mb": memory_mb,
        "status": status,
        "error": error
    }

    with open("logs/production_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def predict_interface(
    age,
    amt_income_total,
    amt_credit,
    loan_duration,
    cnt_children,
    flag_own_car,
    flag_own_realty
):
    start_time = time.time()
    inference_time = None

    try:
        # Validations métier
        if age < 18 or age > 100:
            raise ValueError("L'âge doit être compris entre 18 et 100 ans.")

        if amt_income_total <= 0:
            raise ValueError("Le revenu total doit être positif.")

        if amt_credit <= 0:
            raise ValueError("Le montant du crédit doit être positif.")

        if loan_duration <= 0 or loan_duration > 40:
            raise ValueError("La durée du prêt doit être comprise entre 1 et 40 ans.")

        if cnt_children < 0:
            raise ValueError("Le nombre d'enfants ne peut pas être négatif.")

        if flag_own_car not in [0, 1]:
            raise ValueError("Voiture doit valoir 0 ou 1.")

        if flag_own_realty not in [0, 1]:
            raise ValueError("Bien immobilier doit valoir 0 ou 1.")

        # Conversion métier
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

        # Mesure spécifique de l'inférence
        inference_start = time.time()
        result = predict_credit_score(data)
        inference_time = time.time() - inference_start

        latency = time.time() - start_time
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_mb = get_memory_mb()

        log_prediction(
            data=data,
            prediction=result,
            latency=latency,
            inference_time=inference_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            status="success"
        )

        if result == 0:
            return "Crédit accordé (risque faible)"
        else:
            return "Crédit refusé (risque élevé)"

    except Exception as e:
        latency = time.time() - start_time
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_mb = get_memory_mb()

        log_prediction(
            data={},
            prediction=None,
            latency=latency,
            inference_time=inference_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            status="error",
            error=str(e)
        )

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