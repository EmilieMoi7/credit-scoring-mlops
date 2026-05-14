import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import json
import time

import numpy as np
import psutil

from src.predict import predict_credit_score


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

N_REQUESTS = 50

samples = [
    {
        "DAYS_BIRTH": -16425,
        "AMT_INCOME_TOTAL": 120000,
        "AMT_CREDIT": 100000,
        "AMT_ANNUITY": 6666.67,
        "CNT_CHILDREN": 0,
        "FLAG_OWN_CAR": 1,
        "FLAG_OWN_REALTY": 1,
    },
    {
        "DAYS_BIRTH": -8030,
        "AMT_INCOME_TOTAL": 20000,
        "AMT_CREDIT": 300000,
        "AMT_ANNUITY": 20000,
        "CNT_CHILDREN": 3,
        "FLAG_OWN_CAR": 0,
        "FLAG_OWN_REALTY": 0,
    },
    {
        "DAYS_BIRTH": -10950,
        "AMT_INCOME_TOTAL": 50000,
        "AMT_CREDIT": 250000,
        "AMT_ANNUITY": 8333.33,
        "CNT_CHILDREN": 1,
        "FLAG_OWN_CAR": 0,
        "FLAG_OWN_REALTY": 1,
    },
]


def main():
    latencies = []
    cpu_values = []
    memory_values = []
    errors = 0

    total_start = time.time()

    for i in range(N_REQUESTS):
        sample = samples[i % len(samples)]

        start = time.time()

        try:
            _ = predict_credit_score(sample)
        except Exception:
            errors += 1

        latency = time.time() - start

        latencies.append(latency)
        cpu_values.append(psutil.cpu_percent(interval=None))
        memory_values.append(psutil.Process().memory_info().rss / 1024 / 1024)

    total_time = time.time() - total_start

    metrics = {
        "version": "baseline",
        "n_requests": N_REQUESTS,
        "mean_latency_sec": float(np.mean(latencies)),
        "min_latency_sec": float(np.min(latencies)),
        "max_latency_sec": float(np.max(latencies)),
        "p95_latency_sec": float(np.percentile(latencies, 95)),
        "throughput_req_per_sec": float(N_REQUESTS / total_time),
        "cpu_avg_percent": float(np.mean(cpu_values)),
        "memory_avg_mb": float(np.mean(memory_values)),
        "error_rate_percent": float(errors / N_REQUESTS * 100),
    }

    output_path = REPORTS_DIR / "baseline_metrics.json"

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
    print(f"\nRésultats sauvegardés dans : {output_path}")


if __name__ == "__main__":
    main()