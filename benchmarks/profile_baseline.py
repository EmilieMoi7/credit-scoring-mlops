import sys
from pathlib import Path
import cProfile
import pstats

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.predict import predict_credit_score


sample = {
    "DAYS_BIRTH": -16425,
    "AMT_INCOME_TOTAL": 120000,
    "AMT_CREDIT": 100000,
    "AMT_ANNUITY": 6666.67,
    "CNT_CHILDREN": 0,
    "FLAG_OWN_CAR": 1,
    "FLAG_OWN_REALTY": 1,
}


def run_profile():
    for _ in range(5):
        predict_credit_score(sample)


if __name__ == "__main__":
    output_path = ROOT_DIR / "reports" / "profile_baseline.prof"
    txt_output_path = ROOT_DIR / "reports" / "profile_baseline.txt"

    profiler = cProfile.Profile()
    profiler.enable()

    run_profile()

    profiler.disable()
    profiler.dump_stats(output_path)

    with open(txt_output_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")
        stats.print_stats(30)

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print(f"\nProfiling sauvegardé dans : {output_path}")
    print(f"Rapport texte sauvegardé dans : {txt_output_path}")