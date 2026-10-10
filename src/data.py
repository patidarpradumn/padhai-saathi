"""Data loading. Uses the real UCI file if present, else synthetic data."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = ROOT / "data" / "student-mat.csv"   # UCI Student Performance (sep=';')

FEATURES = ["studytime", "failures", "absences", "G1", "G2",
            "goout", "Dalc", "freetime", "health"]
TARGET = "G3"          # final grade, 0-20
PASS_MARK = 10         # G3 >= 10 -> pass


def make_synthetic(n=600, seed=42):
    rng = np.random.default_rng(seed)
    studytime = rng.integers(1, 5, n)
    failures = rng.choice([0, 1, 2, 3], n, p=[.75, .15, .06, .04])
    absences = rng.poisson(5, n)
    goout = rng.integers(1, 6, n)
    Dalc = rng.integers(1, 6, n)
    freetime = rng.integers(1, 6, n)
    health = rng.integers(1, 6, n)
    ability = rng.normal(11, 3, n) + 0.8 * studytime - 1.8 * failures
    G1 = np.clip(ability + rng.normal(0, 1.5, n), 0, 20).round()
    G2 = np.clip(G1 + rng.normal(0, 1.5, n), 0, 20).round()
    G3 = np.clip(G2 + rng.normal(0, 1.5, n) - 0.05 * absences - 0.2 * (Dalc - 1), 0, 20).round()
    return pd.DataFrame(dict(studytime=studytime, failures=failures, absences=absences,
                             G1=G1, G2=G2, goout=goout, Dalc=Dalc, freetime=freetime,
                             health=health, G3=G3))


def data_source():
    return "real" if REAL_CSV.exists() else "synthetic"


def load_data():
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV, sep=";")
        print(f"[data] Loaded REAL dataset: {REAL_CSV.name} ({len(df)} rows)")
    else:
        df = make_synthetic()
        print("[data] WARNING: real CSV not found -> using SYNTHETIC data. "
              "Put UCI student-mat.csv in data/ for real results.")
    df = df[FEATURES + [TARGET]].dropna().reset_index(drop=True)
    return df
