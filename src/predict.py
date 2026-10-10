"""Shared prediction logic used by the Streamlit app and tests."""
import joblib
import pandas as pd
from data import FEATURES, ROOT
from recommend import recommend

M = ROOT / "models"


def load_models():
    return {
        "lin": joblib.load(M / "linreg.pkl"),
        "tree_r": joblib.load(M / "tree_reg.pkl"),
        "tree_c": joblib.load(M / "tree_clf.pkl"),
        "cl": joblib.load(M / "cluster.pkl"),
    }


def predict_df(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Return df with predicted G3, pass/fail, risk group and top action per row."""
    X = df[FEATURES]
    out = df.copy()
    out["pred_G3"] = models["lin"].predict(X).round(1)
    out["pred_pass"] = models["tree_c"].predict(X)
    cl = models["cl"]
    out["risk"] = [cl["names"][int(c)] for c in cl["km"].predict(cl["scaler"].transform(X[cl["cols"]]))]
    out["top_action"] = [
        recommend(r[FEATURES].to_dict(), r["pred_G3"], int(r["pred_pass"]), r["risk"])[0]["action"]
        for _, r in out.iterrows()
    ]
    return out
