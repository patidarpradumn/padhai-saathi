"""PadhAI Saathi - Streamlit UI. Run: streamlit run app.py  (after python src/train.py)"""
import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data import FEATURES, PASS_MARK
from predict import load_models, predict_df
from recommend import recommend, teacher_alert

st.set_page_config(page_title="PadhAI Saathi", page_icon="🎓", layout="wide")
st.title("🎓 PadhAI Saathi")
st.caption("AI learning-gap detector for rural students - predicts risk early and suggests what to do next.")

if not (ROOT / "models" / "linreg.pkl").exists():
    st.error("Models not found. Run `python src/train.py` first.")
    st.stop()

models = load_models()
meta = json.loads((ROOT / "models" / "metrics.json").read_text())
if meta.get("data_source") != "real":
    st.warning("Models were trained on SYNTHETIC demo data. Put data/student-mat.csv in place and run "
               "`python src/train.py`. Do not report these metrics.")
tab1, tab2, tab3 = st.tabs(["Single student", "Class view (teacher)", "Model performance"])

# ---------------- Tab 1: single student ----------------
with tab1:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Student details")
        name = st.text_input("Name", "Student A")
        d = {
            "studytime": st.slider("Weekly study time (1=<2h ... 4=>10h)", 1, 4, 2),
            "failures": st.slider("Past class failures", 0, 3, 0),
            "absences": st.slider("Absences", 0, 40, 4),
            "G1": st.slider("First period grade (0-20)", 0, 20, 10),
            "G2": st.slider("Second period grade (0-20)", 0, 20, 10),
            "goout": st.slider("Going out (1-5)", 1, 5, 3),
            "Dalc": st.slider("Weekday alcohol (1-5)", 1, 5, 1),
            "freetime": st.slider("Free time (1-5)", 1, 5, 3),
            "health": st.slider("Health (1-5)", 1, 5, 3),
        }
    res = predict_df(pd.DataFrame([d]), models).iloc[0]
    passed, risk, g3 = int(res["pred_pass"]), res["risk"], float(res["pred_G3"])

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted grade", f"{g3:.1f}")
        c2.metric("Pass / Fail", "PASS" if passed else "FAIL")
        c3.metric("Risk group", risk.replace(" risk", ""))
        st.caption("Predicted grade is out of 20.")
        alert = teacher_alert(name, g3, passed, risk)
        actions = recommend(d, g3, passed, risk)
        n_high = sum(1 for x in actions if x["priority"] == "High")
        if alert:
            st.error(alert)
        elif n_high:
            st.warning(f"No automatic alert, but {n_high} high-priority action(s) below.")
        else:
            st.success("On track - no teacher alert.")
        st.subheader("Recommended actions")
        for a in actions:
            st.markdown(f"**[{a['priority']}]** {a['action']}  \n*Why:* {a['why']}")

# ---------------- Tab 2: class view ----------------
with tab2:
    st.write("Upload a class CSV with columns: " + ", ".join(["name"] + FEATURES))
    sample = ROOT / "data" / "sample_class.csv"
    if sample.exists():
        st.download_button("Download sample CSV", sample.read_bytes(), "sample_class.csv")
    up = st.file_uploader("Class CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            out = predict_df(df, models)
            order = {"High risk": 0, "Medium risk": 1, "Low risk": 2}
            out = out.sort_values("risk", key=lambda s: s.map(order))
            st.metric("Students flagged High risk", int((out["risk"] == "High risk").sum()))
            show = [c for c in ["name"] if c in out.columns] + ["pred_G3", "pred_pass", "risk", "top_action"]
            st.dataframe(out[show], use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False), "class_results.csv")

# ---------------- Tab 3: metrics ----------------
with tab3:
    st.json(meta)
    st.caption(f"Pass mark = {PASS_MARK}/20. Clusters are built from habits only (not grades).")
