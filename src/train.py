"""Train regression, classification and clustering models. Run: python src/train.py"""
import json, sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FEATURES, TARGET, PASS_MARK, ROOT, load_data, data_source

MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    y_cls = (y >= PASS_MARK).astype(int)
    Xtr, Xte, ytr, yte, ctr, cte = train_test_split(X, y, y_cls, test_size=0.2, random_state=42)

    metrics = {"data_source": data_source(), "n_rows": int(len(df))}

    # 1) Regression: predict G3
    lin = LinearRegression().fit(Xtr, ytr)
    tree_r = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Xtr, ytr)
    for name, m in [("linear_regression", lin), ("tree_regressor", tree_r)]:
        p = m.predict(Xte)
        metrics[name] = {"rmse": float(np.sqrt(mean_squared_error(yte, p))),
                         "r2": float(r2_score(yte, p))}

    # 2) Classification: pass / fail
    tree_c = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xtr, ctr)
    metrics["tree_classifier"] = {"accuracy": float(accuracy_score(cte, tree_c.predict(Xte)))}
    metrics["baseline_always_pass_accuracy"] = float(max(cte.mean(), 1 - cte.mean()))

    # 3) Clustering: 3 risk groups on behaviour features (NOT grades)
    behav = ["studytime", "failures", "absences", "goout", "Dalc", "freetime", "health"]
    scaler = StandardScaler().fit(df[behav])
    Z = scaler.transform(df[behav])
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
    metrics["kmeans_silhouette"] = float(silhouette_score(Z, km.labels_))

    # label clusters low/medium/high risk by their mean final grade
    means = df.assign(c=km.labels_).groupby("c")[TARGET].mean().sort_values()
    risk_names = {int(c): n for c, n in zip(means.index, ["High risk", "Medium risk", "Low risk"])}
    metrics["cluster_mean_G3"] = {risk_names[int(c)]: float(v) for c, v in means.items()}

    joblib.dump(lin, MODELS / "linreg.pkl")
    joblib.dump(tree_r, MODELS / "tree_reg.pkl")
    joblib.dump(tree_c, MODELS / "tree_clf.pkl")
    joblib.dump({"km": km, "scaler": scaler, "cols": behav, "names": risk_names}, MODELS / "cluster.pkl")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
