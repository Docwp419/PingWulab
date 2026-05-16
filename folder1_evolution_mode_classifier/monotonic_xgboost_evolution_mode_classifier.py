# -*- coding: utf-8 -*-

import os
import re
import math
import warnings
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# ---------------------------- Paths and switches ----------------------------
BASE = r"./evolution_mode_model"
TRAIN_XLSX = os.path.join(BASE, "evolution_mode_training_set.xlsx")
PRED_XLSX = os.path.join(
    BASE,
    "sample_subclone_topology_and_relative_abundance_normalized.xlsx",
)
OUT_XLSX = os.path.join(BASE, "HNSCC_evolution_mode_predictions.xlsx")
MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_monotone_phi.joblib")
THR_PATH = os.path.join(MODEL_DIR, "decision_threshold.txt")

USE_SAVED_MODEL = False   # True: use saved model and threshold for inference if available.
FORCE_RETRAIN = False     # True: force model retraining and ignore saved model files.

# ---------------------------- Table reading utilities ----------------------------
def _std_col(s):
    return str(s).strip().lower().replace(":", ":").replace(" ", "").replace("_", "")


SAMPLE_CANDS = {"sample", "samplename", "sampleid", "id", "case", "caseid", "patient", "patientid"}
CLONE_CANDS = {"clone", "subclone", "subcloneid", "cloneid", "node", "subclonelayer", "clonelayer"}
PROP_CANDS = {"prop", "proportion", "frequency", "abundance", "relativeabundance", "p", "freq"}
LABEL_CANDS = {"evolutionarypattern", "evolutionarymode", "label", "class", "truth", "groundtruth"}


def pick_col(std_map, cands):
    for k in cands:
        if k in std_map:
            return std_map[k]
    return None


def norm_label(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"neutral", "neutral-like", "near-neutral", "near neutral"}:
        return "Neutral"
    if s in {"branched", "branch", "branching", "branched evolution"}:
        return "Branched"
    if s in {"linear", "linear evolution"}:
        return "Linear"
    return np.nan


def is_valid_clone_id(s):
    return isinstance(s, str) and re.match(r"^G\d+(?:\.\d+)*$", s) is not None


def read_long_table(xlsx):
    if not os.path.exists(xlsx):
        raise FileNotFoundError(f"File not found: {xlsx}")

    df0 = pd.read_excel(xlsx, sheet_name=0)
    std = {_std_col(c): c for c in df0.columns}
    sample_col = pick_col(std, SAMPLE_CANDS) or list(df0.columns)[0]
    clone_col = pick_col(std, CLONE_CANDS) or list(df0.columns)[1]
    prop_col = pick_col(std, PROP_CANDS) or list(df0.columns)[2]
    label_col = pick_col(std, LABEL_CANDS)

    df = pd.DataFrame(
        {
            "sample": df0[sample_col].replace({"nan": np.nan, "NaN": np.nan}).ffill(),
            "clone": df0[clone_col].astype(str),
            "prop": pd.to_numeric(
                pd.Series(df0[prop_col])
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", "", regex=False),
                errors="coerce",
                    ),
        }
    )

    if label_col:
        lab0 = df0[label_col].map(norm_label)
        lab = pd.DataFrame({"sample": df["sample"], "lab": lab0}).groupby("sample")["lab"].apply(
            lambda s: s.dropna().iloc[0] if s.dropna().size > 0 else np.nan
        )
        df["label"] = df["sample"].map(lab)

    df.loc[~df["clone"].apply(is_valid_clone_id), "clone"] = np.nan
    df = df.dropna(subset=["clone", "prop"])

    if df["prop"].max() > 1.2:
        df["prop"] = df["prop"] / 100.0

    return df


# ---------------------------- Tree structure and subtree mass ----------------------------
def parent_of(c):
    return c.rsplit(".", 1)[0] if "." in c else None


def ensure_nodes(clones):
    nodes = set()
    for c in clones:
        parts = c.split(".")
        for k in range(1, len(parts) + 1):
            nodes.add(".".join(parts[:k]))
    if any(s.startswith("G0.") for s in nodes) and "G0" not in nodes:
        nodes.add("G0")
    return nodes


def build_parent_maps(nodes):
    parent = {}
    children = {n: [] for n in nodes}

    for n in nodes:
        p = parent_of(n)
        if p and p in nodes:
            parent[n] = p
            children[p].append(n)

    roots = [n for n in nodes if n not in parent]
    root = "G0" if "G0" in nodes else (roots[0] if len(roots) == 1 else roots[0])

    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            depth[v] = depth[u] + 1
            q.append(v)

    for n in nodes:
        if n not in depth:
            depth[n] = 0

    return root, parent, children, depth


def compute_subtree_mass(p, nodes, parent, children, depth):
    mass = {n: 0.0 for n in nodes}

    for n, v in p.items():
        mass[n] += float(v)

    for n, _ in sorted(depth.items(), key=lambda kv: -kv[1]):
        if n in parent:
            mass[parent[n]] += mass[n]

    return mass


def is_linear_tree(children):
    edges = sum(len(v) for v in children.values())
    if edges < 1:
        return False

    for _, kids in children.items():
        if len(kids) >= 2:
            return False

    return True


# ---------------------------- Within-layer vectors and imbalance metrics ----------------------------
def F_of_node(node, children, mass):
    kids = children.get(node, [])
    if len(kids) < 2:
        return None, kids

    F = np.array([mass[k] for k in kids], dtype=float)
    S = F.sum()
    if S <= 0:
        return None, kids

    return F / S, kids


def dmax_rel(F):
    m = len(F)
    mu = 1.0 / m
    return float(np.max(np.abs(F - mu)) / mu)


def cv_rel(F):
    m = len(F)
    mu = 1.0 / m
    return float(np.std(F, ddof=0) / mu)


def gini(F):
    m = len(F)
    Fs = np.sort(F)
    num = 0.0

    for i in range(1, m + 1):
        num += (2 * i - m - 1) * Fs[i - 1]

    den = m * Fs.sum()
    return float(num / den) if den > 0 else np.nan


def chi2_p(F):
    m = len(F)
    if m < 2:
        return 1.0

    mu = 1.0 / m
    chi2 = float(np.sum((F - mu) ** 2 / mu))
    df = m - 1
    z = (((chi2 / df) ** (1 / 3)) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    p = 0.5 * math.erfc(z / math.sqrt(2))

    return float(max(min(p, 1.0), 0.0))


# ---------------------------- PHI and auxiliary global metrics ----------------------------
def phi_edge_sum(nodes, parent, children, mass, edge_len=1.0):
    """
    PHI is computed as the edge-sum expectation:
        PHI = sum_edges 2 * edge_len * f * (1 - f),
    where f is the child-subtree mass carried by each edge.
    """
    contrib = []

    for v in nodes:
        if v in parent:
            f = mass[v]
            c = 2.0 * edge_len * f * (1.0 - f)
            contrib.append(c)

    phi = float(np.sum(contrib)) if contrib else 0.0
    return phi, contrib


def ecc_from_contrib(contrib):
    tot = float(np.sum(contrib)) if len(contrib) > 0 else 0.0
    if tot <= 0:
        return 0.0
    return float(np.max(contrib) / tot)


def direct_mass_map(nodes, children, mass):
    return {n: mass[n] - sum(mass[c] for c in children.get(n, [])) for n in nodes}


def mbd_path_max(nodes, parent, children, direct_mass):
    leaves = [n for n in nodes if len(children.get(n, [])) == 0]

    def path_to_root(n):
        path = [n]
        while True:
            if n not in parent:
                break
            n = parent[n]
            path.append(n)
        return path[::-1]

    best = 0.0
    for leaf in leaves:
        path = path_to_root(leaf)
        s = sum(direct_mass.get(x, 0.0) for x in path)
        best = max(best, float(s))

    return best


def all_pairs_dmax(nodes, parent, children):
    idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    adj = [[] for _ in range(n_nodes)]

    for v in nodes:
        i = idx[v]
        if v in parent:
            u = parent[v]
            j = idx[u]
            adj[i].append(j)
            adj[j].append(i)

    def bfs(s):
        dist = [-1] * n_nodes
        dist[s] = 0
        q = deque([s])

        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    q.append(w)

        return max(dist)

    dmax = 0
    for i in range(n_nodes):
        dmax = max(dmax, bfs(i))

    return dmax


# ---------------------------- Sample-level feature extraction ----------------------------
def features_for_sample(df_g):
    agg = df_g.groupby("clone", as_index=False)["prop"].sum()
    if len(agg) < 2 or agg["prop"].sum() <= 0:
        return None

    agg["prop"] = agg["prop"] / agg["prop"].sum()
    p = dict(zip(agg["clone"], agg["prop"]))

    nodes = ensure_nodes(p.keys())
    root, parent, children, depth = build_parent_maps(nodes)
    mass = compute_subtree_mass(p, nodes, parent, children, depth)

    linear = is_linear_tree(children)
    if linear:
        return {"linear": 1}

    recs = []
    for v, kids in children.items():
        F, _ = F_of_node(v, children, mass)
        if F is None:
            continue
        recs.append(
            [
                v,
                depth.get(v, 0),
                len(F),
                dmax_rel(F),
                cv_rel(F),
                gini(F),
                chi2_p(F),
            ]
        )

    ldf = pd.DataFrame(recs, columns=["node", "depth", "m", "Dmax", "CV", "Gini", "pchi"])
    if ldf.empty:
        ldf = pd.DataFrame(
            [["__none__", 0, 0, 0, 0, 0, 1.0]],
            columns=ldf.columns,
        )

    votes = (
            ldf["Dmax"].rank(ascending=True, pct=True)
            + ldf["CV"].rank(ascending=True, pct=True)
            + ldf["Gini"].rank(ascending=True, pct=True)
            + ldf["pchi"].rank(ascending=False, pct=True)
    )
    worst_idx = int(np.argmin(votes))

    F0, _ = F_of_node(root, children, mass)
    if F0 is not None and len(F0) >= 2:
        Dmax_root = dmax_rel(F0)
        CV_root = cv_rel(F0)
        Gini_root = gini(F0)
        pchi_root = chi2_p(F0)
        n_root_children = int(len(F0))
    else:
        Dmax_root = 0.0
        CV_root = 0.0
        Gini_root = 0.0
        pchi_root = 1.0
        n_root_children = int(len(children.get(root, [])))

    phi, contrib = phi_edge_sum(nodes, parent, children, mass, edge_len=1.0)
    ECC = ecc_from_contrib(contrib)
    direct = direct_mass_map(nodes, children, mass)
    MBD = mbd_path_max(nodes, parent, children, direct)
    E = sum(len(children.get(n, [])) for n in nodes)
    phi_rel = (2.0 * phi / E) if E > 0 else 0.0
    dmax = all_pairs_dmax(list(nodes), parent, children)
    phi_by_dmax = (phi / (dmax / 2.0)) if dmax > 0 else 0.0

    feat = {
        "n_layers": int((ldf["m"] >= 2).sum()),
        "max_depth": int(max(depth.values()) if depth else 0),
        "max_branching": int(max([len(children.get(n, [])) for n in nodes]) if nodes else 0),
        "Dmax_worst": float(ldf.iloc[worst_idx]["Dmax"]),
        "CV_worst": float(ldf.iloc[worst_idx]["CV"]),
        "Gini_worst": float(ldf.iloc[worst_idx]["Gini"]),
        "pchi_worst": float(ldf.iloc[worst_idx]["pchi"]),
        "Dmax_mean": float(ldf["Dmax"].mean()),
        "Gini_q75": float(ldf["Gini"].quantile(0.75)),
        "pchi_q25": float(ldf["pchi"].quantile(0.25)),
        "Dmax_root": float(Dmax_root),
        "CV_root": float(CV_root),
        "Gini_root": float(Gini_root),
        "pchi_root": float(pchi_root),
        "n_root_children": int(n_root_children),
        "PHI": float(phi),
        "PHI_rel": float(phi_rel),
        "PHI_dmax": float(phi_by_dmax),
        "ECC": float(ECC),
        "MBD": float(MBD),
    }

    return feat


# ---------------------------- Monotone constraints ----------------------------
def get_feature_list_and_monotone(feat_df):
    """
    Class convention: Neutral = 1 and Branched = 0.

    Constraint direction:
    - More imbalance or more extreme dominance implies a lower Neutral probability (-1).
    - Greater global evenness after scale correction implies a higher Neutral probability (+1).
    - Ambiguous scale-dependent features are left unconstrained (0).
    """
    ordered = [
        "n_layers",
        "max_depth",
        "max_branching",
        "Dmax_worst",
        "CV_worst",
        "Gini_worst",
        "pchi_worst",
        "Dmax_mean",
        "Gini_q75",
        "pchi_q25",
        "Dmax_root",
        "CV_root",
        "Gini_root",
        "pchi_root",
        "n_root_children",
        "PHI",
        "PHI_rel",
        "PHI_dmax",
        "ECC",
        "MBD",
    ]

    for c in ordered:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    monotone = [
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        +1,
        -1,
        -1,
        +1,
        -1,
        -1,
        -1,
        +1,
        0,
        0,
        +1,
        +1,
        -1,
        -1,
    ]

    return ordered, monotone


# ---------------------------- Training ----------------------------
def train_xgb_model():
    df_tr = read_long_table(TRAIN_XLSX)

    rows = []
    for sid, g in df_tr.groupby("sample", sort=False):
        feat = features_for_sample(g)
        if feat is None:
            continue

        feat["sample"] = sid
        lab = g.get("label", pd.Series(dtype=object)).dropna()
        feat["label"] = lab.iloc[0] if not lab.empty else np.nan
        rows.append(feat)

    Xy = pd.DataFrame(rows)
    Xy = Xy[(Xy.get("linear", 0) != 1) & (Xy["label"].isin(["Neutral", "Branched"]))].copy()
    y = (Xy["label"] == "Neutral").astype(int).values

    X = Xy.drop(columns=[c for c in ["sample", "label", "linear"] if c in Xy.columns]).copy()
    feat_order, mono = get_feature_list_and_monotone(X)
    X = X[feat_order]
    mono_str = "({})".format(",".join(str(int(m)) for m in mono))

    base = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        max_depth=4,
        n_estimators=600,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        monotone_constraints=mono_str,
        scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())),
        random_state=42,
        n_jobs=0,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs_cv = []
    y_cv = []

    for tr, va in skf.split(X, y):
        base.fit(X.iloc[tr], y[tr])
        p = base.predict_proba(X.iloc[va])[:, 1]
        probs_cv.append(p)
        y_cv.append(y[va])

    probs_cv = np.concatenate(probs_cv)
    y_cv = np.concatenate(y_cv)

    prec, rec, thr = precision_recall_curve(y_cv, probs_cv)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_thr = float(np.clip(thr[best_idx - 1], 0.05, 0.95)) if 0 < best_idx < len(thr) else 0.5

    print("[CV] BalancedAcc =", balanced_accuracy_score(y_cv, (probs_cv >= best_thr).astype(int)))
    print("[CV] F1(Neutral) =", f1[best_idx])

    try:
        print("[CV] ROC-AUC =", roc_auc_score(y_cv, probs_cv))
    except Exception:
        pass

    base.fit(X, y)
    joblib.dump({"model": base, "features": feat_order, "monotone": mono}, MODEL_PATH)

    with open(THR_PATH, "w", encoding="utf-8") as f:
        f.write(str(best_thr))

    print(f"[OK] Model saved: {MODEL_PATH}; threshold: {best_thr:.3f}")


# ---------------------------- Inference ----------------------------
def predict_with_saved_model():
    pkg = joblib.load(MODEL_PATH)
    clf, feat_order = pkg["model"], pkg["features"]

    try:
        with open(THR_PATH, "r", encoding="utf-8") as f:
            thr = float(f.read().strip())
    except Exception:
        thr = 0.5

    df_pred = read_long_table(PRED_XLSX)
    out_rows = []
    detail_rows = []

    for sid, g in df_pred.groupby("sample", sort=False):
        feat = features_for_sample(g)
        if feat is None:
            out_rows.append({"sample": sid, "Prediction": "ERROR: no data"})
            continue

        if feat.get("linear", 0) == 1:
            out_rows.append(
                {
                    "sample": sid,
                    "Prediction": "Linear",
                    "prob_neutral": np.nan,
                    "threshold": thr,
                }
            )
            continue

        x = pd.DataFrame([feat])
        for c in feat_order:
            if c not in x.columns:
                x[c] = 0.0
        x = x[feat_order].fillna(0.0)

        p = float(clf.predict_proba(x)[0, 1])
        pred = "Neutral" if p >= thr else "Branched"
        out_rows.append({"sample": sid, "Prediction": pred, "prob_neutral": p, "threshold": thr})

        detail = {"sample": sid, **{k: float(x.iloc[0][k]) for k in feat_order}}
        detail_rows.append(detail)

    df_out = pd.DataFrame(out_rows).sort_values("sample")

    with pd.ExcelWriter(OUT_XLSX) as w:
        df_out.to_excel(w, sheet_name="predictions", index=False)

        if detail_rows:
            pd.DataFrame(detail_rows).to_excel(w, sheet_name="features_used", index=False)

        model_info = pd.DataFrame(
            [
                {
                    "model_path": MODEL_PATH,
                    "threshold": thr,
                    "objective": "binary:logistic (Neutral=1)",
                    "monotone_constraints": "see joblib",
                    "train_file": TRAIN_XLSX,
                    "predict_file": PRED_XLSX,
                }
            ]
        )
        model_info.to_excel(w, sheet_name="model_info", index=False)

    print(f"[OK] Prediction results written to: {OUT_XLSX}")


# ---------------------------- Main entry point ----------------------------
def main():
    warnings.filterwarnings("ignore")
    need_train = (
            FORCE_RETRAIN
            or (not USE_SAVED_MODEL)
            or (not os.path.exists(MODEL_PATH))
            or (not os.path.exists(THR_PATH))
    )

    if need_train:
        print("[INFO] Training model with cross-validated threshold selection...")
        train_xgb_model()
    else:
        print("[INFO] Saved model and threshold found. Skipping training and running inference directly.")

    predict_with_saved_model()


if __name__ == "__main__":
    main()