# -*- coding: utf-8 -*-
"""
Standalone curves for Ordinal Logit & Monotone-GBDT (with robust fallback)
- If main workbook exists -> read features from it
- Else -> rebuild features from raw matrix '突变基因数据矩阵 V2.xlsx'
- Cleans all NaN/Inf in features before scaling and before LR fallback
- Outputs PDFs to: C:\CCF校正样本单一文件\GBDT分析V1
- Python 3.8
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, norm
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

# ============== Paths ==============
BASE_DIR   = r"C:\CCF校正样本单一文件"
OUT_DIR    = os.path.join(BASE_DIR, "GBDT分析V1")
MAIN_XLSX  = os.path.join(OUT_DIR, "HNSCC_TMB_integrated_results.xlsx")
INPUT_XLSX = os.path.join(BASE_DIR, "突变基因数据矩阵 V2.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-12

# ============== Safe sigmoid ==============
def _safe_sigmoid(u):
    u = np.clip(u, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-u))

# ============== Ordinal Logit (proportional odds) ==============
def nll_proportional_odds_bounded(theta, X, y, l2=1e-3):
    d = X.shape[1]
    w  = theta[:d]
    t1 = theta[d]
    d2 = theta[d+1]
    t2 = t1 + np.exp(d2)  # enforce t2>t1
    a  = X @ w
    c0 = _safe_sigmoid(t1 - a)
    c1 = _safe_sigmoid(t2 - a)
    p0 = c0
    p1 = np.clip(c1 - c0, EPS, 1.0)
    p2 = 1.0 - c1
    p  = np.choose(y, [p0, p1, p2])
    p  = np.clip(p, EPS, 1.0 - EPS)
    return -np.sum(np.log(p)) + 0.5 * l2 * np.sum(w * w)

def fit_ordinal_logit_stable(X, y):
    """Proportional-odds with bounds, L2 grid, multi-start."""
    n, d = X.shape
    bounds = [(-10.0, 10.0)] * d + [(-10.0, 10.0)] + [(-3.0, 3.0)]
    inits  = [
        np.r_[np.zeros(d), 0.0, 0.0],
        np.r_[np.random.RandomState(0).normal(0, 0.1, size=d), 0.0, 0.0],
        np.r_[np.random.RandomState(1).normal(0, 0.2, size=d), 0.5, -0.5],
    ]
    l2_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best = None
    for l2 in l2_grid:
        for theta0 in inits:
            res = minimize(
                nll_proportional_odds_bounded, theta0,
                args=(X, y, l2), method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 2000}
            )
            if not res.success:
                continue
            th = res.x
            w  = th[:d]; t1 = th[d]; t2 = t1 + np.exp(th[d+1])
            a  = X @ w
            if np.all(np.isfinite(a)) and (np.nanstd(a) > 1e-9):
                best = {"w": w, "t1": t1, "t2": t2, "theta": th, "l2": l2}
                break
        if best is not None:
            break
    return best

# ============== Fallback: two cumulative LRs + cut optimization ==============
def _optimize_cuts_on_scores(a, y):
    """Optimize t1 < t2 on 1D score a to minimize NLL."""
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=int)

    def _safe_median(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if arr.size > 0 else 0.0

    a0 = a[y==0]; a1 = a[y==1]; a2 = a[y==2]
    m0 = _safe_median(a0); m1 = _safe_median(a1); m2 = _safe_median(a2)
    t1_init = 0.5*(m0 + m1)
    t2_init = 0.5*(m1 + m2)
    if not np.isfinite(t1_init): t1_init = 0.0
    if (not np.isfinite(t2_init)) or (t2_init <= t1_init): t2_init = t1_init + 1.0

    def nll_phi(phi):
        t1 = phi[0]
        d2 = phi[1]
        t2 = t1 + np.exp(d2)
        c0 = _safe_sigmoid(t1 - a)
        c1 = _safe_sigmoid(t2 - a)
        p0 = c0
        p1 = np.clip(c1 - c0, EPS, 1.0)
        p2 = 1.0 - c1
        p  = np.choose(y, [p0, p1, p2])
        p  = np.clip(p, EPS, 1.0 - EPS)
        return -np.sum(np.log(p))

    phi0 = np.array([t1_init, np.log(max(t2_init - t1_init, 1e-3))], dtype=float)
    res = minimize(nll_phi, phi0, method="L-BFGS-B", bounds=[(-10,10), (-5,5)],
                   options={"maxiter": 3000})
    if not res.success:
        return t1_init, t2_init
    t1 = res.x[0]
    t2 = t1 + np.exp(res.x[1])
    return float(t1), float(t2)

def fit_ordinal_or_fallback(X, y):
    """
    Try robust proportional-odds; if fails, fallback to:
      - two cumulative LRs (P(y<=0), P(y<=1)) with L2, liblinear,
      - shared slope w = (b0 + b1) / 2,
      - optimize cutpoints t1, t2 on a = X @ w.
    Returns dict: {"w","t1","t2","is_fallback":bool}
    """
    # —— 再保险清洗，避免 NaN —— #
    Xc = np.array(X, dtype=float)
    Xc[~np.isfinite(Xc)] = 0.0

    model = fit_ordinal_logit_stable(Xc, y)
    if model is not None:
        model["is_fallback"] = False
        return model

    y_le0 = (y <= 0).astype(int)
    y_le1 = (y <= 1).astype(int)

    lr0 = LogisticRegression(penalty="l2", C=0.5, solver="liblinear",
                             max_iter=5000, class_weight="balanced", random_state=42)
    lr1 = LogisticRegression(penalty="l2", C=0.5, solver="liblinear",
                             max_iter=5000, class_weight="balanced", random_state=42)

    lr0.fit(Xc, y_le0); b0 = lr0.coef_.ravel()
    lr1.fit(Xc, y_le1); b1 = lr1.coef_.ravel()

    w = 0.5*(b0 + b1)
    if not np.all(np.isfinite(w)) or np.linalg.norm(w) < 1e-8:
        w = (b0 if np.linalg.norm(b0) > 1e-8 else b1)
        if np.linalg.norm(w) < 1e-8:
            w = np.ones(Xc.shape[1], dtype=float) * 1e-3

    a = Xc @ w
    if not np.all(np.isfinite(a)) or np.nanstd(a) < 1e-9:
        a = a + 1e-6*np.random.RandomState(0).normal(size=a.shape)

    t1, t2 = _optimize_cuts_on_scores(a, y)
    return {"w": w.astype(float), "t1": float(t1), "t2": float(t2), "is_fallback": True}

# ============== Gene trend weighting (freq × intensity) ==============
def cochran_armitage_trend(group_labels, mutated_flags):
    g = np.asarray(group_labels, dtype=float)
    y = np.asarray(mutated_flags, dtype=float)
    uniq = np.unique(g)
    weights = np.array([0.0, 1.0, 2.0])
    xk, nk, wk = [], [], []
    for k in uniq:
        mask = (g == k)
        nk.append(np.sum(mask))
        xk.append(np.sum(y[mask]))
        wk.append(weights[int(k)])
    xk = np.array(xk, dtype=float)
    nk = np.array(nk, dtype=float)
    wk = np.array(wk, dtype=float)
    N = float(np.sum(nk)); X = float(np.sum(xk))
    if N <= 1 or X == 0 or X == N:
        return np.nan, 1.0, 0.0
    w_bar = np.sum(nk * wk) / N
    s_w = np.sum(nk * (wk - w_bar) ** 2)
    numerator = np.sum(wk * (xk - nk * (X / N)))
    var = (X * (N - X) * s_w) / (N * (N - 1.0))
    if var <= 0:
        return np.nan, 1.0, 0.0
    Z = numerator / math.sqrt(var)
    p = 2.0 * (1.0 - norm.cdf(abs(Z)))
    slope_sign = 1.0 if numerator > 0 else (-1.0 if numerator < 0 else 0.0)
    return Z, p, slope_sign

# ============== Feature loaders ==============
def load_features_from_workbook():
    if not os.path.exists(MAIN_XLSX):
        return None
    df_ps = pd.read_excel(MAIN_XLSX, sheet_name="per_sample")
    needed = ["PI_mut_z", "TMB_z", "TMB_hinge", "group"]
    if not all(c in df_ps.columns for c in needed):
        return None
    raw = df_ps[["PI_mut_z", "TMB_z", "TMB_hinge"]].astype(float).values
    # —— 清洗 NaN/Inf —— #
    raw = np.where(np.isfinite(raw), raw, 0.0)
    y   = df_ps["group"].astype(int).values
    scaler = StandardScaler().fit(raw)
    Xstd = scaler.transform(raw)
    print("Loaded features from workbook:", MAIN_XLSX)
    return {"raw": raw, "X": Xstd, "y": y, "scaler": scaler, "df_ps": df_ps}

def rebuild_features_from_raw():
    if not os.path.exists(INPUT_XLSX):
        return None
    df = pd.read_excel(INPUT_XLSX)
    df.columns = [str(c).strip() for c in df.columns]
    if "group" not in df.columns or "patient" not in df.columns:
        raise RuntimeError("Input matrix missing 'group' or 'patient'.")
    # TMB column
    tmb_col = None
    for c in df.columns:
        if "tmb" in c.lower():
            tmb_col = c; break
    if tmb_col is None:
        raise RuntimeError("No TMB column found in raw matrix.")

    meta_cols = {"patient", "group", "mutation_type", tmb_col}
    gene_cols = []
    for c in df.columns:
        if c in meta_cols: continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        gene_cols.append(c)

    df = df.dropna(subset=["group"]).copy()
    df["group"] = pd.to_numeric(df["group"], errors="coerce").astype(int)

    # VAF rescale
    VAF_MAX = max(1.0, float(np.nanmax(df[gene_cols].values)))
    df_scaled = df.copy()
    df_scaled[gene_cols] = df_scaled[gene_cols] / VAF_MAX

    # TMB z
    tmb_raw = pd.to_numeric(df_scaled[tmb_col], errors="coerce").values.astype(float)
    tmb_log1p = np.log1p(np.clip(tmb_raw, 0, None))
    tmb_mean = float(np.nanmean(tmb_log1p))
    tmb_std  = float(np.nanstd(tmb_log1p)) + EPS
    tmb_z = (tmb_log1p - tmb_mean) / tmb_std
    df_scaled["TMB_z"] = tmb_z

    # Gene weights (freq × intensity)
    groups = df_scaled["group"].values
    records = []
    for gname in gene_cols:
        v = df_scaled[gname].values
        mutated = (v > 0).astype(int)
        Z, p_freq, s_freq = cochran_armitage_trend(groups, mutated)
        mask_pos = v > 0
        tau, p_tau = (np.nan, 1.0)
        s_vaf = 0.0
        if np.sum(mask_pos) >= 3:
            tau, p_tau = kendalltau(groups[mask_pos], v[mask_pos])
            if not np.isnan(tau):
                s_vaf = 1.0 if tau > 0 else (-1.0 if tau < 0 else 0.0)
        logp_freq = -math.log10(max(p_freq, 1e-300)) if not np.isnan(p_freq) else 0.0
        logp_vaf  = -math.log10(max(p_tau,  1e-300)) if not np.isnan(p_tau)  else 0.0
        weight = logp_freq * logp_vaf if (s_freq > 0 and s_vaf > 0) else 0.0
        records.append({"gene": gname, "weight": weight})
    gene_trends = pd.DataFrame.from_records(records).sort_values("weight", ascending=False)
    pos = gene_trends[gene_trends["weight"] > 0.0]
    if pos.empty:
        pos = gene_trends.head(20)
    weights_map = {r["gene"]: float(r["weight"]) for _, r in pos.iterrows()}
    W_sum = sum(weights_map.values()) if len(weights_map) > 0 else 1.0

    def compute_PI(row):
        s = 0.0
        for gname, w in weights_map.items():
            val = row.get(gname, np.nan)
            if pd.isna(val): continue
            s += w * float(val)
        return s / W_sum

    df_scaled["PI_mut"] = df_scaled.apply(compute_PI, axis=1)
    pi_mean = float(np.nanmean(df_scaled["PI_mut"].values))
    pi_std  = float(np.nanstd(df_scaled["PI_mut"].values)) + EPS
    df_scaled["PI_mut_z"] = (df_scaled["PI_mut"] - pi_mean) / pi_std

    # Hinge on TAT median
    tmb_tat = df_scaled.loc[df_scaled["group"] == 1, "TMB_z"].values
    c = float(np.median(tmb_tat)) if tmb_tat.size > 0 else 0.0
    tmb_hinge = np.maximum(df_scaled["TMB_z"].values - c, 0.0)
    df_scaled["TMB_hinge"] = tmb_hinge

    # Build feature matrix and standardize
    raw = df_scaled[["PI_mut_z", "TMB_z", "TMB_hinge"]].astype(float).values
    # —— 清洗 NaN/Inf —— #
    raw = np.where(np.isfinite(raw), raw, 0.0)
    y   = df_scaled["group"].astype(int).values
    scaler = StandardScaler().fit(raw)
    Xstd = scaler.transform(raw)

    df_ps = pd.DataFrame({
        "patient": df_scaled["patient"].astype(str).values,
        "group": y,
        "PI_mut_z": raw[:,0],
        "TMB_z": raw[:,1],
        "TMB_hinge": raw[:,2]
    })
    print("Rebuilt features from raw matrix:", INPUT_XLSX)
    return {"raw": raw, "X": Xstd, "y": y, "scaler": scaler, "df_ps": df_ps}

# Try workbook -> else rebuild from raw
loaded = load_features_from_workbook()
if loaded is None:
    loaded = rebuild_features_from_raw()
if loaded is None:
    raise RuntimeError("Neither main workbook nor raw matrix was found. Please check file paths.")

raw_all = loaded["raw"]
X_all   = loaded["X"]
y_all   = loaded["y"]
scaler  = loaded["scaler"]
df_ps   = loaded["df_ps"]

# ============== Fit models on full data (with fallback) ==============
ord_model = fit_ordinal_or_fallback(X_all, y_all)
if ord_model.get("is_fallback", False):
    print("[WARN] Ordinal Logit did not converge; used fallback (two cumulative LRs + cut optimization).")
w_full  = ord_model["w"]; t1_full = ord_model["t1"]; t2_full = ord_model["t2"]

def ord_probs_from_raw(raw_mat):
    Xstd = scaler.transform(raw_mat)
    a = Xstd @ w_full
    c0 = _safe_sigmoid(t1_full - a)
    c1 = _safe_sigmoid(t2_full - a)
    p0 = c0
    p1 = np.clip(c1 - c0, EPS, 1.0)
    p2 = 1.0 - c1
    return p0, p1, p2

# Monotone-GBDT
mono_cst = [1, 1, 1]
try:
    hgb = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.09, max_iter=300,
        random_state=42, monotonic_cst=mono_cst
    )
    _ = hgb.get_params()
except TypeError:
    hgb = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.09, max_iter=300,
        random_state=42
    )
hgb.fit(X_all, y_all)

# ============== Curves: one feature at a time ==============
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

feature_names = ["PI_mut_z", "TMB_z", "TMB_hinge"]

def percentile_grid(vals, n=250):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(-1.0, 1.0, n)
    lo = np.nanpercentile(vals, 1)
    hi = np.nanpercentile(vals, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(vals); hi = np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -1.0, 1.0
    return np.linspace(lo, hi, n)

base_raw = np.array([np.nanmedian(df_ps[c].values) for c in feature_names], dtype=float)
col_arrays = [df_ps[c].values for c in feature_names]

def save_curve_pdf(x_grid, y_curve, xlabel, ylabel, filename, ylimit=None, hlines=None):
    fig = plt.figure(figsize=(3.54, 2.6))  # ~90mm single-column
    ax = fig.add_subplot(111)
    ax.plot(x_grid, y_curve, linewidth=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(*ylimit)
    if hlines:
        for yv in hlines:
            ax.axhline(yv, linewidth=0.8)  # solid reference lines
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), format="pdf", bbox_inches="tight")
    plt.close(fig)

for j, fname in enumerate(feature_names):
    grid = percentile_grid(col_arrays[j], n=250)
    raw_mat = np.tile(base_raw, (grid.shape[0], 1))
    raw_mat[:, j] = grid

    # Ordinal Logit probabilities
    p0, p1, p2 = ord_probs_from_raw(raw_mat)
    save_curve_pdf(grid, p0, fname, "P(class=0)", f"OrdLogit_curve_{fname}_p0.pdf", ylimit=(0,1))
    save_curve_pdf(grid, p1, fname, "P(class=1)", f"OrdLogit_curve_{fname}_p1.pdf", ylimit=(0,1))
    save_curve_pdf(grid, p2, fname, "P(class=2)", f"OrdLogit_curve_{fname}_p2.pdf", ylimit=(0,1))

    # GBDT regression score
    Xstd = scaler.transform(raw_mat)
    score = hgb.predict(Xstd)
    save_curve_pdf(grid, score, fname, "GBDT regression score",
                   f"GBDT_curve_{fname}_score.pdf", ylimit=None, hlines=[0.5, 1.5])

print("All model curve PDFs saved to:", OUT_DIR)
