"""Shared helpers for the causal customer-retention project.

Conventions used throughout
---------------------------
* ``t`` is the treatment indicator (1 = contacted by the retention campaign).
* ``y`` is the observed outcome (1 = customer churned within the follow-up window).
* A *successful* intervention therefore has a NEGATIVE effect on ``y``.
* For uplift/targeting we score customers by the estimated **reduction in churn
  probability** caused by treatment, ``P(y=1|t=0,x) - P(y=1|t=1,x)`` (= -CATE),
  so that "higher score = better to contact".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "figures"

OPENML_ID = 45580
TREATMENT = "t"
OUTCOME = "y"

# Reference categorical palette (validated, light mode) - fixed order, never cycled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#8c8b87"
INK, INK_2 = "#0b0b0b", "#52514e"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(cache: bool = True) -> pd.DataFrame:
    """Load the Orange Belgium churn-uplift data (OpenML 45580).

    The raw data is licensed CC BY-NC-ND 4.0, so it is downloaded on first use
    and cached locally in ``data/raw/`` (git-ignored) rather than committed.
    """
    path = RAW_DIR / "churn_uplift.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)

    from sklearn.datasets import fetch_openml

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto",
                      data_home=str(RAW_DIR)).frame
    df[TREATMENT] = df[TREATMENT].astype(int)
    df[OUTCOME] = df[OUTCOME].astype(int)
    for c in df.columns:
        if c.startswith("FACTOR"):
            df[c] = df[c].astype(str)
    if cache:
        df.to_parquet(path, index=False)
    return df


def feature_lists(df: pd.DataFrame, var_tol: float = 1e-8, min_minority: int = 30):
    """Return (numeric, categorical) covariate names, dropping uninformative columns.

    * numeric PCA components with (numerically) zero variance are dropped;
    * categorical factors that are constant, or whose non-modal levels together
      cover fewer than ``min_minority`` customers, are dropped.
    """
    num = [c for c in df.columns if c.startswith("PC") and df[c].var() > var_tol]
    cat = []
    for c in (c for c in df.columns if c.startswith("FACTOR")):
        counts = df[c].value_counts()
        if len(counts) > 1 and (len(df) - counts.iloc[0]) >= min_minority:
            cat.append(c)
    return num, cat


def make_preprocessor(num: list[str], cat: list[str], min_frequency: int = 50):
    """PCs pass through unchanged; factors are one-hot encoded with rare levels pooled."""
    return ColumnTransformer(
        [
            ("num", "passthrough", num),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=min_frequency,
                                  sparse_output=False), cat),
        ],
        verbose_feature_names_out=False,
    )


def read_sql_queries(path=PROJECT_ROOT / "sql" / "customer_summary.sql") -> dict:
    """Parse a .sql file into {name: query} using '-- name: <name>' markers."""
    queries, name, buf = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("-- name:"):
            if name:
                queries[name] = "\n".join(buf).strip()
            name, buf = line.split(":", 1)[1].strip(), []
        elif name:
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip()
    return queries


def build_sqlite(df: pd.DataFrame):
    """Load the data into an in-memory SQLite DB as `customers` and `campaign` tables."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    d = df.reset_index(drop=True).rename_axis("customer_id").reset_index()
    attr_cols = ["customer_id"] + [c for c in d.columns if c.startswith("FACTOR")] + ["PC1", "PC2", "PC3"]
    d[attr_cols].to_sql("customers", conn, index=False)
    d[["customer_id", TREATMENT, OUTCOME]].to_sql("campaign", conn, index=False)
    return conn


def collapse_rare(s: pd.Series, min_count: int = 100, other: str = "Other") -> pd.Series:
    """Pool levels of a categorical with fewer than ``min_count`` observations."""
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    return s.where(~s.isin(rare), other)


# ---------------------------------------------------------------------------
# Classical experimental analysis
# ---------------------------------------------------------------------------
def ate_diff_in_means(y, t, alpha: float = 0.05) -> dict:
    """Difference-in-means ATE for a binary outcome in a randomised experiment.

    Returns arm rates, the risk difference (treated - control) with a Wald CI and
    a Newcombe hybrid-score CI, the two-proportion z-test, the risk ratio with a
    log-scale CI, and Cohen's h.
    """
    y, t = np.asarray(y), np.asarray(t)
    y1, y0 = y[t == 1], y[t == 0]
    n1, n0 = len(y1), len(y0)
    p1, p0 = y1.mean(), y0.mean()
    z = stats.norm.ppf(1 - alpha / 2)

    diff = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)

    # Pooled-variance z-test (H0: p1 == p0).
    pp = y.mean()
    se_pool = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    z_stat = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z_stat))

    # Newcombe (1998) hybrid score interval built from Wilson intervals.
    def wilson(p, n):
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        return centre - half, centre + half

    l1, u1 = wilson(p1, n1)
    l0, u0 = wilson(p0, n0)
    newcombe = (diff - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2),
                diff + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2))

    rr = p1 / p0
    se_log_rr = np.sqrt((1 - p1) / (n1 * p1) + (1 - p0) / (n0 * p0))
    rr_ci = (rr * np.exp(-z * se_log_rr), rr * np.exp(z * se_log_rr))

    cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p0))

    return {
        "n_treated": n1, "n_control": n0,
        "churn_treated": p1, "churn_control": p0,
        "ate": diff, "se": se,
        "ci_wald": (diff - z * se, diff + z * se),
        "ci_newcombe": newcombe,
        "z": z_stat, "p_value": p_value,
        "risk_ratio": rr, "rr_ci": rr_ci,
        "cohens_h": cohens_h,
    }


def bootstrap_ci(stat_fn, *arrays, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0, stratify=None):
    """Percentile bootstrap CI for ``stat_fn(*resampled_arrays)``.

    ``stratify`` (e.g. the treatment vector) resamples within strata so that
    arm sizes are held fixed, mirroring the experimental design.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if stratify is None:
        groups = [np.arange(n)]
    else:
        stratify = np.asarray(stratify)
        groups = [np.flatnonzero(stratify == g) for g in np.unique(stratify)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        draws[b] = stat_fn(*(a[idx] for a in arrays))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return lo, hi, draws


# ---------------------------------------------------------------------------
# Model factories (one place, so every notebook uses identical specifications)
# ---------------------------------------------------------------------------
# Only ~400 churners exist, so the gradient-boosting models are deliberately
# shallow and heavily regularised.
GBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=8,
                  min_child_samples=80, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)


def make_gbm_classifier(random_state: int = 0):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**GBM_PARAMS, random_state=random_state)


def make_gbm_regressor(random_state: int = 0):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**GBM_PARAMS, random_state=random_state)


def make_logit(C: float = 0.01):
    """L2-penalised logistic regression on standardised features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def make_propensity_model(random_state: int = 0):
    """Propensity model P(t=1 | x). Chosen in notebook 03 because, among the
    specifications compared, it gave the best covariate balance after weighting."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=8,
                          min_child_samples=300, verbose=-1, random_state=random_state)


def weighted_smd(X: pd.DataFrame, t, w=None) -> pd.Series:
    """Standardised mean differences (treated - control), optionally IPW-weighted.
    The pooled SD is always the unweighted one, so raw and weighted SMDs are comparable."""
    t = np.asarray(t)
    w = np.ones(len(t)) if w is None else np.asarray(w)
    A = X.to_numpy(dtype=float)
    m1 = np.average(A[t == 1], axis=0, weights=w[t == 1])
    m0 = np.average(A[t == 0], axis=0, weights=w[t == 0])
    sd = np.sqrt((A[t == 1].var(axis=0) + A[t == 0].var(axis=0)) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        smd = np.where(sd > 0, (m1 - m0) / sd, 0.0)
    return pd.Series(smd, index=X.columns, name="smd")


# ---------------------------------------------------------------------------
# Meta-learners (cross-fitted so every CATE is an out-of-sample prediction)
# ---------------------------------------------------------------------------
def _folds(X, t, y, n_splits, seed):
    strata = np.asarray(t) * 2 + np.asarray(y)  # keep arm x outcome balance per fold
    return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strata)


def s_learner(make_model, X, t, y, n_splits: int = 5, seed: int = 0,
              interactions: bool = False):
    """S-learner: one model y ~ f(X, t); CATE = f(x, 1) - f(x, 0).

    ``interactions=True`` appends t * X columns. A linear model needs them to
    express effects that vary with X; tree models can find interactions themselves.
    Returns out-of-fold arrays (mu0, mu1, cate) where cate is on the churn scale
    (negative = treatment lowers churn).
    """
    X, t, y = np.asarray(X, dtype=float), np.asarray(t), np.asarray(y)

    def design(Xp, tp):
        cols = [Xp, tp[:, None]] + ([Xp * tp[:, None]] if interactions else [])
        return np.hstack(cols)

    mu0, mu1 = np.empty(len(y)), np.empty(len(y))
    for tr, te in _folds(X, t, y, n_splits, seed):
        m = make_model().fit(design(X[tr], t[tr].astype(float)), y[tr])
        mu0[te] = m.predict_proba(design(X[te], np.zeros(len(te))))[:, 1]
        mu1[te] = m.predict_proba(design(X[te], np.ones(len(te))))[:, 1]
    return mu0, mu1, mu1 - mu0


def t_learner(make_model, X, t, y, n_splits: int = 5, seed: int = 0):
    """T-learner: separate outcome models for treated and control customers."""
    X, t, y = np.asarray(X, dtype=float), np.asarray(t), np.asarray(y)
    mu0, mu1 = np.empty(len(y)), np.empty(len(y))
    for tr, te in _folds(X, t, y, n_splits, seed):
        trc, trt = tr[t[tr] == 0], tr[t[tr] == 1]
        mu0[te] = make_model().fit(X[trc], y[trc]).predict_proba(X[te])[:, 1]
        mu1[te] = make_model().fit(X[trt], y[trt]).predict_proba(X[te])[:, 1]
    return mu0, mu1, mu1 - mu0


def x_learner(make_clf, make_reg, X, t, y, propensity,
              n_splits: int = 5, seed: int = 0):
    """X-learner (Kunzel et al., 2019), suited to unbalanced arms.

    Stage 1: T-learner outcome models mu0, mu1.
    Stage 2: impute individual effects D1 = y - mu0(x) for treated, D0 = mu1(x) - y
             for control, and regress each on X.
    Stage 3: combine tau = e(x) * tau0 + (1 - e(x)) * tau1, where ``propensity``
             is P(t=1 | x) - a scalar or one value per customer.
    All stages are fitted on the training folds only.
    """
    X, t, y = np.asarray(X, dtype=float), np.asarray(t), np.asarray(y)
    e = np.broadcast_to(np.asarray(propensity, dtype=float), y.shape)
    cate = np.empty(len(y))
    for tr, te in _folds(X, t, y, n_splits, seed):
        trc, trt = tr[t[tr] == 0], tr[t[tr] == 1]
        m0 = make_clf().fit(X[trc], y[trc])
        m1 = make_clf().fit(X[trt], y[trt])
        d1 = y[trt] - m0.predict_proba(X[trt])[:, 1]
        d0 = m1.predict_proba(X[trc])[:, 1] - y[trc]
        tau1 = make_reg().fit(X[trt], d1).predict(X[te])
        tau0 = make_reg().fit(X[trc], d0).predict(X[te])
        cate[te] = e[te] * tau0 + (1 - e[te]) * tau1
    return cate


def oof_predict_proba(make_model, X, y, strata=None, n_splits: int = 5, seed: int = 0):
    """Out-of-fold predicted probabilities for an ordinary classifier."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    strata = y if strata is None else np.asarray(strata)
    p = np.empty(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strata):
        p[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


# ---------------------------------------------------------------------------
# Covariate-adjusted ATE (treatment assignment probability varies with X)
# ---------------------------------------------------------------------------
def ipw_weights(t, e, clip: float = 0.02):
    """Inverse-propensity weights: 1/e for treated, 1/(1-e) for control.

    Re-weights each arm to resemble the full population, removing imbalance
    caused by covariates that drive assignment. ``e`` is clipped for stability.
    """
    t, e = np.asarray(t), np.clip(np.asarray(e), clip, 1 - clip)
    return np.where(t == 1, 1 / e, 1 / (1 - e))


def aipw_ate(y, t, e, mu0, mu1, clip: float = 0.02, alpha: float = 0.05) -> dict:
    """Augmented IPW (doubly robust) ATE with an influence-function standard error.

    Consistent if EITHER the propensity model ``e`` OR the outcome models
    ``mu0``/``mu1`` are correctly specified. Inputs should be cross-fitted.
    """
    y, t = np.asarray(y), np.asarray(t)
    e = np.clip(np.asarray(e), clip, 1 - clip)
    psi = (mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e))
    ate = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(len(y))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"ate": ate, "se": se, "ci": (ate - z * se, ate + z * se),
            "p_value": 2 * stats.norm.sf(abs(ate / se))}


# ---------------------------------------------------------------------------
# Uplift evaluation
# ---------------------------------------------------------------------------
# Every function below accepts optional per-customer weights ``w``. With w = None
# the arms are compared as observed (valid under simple randomisation). Passing
# IPW weights (``ipw_weights``) compares re-weighted arms, which is required when
# the probability of treatment differs across customers.

def _arm_rates(y, t, w):
    w = np.ones(len(y)) if w is None else np.asarray(w)
    p1 = np.sum(w * y * t) / np.sum(w * t)
    p0 = np.sum(w * y * (1 - t)) / np.sum(w * (1 - t))
    return p1, p0


def uplift_by_group(score, y, t, n_groups: int = 10, w=None) -> pd.DataFrame:
    """Observed churn by arm within score quantile groups (group 1 = highest score).

    ``observed_reduction`` = churn(control) - churn(treated) in that group: the
    empirical churn reduction associated with contact, i.e. an estimate of the
    group's average treatment effect (sign flipped so positive = fewer churners).
    """
    y, t = np.asarray(y), np.asarray(t)
    w = np.ones(len(y)) if w is None else np.asarray(w)
    rank = pd.Series(score).rank(method="first", ascending=False).to_numpy()
    group = pd.qcut(rank, n_groups, labels=False) + 1
    rows = []
    for g in range(1, n_groups + 1):
        m = group == g
        p1, p0 = _arm_rates(y[m], t[m], w[m])
        n1, n0 = int(t[m].sum()), int((1 - t[m]).sum())
        se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        rows.append({"group": g, "n": int(m.sum()), "n_treated": n1, "n_control": n0,
                     "churn_treated": p1, "churn_control": p0,
                     "observed_reduction": p0 - p1, "se": se,
                     "mean_score": np.asarray(score)[m].mean()})
    return pd.DataFrame(rows)


def _sorted_cumsums(score, y, t, w):
    score, y, t = map(np.asarray, (score, y, t))
    w = np.ones(len(y)) if w is None else np.asarray(w)
    o = np.argsort(-score, kind="stable")
    y, t, w = y[o], t[o], w[o]
    return (np.cumsum(w * t), np.cumsum(w * (1 - t)),
            np.cumsum(w * y * t), np.cumsum(w * y * (1 - t)))


def uplift_curve(score, y, t, w=None):
    """Cumulative uplift curve for churn reduction.

    For the top fraction k of customers ranked by ``score``:
        gain(k) = [churn_control(k) - churn_treated(k)] * N(k)
    = estimated churners prevented if everyone in the top-k were contacted rather
    than nobody. Rates are computed per arm, which handles the unequal arm sizes.
    Returns (fractions, gain), both starting at 0.
    """
    ct, cc, yt, yc = _sorted_cumsums(score, y, t, w)
    n = len(ct)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_diff = np.where(cc > 0, yc / cc, 0) - np.where(ct > 0, yt / ct, 0)
    gain = rate_diff * np.arange(1, n + 1)
    return np.r_[0, np.arange(1, n + 1) / n], np.r_[0, gain]


def qini_curve(score, y, t, w=None):
    """Qini curve (Radcliffe, 2007) for churn reduction.

    Q(k) = Y_c(k) * N_t(k) / N_c(k) - Y_t(k): churners prevented among the
    *treated* customers in the top k, with control churners rescaled to the
    treated group size.
    """
    ct, cc, yt, yc = _sorted_cumsums(score, y, t, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(cc > 0, yc * ct / cc, 0) - yt
    return np.r_[0, np.arange(1, len(ct) + 1) / len(ct)], np.r_[0, q]


def area_vs_random(score, y, t, w=None, curve=uplift_curve) -> float:
    """Area between a model's uplift (or Qini) curve and the random-targeting line,
    normalised by N. Positive = the ranking beats random targeting."""
    x, g = curve(score, y, t, w)
    return float(np.trapezoid(g - x * g[-1], x)) / len(np.asarray(y))


def qini_coefficient(score, y, t, w=None) -> float:
    """Qini coefficient: area between the Qini curve and the random line, / N."""
    return area_vs_random(score, y, t, w, curve=qini_curve)


def targeting_value(score, y, t, budget: float, w=None) -> dict:
    """Churners prevented if the top ``budget`` share by ``score`` is contacted.

    Estimated within the selected customers as
    (churn_control - churn_treated) * N_selected, using both experimental arms.
    """
    score, y, t = map(np.asarray, (score, y, t))
    w = np.ones(len(y)) if w is None else np.asarray(w)
    k = int(round(budget * len(y)))
    sel = np.argsort(-score, kind="stable")[:k]
    p1, p0 = _arm_rates(y[sel], t[sel], w[sel])
    return {"budget": budget, "n_contacted": k,
            "churn_control": p0, "churn_treated": p1,
            "reduction_pp": 100 * (p0 - p1),
            "churners_prevented": (p0 - p1) * k}


def bootstrap_targeting(scores: dict, y, t, budgets, w=None, n_boot: int = 1000,
                        seed: int = 0) -> pd.DataFrame:
    """Bootstrap the churners-prevented estimate per strategy and budget.

    Scores are held fixed (they are out-of-fold predictions); the bootstrap
    captures sampling noise in the evaluation data, not model-fitting variability.
    """
    rng = np.random.default_rng(seed)
    y, t = np.asarray(y), np.asarray(t)
    w = np.ones(len(y)) if w is None else np.asarray(w)
    n = len(y)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for name, s in scores.items():
            s_b = np.asarray(s)[idx]
            for bud in budgets:
                v = targeting_value(s_b, y[idx], t[idx], bud, w[idx])
                rows.append({"boot": b, "strategy": name, "budget": bud,
                             "churners_prevented": v["churners_prevented"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic data with a known CATE (to validate the methodology)
# ---------------------------------------------------------------------------
def make_synthetic(n: int = 12000, p: int = 10, p_treat: float = 0.76,
                   effect_scale: float = 1.0, seed: int = 0):
    """Simulated RCT resembling the real data (~3-4% churn, 76% treated) with a
    *known* heterogeneous effect: contact strongly reduces churn for customers
    with x0 > 0.5 and slightly increases it ("sleeping dogs") when x1 > 1.
    ``effect_scale`` shrinks the effect (on the logit scale) to mimic weaker signals.
    Returns X, t, y, true_cate (churn scale).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    t = rng.binomial(1, p_treat, n)
    base = -3.6 + 0.6 * X[:, 0] + 0.4 * X[:, 2]
    effect = effect_scale * (-1.6 * (X[:, 0] > 0.5) + 0.8 * (X[:, 1] > 1))
    sig = lambda z: 1 / (1 + np.exp(-z))
    p0, p1 = sig(base), sig(base + effect)
    y = rng.binomial(1, np.where(t == 1, p1, p0))
    return X, t, y, p1 - p0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def set_style():
    """Quiet, consistent matplotlib style: recessive grid/axes, ink-coloured text."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "font.family": "sans-serif",
    })


def savefig(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
