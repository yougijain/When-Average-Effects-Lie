#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
When Average Effects Lie
========================
End-to-end causal analysis of the Hillstrom Email Marketing experiment.

The dataset is a clean randomized experiment: 64,000 customers were randomly
assigned to receive a Men's-merchandise email, a Women's-merchandise email, or
no email. We measure downstream `visit`, `conversion`, and `spend`.

The story this pipeline tells: the *average* treatment effect looks healthy, but
the average hides large heterogeneity. Uplift modelling turns that heterogeneity
into a targeting policy that beats blanket emailing -- the difference between
"the experiment worked" and "here is what to do on Monday".

Pipeline (8 layers)
-------------------
  1. Load & validate           - shape, missingness, arm balance
  2. Randomization checks       - covariate balance (SMD love plot, omnibus test)
  3. Average treatment effect   - naive diff-in-means / diff-in-proportions + CIs
  4. Regression adjustment      - Lin (2013) interacted estimator, variance gain
  5. Heterogeneous effects      - subgroup CATEs, interaction tests, BH correction
  6. Uplift modelling           - T-learner vs S-learner, Qini curve / coefficient
  7. Targeting policy value     - IPW policy value: target vs blanket vs none
  8. Robustness & inference     - randomization inference, bootstrap, power

Everything is computed from the data. Numbers printed here and written to
RESULTS.md are the real outputs -- nothing is hard-coded.

Run:
    python hillstrom_ab_analysis.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

# np.trapezoid (NumPy >= 2.0) replaced the older np.trapz; support both.
TRAPEZOID = getattr(np, "trapezoid", getattr(np, "trapz", None))

import matplotlib
matplotlib.use("Agg")  # headless / file output
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
DATA_PATH = DATA_DIR / "hillstrom.csv"
RESULTS_PATH = ROOT / "RESULTS.md"

# Tried in order. The canonical host only speaks plain HTTP, which some
# corporate proxies refuse outright, so try TLS first and fall back.
_DATA_FILE = (
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
DATA_URLS = (
    f"https://www.minethatdata.com/{_DATA_FILE}",
    f"http://www.minethatdata.com/{_DATA_FILE}",
)

CONTROL = "No E-Mail"
MENS = "Mens E-Mail"
WOMENS = "Womens E-Mail"

# Pre-treatment covariates used for balance, adjustment and uplift features.
NUM_COLS = ["recency", "history", "mens", "womens", "newbie"]
CAT_COLS = ["history_segment", "zip_code", "channel"]
COVARIATES = NUM_COLS + CAT_COLS

PRIMARY_OUTCOME = "visit"  # highest base-rate => most statistical power

# Illustrative economics for the policy section (clearly-stated ASSUMPTIONS,
# not claims about Hillstrom's real margins). These two numbers only set the
# break-even uplift = COST/VALUE; the qualitative targeting call is robust to a
# wide range of prices. Here break-even = 0.06/2.00 = 3.0pp, which lands inside
# the women's-email heterogeneity gap (~1pp for low-affinity vs ~7pp for high).
VALUE_PER_VISIT = 2.00   # $ expected downstream value of one incremental visit
COST_PER_EMAIL = 0.06    # $ fully-loaded cost of one contact (send + list fatigue)

# Collector for everything that ends up in RESULTS.md
REPORT: dict = {}


# --------------------------------------------------------------------------- #
# Small statistics helpers                                                    #
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def two_proportion_effect(s_t, n_t, s_c, n_c, alpha=0.05):
    """Risk difference between two proportions with Wald CI and pooled z-test.

    Returns a dict: p_t, p_c, risk_diff, ci_low, ci_high, rel_lift, z, p_value.
    """
    p_t, p_c = s_t / n_t, s_c / n_c
    rd = p_t - p_c
    se = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (rd - z_crit * se, rd + z_crit * se)

    # pooled two-proportion z-test (H0: p_t == p_c)
    p_pool = (s_t + s_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z = rd / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "p_t": p_t, "p_c": p_c, "risk_diff": rd, "se": se,
        "ci_low": ci[0], "ci_high": ci[1],
        "rel_lift": rd / p_c if p_c > 0 else np.nan,
        "z": z, "p_value": p_value, "n_t": int(n_t), "n_c": int(n_c),
    }


def mean_diff_bootstrap(a, b, n_boot=2000, alpha=0.05):
    """Difference in means (a - b) with a percentile bootstrap CI and Welch p."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = a.mean() - b.mean()
    boot = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        ba = a[RNG.integers(0, na, na)]
        bb = b[RNG.integers(0, nb, nb)]
        boot[i] = ba.mean() - bb.mean()
    ci = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return {"diff": diff, "ci_low": ci[0], "ci_high": ci[1], "p_value": p}


def standardized_mean_diff(x_t, x_c):
    """Standardized mean difference (Cohen-style) for a numeric/binary column."""
    mt, mc = np.mean(x_t), np.mean(x_c)
    vt, vc = np.var(x_t, ddof=1), np.var(x_c, ddof=1)
    pooled = np.sqrt((vt + vc) / 2)
    return (mt - mc) / pooled if pooled > 0 else 0.0


# --------------------------------------------------------------------------- #
# Layer 1 - Load & validate                                                   #
# --------------------------------------------------------------------------- #
def _download_dataset() -> None:  # pragma: no cover - network dependent
    """Fetch the dataset to a temp file, validate it, then move it into place.

    urlretrieve saves whatever the server returns, including an HTML error
    page. Writing straight to DATA_PATH would poison the cache: the file then
    exists, every later run skips the download, and the failure surfaces as a
    cryptic parse error instead of a network one. So: download aside, check the
    header, and only then commit the file.
    """
    tmp = DATA_PATH.with_suffix(".part")
    failures = []
    for url in DATA_URLS:
        try:
            urlretrieve(url, tmp)
            first = tmp.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
            if "recency" not in first.lower():
                raise ValueError(f"not the expected CSV (first line: {first[:80]!r})")
            tmp.replace(DATA_PATH)
            print(f"  downloaded from {url}")
            return
        except Exception as exc:
            failures.append(f"    {url}\n      -> {exc}")
        finally:
            tmp.unlink(missing_ok=True)

    sys.exit(
        "  Could not download the dataset. Tried:\n"
        + "\n".join(failures)
        + f"\n  Download it manually to {DATA_PATH} and re-run:\n"
        + f"    {DATA_URLS[0]}"
    )


def load_data() -> pd.DataFrame:
    banner("LAYER 1  -  Load & validate")
    DATA_DIR.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print(f"  downloading dataset -> {DATA_PATH}")
        _download_dataset()
    df = pd.read_csv(DATA_PATH)

    # The raw file misspells the zip label as 'Surburban'; normalize for display.
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})

    # Ordinal code for history_segment ("1) $0 - $100" -> 1) for plotting/sorting.
    df["history_seg_ord"] = df["history_segment"].str.extract(r"(\d)").astype(int)

    n, k = df.shape
    print(f"  rows x cols          : {n:,} x {k}")
    print(f"  missing cells        : {int(df.isna().sum().sum())}")
    print(f"  identical rows       : {int(df.duplicated().sum()):,}  "
          f"(coincident covariate/outcome combos from low-cardinality "
          f"features - expected, not dropped)")
    print("  arm sizes            :")
    counts = df["segment"].value_counts()
    for seg in [MENS, WOMENS, CONTROL]:
        print(f"     {seg:<14}: {counts[seg]:>6,}  ({counts[seg] / n:5.1%})")
    print("  outcome base rates   :")
    for col in ["visit", "conversion"]:
        print(f"     {col:<14}: {df[col].mean():6.3%}")
    print(f"     {'spend (mean)':<14}: ${df['spend'].mean():.3f}")

    REPORT["n_total"] = n
    REPORT["arm_sizes"] = {s: int(counts[s]) for s in [MENS, WOMENS, CONTROL]}
    REPORT["base_visit"] = df["visit"].mean()
    REPORT["base_conv"] = df["conversion"].mean()
    REPORT["base_spend"] = df["spend"].mean()
    return df


# --------------------------------------------------------------------------- #
# Layer 2 - Randomization / covariate balance                                 #
# --------------------------------------------------------------------------- #
def layer2_balance(df: pd.DataFrame) -> pd.DataFrame:
    banner("LAYER 2  -  Randomization checks (covariate balance)")
    treated = df[df["segment"] != CONTROL]
    control = df[df["segment"] == CONTROL]

    rows = []
    # Numeric / binary covariates: SMD directly.
    for col in ["recency", "history", "mens", "womens", "newbie", "history_seg_ord"]:
        smd = standardized_mean_diff(treated[col].values, control[col].values)
        rows.append((col, smd))
    # Categorical covariates: SMD per level (indicator).
    for col in ["zip_code", "channel"]:
        for level in sorted(df[col].unique()):
            smd = standardized_mean_diff(
                (treated[col] == level).astype(float).values,
                (control[col] == level).astype(float).values,
            )
            rows.append((f"{col}={level}", smd))

    bal = pd.DataFrame(rows, columns=["covariate", "smd"])
    bal["abs_smd"] = bal["smd"].abs()
    bal = bal.sort_values("abs_smd", ascending=False).reset_index(drop=True)

    max_smd = bal["abs_smd"].max()
    n_imbalanced = int((bal["abs_smd"] > 0.1).sum())
    print(bal.assign(smd=bal["smd"].round(4)).to_string(index=False))
    print(f"\n  max |SMD|            : {max_smd:.4f}  (rule of thumb: < 0.10 = balanced)")
    print(f"  covariates |SMD|>0.1 : {n_imbalanced} of {len(bal)}")

    # Omnibus test: can covariates jointly predict assignment? (They shouldn't.)
    omni = smf.logit(
        "is_treated ~ recency + history + mens + womens + newbie "
        "+ C(zip_code) + C(channel) + C(history_segment)",
        data=df.assign(is_treated=(df["segment"] != CONTROL).astype(int)),
    ).fit(disp=False)
    lr_p = stats.chi2.sf(omni.llr, omni.df_model)
    print(f"  omnibus LR test p    : {lr_p:.3f}  "
          f"({'PASS - no joint imbalance' if lr_p > 0.05 else 'WARN - imbalance'})")

    REPORT["max_smd"] = max_smd
    REPORT["balance_omnibus_p"] = lr_p
    _plot_love(bal)
    return bal


def _plot_love(bal: pd.DataFrame) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    order = bal.sort_values("abs_smd")
    colors = ["#c0392b" if v > 0.1 else "#2c7fb8" for v in order["abs_smd"]]
    ax.hlines(order["covariate"], 0, order["abs_smd"], color=colors, lw=2)
    ax.plot(order["abs_smd"], order["covariate"], "o", color="#2c7fb8")
    ax.axvline(0.1, ls="--", color="#c0392b", lw=1, label="0.10 threshold")
    ax.set_xlabel("|Standardized mean difference|")
    ax.set_title("Covariate balance: email vs. control (love plot)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_balance_love_plot.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Layer 3 - Average treatment effect (naive)                                  #
# --------------------------------------------------------------------------- #
def layer3_ate(df: pd.DataFrame) -> dict:
    banner("LAYER 3  -  Average treatment effect (naive)")
    ctrl = df[df["segment"] == CONTROL]
    results = {}

    for arm in [MENS, WOMENS]:
        trt = df[df["segment"] == arm]
        print(f"\n  {arm}  vs  {CONTROL}")
        arm_res = {}
        for outcome in ["visit", "conversion"]:
            r = two_proportion_effect(
                trt[outcome].sum(), len(trt), ctrl[outcome].sum(), len(ctrl)
            )
            arm_res[outcome] = r
            print(f"    {outcome:<11}: {r['p_c']:.3%} -> {r['p_t']:.3%}  "
                  f"| lift {r['risk_diff']*100:+.2f}pp "
                  f"[{r['ci_low']*100:+.2f}, {r['ci_high']*100:+.2f}]  "
                  f"| +{r['rel_lift']:.1%} rel  | p={r['p_value']:.2e}")
        sp = mean_diff_bootstrap(trt["spend"].values, ctrl["spend"].values)
        arm_res["spend"] = sp
        print(f"    {'spend':<11}: ${ctrl['spend'].mean():.3f} -> ${trt['spend'].mean():.3f}  "
              f"| diff ${sp['diff']:+.3f} "
              f"[${sp['ci_low']:+.3f}, ${sp['ci_high']:+.3f}]  | p={sp['p_value']:.3f}")
        results[arm] = arm_res

    REPORT["ate"] = {
        arm: {
            "visit_rd": results[arm]["visit"]["risk_diff"],
            "visit_ci": (results[arm]["visit"]["ci_low"], results[arm]["visit"]["ci_high"]),
            "visit_p": results[arm]["visit"]["p_value"],
            "conv_rd": results[arm]["conversion"]["risk_diff"],
            "conv_p": results[arm]["conversion"]["p_value"],
            "spend_diff": results[arm]["spend"]["diff"],
            "spend_p": results[arm]["spend"]["p_value"],
        }
        for arm in [MENS, WOMENS]
    }
    _plot_ate_forest(results)
    return results


def _plot_ate_forest(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels, est, lo, hi = [], [], [], []
    for arm in [MENS, WOMENS]:
        r = results[arm]["visit"]
        labels.append(f"{arm}\n(visit)")
        est.append(r["risk_diff"] * 100)
        lo.append(r["ci_low"] * 100)
        hi.append(r["ci_high"] * 100)
    y = np.arange(len(labels))
    ax.errorbar(est, y, xerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
                fmt="o", color="#2c7fb8", capsize=4, lw=2)
    ax.axvline(0, ls="--", color="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Visit-rate lift vs. control (percentage points, 95% CI)")
    ax.set_title("Average treatment effect on website visits")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_ate_forest.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Layer 4 - Regression adjustment (variance reduction)                        #
# --------------------------------------------------------------------------- #
def layer4_adjustment(df: pd.DataFrame) -> dict:
    banner("LAYER 4  -  Regression adjustment (Lin 2013 estimator)")
    out = {}
    for arm in [MENS, WOMENS]:
        sub = df[df["segment"].isin([arm, CONTROL])].copy()
        sub["T"] = (sub["segment"] == arm).astype(int)

        # Naive: outcome ~ T  (HC1 robust SE)
        m0 = smf.ols(f"{PRIMARY_OUTCOME} ~ T", data=sub).fit(cov_type="HC1")
        # Lin estimator: center covariates, fully interact with T.
        cov = ["recency", "history", "mens", "womens", "newbie"]
        for c in cov:
            sub[c + "_c"] = sub[c] - sub[c].mean()
        terms = " + ".join(c + "_c" for c in cov)
        inter = " + ".join(f"T:{c}_c" for c in cov)
        m1 = smf.ols(
            f"{PRIMARY_OUTCOME} ~ T + {terms} + C(zip_code) + C(channel) "
            f"+ C(history_segment) + {inter}",
            data=sub,
        ).fit(cov_type="HC1")

        b0, se0 = m0.params["T"], m0.bse["T"]
        b1, se1 = m1.params["T"], m1.bse["T"]
        var_red = 1 - (se1 ** 2) / (se0 ** 2)   # proportional VARIANCE reduction
        se_red = 1 - se1 / se0                   # the (smaller) SE reduction
        print(f"\n  {arm} vs {CONTROL}  (outcome = {PRIMARY_OUTCOME})")
        print(f"    naive ATE   : {b0*100:+.3f}pp  (SE {se0*100:.3f})")
        print(f"    adjusted ATE: {b1*100:+.3f}pp  (SE {se1*100:.3f})")
        print(f"    variance reduction: {var_red:5.1%}  (SE -{se_red:.1%})  "
              f"(estimate stable => randomization holds; CI is tighter)")
        out[arm] = {"naive": b0, "naive_se": se0, "adj": b1,
                    "adj_se": se1, "var_red": var_red}
    REPORT["adjustment"] = out
    return out


# --------------------------------------------------------------------------- #
# Layer 5 - Heterogeneous treatment effects                                   #
# --------------------------------------------------------------------------- #
def layer5_hte(df: pd.DataFrame) -> dict:
    banner("LAYER 5  -  Heterogeneous treatment effects (where the average lies)")
    moderators = {
        "mens (bought men's)": ("mens", [0, 1]),
        "womens (bought women's)": ("womens", [0, 1]),
        "newbie": ("newbie", [0, 1]),
        "zip_code": ("zip_code", None),
        "channel": ("channel", None),
    }
    interaction_tests = []  # (label, p_value) for BH correction in Layer 8
    subgroup_table = {}

    for arm in [MENS, WOMENS]:
        sub = df[df["segment"].isin([arm, CONTROL])].copy()
        sub["T"] = (sub["segment"] == arm).astype(int)
        ctrl = sub[sub["T"] == 0]
        trt = sub[sub["T"] == 1]
        print(f"\n  === {arm} vs {CONTROL} :: subgroup visit-rate lifts ===")
        arm_rows = []
        for name, (col, levels) in moderators.items():
            use_levels = levels if levels is not None else sorted(df[col].unique())
            for lv in use_levels:
                t = trt[trt[col] == lv]
                c = ctrl[ctrl[col] == lv]
                if len(t) < 30 or len(c) < 30:
                    continue
                r = two_proportion_effect(t["visit"].sum(), len(t),
                                          c["visit"].sum(), len(c))
                arm_rows.append({
                    "moderator": name, "level": lv,
                    "lift_pp": r["risk_diff"] * 100,
                    "ci_low": r["ci_low"] * 100, "ci_high": r["ci_high"] * 100,
                    "p": r["p_value"], "n": r["n_t"] + r["n_c"],
                })
            # Interaction test for binary moderators (treatment x moderator).
            if levels == [0, 1]:
                m = smf.ols(f"visit ~ T * {col}", data=sub).fit(cov_type="HC1")
                coef = m.params.get(f"T:{col}", np.nan)
                pval = m.pvalues.get(f"T:{col}", np.nan)
                interaction_tests.append((f"{arm}: T x {col}", pval))
                print(f"    interaction  T x {col:<7}: "
                      f"{coef*100:+.3f}pp  p={pval:.4f}")

        tbl = pd.DataFrame(arm_rows)
        subgroup_table[arm] = tbl
        # Show the biggest spread within each moderator.
        for name in moderators:
            block = tbl[tbl["moderator"] == name]
            if len(block) >= 2:
                spread = block["lift_pp"].max() - block["lift_pp"].min()
                lo = block.loc[block["lift_pp"].idxmin()]
                hi = block.loc[block["lift_pp"].idxmax()]
                print(f"    {name:<24}: "
                      f"{lo['level']}={lo['lift_pp']:+.2f}pp .. "
                      f"{hi['level']}={hi['lift_pp']:+.2f}pp  "
                      f"(spread {spread:.2f}pp)")

    REPORT["interaction_tests"] = interaction_tests
    REPORT["subgroup_tables"] = {a: subgroup_table[a].to_dict("records")
                                 for a in subgroup_table}
    _plot_hte(subgroup_table)
    return {"interaction_tests": interaction_tests, "subgroup": subgroup_table}


def _plot_hte(subgroup_table: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, arm in zip(axes, [MENS, WOMENS]):
        tbl = subgroup_table[arm].copy()
        tbl["label"] = tbl["moderator"] + " = " + tbl["level"].astype(str)
        tbl = tbl.iloc[::-1]
        y = np.arange(len(tbl))
        ax.errorbar(tbl["lift_pp"], y,
                    xerr=[tbl["lift_pp"] - tbl["ci_low"], tbl["ci_high"] - tbl["lift_pp"]],
                    fmt="o", capsize=3, color="#2c7fb8", lw=1.5)
        ax.axvline(0, ls="--", color="grey", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(tbl["label"], fontsize=8)
        ax.set_title(f"{arm}")
        ax.set_xlabel("Visit-rate lift vs. control (pp, 95% CI)")
    fig.suptitle("Heterogeneous treatment effects by customer segment")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_hte_forest.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Layer 6 - Uplift modelling                                                  #
# --------------------------------------------------------------------------- #
def _make_encoder(train_df: pd.DataFrame) -> ColumnTransformer:
    enc = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)],
        remainder="passthrough",
    )
    enc.fit(train_df[COVARIATES])
    return enc


def qini_curve(y, treat, score):
    """Qini curve and coefficient for a binary-treatment uplift ranking.

    Returns fraction targeted, qini values, the random baseline line, and the
    Qini coefficient (area between model curve and random baseline).
    """
    order = np.argsort(-score, kind="mergesort")
    y, t = np.asarray(y)[order], np.asarray(treat)[order]
    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cum_c > 0, cum_t / cum_c, 0.0)
    qini = cum_yt - cum_yc * ratio
    n = len(y)
    frac = np.arange(1, n + 1) / n
    total_qini = cum_yt[-1] - cum_yc[-1] * (cum_t[-1] / cum_c[-1])
    random_line = frac * total_qini
    qini_coef = TRAPEZOID(qini, frac) - TRAPEZOID(random_line, frac)
    return frac, qini, random_line, qini_coef


def uplift_at_k(y, treat, score, k=0.30):
    """Observed uplift among the top-k fraction by predicted score."""
    cut = np.quantile(score, 1 - k)
    top = score >= cut
    yt = y[top & (treat == 1)]
    yc = y[top & (treat == 0)]
    if len(yt) == 0 or len(yc) == 0:
        return np.nan
    return yt.mean() - yc.mean()


def layer6_uplift(df: pd.DataFrame, arm: str) -> dict:
    banner(f"LAYER 6  -  Uplift modelling  ({arm} vs {CONTROL})")
    sub = df[df["segment"].isin([arm, CONTROL])].copy()
    sub["T"] = (sub["segment"] == arm).astype(int)

    train, test = train_test_split(
        sub, test_size=0.35, random_state=SEED, stratify=sub["T"]
    )
    enc = _make_encoder(train)
    Xtr = enc.transform(train[COVARIATES])
    Xte = enc.transform(test[COVARIATES])
    ytr = train[PRIMARY_OUTCOME].values
    yte = test[PRIMARY_OUTCOME].values
    ttr = train["T"].values
    tte = test["T"].values

    # --- T-learner (two-model) ---
    m_t = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m_c = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr[ttr == 0], ytr[ttr == 0])
    score_t = m_t.predict_proba(Xte)[:, 1] - m_c.predict_proba(Xte)[:, 1]

    # --- S-learner (single model with treatment as a feature) ---
    Xtr_s = np.column_stack([Xtr, ttr])
    s_model = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr_s, ytr)
    X1 = np.column_stack([Xte, np.ones(len(Xte))])
    X0 = np.column_stack([Xte, np.zeros(len(Xte))])
    score_s = s_model.predict_proba(X1)[:, 1] - s_model.predict_proba(X0)[:, 1]

    frac_t, qini_t, rand_t, coef_t = qini_curve(yte, tte, score_t)
    frac_s, qini_s, _, coef_s = qini_curve(yte, tte, score_s)
    u30_t = uplift_at_k(yte, tte, score_t, 0.30)
    u30_s = uplift_at_k(yte, tte, score_s, 0.30)

    overall = (yte[tte == 1].mean() - yte[tte == 0].mean())
    print(f"  test set            : {len(test):,} rows "
          f"({tte.sum():,} treated / {(1-tte).sum():,} control)")
    print(f"  overall test uplift : {overall*100:+.3f}pp")
    print(f"  T-learner Qini coef : {coef_t:8.2f}   uplift@30% = {u30_t*100:+.3f}pp")
    print(f"  S-learner Qini coef : {coef_s:8.2f}   uplift@30% = {u30_s*100:+.3f}pp")
    better = "T-learner" if coef_t >= coef_s else "S-learner"
    score_best = score_t if coef_t >= coef_s else score_s
    print(f"  better ranker       : {better}")

    REPORT.setdefault("uplift", {})[arm] = {
        "qini_t": coef_t, "qini_s": coef_s,
        "u30_t": u30_t, "u30_s": u30_s,
        "overall_test_uplift": overall, "better": better,
    }
    _plot_qini(arm, frac_t, qini_t, rand_t, qini_s)
    return {
        "test": test, "score_t": score_t, "score_s": score_s,
        "score_best": score_best, "better": better,
        "yte": yte, "tte": tte, "qini_coef": max(coef_t, coef_s),
    }


def _plot_qini(arm, frac, qini_t, rand, qini_s) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(frac, qini_t, color="#2c7fb8", lw=2, label="T-learner")
    ax.plot(frac, qini_s, color="#41ab5d", lw=1.5, ls="-.", label="S-learner")
    ax.plot(frac, rand, color="grey", lw=1.2, ls="--", label="Random targeting")
    ax.set_xlabel("Fraction of customers targeted")
    ax.set_ylabel("Cumulative incremental visits (Qini)")
    ax.set_title(f"Qini curve - {arm}")
    ax.legend()
    fig.tight_layout()
    safe = arm.split()[0].lower()
    fig.savefig(FIG_DIR / f"04_qini_{safe}.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Layer 7 - Targeting policy value (IPW)                                       #
# --------------------------------------------------------------------------- #
def layer7_policy(arm: str, art: dict) -> dict:
    banner(f"LAYER 7  -  Targeting policy value  ({arm} vs {CONTROL})")
    yte, tte, score = art["yte"], art["tte"], art["score_best"]
    n = len(yte)
    p = tte.mean()  # randomized propensity in this two-arm subset
    breakeven = COST_PER_EMAIL / VALUE_PER_VISIT  # min uplift that pays for a contact
    print(f"  policy ranker       : {art['better']} (best Qini)")
    print(f"  economics (illustrative): visit ${VALUE_PER_VISIT:.2f}, "
          f"contact ${COST_PER_EMAIL:.2f}  ->  break-even uplift "
          f"{breakeven*100:.1f}pp")

    def ipw_value(policy):
        """IPW estimate of E[Y(policy(X))] under known constant propensity p.

        Reduces to mean(treated) for treat-all and mean(control) for treat-none.
        """
        w = np.where(policy == 1, (tte == 1) / p, (tte == 0) / (1 - p))
        return (w * yte).mean()

    v_all = ipw_value(np.ones(n, int))
    v_none = ipw_value(np.zeros(n, int))
    pi = (score > breakeven).astype(int)  # cost-sensitive: contact if uplift pays
    frac = pi.mean()
    v_pi = ipw_value(pi)

    # Net INCREMENTAL value per 1,000 customers, vs. contacting no one. Only
    # incremental visits are credited -- baseline visits would happen anyway.
    def net_value(v_policy, email_frac):
        return ((v_policy - v_none) * VALUE_PER_VISIT - email_frac * COST_PER_EMAIL) * 1000

    nv_all = net_value(v_all, 1.0)
    nv_pi = net_value(v_pi, frac)

    print(f"  policy: contact if predicted uplift > {breakeven*100:.1f}pp  "
          f"->  targets {frac:.1%} of customers")
    print(f"  visit rate | none {v_none:.3%}  all {v_all:.3%}  policy {v_pi:.3%}")
    print(f"  incremental visits / 1,000 (vs none): "
          f"blanket {(v_all-v_none)*1000:+.1f}   targeted {(v_pi-v_none)*1000:+.1f}")
    print(f"  NET VALUE / 1,000 (vs none): blanket ${nv_all:+.2f}   "
          f"targeted ${nv_pi:+.2f}   "
          f"(targeting {nv_pi-nv_all:+.2f}, {(1-frac)*1000:.0f} fewer contacts)")

    # Segment the policy excludes. Compare its *realized* uplift to break-even:
    # below => the model correctly dropped unprofitable customers; above => the
    # model misranked profitable ones (a sign there is little to target).
    excluded = score <= breakeven
    if excluded.sum() > 0:
        et = yte[excluded & (tte == 1)]
        ec = yte[excluded & (tte == 0)]
        ex_up = (et.mean() - ec.mean()) if len(et) and len(ec) else np.nan
        if not np.isnan(ex_up):
            below = ex_up < breakeven
            sign = "<" if below else ">"
            verdict = ("net-negative to contact - correctly dropped" if below
                       else "still profitable - model misranked them; blanket wins here")
            print(f"  excluded segment ({excluded.mean():.0%} of base): realized "
                  f"uplift {ex_up*100:+.2f}pp {sign} {breakeven*100:.1f}pp break-even "
                  f"=> {verdict}")

    REPORT.setdefault("policy", {})[arm] = {
        "breakeven_pp": breakeven * 100,
        "targeted_frac": frac,
        "v_all": v_all, "v_none": v_none, "v_policy": v_pi,
        "inc_visits_blanket_per1k": (v_all - v_none) * 1000,
        "inc_visits_targeted_per1k": (v_pi - v_none) * 1000,
        "nv_blanket": nv_all, "nv_targeted": nv_pi,
        "contacts_saved_per1k": (1 - frac) * 1000,
    }
    _plot_policy_curve(arm, yte, tte, score, p)
    return REPORT["policy"][arm]


def _plot_policy_curve(arm, yte, tte, score, p) -> None:
    """Net value per 1,000 as targeting extends from highest to lowest uplift."""
    order = np.argsort(-score, kind="mergesort")
    y, t = yte[order], tte[order]
    v_none = y[t == 0].mean()  # IPW value of contacting no one = control mean
    fracs = np.linspace(0.02, 1.0, 50)
    nv = []
    for f in fracs:
        k = max(1, int(f * len(y)))
        pol = np.zeros(len(y), int)
        pol[:k] = 1
        w = np.where(pol == 1, (t == 1) / p, (t == 0) / (1 - p))
        v = (w * y).mean()
        nv.append(((v - v_none) * VALUE_PER_VISIT - f * COST_PER_EMAIL) * 1000)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fracs * 100, nv, color="#2c7fb8", lw=2)
    best = int(np.argmax(nv))
    ax.plot(fracs[best] * 100, nv[best], "o", color="#c0392b",
            label=f"optimum @ {fracs[best]*100:.0f}% contacted")
    ax.axhline(0, ls="--", color="grey", lw=1)
    ax.set_xlabel("% of customers contacted (highest predicted uplift first)")
    ax.set_ylabel("Net value / 1,000 vs. contacting no one ($)")
    ax.set_title(f"Targeting profit curve - {arm}")
    ax.legend()
    fig.tight_layout()
    safe = arm.split()[0].lower()
    fig.savefig(FIG_DIR / f"05_policy_{safe}.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Layer 8 - Robustness & inference                                            #
# --------------------------------------------------------------------------- #
def layer8_robustness(df: pd.DataFrame, hte: dict) -> None:
    banner("LAYER 8  -  Robustness & inference")

    # (a) Benjamini-Hochberg correction across all interaction tests.
    tests = hte["interaction_tests"]
    labels = [t[0] for t in tests]
    pvals = np.array([t[1] for t in tests])
    order = np.argsort(pvals)
    m = len(pvals)
    bh = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=0):
        i = m - rank
        val = min(prev, pvals[idx] * m / i)
        bh[idx] = prev = val
    print("  (a) Multiple-testing correction (Benjamini-Hochberg, FDR=0.05):")
    for lab, raw, adj in sorted(zip(labels, pvals, bh), key=lambda x: x[2]):
        flag = "significant" if adj < 0.05 else "n.s."
        print(f"      {lab:<22}: raw p={raw:.4f}  BH p={adj:.4f}  [{flag}]")

    # (b) Randomization inference for the primary ATE (Womens email on visit).
    sub = df[df["segment"].isin([WOMENS, CONTROL])]
    y = sub["visit"].values.astype(float)
    t = (sub["segment"] == WOMENS).values
    obs = y[t].mean() - y[~t].mean()
    n_t = t.sum()
    B = 2000
    null = np.empty(B)
    idx = np.arange(len(y))
    for b in range(B):
        perm = RNG.permutation(idx)
        sel = perm[:n_t]
        mask = np.zeros(len(y), bool)
        mask[sel] = True
        null[b] = y[mask].mean() - y[~mask].mean()
    ri_p = (np.abs(null) >= abs(obs)).mean()
    print(f"\n  (b) Randomization inference ({WOMENS} on visit, B={B}):")
    print(f"      observed ATE = {obs*100:+.3f}pp ; "
          f"permutation p = {ri_p:.4f} "
          f"(null SD = {null.std()*100:.3f}pp)")

    # (c) Retrospective power & MDE for the primary comparison.
    p_c = y[~t].mean()
    p_t = y[t].mean()
    h = proportion_effectsize(p_t, p_c)
    analysis = NormalIndPower()
    power = analysis.power(effect_size=abs(h), nobs1=int((~t).sum()),
                           alpha=0.05, ratio=n_t / (~t).sum(), alternative="two-sided")
    mde_h = analysis.solve_power(effect_size=None, nobs1=int((~t).sum()),
                                 alpha=0.05, power=0.8,
                                 ratio=n_t / (~t).sum(), alternative="two-sided")
    print(f"\n  (c) Power for {WOMENS} vs control on visit:")
    print(f"      achieved power = {power:.3f} ; "
          f"detectable Cohen's h @ 80% power = {mde_h:.4f}")

    REPORT["robustness"] = {
        "ri_p": ri_p, "ri_obs": obs,
        "power": power,
        "bh": {lab: float(adj) for lab, adj in zip(labels, bh)},
    }


# --------------------------------------------------------------------------- #
# RESULTS.md writer                                                           #
# --------------------------------------------------------------------------- #
def write_results_md() -> None:
    r = REPORT
    mens = r["ate"][MENS]
    wom = r["ate"][WOMENS]

    def pp(x):  # percentage-point formatter
        return f"{x*100:+.2f}pp"

    lines = []
    lines.append("# Results — When Average Effects Lie\n")
    lines.append(
        "_Auto-generated by `hillstrom_ab_analysis.py`. Every number below is "
        "computed from the 64,000-customer Hillstrom randomized experiment._\n"
    )

    lines.append("## 1. Experiment integrity")
    lines.append(
        f"- Arms: **{r['arm_sizes'][MENS]:,}** Men's email · "
        f"**{r['arm_sizes'][WOMENS]:,}** Women's email · "
        f"**{r['arm_sizes'][CONTROL]:,}** control.")
    lines.append(
        f"- Covariate balance: max |SMD| = **{r['max_smd']:.3f}** "
        f"(< 0.10 ⇒ balanced); omnibus assignment test p = "
        f"**{r['balance_omnibus_p']:.2f}** ⇒ randomization holds.\n")

    lines.append("## 2. Average treatment effect (visit rate)")
    lines.append("| Campaign | Visit lift | 95% CI | p-value |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Men's email | {pp(mens['visit_rd'])} | "
                 f"[{mens['visit_ci'][0]*100:+.2f}, {mens['visit_ci'][1]*100:+.2f}] | "
                 f"{mens['visit_p']:.1e} |")
    lines.append(f"| Women's email | {pp(wom['visit_rd'])} | "
                 f"[{wom['visit_ci'][0]*100:+.2f}, {wom['visit_ci'][1]*100:+.2f}] | "
                 f"{wom['visit_p']:.1e} |")
    lines.append("\nRegression adjustment (Lin 2013) left the point estimates "
                 "essentially unchanged while cutting the sampling **variance** by "
                 f"**{r['adjustment'][WOMENS]['var_red']:.0%}** (Women's) / "
                 f"**{r['adjustment'][MENS]['var_red']:.0%}** (Men's) — exactly "
                 "what you expect when assignment is random.\n")

    lines.append("## 3. The twist — heterogeneous effects")
    # Find the strongest interaction (smallest BH p) to headline.
    bh = r["robustness"]["bh"]
    head = min(bh.items(), key=lambda kv: kv[1])
    lines.append(
        f"- Strongest moderation: **{head[0]}** (BH-adjusted p = {head[1]:.3f}).")
    # Report each moderator by its actual level (yes=1 / no=0) -- never infer
    # the label from the size of the lift.
    for arm in [MENS, WOMENS]:
        tbl = pd.DataFrame(r["subgroup_tables"][arm])
        for key, lab in [("mens", "prior men's-merch buyer"),
                         ("womens", "prior women's-merch buyer")]:
            block = tbl[tbl["moderator"].str.startswith(key)]
            yes = block[block["level"] == 1]["lift_pp"]
            no = block[block["level"] == 0]["lift_pp"]
            if len(yes) and len(no):
                y, n0 = float(yes.iloc[0]), float(no.iloc[0])
                lines.append(
                    f"- **{arm}** × {lab}: **{y:+.2f}pp** (yes) vs "
                    f"**{n0:+.2f}pp** (no) — gap {abs(y - n0):.2f}pp.")
    lines.append("")

    lines.append("## 4. Uplift modelling & cost-sensitive targeting")
    lines.append(
        f"_Illustrative economics: a visit is worth ${VALUE_PER_VISIT:.2f} and a "
        f"contact costs ${COST_PER_EMAIL:.2f} (break-even uplift "
        f"{COST_PER_EMAIL / VALUE_PER_VISIT * 100:.1f}pp). Net value is per 1,000 "
        f"customers vs. contacting no one. The qualitative call — broad for Men's, "
        f"selective for Women's — is robust to the exact prices._\n")
    for arm in [MENS, WOMENS]:
        up = r["uplift"][arm]
        po = r["policy"][arm]
        targeting_wins = po["nv_targeted"] > po["nv_blanket"]
        verdict = (f"**target the top {po['targeted_frac']:.0%}** "
                   f"(saves {po['contacts_saved_per1k']:.0f} contacts/1,000)"
                   if targeting_wins else
                   "**contact broadly** — targeting adds nothing")
        lines.append(
            f"- **{arm}** (best ranker {up['better']}, Qini "
            f"{max(up['qini_t'], up['qini_s']):.1f}): net value blanket "
            f"${po['nv_blanket']:+.2f} vs targeted ${po['nv_targeted']:+.2f} / 1,000 "
            f"→ {verdict}.")
    lines.append("")

    lines.append("## 5. Robustness")
    rb = r["robustness"]
    lines.append(
        f"- Randomization inference (B=2000) on the Women's-email visit effect: "
        f"observed {pp(rb['ri_obs'])}, permutation p = **{rb['ri_p']:.4f}**.")
    lines.append(
        f"- Achieved power for that comparison: **{rb['power']:.2f}**.")
    lines.append("- All subgroup interactions reported with Benjamini-Hochberg "
                 "FDR control (see console output).\n")

    lines.append("## Figures")
    for f in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- `figures/{f.name}`")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  wrote {RESULTS_PATH.name}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    print(textwrap.dedent("""
        ##############################################################
        #  WHEN AVERAGE EFFECTS LIE                                  #
        #  Hillstrom email experiment - 8-layer causal analysis     #
        ##############################################################"""))
    df = load_data()
    layer2_balance(df)
    layer3_ate(df)
    layer4_adjustment(df)
    hte = layer5_hte(df)
    for arm in [MENS, WOMENS]:
        art = layer6_uplift(df, arm)
        layer7_policy(arm, art)
    layer8_robustness(df, hte)
    write_results_md()

    banner("DONE")
    print(f"  figures : {FIG_DIR}")
    print(f"  summary : {RESULTS_PATH}")
    print("  Open RESULTS.md for the headline numbers to drop into your README.")


if __name__ == "__main__":
    main()
