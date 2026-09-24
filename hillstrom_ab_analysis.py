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

import gzip
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
from matplotlib.ticker import PercentFormatter

import statsmodels.formula.api as smf
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split


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
# corporate proxies refuse outright, so try TLS first and fall back. The third
# entry is the gzipped copy scikit-uplift distributes, for networks that refuse
# minethatdata.com altogether; it is the same 64,000 rows with the same arm
# sizes. Nothing is trusted on the strength of its URL - every download is
# checked against EXPECTED_ROWS and EXPECTED_ARMS before it is cached, so a
# mirror that reindexed or resampled the file fails loudly instead of flowing
# into the numbers below.
_DATA_FILE = (
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
DATA_URLS = (
    f"https://www.minethatdata.com/{_DATA_FILE}",
    f"http://www.minethatdata.com/{_DATA_FILE}",
    "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
)

CONTROL = "No E-Mail"
MENS = "Mens E-Mail"
WOMENS = "Womens E-Mail"

# The published experiment, used to authenticate a downloaded file.
EXPECTED_ROWS = 64_000
EXPECTED_ARMS = {MENS: 21_307, WOMENS: 21_387, CONTROL: 21_306}

# Pre-treatment covariates used for balance, adjustment and uplift features.
NUM_COLS = ["recency", "history", "mens", "womens", "newbie"]
CAT_COLS = ["history_segment", "zip_code", "channel"]
COVARIATES = NUM_COLS + CAT_COLS

PRIMARY_OUTCOME = "visit"  # highest base-rate => most statistical power

# Share of each two-arm subset held out for reporting, and the number of folds
# used to choose between the learners on what is left. Selection and reporting
# never share a row: see _select_learner.
REPORT_FRAC = 0.35
N_SELECT_FOLDS = 5

# Deciles of predicted uplift for the calibration check in Layer 6b.
N_CALIB_BINS = 10

# Bootstrap resamples of the reporting split for the Layer 7b intervals.
N_BOOT_REPORT = 2000

# Illustrative economics for the policy section (clearly-stated ASSUMPTIONS,
# not claims about Hillstrom's real margins). These two numbers only set the
# break-even uplift = COST/VALUE; the qualitative targeting call is robust to a
# wide range of prices. Here break-even = 0.06/2.00 = 3.0pp, which lands inside
# the women's-email heterogeneity gap (~1pp for low-affinity vs ~7pp for high).
VALUE_PER_VISIT = 2.00   # $ expected downstream value of one incremental visit
COST_PER_EMAIL = 0.06    # $ fully-loaded cost of one contact (send + list fatigue)

# --------------------------------------------------------------------------- #
# Figure style - matches the write-up page (index.html)                       #
# --------------------------------------------------------------------------- #
# Same paper, ink and rule colours as the page's CSS. BLUE and RUST are the two
# categorical slots (T-learner / S-learner); they were checked for colour-vision
# separation (Delta E 18.5 under protanopia) and for chroma, which is why BLUE is
# a step more saturated than the page's original link colour - the page now uses
# this value too, so chart and prose share one blue.
PAPER, INK, INK2, INK3, RULE = "#fdfdfb", "#1b1b1a", "#4a4a46", "#6f6f68", "#d8d6cf"
BLUE, RUST = "#1a5e94", "#b4561f"
DASH = (0, (4, 3))                       # for threshold / reference lines only
DOT = (0, (1, 2.5))                      # second reference style, where DASH is taken
FONT_FILE = ROOT / "fonts" / "SourceSerif4-normal.ttf"
ARM_LABEL = {MENS: "Men's email", WOMENS: "Women's email"}


def _setup_style() -> None:
    """Editorial defaults: the page's typeface and palette, no chart chrome."""
    from matplotlib import font_manager
    family = "DejaVu Serif"
    if FONT_FILE.exists():
        font_manager.fontManager.addfont(str(FONT_FILE))
        family = "Source Serif 4"
    plt.rcParams.update({
        "font.family": family, "font.size": 9.5, "text.color": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.titlepad": 10,
        "axes.labelsize": 9, "axes.labelcolor": INK2,
        "axes.edgecolor": RULE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.facecolor": PAPER, "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
        "xtick.color": RULE, "ytick.color": RULE,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": RULE, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    })


def _pretty_covariate(name: str) -> str:
    """'zip_code=Rural' -> 'ZIP code: Rural'; 'mens' -> \"Prior men's purchase\"."""
    base = {
        "recency": "Recency (months)", "history": "Spend history ($)",
        "history_seg_ord": "History segment", "mens": "Prior men's purchase",
        "womens": "Prior women's purchase", "newbie": "New customer",
    }
    if "=" in name:
        col, level = name.split("=", 1)
        return f"{'ZIP code' if col == 'zip_code' else col.capitalize()}: {level}"
    return base.get(name, name)


def _pretty_segment(moderator: str, level) -> str:
    """Subgroup row label: the moderator's display name plus its level."""
    if moderator.startswith("mens"):
        return "Prior men's purchase" if level == 1 else "No prior men's purchase"
    if moderator.startswith("womens"):
        return "Prior women's purchase" if level == 1 else "No prior women's purchase"
    if moderator.startswith("newbie"):
        return "New customer" if level == 1 else "Existing customer"
    if moderator.startswith("zip"):
        return f"ZIP code: {level}"
    return f"{moderator.capitalize()}: {level}"


def _hide_left_spine(ax) -> None:
    """Dot-and-interval plots read better without a y-axis rule and tick marks."""
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

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
def _validate_dataset(path: Path) -> None:
    """Raise unless `path` really is the Hillstrom experiment.

    A download can succeed and still be the wrong bytes: an HTML error page, or
    a mirror that reindexed, resampled or re-shuffled the file. The columns
    catch the first. The row count and the three arm sizes catch the second,
    which is the failure that would otherwise pass silently into every estimate
    downstream - a resampled file still parses, still runs, and quietly reports
    a different experiment.
    """
    df = pd.read_csv(path)
    required = [*COVARIATES, "segment", "visit", "conversion", "spend"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"not the expected CSV (missing columns: {missing})")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} rows, found {len(df):,}")
    arms = {str(k): int(v) for k, v in df["segment"].value_counts().items()}
    if arms != EXPECTED_ARMS:
        raise ValueError(f"expected arm sizes {EXPECTED_ARMS}, found {arms}")


def _download_dataset() -> None:  # pragma: no cover - network dependent
    """Fetch the dataset to a temp file, validate it, then move it into place.

    urlretrieve saves whatever the server returns, including an HTML error
    page. Writing straight to DATA_PATH would poison the cache: the file then
    exists, every later run skips the download, and the failure surfaces as a
    cryptic parse error instead of a network one. So: download aside, check it
    is the right experiment, and only then commit the file.
    """
    tmp = DATA_PATH.with_suffix(".part")
    failures = []
    for url in DATA_URLS:
        try:
            urlretrieve(url, tmp)
            if url.endswith(".gz"):
                tmp.write_bytes(gzip.decompress(tmp.read_bytes()))
            _validate_dataset(tmp)
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
    order = bal.sort_values("abs_smd")
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hlines(y, 0, order["abs_smd"], color=BLUE, lw=1.4)
    ax.plot(order["abs_smd"], y, "o", color=BLUE, ms=6, mec=PAPER, mew=1.2)
    # The threshold is a reference, not a series: dashed, labelled in place, no legend.
    ax.axvline(0.10, ls=DASH, color=INK3, lw=0.9)
    ax.text(0.097, y[-1] + 0.15, "0.10, the conventional\nimbalance threshold",
            ha="right", va="top", fontsize=8, color=INK3, linespacing=1.3)
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_covariate(c) for c in order["covariate"]])
    ax.set_xlim(0, 0.108)
    _hide_left_spine(ax)
    ax.set_xlabel("Absolute standardized mean difference, email arms vs. control")
    ax.set_title("Covariate balance")
    fig.savefig(FIG_DIR / "01_balance_love_plot.png")
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
    labels, est, lo, hi = [], [], [], []
    for arm in [MENS, WOMENS]:
        r = results[arm]["visit"]
        labels.append(ARM_LABEL[arm])
        est.append(r["risk_diff"] * 100)
        lo.append(r["ci_low"] * 100)
        hi.append(r["ci_high"] * 100)
    est, lo, hi = map(np.array, (est, lo, hi))
    y = np.arange(len(labels))[::-1]           # Men's on top, as in the tables
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.axvline(0, color=INK3, lw=0.8)
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color=BLUE, ecolor=INK2,
                elinewidth=1.2, capsize=3, ms=6, mec=PAPER, mew=1.2)
    for yi, e, h in zip(y, est, hi):           # two points: label both
        ax.annotate(f"{e:+.2f} pp", (h, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.5, hi.max() + 1.8)
    _hide_left_spine(ax)
    ax.set_xlabel("Lift in visit rate vs. no email (percentage points, 95% CI)")
    ax.set_title("Average treatment effect on website visits")
    fig.savefig(FIG_DIR / "02_ate_forest.png")
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, layout="constrained")
    for ax, arm in zip(axes, [MENS, WOMENS]):
        tbl = subgroup_table[arm].iloc[::-1]
        y = np.arange(len(tbl))
        ax.axvline(0, color=INK3, lw=0.8)
        ax.errorbar(tbl["lift_pp"], y,
                    xerr=[tbl["lift_pp"] - tbl["ci_low"], tbl["ci_high"] - tbl["lift_pp"]],
                    fmt="o", color=BLUE, ecolor=INK2, elinewidth=1.1, capsize=2.5,
                    ms=5.5, mec=PAPER, mew=1.2)
        # Direct-label only the purchase-history rows: they are what the text is about.
        for yi, (mod, lift, hi) in enumerate(zip(tbl["moderator"], tbl["lift_pp"], tbl["ci_high"])):
            if mod.startswith(("mens", "womens")):
                ax.annotate(f"{lift:+.1f}", (hi, yi), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([_pretty_segment(m, l) for m, l in zip(tbl["moderator"], tbl["level"])],
                           fontsize=8.5)
        _hide_left_spine(ax)
        ax.set_title(ARM_LABEL[arm])
        ax.set_xlabel("Lift in visit rate vs. no email (pp, 95% CI)")
    fig.suptitle("Treatment effect by customer segment", x=0.0, ha="left", fontsize=11)
    fig.savefig(FIG_DIR / "03_hte_forest.png")
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

    The coefficient is in incremental visits, so it scales with the size of the
    set it is computed on: 60 on 15,000 rows and 60 on 30,000 rows are not the
    same quality of ranking. Anywhere two sets of different size are compared
    below, the per-1,000 figure is what is comparable.
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


def _fit_learners(train_df: pd.DataFrame, score_df: pd.DataFrame):
    """Fit a T-learner and an S-learner on `train_df`, score `score_df` with both.

    Both get the same features and the same estimator, so the only thing being
    compared is how treatment enters: two separate outcome models, or one model
    with treatment as a feature.
    """
    enc = _make_encoder(train_df)
    Xtr = enc.transform(train_df[COVARIATES])
    Xsc = enc.transform(score_df[COVARIATES])
    ytr = train_df[PRIMARY_OUTCOME].values
    ttr = train_df["T"].values

    # --- T-learner (two-model) ---
    m_t = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m_c = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr[ttr == 0], ytr[ttr == 0])
    score_t = m_t.predict_proba(Xsc)[:, 1] - m_c.predict_proba(Xsc)[:, 1]

    # --- S-learner (single model with treatment as a feature) ---
    s_model = HistGradientBoostingClassifier(random_state=SEED).fit(
        np.column_stack([Xtr, ttr]), ytr
    )
    X1 = np.column_stack([Xsc, np.ones(len(Xsc))])
    X0 = np.column_stack([Xsc, np.zeros(len(Xsc))])
    score_s = s_model.predict_proba(X1)[:, 1] - s_model.predict_proba(X0)[:, 1]
    return score_t, score_s


def _select_learner(train: pd.DataFrame) -> dict:
    """Choose T- vs S-learner by cross-fitted Qini, using training rows only.

    This function is the whole point of the three-set discipline. Picking the
    learner by Qini on the same held-out split that then reports the winner's
    Qini - which is what this script used to do - biases the reported number
    upward: the split that chose the maximum of two candidates is also the
    split asked how good the maximum is. Here every training row is scored by
    models that never saw it, the out-of-fold scores are pooled, and the Qini
    of that pooled ranking decides. The reporting set is not consulted.

    Cross-fitting rather than a third slice because it spends no data: all
    27,000-odd training rows inform the choice, and the fold models are trained
    on 80% of the training portion, close to the 100% the winner is refit on.
    """
    y = train[PRIMARY_OUTCOME].values
    t = train["T"].values
    oof_t = np.empty(len(train))
    oof_s = np.empty(len(train))
    # Stratify on the (arm, outcome) pair, not on the arm alone: at a 14.7%
    # base rate, arm-only folds end up with visibly different positive counts.
    folds = StratifiedKFold(n_splits=N_SELECT_FOLDS, shuffle=True, random_state=SEED)
    for fit_idx, held_idx in folds.split(train, t * 2 + y):
        oof_t[held_idx], oof_s[held_idx] = _fit_learners(
            train.iloc[fit_idx], train.iloc[held_idx]
        )
    coef_t = qini_curve(y, t, oof_t)[3]
    coef_s = qini_curve(y, t, oof_s)[3]
    return {
        "qini_t": coef_t,
        "qini_s": coef_s,
        "selected": "T-learner" if coef_t >= coef_s else "S-learner",
        "n": len(train),
    }


def layer6_uplift(df: pd.DataFrame, arm: str) -> dict:
    banner(f"LAYER 6  -  Uplift modelling  ({arm} vs {CONTROL})")
    sub = df[df["segment"].isin([arm, CONTROL])].copy()
    sub["T"] = (sub["segment"] == arm).astype(int)

    train, report = train_test_split(
        sub, test_size=REPORT_FRAC, random_state=SEED, stratify=sub["T"]
    )

    # 1. Choose the learner without touching the reporting set.
    sel = _select_learner(train)

    # 2. Refit both on the full training portion and score the reporting set
    #    once. Both are scored so the optimism of same-set selection can be
    #    quantified rather than asserted.
    score_t, score_s = _fit_learners(train, report)
    yte = report[PRIMARY_OUTCOME].values
    tte = report["T"].values

    frac_t, qini_t, rand_t, coef_t = qini_curve(yte, tte, score_t)
    _, qini_s, _, coef_s = qini_curve(yte, tte, score_s)
    u30_t = uplift_at_k(yte, tte, score_t, 0.30)
    u30_s = uplift_at_k(yte, tte, score_s, 0.30)

    chosen = sel["selected"]
    score_best = score_t if chosen == "T-learner" else score_s
    reported = coef_t if chosen == "T-learner" else coef_s
    naive = max(coef_t, coef_s)       # what same-set selection would have quoted
    optimism = naive - reported

    per_k = lambda coef, n: coef / n * 1000
    overall = yte[tte == 1].mean() - yte[tte == 0].mean()
    print(f"  reporting set       : {len(report):,} rows "
          f"({tte.sum():,} treated / {(1 - tte).sum():,} control)")
    print(f"  overall test uplift : {overall * 100:+.3f}pp")
    print(f"  selection ({N_SELECT_FOLDS}-fold out-of-fold, {sel['n']:,} training rows):")
    print(f"    T-learner Qini    : {sel['qini_t']:8.2f}  "
          f"({per_k(sel['qini_t'], sel['n']):.2f} per 1,000)")
    print(f"    S-learner Qini    : {sel['qini_s']:8.2f}  "
          f"({per_k(sel['qini_s'], sel['n']):.2f} per 1,000)")
    print(f"    -> selected       : {chosen}")
    print(f"  reporting (untouched {len(report):,} rows):")
    print(f"    T-learner Qini    : {coef_t:8.2f}  "
          f"({per_k(coef_t, len(report)):.2f} per 1,000)   "
          f"uplift@30% = {u30_t * 100:+.3f}pp")
    print(f"    S-learner Qini    : {coef_s:8.2f}  "
          f"({per_k(coef_s, len(report)):.2f} per 1,000)   "
          f"uplift@30% = {u30_s * 100:+.3f}pp")
    print(f"  REPORTED Qini ({chosen}, selected elsewhere) : {reported:.2f}")
    print(f"  same-set selection would have quoted        : {naive:.2f}  "
          f"(winner's curse {optimism:+.2f})")

    REPORT.setdefault("uplift", {})[arm] = {
        "qini_sel_t": sel["qini_t"], "qini_sel_s": sel["qini_s"],
        "n_select": sel["n"],
        "qini_t": coef_t, "qini_s": coef_s, "n_report": len(report),
        "u30_t": u30_t, "u30_s": u30_s,
        "overall_test_uplift": overall,
        "selected": chosen, "qini_reported": reported,
        "qini_naive": naive, "optimism": optimism,
    }
    _plot_qini(arm, frac_t, qini_t, rand_t, qini_s, chosen)
    return {
        "report": report, "score_t": score_t, "score_s": score_s,
        "score_best": score_best, "selected": chosen,
        "yte": yte, "tte": tte, "qini_reported": reported,
    }


def _plot_qini(arm, frac, qini_t, rand, qini_s, chosen) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(axis="y")
    ax.plot(frac, rand, color=INK3, lw=0.9, ls=DASH, label="Random targeting")
    # The selected learner is named in the legend so the chart cannot be read as
    # "whichever curve is higher here is the one we shipped" - it was chosen on
    # other data, and on this set it need not be the higher curve.
    mark = lambda name: f"{name} (selected)" if name == chosen else name
    ax.plot(frac, qini_t, color=BLUE, label=mark("T-learner"))
    ax.plot(frac, qini_s, color=RUST, label=mark("S-learner"))
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of customers targeted, highest predicted uplift first")
    ax.set_ylabel("Cumulative incremental visits")
    ax.set_title(f"Qini curves, {ARM_LABEL[arm]}")
    handles, labels = ax.get_legend_handles_labels()   # series first, reference last
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="upper left")
    safe = arm.split()[0].lower()
    fig.savefig(FIG_DIR / f"04_qini_{safe}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Layer 6b - Calibration of predicted uplift                                   #
# --------------------------------------------------------------------------- #
def layer6b_calibration(arm: str, art: dict) -> dict:
    """Do the predicted uplifts mean anything, or are they just a ranking?

    A Qini curve answers only the ranking question: it is invariant to any
    monotone transform of the score, so a model that ranks perfectly and
    predicts every uplift as 0.3pp scores exactly as well as one that predicts
    the truth. Layer 7 does not spend a ranking, though - it compares each
    predicted uplift to a break-even in percentage points and contacts the
    customer if it clears. That decision is only as good as the magnitudes, so
    they get checked: bucket the reporting set into deciles of predicted
    uplift, and put predicted next to observed in each.
    """
    banner(f"LAYER 6b -  Calibration of predicted uplift  ({arm} vs {CONTROL})")
    yte, tte, score = art["yte"], art["tte"], art["score_best"]

    # Bin on ranks rather than on the score itself: gradient-boosted scores tie
    # on identical covariate patterns, and ties would otherwise collapse the
    # deciles into unequal buckets.
    bins = pd.qcut(stats.rankdata(score, method="ordinal"), N_CALIB_BINS, labels=False)
    rows = []
    for b in range(N_CALIB_BINS):
        m = bins == b
        y1, y0 = yte[m & (tte == 1)], yte[m & (tte == 0)]
        if len(y1) < 2 or len(y0) < 2:
            continue
        p1, p0 = y1.mean(), y0.mean()
        rows.append({
            "decile": b + 1,
            "n": int(m.sum()),
            "predicted": float(score[m].mean()),
            "observed": float(p1 - p0),
            "se": float(np.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0))),
        })
    tab = pd.DataFrame(rows)
    pred, obs, se = tab["predicted"].values, tab["observed"].values, tab["se"].values

    # Weighted by precision: the extreme deciles are the noisiest and should not
    # drive the line. Perfect calibration is slope 1, intercept 0.
    slope, intercept = np.polyfit(pred, obs, 1, w=1.0 / se)
    rho = stats.spearmanr(pred, obs).statistic
    mace = np.mean(np.abs(obs - pred))
    covered = np.mean(np.abs(obs - pred) <= 1.96 * se)

    print(f"  decile   n      predicted   observed        95% CI")
    for r in tab.itertuples():
        lo, hi = (r.observed - 1.96 * r.se) * 100, (r.observed + 1.96 * r.se) * 100
        print(f"    {r.decile:2d}   {r.n:5,}    {r.predicted * 100:+7.2f}pp  "
              f"{r.observed * 100:+7.2f}pp   [{lo:+6.2f}, {hi:+6.2f}]")
    print(f"  calibration slope   : {slope:.2f}  (1.00 = predictions are on scale; "
          f"< 1 = over-confident spread)")
    print(f"  intercept           : {intercept * 100:+.2f}pp")
    print(f"  Spearman rho        : {rho:+.2f}  (ranking across deciles)")
    print(f"  mean |pred - obs|   : {mace * 100:.2f}pp")
    print(f"  deciles whose 95% CI covers their prediction: {covered:.0%}")

    REPORT.setdefault("calibration", {})[arm] = {
        "slope": float(slope), "intercept": float(intercept),
        "spearman": float(rho), "mace": float(mace), "covered": float(covered),
        "table": tab,
    }
    _plot_calibration(arm, tab, slope, intercept)
    return REPORT["calibration"][arm]


def _plot_calibration(arm, tab, slope, intercept) -> None:
    pred = tab["predicted"].values * 100
    obs = tab["observed"].values * 100
    err = tab["se"].values * 100 * 1.96
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(axis="y")
    lo = min(pred.min(), (obs - err).min(), 0.0)
    hi = max(pred.max(), (obs + err).max())
    pad = 0.08 * (hi - lo)
    span = np.array([lo - pad, hi + pad])
    # Identity first, so the data sits on top of it.
    ax.plot(span, span, color=INK3, lw=0.9, ls=DASH, label="Perfect calibration")
    ax.plot(span, intercept * 100 + slope * span, color=RUST, lw=1.1,
            label=f"Fitted, slope {slope:.2f}")
    ax.errorbar(pred, obs, yerr=err, fmt="o", color=BLUE, ms=5, lw=0.9,
                capsize=2.5, mec=PAPER, mew=0.8, label="Decile of predicted uplift")
    # Where Layer 7 cuts: deciles to its right are the ones the policy contacts,
    # so that is the region whose magnitudes have to be right for the policy to
    # be right, which is the whole reason this chart exists. It goes in the
    # legend rather than on the line - an in-place label here lands on a decile
    # or its interval whichever way it is turned.
    breakeven = COST_PER_EMAIL / VALUE_PER_VISIT * 100
    if span[0] < breakeven < span[1]:
        ax.axvline(breakeven, color=INK3, lw=0.8, ls=DOT,
                   label=f"Policy cuts here ({breakeven:.1f}pp)")
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_xlabel("Mean predicted uplift in the decile (pp)")
    ax.set_ylabel("Observed uplift in the decile (pp)")
    ax.set_title(f"Predicted vs. observed uplift by decile, {ARM_LABEL[arm]}")
    # Data first, references after, ordered by label rather than by index:
    # matplotlib returns lines before error-bar containers whatever order they
    # were drawn in, so index arithmetic here silently reorders itself.
    order = ["Decile of predicted uplift", "Perfect calibration"]
    handles, labels = ax.get_legend_handles_labels()
    rank = {lab: i for i, lab in enumerate(order)}
    pairs = sorted(zip(labels, handles), key=lambda kv: rank.get(kv[0], len(order)))
    # Paper-coloured knockout: the break-even rule is a full-height vertical and
    # would otherwise run straight through the legend's own text.
    ax.legend([h for _, h in pairs], [l for l, _ in pairs], loc="upper left",
              frameon=True, framealpha=1.0, facecolor=PAPER, edgecolor="none")
    safe = arm.split()[0].lower()
    fig.savefig(FIG_DIR / f"06_calibration_{safe}.png")
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
    print(f"  policy ranker       : {art['selected']} (selected out-of-fold)")
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
    _plot_policy_curve(arm, yte, tte, score, p, rule_frac=frac)
    return REPORT["policy"][arm]


def _plot_policy_curve(arm, yte, tte, score, p, rule_frac=None) -> None:
    """Net value per 1,000 as targeting extends from highest to lowest uplift.

    rule_frac, if given, is the share the cost-sensitive rule actually contacts;
    it is drawn as a reference line so the chart shows the gap between the rule
    and the curve's empirical peak instead of leaving it to the caption.
    """
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
    nv = np.array(nv)
    best = int(np.argmax(nv))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(axis="y")
    ax.axhline(0, color=INK3, lw=0.8)
    ax.plot(fracs * 100, nv, color=BLUE)
    # One series, so no legend: the peak is annotated in place. Labels get a
    # paper-coloured knockout so the reference line can never run through text.
    knockout = dict(boxstyle="square,pad=0.18", fc=PAPER, ec="none")
    ax.plot(fracs[best] * 100, nv[best], "o", color=INK, ms=6.5, mec=PAPER, mew=1.4)
    # The peak is the curve's maximum, so the open space is below it on the side
    # the curve rose from. Set the label there and join it to the point with a
    # thin leader rather than crowding the point itself.
    px, py = fracs[best] * 100, nv[best]
    yspan = nv.max() - min(nv.min(), 0.0)
    to_left = px > 35
    ax.annotate(f"peak: {px:.0f}% contacted, ${py:.2f}", (px, py),
                xytext=(px - 22 if to_left else px + 22, py - 0.2 * yspan),
                textcoords="data", ha="right" if to_left else "left", va="center",
                fontsize=8.5, color=INK2, bbox=knockout,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.7, shrinkA=2, shrinkB=5))
    if rule_frac is not None:
        ax.axvline(rule_frac * 100, color=INK3, lw=0.9, ls=DASH)
        ax.text(rule_frac * 100 - 1.2, nv.min() if nv.min() < 0 else 0.6,
                f"break-even rule\ncontacts {rule_frac*100:.0f}%",
                ha="right", va="bottom", fontsize=8, color=INK3, linespacing=1.3,
                bbox=knockout)
    ax.set_xlim(0, 102)
    ax.set_xlabel("Share of customers contacted, highest predicted uplift first (%)")
    ax.set_ylabel("Net value per 1,000 vs. contacting no one ($)")
    ax.set_title(f"Value of targeting, {ARM_LABEL[arm]}")
    safe = arm.split()[0].lower()
    fig.savefig(FIG_DIR / f"05_policy_{safe}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Layer 7b - Sampling uncertainty on the reporting split                       #
# --------------------------------------------------------------------------- #
def _reporting_stats(y, t, score, policy):
    """Qini coefficient and net values for one (resampled) reporting split.

    The propensity is re-read from the sample rather than fixed at the design
    constant, matching what Layer 7 does with the real split: it is the same
    estimator, so the bootstrap centres on the point estimate instead of
    drifting away from it as resamples land with a different treated share.
    """
    p = t.mean()

    def ipw(pol):
        w = np.where(pol == 1, (t == 1) / p, (t == 0) / (1 - p))
        return (w * y).mean()

    v_none = ipw(np.zeros(len(y), int))
    v_all = ipw(np.ones(len(y), int))
    v_pi = ipw(policy)
    frac = policy.mean()

    def net(v, email_frac):
        return ((v - v_none) * VALUE_PER_VISIT - email_frac * COST_PER_EMAIL) * 1000

    return qini_curve(y, t, score)[3], net(v_all, 1.0), net(v_pi, frac), frac


def layer7b_uncertainty(arm: str, art: dict) -> dict:
    """How much of the reporting split's story is sampling noise?

    Layer 6 and Layer 7 report a Qini coefficient and two dollar figures as
    point estimates, which invites reading them as exact. They come from one
    14,900-row draw. This resamples that draw and reports what the numbers
    would have looked like on a different one.

    The scores are held fixed across resamples, so the models are not refit:
    these are intervals on the estimates *given this fitted model*, not on the
    modelling procedure as a whole. Widening them to cover model fitting would
    mean refitting inside every resample and needs the training split too,
    which is a different and much more expensive claim.
    """
    banner(f"LAYER 7b -  Sampling uncertainty  ({arm} vs {CONTROL})")
    yte, tte, score = art["yte"], art["tte"], art["score_best"]
    breakeven = COST_PER_EMAIL / VALUE_PER_VISIT
    policy = (score > breakeven).astype(int)
    n = len(yte)

    # A private generator: RNG is the module-level stream Layer 8's permutation
    # test draws from, and borrowing it here would shift every later draw.
    rng = np.random.default_rng(SEED)
    boot = np.empty((N_BOOT_REPORT, 4))
    for b in range(N_BOOT_REPORT):
        idx = rng.integers(0, n, n)
        boot[b] = _reporting_stats(yte[idx], tte[idx], score[idx], policy[idx])
    qini_b, nv_blanket_b, nv_targeted_b, frac_b = boot.T
    gain_b = nv_targeted_b - nv_blanket_b

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    obs = _reporting_stats(yte, tte, score, policy)
    gain_obs = obs[2] - obs[1]
    p_better = float((gain_b > 0).mean())

    print(f"  {N_BOOT_REPORT:,} resamples of the {n:,}-row reporting split, "
          f"scores held fixed")
    print(f"  Qini ({art['selected']})   : {obs[0]:7.2f}  "
          f"95% CI [{ci(qini_b)[0]:.2f}, {ci(qini_b)[1]:.2f}]")
    print(f"  net value, blanket  : ${obs[1]:+7.2f}  "
          f"95% CI [${ci(nv_blanket_b)[0]:+.2f}, ${ci(nv_blanket_b)[1]:+.2f}]")
    print(f"  net value, targeted : ${obs[2]:+7.2f}  "
          f"95% CI [${ci(nv_targeted_b)[0]:+.2f}, ${ci(nv_targeted_b)[1]:+.2f}]")
    print(f"  GAIN from targeting : ${gain_obs:+7.2f}  "
          f"95% CI [${ci(gain_b)[0]:+.2f}, ${ci(gain_b)[1]:+.2f}]")
    print(f"  P(targeting beats blanket) = {p_better:.3f}")
    # The ratio is the number people reach for and the one the data supports
    # least: it divides by a quantity whose own interval can straddle zero.
    blanket_lo, blanket_hi = ci(nv_blanket_b)
    if blanket_lo <= 0 <= blanket_hi:
        print("  ratio (targeted/blanket) not reported: the blanket figure's own "
              "interval covers $0, so the ratio is unbounded. Quote the "
              "difference.")

    REPORT.setdefault("uncertainty", {})[arm] = {
        "n_boot": N_BOOT_REPORT, "n_report": n,
        "qini": obs[0], "qini_ci": ci(qini_b),
        "nv_blanket": obs[1], "nv_blanket_ci": ci(nv_blanket_b),
        "nv_targeted": obs[2], "nv_targeted_ci": ci(nv_targeted_b),
        "gain": gain_obs, "gain_ci": ci(gain_b), "p_better": p_better,
        "ratio_unbounded": bool(blanket_lo <= 0 <= blanket_hi),
    }
    _plot_uncertainty(arm, gain_b, gain_obs, ci(gain_b))
    return REPORT["uncertainty"][arm]


def _plot_uncertainty(arm, gain_b, gain_obs, gain_ci) -> None:
    """Where the gain from targeting lands across resamples of this split.

    Every dollar sign here is escaped. Matplotlib reads a *pair* of unescaped
    "$" as mathtext delimiters, so "$2.99 to $30.60" silently renders as italic
    maths with the signs eaten.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.grid(axis="y")
    counts, _, _ = ax.hist(gain_b, bins=46, color=BLUE, alpha=0.85,
                           edgecolor=PAPER, linewidth=0.4)
    # Headroom above the tallest bar so the labels sit on paper, not on ink.
    top = counts.max() * 1.22
    ax.set_ylim(0, top)
    ax.axvline(0, color=INK3, lw=0.9, ls=DASH)
    ax.axvline(gain_obs, color=RUST, lw=1.3)

    # Each label goes on the far side of its own line, away from the other one.
    # When the gain is negative its line sits left of zero, and two labels both
    # reaching inward would land on top of each other. Different heights as
    # well, so the two stay legible even when the lines nearly coincide.
    knockout = dict(boxstyle="square,pad=0.18", fc=PAPER, ec="none")
    obs_left = gain_obs < 0
    ax.annotate(f"observed \\${gain_obs:+.2f}", (gain_obs, top * 0.97),
                xytext=(-5 if obs_left else 5, 0), textcoords="offset points",
                ha="right" if obs_left else "left", va="top",
                fontsize=8.5, color=RUST, bbox=knockout)
    ax.annotate("no gain", (0, top * 0.88),
                xytext=(5 if obs_left else -5, 0), textcoords="offset points",
                ha="left" if obs_left else "right", va="top",
                fontsize=8, color=INK3, bbox=knockout)

    # The interval as a rule inside the axes, under the distribution: the bars
    # already carry the ink, and a shaded band behind them reads as a series.
    # Inside rather than below, so it cannot collide with the axis label.
    ax.plot(gain_ci, [top * 0.045] * 2, color=INK, lw=1.8, solid_capstyle="butt")
    ax.text(np.mean(gain_ci), top * 0.075,
            f"95% CI  \\${gain_ci[0]:+.2f} to \\${gain_ci[1]:+.2f}",
            ha="center", va="bottom", fontsize=8, color=INK2, bbox=knockout)

    ax.set_yticks([])
    _hide_left_spine(ax)
    ax.set_xlabel("Net value of targeting minus blanket, per 1,000 (\\$)")
    ax.set_title(f"Gain from targeting across resamples, {ARM_LABEL[arm]}")
    safe = arm.split()[0].lower()
    fig.savefig(FIG_DIR / f"07_uncertainty_{safe}.png")
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

    def money(x, sign=False):
        """Sign outside the currency: -$4.92, not $-4.92.

        Used everywhere a bootstrap interval can run negative. `sign=True`
        keeps an explicit + on positives, for quantities whose sign is the
        claim being made.
        """
        lead = "-" if x < 0 else ("+" if sign else "")
        return f"{lead}${abs(x):.2f}"

    lines = []
    lines.append("# Results — When Average Effects Lie\n")
    lines.append(
        "_Auto-generated by `hillstrom_ab_analysis.py`. Every number below is "
        "computed from the 64,000-customer Hillstrom randomized experiment._\n"
    )

    def seg_lift(arm, key, level):
        """Subgroup lift in pp, looked up by the moderator's actual level."""
        tbl = pd.DataFrame(r["subgroup_tables"][arm])
        hit = tbl[tbl["moderator"].str.startswith(key) & (tbl["level"] == level)]
        return float(hit["lift_pp"].iloc[0]) if len(hit) else float("nan")

    wq, wp = r["uplift"][WOMENS], r["policy"][WOMENS]
    wu = r["uncertainty"][WOMENS]
    lines.append("## Headline figures")
    lines.append(
        "_Every number the README and the write-up quote, in one place, so "
        "there is a single thing to check them against. Square brackets are "
        "95% bootstrap intervals over the reporting split (section 7)._\n")
    lines.append("| Figure | Value |")
    lines.append("|---|---|")
    lines.append(f"| Randomized trial | **{r['n_total']:,}** customers, three arms |")
    lines.append(f"| Covariate balance, max abs. SMD | **{r['max_smd']:.3f}** |")
    lines.append(f"| Women's-email ATE on visits | **{pp(wom['visit_rd'])}** |")
    lines.append(f"| ...among prior women's-merch buyers | "
                 f"**{seg_lift(WOMENS, 'womens', 1):+.2f}pp** |")
    lines.append(f"| ...among everyone else | "
                 f"**{seg_lift(WOMENS, 'womens', 0):+.2f}pp** |")
    lines.append(f"| Qini, Women's email (reporting set, {wq['selected']}) | "
                 f"**{wq['qini_reported']:.1f}** "
                 f"[{wu['qini_ci'][0]:.1f}, {wu['qini_ci'][1]:.1f}] |")
    # No sign on the two levels: they are the figures quoted verbatim
    # elsewhere, and a leading "+" on a dollar amount reads as a typo rather
    # than as a sign. The gain keeps its sign, because its sign is the claim.
    lines.append(f"| Net value, uplift-targeted | "
                 f"**{money(wp['nv_targeted'])}** per 1,000 "
                 f"[{money(wu['nv_targeted_ci'][0])}, "
                 f"{money(wu['nv_targeted_ci'][1])}] |")
    lines.append(f"| Net value, blanket | **{money(wp['nv_blanket'])}** per 1,000 "
                 f"[{money(wu['nv_blanket_ci'][0])}, "
                 f"{money(wu['nv_blanket_ci'][1])}] |")
    lines.append(f"| **Gain from targeting** | "
                 f"**{money(wu['gain'], sign=True)}** per 1,000 "
                 f"[{money(wu['gain_ci'][0], sign=True)}, "
                 f"{money(wu['gain_ci'][1], sign=True)}] |")
    lines.append(f"| Contacts saved by targeting | "
                 f"**{wp['contacts_saved_per1k']:.0f}** per 1,000 |")
    lines.append("")

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
                 f"**{r['adjustment'][MENS]['var_red']:.0%}** (Men's), exactly "
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
                    f"**{n0:+.2f}pp** (no); gap {abs(y - n0):.2f}pp.")
    lines.append("")

    lines.append("## 4. Choosing the uplift learner without the reporting set")
    lines.append(
        f"_The T-learner / S-learner choice is made by cross-fitted Qini on the "
        f"training portion ({N_SELECT_FOLDS}-fold, every row scored by models "
        f"that never saw it). The reporting set is not consulted. Qini is in "
        f"incremental visits and scales with the size of the set it is computed "
        f"on, so the per-1,000 figure is what compares across the two columns._\n")
    for arm in [MENS, WOMENS]:
        up = r["uplift"][arm]
        lines.append(f"**{arm}**\n")
        lines.append(f"| Learner | Selection set (out-of-fold, n={up['n_select']:,}) "
                     f"| Reporting set (untouched, n={up['n_report']:,}) |")
        lines.append("|---|---|---|")
        for name, ks, kr in [("T-learner", "qini_sel_t", "qini_t"),
                             ("S-learner", "qini_sel_s", "qini_s")]:
            lines.append(
                f"| {name} | {up[ks]:.1f} "
                f"({up[ks] / up['n_select'] * 1000:.2f} per 1,000) | {up[kr]:.1f} "
                f"({up[kr] / up['n_report'] * 1000:.2f} per 1,000) |")
        if up["optimism"] > 0.05:
            tail = (f"Choosing on the reporting set, as this script used to do, "
                    f"would have quoted **{up['qini_naive']:.1f}** instead, "
                    f"**{up['optimism']:+.1f}** of winner's curse over two "
                    f"candidates.")
        else:
            tail = ("Choosing on the reporting set, as this script used to do, "
                    "would have picked the same learner and quoted the same "
                    "figure: the optimism it was exposed to is zero here. That "
                    "is now a result rather than an assumption, which is the "
                    "point. The exposure was real either way.")
        unc = r["uncertainty"][arm]
        lines.append(
            f"\nSelected: **{up['selected']}**. Its Qini on the untouched "
            f"reporting set, the quotable number, is "
            f"**{up['qini_reported']:.1f}** (95% CI "
            f"{unc['qini_ci'][0]:.1f} to {unc['qini_ci'][1]:.1f}). {tail}\n")

    lines.append("## 5. Uplift modelling & cost-sensitive targeting")
    lines.append(
        f"_Illustrative economics: a visit is worth ${VALUE_PER_VISIT:.2f} and a "
        f"contact costs ${COST_PER_EMAIL:.2f} (break-even uplift "
        f"{COST_PER_EMAIL / VALUE_PER_VISIT * 100:.1f}pp). The threshold is that "
        f"break-even and nothing else, tuned on no split. Net value is per "
        f"1,000 customers vs. contacting no one. The qualitative call, broad "
        f"for Men's and selective for Women's, is robust to the exact "
        f"prices._\n")
    for arm in [MENS, WOMENS]:
        up = r["uplift"][arm]
        po = r["policy"][arm]
        unc = r["uncertainty"][arm]
        lo, hi = unc["gain_ci"]
        # The verdict reads the interval, not the point estimate. A gain whose
        # interval covers zero is not a small gain, it is an undetermined one,
        # and "targeting adds nothing" claims more than the data supports.
        if lo > 0:
            verdict = (f"**target the top {po['targeted_frac']:.0%}** "
                       f"(saves {po['contacts_saved_per1k']:.0f} contacts/1,000)")
        elif hi < 0:
            verdict = "**contact broadly**; targeting measurably loses money"
        else:
            verdict = ("**contact broadly**; the gain from targeting is not "
                       "distinguishable from zero")
        lines.append(
            f"- **{arm}** (ranker {up['selected']}, Qini "
            f"{up['qini_reported']:.1f}): net value blanket "
            f"{money(po['nv_blanket'])} vs targeted "
            f"{money(po['nv_targeted'])} / 1,000, a gain of "
            f"**{money(unc['gain'], sign=True)}** "
            f"[{money(lo, sign=True)}, {money(hi, sign=True)}] "
            f"→ {verdict}.")
    lines.append("")

    lines.append("## 6. Calibration of predicted uplift")
    lines.append(
        f"_Qini is invariant to any monotone transform of the score, so it "
        f"certifies the ranking and says nothing about the magnitudes. Section 5 "
        f"spends the magnitudes: a customer is contacted when predicted uplift "
        f"clears {COST_PER_EMAIL / VALUE_PER_VISIT * 100:.1f}pp. Deciles of "
        f"predicted uplift on the reporting set, predicted against observed:_\n")
    for arm in [MENS, WOMENS]:
        cal = r["calibration"][arm]
        safe = arm.split()[0].lower()
        lines.append(
            f"- **{arm}**: calibration slope **{cal['slope']:.2f}** "
            f"(1.00 = predictions on scale), intercept "
            f"{cal['intercept'] * 100:+.2f}pp, Spearman rho "
            f"**{cal['spearman']:+.2f}** across deciles, mean absolute error "
            f"**{cal['mace'] * 100:.2f}pp**, and {cal['covered']:.0%} of deciles "
            f"have a 95% CI covering their own prediction "
            f"(`figures/06_calibration_{safe}.png`).")
    lines.append("")

    lines.append("## 7. How much of this is sampling noise?")
    lines.append(
        f"_The reporting split is one {r['uncertainty'][WOMENS]['n_report']:,}-row "
        f"draw. Resampling it {r['uncertainty'][WOMENS]['n_boot']:,} times, with "
        f"the fitted scores held fixed, says what the figures above would have "
        f"looked like on a different draw. These are intervals on the estimates "
        f"**given this fitted model**; widening them to cover model fitting "
        f"would mean refitting inside every resample, which is a different and "
        f"far more expensive claim._\n")
    lines.append("| Campaign | Qini | Net value, blanket | Net value, targeted "
                 "| Gain | P(gain > 0) |")
    lines.append("|---|---|---|---|---|---|")
    for arm in [MENS, WOMENS]:
        u = r["uncertainty"][arm]
        lines.append(
            f"| {arm} | {u['qini']:.1f} "
            f"[{u['qini_ci'][0]:.1f}, {u['qini_ci'][1]:.1f}] "
            f"| {money(u['nv_blanket'])} "
            f"[{money(u['nv_blanket_ci'][0])}, {money(u['nv_blanket_ci'][1])}] "
            f"| {money(u['nv_targeted'])} "
            f"[{money(u['nv_targeted_ci'][0])}, {money(u['nv_targeted_ci'][1])}] "
            f"| **{money(u['gain'], sign=True)}** "
            f"[{money(u['gain_ci'][0], sign=True)}, "
            f"{money(u['gain_ci'][1], sign=True)}] "
            f"| {u['p_better']:.3f} |")
    lines.append("")
    mu, wu2 = r["uncertainty"][MENS], r["uncertainty"][WOMENS]
    lines.append(
        f"Two things the point estimates hid. The Men's-email Qini of "
        f"{mu['qini']:.1f} has an interval of "
        f"[{mu['qini_ci'][0]:.1f}, {mu['qini_ci'][1]:.1f}], which covers zero. "
        f"That ranker is not merely weak; it is indistinguishable from no "
        f"ranking at all. And its gain from targeting, "
        f"{money(mu['gain'], sign=True)} "
        f"[{money(mu['gain_ci'][0], sign=True)}, "
        f"{money(mu['gain_ci'][1], sign=True)}], covers zero too: the honest reading is not "
        f"\"targeting loses a little\" but \"this split cannot tell\". Both point "
        f"the same way as the calibration slope of "
        f"{r['calibration'][MENS]['slope']:.2f}. Contact broadly.\n")
    if wu2["ratio_unbounded"]:
        lines.append(
            f"For the Women's email the gain is real: "
            f"{money(wu2['gain'], sign=True)} per 1,000, interval "
            f"[{money(wu2['gain_ci'][0], sign=True)}, "
            f"{money(wu2['gain_ci'][1], sign=True)}], "
            f"clear of zero in {wu2['p_better']:.1%} of resamples. The **ratio** "
            f"is not reportable, though, and it is the number a summary reaches "
            f"for first. Targeted over blanket is "
            f"{wu2['nv_targeted'] / wu2['nv_blanket']:.1f}x at the point "
            f"estimate, but the blanket figure's own interval "
            f"[{money(wu2['nv_blanket_ci'][0])}, "
            f"{money(wu2['nv_blanket_ci'][1])}] "
            f"covers $0, which leaves the ratio unbounded. Quote the "
            f"difference, not the multiple.\n")

    lines.append("## 8. Robustness")
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
    _setup_style()
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
        layer6b_calibration(arm, art)
        layer7_policy(arm, art)
        layer7b_uncertainty(arm, art)
    layer8_robustness(df, hte)
    write_results_md()

    banner("DONE")
    print(f"  figures : {FIG_DIR}")
    print(f"  summary : {RESULTS_PATH}")
    print("  Open RESULTS.md for the headline numbers to drop into your README.")


if __name__ == "__main__":
    main()
