# When Average Effects Lie
### Turning a routine email A/B test into a targeting strategy with causal + uplift modelling

**[→ Read the interactive write-up](https://yougijain.github.io/When-Average-Effects-Lie/)**
 · [Generated results](RESULTS.md) · [The analysis script](hillstrom_ab_analysis.py) · [Changelog](CHANGELOG.md)

**The business question:** A retailer runs two promotional emails — one featuring
men's merchandise, one featuring women's. Both "work" on average. So which
customers should actually receive which email next quarter — and who should we
stop emailing entirely?

The naive answer ("both lift sales, send both to everyone") leaves money on the
table. This project shows why, using the
[Hillstrom MineThatData email experiment](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html):
a real randomized trial of **64,000 customers** split across a Men's-email arm, a
Women's-email arm, and a no-email control.

---

## TL;DR — the finding

> **The Women's email's healthy +4.5pp average lift is an illusion of aggregation.**
> It moves website visits **+7.3pp** for customers who have bought women's
> merchandise before, but only **+1.1pp** for those who haven't — and just
> **+2.2pp** for men's-merchandise buyers. The Men's email, by contrast, lifts
> *everyone* by ~7.7pp fairly uniformly.

So the two campaigns need opposite playbooks:

| Campaign | Avg. visit lift | Heterogeneity | Recommended policy |
|---|---|---|---|
| **Men's email** | **+7.66pp** `[+7.00, +8.32]` | low (interactions n.s. after FDR) | **Contact broadly** — targeting adds nothing |
| **Women's email** | **+4.52pp** `[+3.89, +5.16]` | **high** (T×affinity p < 1e-4) | **Target the ~59%** with real affinity — **doubles** net value while sending **415 fewer contacts per 1,000** |

Under illustrative economics ($2 / incremental visit, $0.06 / contact ⇒ 3pp
break-even), the uplift-targeted Women's policy returns **\$33.85 vs \$16.90 per
1,000** for blanket emailing. The qualitative call — *broad for Men's, selective
for Women's* — holds across a wide range of those prices.

That's the story an interviewer remembers: not "the experiment ran clean," but
"the average would have led you to the wrong decision, and here's the policy that
fixes it."

![HTE forest plot](figures/03_hte_forest.png)

---

## How it's built — 8 layers

The single script [`hillstrom_ab_analysis.py`](hillstrom_ab_analysis.py) runs the
whole pipeline and writes [`RESULTS.md`](RESULTS.md) plus nine figures. Every
number is computed from the data — nothing is hard-coded.

| # | Layer | What it does | Why it matters in an interview |
|---|---|---|---|
| 1 | **Load & validate** | shape, missingness, arm sizes, base rates | shows you check data before trusting it |
| 2 | **Randomization checks** | covariate balance (SMD love plot), omnibus assignment test | proves causal claims are *earned*, not assumed (max \|SMD\| = 0.009, omnibus p = 0.76) |
| 3 | **Average treatment effect** | diff-in-proportions + Wald CIs; bootstrap for spend | the "textbook" A/B result everyone expects |
| 4 | **Regression adjustment** | Lin (2013) interacted estimator, HC1 robust SE | variance reduction the right way; estimate stable ⇒ randomization confirmed |
| 5 | **Heterogeneous effects** | subgroup CATEs + treatment×covariate interactions | **the twist** — where the average lies |
| 6 | **Uplift modelling** | T-learner vs S-learner chosen by cross-fitted Qini, Qini curve / coefficient, uplift@k, decile calibration | individual-level treatment effects, not just averages — and a model chosen without spending the reporting split |
| 7 | **Targeting policy value** | IPW policy value, cost-sensitive break-even threshold | converts the model into a decision with a dollar figure |
| 8 | **Robustness & inference** | randomization inference, Benjamini-Hochberg FDR, retrospective power | the rigour that separates a real analysis from a notebook |

### Figures produced
`01_balance_love_plot` · `02_ate_forest` · `03_hte_forest` ·
`04_qini_mens` / `04_qini_womens` · `05_policy_mens` / `05_policy_womens` ·
`06_calibration_mens` / `06_calibration_womens`

---

## Run it

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
python hillstrom_ab_analysis.py
```

The script auto-downloads the dataset (~4 MB) to `data/` on first run and caches
it, falling back to a gzipped mirror where the canonical host is blocked, and
refusing to cache anything that isn't the published experiment — required
columns, 64,000 rows, and the three arm sizes are all checked first. Full run is
well under a minute. Open `RESULTS.md` for the headline numbers
and `figures/` for the charts.

### Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The tests run on synthetic data and never touch the network. They cover the
machinery whose failures wouldn't show up as a crash: the Qini coefficient
(does it reward the true ranking over a null of random ones, is it really
invariant to monotone rescaling — the property that makes the calibration check
necessary — and how does it break ties), the download validator, and learner
selection.

`requirements.txt` is pinned to the exact versions the committed `RESULTS.md`
and figures were produced with, and needs Python 3.12 or newer. Under those pins
the whole pipeline reproduces the committed outputs byte for byte, figures
included. Layers 1–5 also reproduce to the last digit under any compatible
stack; Layer 6's gradient-boosted models are the one place numbers have been
seen to move between environments, which is what the pins are for. An earlier
run whose uplift numbers no longer reproduce is written up in
[`CHANGELOG.md`](CHANGELOG.md).

### The write-up site
`index.html` is a single self-contained page: no build step, no JavaScript, no
CDN calls. Its only assets are the committed figures in `figures/` and two
self-hosted font files in `fonts/` (Source Serif 4, SIL Open Font License).
Open it locally by double-clicking it, or read the
[hosted version](https://yougijain.github.io/When-Average-Effects-Lie/).

---

## A few methodology choices worth defending

- **Primary outcome = `visit`.** Highest base rate (~14.7%), so the most
  statistical power. `conversion` (~0.9%) and `spend` are reported too, but the
  decision rests on the well-powered metric.
- **T-learner *and* S-learner, chosen off the reporting split.** Both are
  reported, and the choice between them is made by 5-fold cross-fitted Qini on
  the training portion — every row scored by models that never saw it — so the
  reporting split is never asked both to pick the maximum of two candidates and
  to say how good that maximum is. (Men's: T-learner, Qini 2.6 — little to
  rank. Women's: S-learner, Qini 60.4 — lots to rank.) The huge gap in Qini
  *is* the heterogeneity story in one number.
- **Calibration, not just ranking.** Qini is invariant to any monotone
  transform of the score, so it certifies the ranking and says nothing about
  the magnitudes — and the targeting rule spends the magnitudes, comparing each
  predicted uplift to a 3pp break-even. Deciles of predicted uplift are plotted
  against observed: Women's comes in at slope 0.74 (on scale, somewhat
  over-spread), Men's at 0.07 (no magnitude signal at all, which is the flat
  Qini curve restated in units).
- **IPW policy value.** Because treatment was randomized, the propensity is a
  known constant, so the inverse-propensity policy-value estimator is unbiased —
  no outcome model required to score the policy. The cut is set at the
  cost/value break-even, the economically correct threshold (not just uplift > 0).
- **Multiple testing.** Every subgroup interaction is reported with a
  Benjamini-Hochberg FDR correction; only the two Women's-email interactions
  survive — so the headline isn't a fishing-expedition artifact.
- **Randomization inference.** A 2,000-permutation exact test backstops the
  parametric p-values for the primary effect (permutation p = 0.0000).

## Honest limitations
- The dollar figures in Layer 7 are **illustrative assumptions**, not Hillstrom's
  real margins — they exist to demonstrate cost-sensitive targeting. The
  *qualitative* recommendation is robust to the exact prices.
- Uplift models are fit on 65% of each two-arm subset and scored on the held-out
  35%. Neither the learner choice nor the threshold is tuned on that 35% — the
  first is cross-fitted on the training portion, the second is fixed at the
  economic break-even. What remains is that one split supplies the Qini, the
  calibration and the dollar figures; three reads on one sample move together,
  and separating them needs another sample rather than another estimator.
- The decile the targeting rule cuts through is also the worst-calibrated one:
  mean predicted uplift 3.7pp against an observed −1.6pp [−4.8, +1.5]. Roughly
  1,500 customers in the held-out split are contacted who probably shouldn't
  be. That is the standing cost of a fixed break-even threshold — it lands
  where the predictions are noisiest — and it is preferable to moving the cut
  to wherever this split happens to peak.
- Layer 7's net-value figures are computed on that 35% held-out split, where the
  realized lift runs a little under the full-sample ATE (~3.8pp vs +4.52pp for
  Women's, ~7.0pp vs +7.66pp for Men's). That is sampling noise, not a
  contradiction — but it is why blanket net value (\$16.90 / 1,000 for Women's)
  doesn't reproduce if you multiply the headline ATE by \$2.
- "Affinity" here is proxied by past-purchase flags already in the data. The
  models already use every column Hillstrom records — recency, spend history
  and its segment, channel, zip type, tenure — so there is no unused feature
  left to add; purchase *frequency* simply isn't in the file. Sharpening the
  policy means features from outside this dataset, not better use of it.

---

## Why a psychology background is an asset here
Experimental design *is* the psychology research toolkit: randomization and
covariate balance, moderation/interaction effects (the women's-email × affinity
finding is textbook moderation), novelty/behavioural heterogeneity, multiple-
comparison discipline, and pre-registration logic. The hard part of causal data
science isn't the estimator — it's the experimental reasoning about what the
number *means*, which is exactly what an experimental-psych training drills.

---

*Dataset: Kevin Hillstrom, MineThatData E-Mail Analytics and Data Mining
Challenge (2008). Public domain. Code released under the [MIT License](LICENSE).*
