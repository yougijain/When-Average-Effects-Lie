# Changelog

Notable changes to the analysis and to the numbers it reports. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project is a
write-up rather than a released package, so entries are dated rather than
versioned.

## 2026-10-07 — Third uplift candidate

### Added
- **A DR-learner joins the T- and S-learners.** It builds the augmented
  inverse-propensity pseudo-outcome
  `mu1 - mu0 + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e)` and regresses it on the
  covariates. The propensity is known exactly in a randomized trial, so the
  doubly-robust half is free; what the outcome models buy is variance, not
  correctness, and they are cross-fitted within the training portion (5 folds,
  separate from the 5 that choose between learners) so they cannot shrink
  their own residuals.

### Changed
- **The winner's-curse demonstration now measures a non-zero curse.** With two
  candidates, same-set selection happened to pick the same learner as
  cross-fitting on both arms, so the measured optimism was zero twice over: the
  right procedure with nothing to point at. The DR-learner ranks worst of the
  three out-of-fold and best of the three on the reporting split, which is the
  pattern the three-set discipline exists to catch. On the Men's arm, choosing
  on the reporting set would now quote Qini **19.3** where the honest figure is
  **2.6**, a gap of **+16.7**. The Women's arm still measures zero, so the
  claim the script can make is that the exposure is real and arm-dependent, not
  that it always bites.
- Selection is unchanged in method, and so are the selected learners
  (T-learner on Men's, S-learner on Women's), the reported Qini figures, the
  ATE estimates and every policy and dollar figure. The third candidate adds a
  row to the selection tables and a line to the Qini figures; it does not move
  a headline number.

## 2026-09-21 — Uplift hardening

### Changed
- **Learner selection no longer touches the reporting set.** The T-learner /
  S-learner choice used to be made by Qini on the same held-out 35% that then
  reported the winner's Qini, so the reported figure carried a winner's curse
  over two candidates. Selection now happens by 5-fold cross-fitted Qini on the
  training rows, every row scored by fold models that never saw it; the winner
  is then refit on the full training portion and scored once on the untouched
  reporting set. Cross-fitting rather than a third slice because it spends no
  data: all 27,000-odd training rows inform the choice, and each fold model
  sees 80% of the training portion, close to the 100% the winner is refit on.
  `RESULTS.md` prints the selection-set and reporting-set Qini side by side,
  plus the figure the old procedure would have quoted, so the size of the
  optimism is visible rather than argued about.
- The targeting threshold is unchanged and still fixed at the economic
  break-even (cost / value), not tuned on any split.

### Added
- **Calibration of predicted uplift.** The reporting set is bucketed into
  deciles of predicted uplift and predicted lift is plotted against observed
  lift with Wald intervals (`figures/06_calibration_*.png`). A Qini curve only
  establishes that the *ranking* is useful; this is the check that the
  *magnitudes* mean something, which is what the policy arithmetic in Layer 7
  actually spends.
- A headline-figures block at the top of `RESULTS.md` collecting the nine
  numbers the write-up quotes, so there is one place to check them against.
- A gzipped mirror as a third download fallback, and validation of row count
  and arm sizes before a downloaded file is cached.

## 2026-09-18 — Reproducibility correction

### Fixed
- The uplift and policy figures in the committed `RESULTS.md` were adopted from
  an earlier run that no longer reproduces. That run recorded Qini 6.5 / 61.8,
  a targeted Women's value of \$36.17, 409 contacts saved and a targeted Men's
  value of \$75.63. Neither the pinned environment nor an unpinned one
  reproduces those numbers: both give Qini 2.6 / 60.4, \$33.85 targeted vs
  \$16.90 blanket, and 415 contacts saved, identically: on Python 3.13 under
  the exact pins, on Python 3.11 under a different numpy / scikit-learn, and at
  1, 2, 4 and 8 threads. Layers 1–5 reproduced to the digit throughout; the
  drift was confined to Layer 6's gradient-boosted models.
- `RESULTS.md`, the figures, the README and the write-up now carry the
  reproducible set. The qualitative result, broad for Men's and selective for
  Women's, was unchanged by the correction.

### Changed
- `requirements.txt` pinned to the exact versions the committed outputs were
  generated with, and the Python 3.12 floor stated (numpy 2.5 and scipy 1.18
  ship no wheels for 3.11).
