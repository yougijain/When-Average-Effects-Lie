# Changelog

Notable changes to the analysis and to the numbers it reports. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project is a
write-up rather than a released package, so entries are dated rather than
versioned.

## 2026-09-21 — Uplift hardening

### Changed
- **Learner selection no longer touches the reporting set.** The T-learner /
  S-learner choice used to be made by Qini on the same held-out 35% that then
  reported the winner's Qini, so the reported figure carried a winner's curse
  over two candidates. Selection now happens on a separate hold-out carved out
  of the training portion; the winner is refit on the full training portion and
  scored once on the untouched reporting set. `RESULTS.md` prints the
  selection-set and reporting-set Qini side by side, plus the figure the old
  procedure would have quoted, so the size of the optimism is visible rather
  than argued about.
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
