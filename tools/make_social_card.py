#!/usr/bin/env python
"""Render social-card.png, the 1280x640 Open Graph image for the repo and page.

Built for a feed thumbnail rather than a text column. The analysis figures are
drawn for a 640px column at 8.5pt, and shrink to grey mush in a LinkedIn or
Slack unfurl; this uses two comparisons and type sized to survive being scaled
to a few hundred pixels wide.

The three numbers are read out of RESULTS.md rather than typed in here, so the
card cannot quietly disagree with the analysis the way a pasted figure would.

Run:
    python tools/make_social_card.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "RESULTS.md"
OUT = ROOT / "social-card.png"
FONT_FILE = ROOT / "fonts" / "SourceSerif4-normal.ttf"

# Same paper, ink and accents as index.html and the analysis figures.
PAPER, INK, INK2, INK3 = "#fdfdfb", "#1b1b1a", "#4a4a46", "#6f6f68"
BLUE, RUST = "#1a5e94", "#b4561f"

# Rows of the RESULTS.md headline table, by their label.
WANTED = {
    "ate": "Women's-email ATE on visits",
    "buyers": "...among prior women's-merch buyers",
    "others": "...among everyone else",
}


def read_headline_figures() -> dict[str, float]:
    """Pull the three lift figures out of the RESULTS.md headline block."""
    if not RESULTS.exists():
        sys.exit(f"{RESULTS.name} not found; run hillstrom_ab_analysis.py first.")
    text = RESULTS.read_text(encoding="utf-8")
    out = {}
    for key, label in WANTED.items():
        m = re.search(rf"^\| {re.escape(label)} \| \*\*([+-][\d.]+)pp\*\*",
                      text, re.M)
        if not m:
            sys.exit(f"no headline row for {label!r} in {RESULTS.name}; the "
                     f"table changed and this card would go stale silently.")
        out[key] = float(m.group(1))
    return out


def main() -> None:
    v = read_headline_figures()
    if FONT_FILE.exists():
        font_manager.fontManager.addfont(str(FONT_FILE))
        family = "Source Serif 4"
    else:
        family = "DejaVu Serif"
    plt.rcParams.update({"font.family": family, "text.color": INK})

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    L = 0.058
    ax.text(L, 0.885, "When Average Effects Lie", fontsize=48, color=INK,
            va="center")
    ax.text(L, 0.772, f"The Women's email's {v['ate']:+.1f}-point average hides "
                      f"two very different campaigns.",
            fontsize=20.5, color=INK2, va="center")

    # 8pp of lift fills the track, which leaves the larger bar short of the
    # value label rather than colliding with it.
    SPAN, MAX = 0.585, 8.0
    rows = [
        ("Customers who bought women's merchandise before", v["buyers"], BLUE, 0.485),
        ("Everyone else", v["others"], RUST, 0.215),
    ]
    for label, val, colour, y in rows:
        ax.text(L, y + 0.093, label, fontsize=18.5, color=INK2, va="center")
        w = val / MAX * SPAN
        ax.add_patch(FancyBboxPatch((L, y - 0.048), w, 0.096,
                                    boxstyle="round,pad=0,rounding_size=0.009",
                                    facecolor=colour, edgecolor="none",
                                    transform=ax.transAxes))
        ax.text(L + w + 0.018, y, f"{val:+.2f}pp", fontsize=39, color=colour,
                ha="left", va="center")

    ax.text(L, 0.058, "Uplift modelling and cost-sensitive targeting on a "
                      "64,000-customer randomized trial",
            fontsize=15.5, color=INK3, va="center")

    # bbox_inches must stay None: "tight" would crop to the ink and lose the
    # exact 1280x640 that the Open Graph slot wants.
    fig.savefig(OUT, dpi=100, facecolor=PAPER, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"  wrote {OUT.name}  (figures read from {RESULTS.name})")


if __name__ == "__main__":
    main()
