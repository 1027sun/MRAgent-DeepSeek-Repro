"""Three-arm comparison figure: MRAgent vs Full-context vs Passive on conv-26.

Reads the per-question judge labels written by eval/evaluate_reasoning.py and plots
accuracy with Wilson 95% confidence intervals. Bars are annotated with n so the
13-question open-domain cell is not read as if it carried the weight of the 70-question
single-hop cell.

Run from the repo root:  python plot_three_arms.py
"""
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    # optional: the SciPilot figure skill provides journal style presets.
    # Without it we fall back to a plain serif/sans setup below.
    from setup_style import setup_style  # type: ignore
    setup_style(journal="nature", lang="zh")
except Exception:  # noqa: BLE001
    from matplotlib import font_manager as _fm
    _have = {f.name for f in _fm.fontManager.ttflist}
    _cjk = next((n for n in ("Noto Sans CJK SC", "Source Han Sans SC", "SimHei",
                             "Microsoft YaHei") if n in _have), None)
    if _cjk:
        plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.6, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })

# --------------------------------------------------------------------------- data
ARMS = [
    ("ds26",    "MRAgent",      "#0072B2", ""),
    ("fullctx", "Full-context", "#E69F00", "///"),
    ("passive", "Passive",      "#009E73", "..."),
]
CAT_ORDER = [1, 2, 3, 4]
CAT_NAME = {1: "多跳", 2: "时间", 3: "开放域", 4: "单跳"}


def load(arm_key):
    path = f"results/result_judge_locomo_deepseek_{arm_key}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows


def wilson(k, n, z=1.959964):
    """Wilson score interval — behaves sensibly for the n=13 cell."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


data = {}
for key, _label, _c, _h in ARMS:
    rows = load(key)
    per_cat = {}
    for cat in CAT_ORDER:
        sub = [r for r in rows if r["category"] == cat]
        k = sum(int(r["llm_score"]) for r in sub)
        per_cat[cat] = (k, len(sub))
    k_all = sum(int(r["llm_score"]) for r in rows)
    data[key] = {"cat": per_cat, "all": (k_all, len(rows))}

ns = {cat: data[ARMS[0][0]]["cat"][cat][1] for cat in CAT_ORDER}

# --------------------------------------------------------------------------- plot
fig = plt.figure(figsize=(7.2, 2.9))
fig.set_layout_engine("none")   # explicit gridspec geometry, not constrained_layout
gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.0], wspace=0.30,
                      left=0.075, right=0.985, top=0.80, bottom=0.20)
ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

NB = len(ARMS)
width = 0.24
x = np.arange(len(CAT_ORDER))


OFFSETS = [-width, 0.0, width]
LABELS = []          # (x_data, lo_pct, hi_pct, text_artist) — for the QA check below
# "ci"  = place the value label above the upper confidence bound (correct)
# "bar" = above the bar top (the original bug; kept so the QA check can be validated)
LABEL_MODE = os.environ.get("LABEL_MODE", "ci")


def draw(ax, positions, keys):
    """positions: x centres of the groups. keys: category ids ([] for a single group)."""
    for (key, label, colour, hatch), dx in zip(ARMS, OFFSETS):
        xs, ps, errs, los, his = [], [], [], [], []
        for pos, k in zip(positions, keys):
            kk, nn = data[key]["cat"][k] if isinstance(k, int) else data[key]["all"]
            p, lo, hi = wilson(kk, nn)
            xs.append(pos + dx)
            ps.append(p * 100)
            errs.append([(p - lo) * 100, (hi - p) * 100])
            los.append(lo * 100)
            his.append(hi * 100)
        errs = np.array(errs).T
        ax.bar(xs, ps, width=width * 0.92, color=colour, hatch=hatch,
               edgecolor="black", linewidth=0.4, label=label, zorder=3)
        ax.errorbar(xs, ps, yerr=errs, fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
        # value labels must clear the UPPER CONFIDENCE BOUND, not the bar top —
        # otherwise the text sits inside the error bar's vertical line.
        for xx, pp, lo, hi in zip(xs, ps, los, his):
            y = (pp + 3.2) if LABEL_MODE == "bar" else (hi + 1.8)
            ART = ax.text(xx, y, f"{pp:.1f}", ha="center", va="bottom", fontsize=5.6)
            LABELS.append((xx, lo, hi, ART))


draw(ax, x, CAT_ORDER)

ax.set_xticks(x)
ax.set_xticklabels([f"{CAT_NAME[c]}\n(n={ns[c]})" for c in CAT_ORDER], fontsize=6.6)
ax.set_ylabel("LLM-judge 准确率 (%)")
ax.set_ylim(0, 118)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
ax.grid(axis="y", linewidth=0.35, alpha=0.45, zorder=0)
ax.set_axisbelow(True)
handle, label = ax.get_legend_handles_labels()
fig.legend(handle, label, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3,
           frameon=False, fontsize=6.6, handlelength=1.4, columnspacing=2.0)

x2 = np.array([0.0])
draw(ax2, x2, [None])
ax2.set_xticks([0.0])
ax2.set_xticklabels([f"总体\n(n={data[ARMS[0][0]]['all'][1]})"], fontsize=6.6)
ax2.set_ylim(0, 118)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_yticklabels(["0", "20", "40", "60", "80", "100"])
ax2.grid(axis="y", linewidth=0.35, alpha=0.45, zorder=0)
ax2.set_axisbelow(True)

ax.text(-0.105, 1.045, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
ax2.text(-0.30, 1.045, "b", transform=ax2.transAxes, fontsize=9, fontweight="bold", va="top")

out_dir = "figs"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------- visual self-check (QA)
def _label(art):
    if hasattr(art, "get_text"):
        return art.get_text()
    return "<legend>"


def qa(fig, axes):
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    figbb = fig.bbox
    problems = []

    # value labels vs. the error bar's vertical span (the bug the text-vs-text check missed)
    owner = {id(t): ax for ax in axes for t in ax.texts}
    for xd, lo, hi, art in LABELS:
        ax = owner.get(id(art), axes[0])
        tb = art.get_window_extent(renderer=r)
        (x0, y0) = ax.transData.transform((xd - 0.06, lo))
        (x1, y1) = ax.transData.transform((xd + 0.06, hi))
        strip = Bbox([[min(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)]])
        ov = Bbox.intersection(tb, strip)
        if ov is not None and ov.width > 0.5 and ov.height > 0.5:
            problems.append(
                f"TEXT-OVER-ERRORBAR {art.get_text()!r} overlaps its error bar "
                f"({ov.width:.1f}x{ov.height:.1f}px)")

    arts = []
    if fig.legends:
        arts += list(fig.legends)
    for ax in axes:
        arts += [t for t in ax.texts if t.get_text().strip()]
        arts += [ax.xaxis.label, ax.yaxis.label]
        arts += [t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text().strip()]
        leg = ax.get_legend()
        if leg is not None:
            arts.append(leg)

    for t in arts:
        bb = t.get_window_extent(renderer=r)
        if bb.x0 < -1.5 or bb.y0 < -1.5 or bb.x1 > figbb.x1 + 1.5 or bb.y1 > figbb.y1 + 1.5:
            problems.append(f"OUT-OF-CANVAS {_label(t)[:24]!r} bbox={tuple(round(v,1) for v in bb.extents)}")

    for i in range(len(arts)):
        for j in range(i + 1, len(arts)):
            a, b = arts[i], arts[j]
            ov = Bbox.intersection(a.get_window_extent(r), b.get_window_extent(r))
            if ov is not None and ov.width > 1.5 and ov.height > 1.5:
                problems.append(f"TEXT-OVERLAP {_label(a)[:18]!r} <-> {_label(b)[:18]!r} "
                                f"({ov.width:.0f}x{ov.height:.0f}px)")

    # legend vs. bars
    for ax in axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lbb = leg.get_window_extent(r)
        for p in ax.patches:
            ov = Bbox.intersection(lbb, p.get_window_extent(r))
            if ov is not None and ov.width > 2 and ov.height > 2:
                problems.append(f"LEGEND-OVER-BAR overlap {ov.width:.0f}x{ov.height:.0f}px")
                break
    return problems


import warnings  # noqa: E402

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fig.canvas.draw()
    glyph_warnings = [str(w.message) for w in caught if "missing from font" in str(w.message)]

issues = qa(fig, [ax, ax2])

try:
    from visual_qa import render_preview      # optional, from the SciPilot skill
    render_preview(fig, os.path.join(out_dir, "_preview_three_arms.png"))
except Exception:  # noqa: BLE001
    fig.savefig(os.path.join(out_dir, "_preview_three_arms.png"), dpi=150)

print("=== visual QA ===")
print("missing-glyph warnings :", len(glyph_warnings))
for g in glyph_warnings[:5]:
    print("   ", g[:130])
print("layout issues          :", len(issues))
for it in issues[:12]:
    print("   ", it)
if not issues and not glyph_warnings:
    print("   (no clipping, no text overlap, legend clear of bars)")

basename = os.path.join(out_dir, "three_arms_conv26")
print("figsize before export:", tuple(round(v, 3) for v in fig.get_size_inches()),
      "dpi:", fig.dpi)
# setup_style() sets savefig.bbox='tight', and bbox_inches=None falls back to that
# rcParam — which silently crops the figure below the target size. Passing an explicit
# full-canvas Bbox is the only reliable way to get an exact 7.2 x 2.9 in output.
from matplotlib.transforms import Bbox  # noqa: E402
W_IN, H_IN = 7.2, 2.9
fig.set_size_inches(W_IN, H_IN)
FULL_BBOX = Bbox.from_bounds(0, 0, W_IN, H_IN)
for _fmt in ("pdf", "svg", "png"):
    fig.savefig(f"{basename}.{_fmt}", dpi=300, bbox_inches=FULL_BBOX,
                transparent=False)

from PIL import Image  # noqa: E402
_im = Image.open(basename + ".png").convert("RGB")
print("saved png pixels   :", _im.size,
      "=>", round(_im.size[0] / 300, 3), "x", round(_im.size[1] / 300, 3), "in")
_im.convert("L").save(basename + "_grayscale.png", dpi=(300, 300))

print("=== exported ===")
for ext in ("pdf", "svg", "png"):
    p = f"{basename}.{ext}"
    if os.path.exists(p):
        print(f"  {p}  {os.path.getsize(p)/1024:.0f} KB")

# ----------------------------------------------------------------- numbers report
print("\n%-13s %-7s %-9s %s" % ("arm", "cat", "acc", "95% CI (Wilson)"))
for key, label, _c, _h in ARMS:
    for cat in CAT_ORDER + ["ALL"]:
        kk, nn = (data[key]["cat"][cat] if cat in CAT_ORDER else data[key]["all"])
        p, lo, hi = wilson(kk, nn)
        print("%-13s %-7s %6.1f%%   [%5.1f, %5.1f]  n=%d"
              % (label, CAT_NAME.get(cat, cat), p * 100, lo * 100, hi * 100, nn))
    print()
