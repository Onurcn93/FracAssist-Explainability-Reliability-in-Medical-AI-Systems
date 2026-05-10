"""
gel/gel_concept_visualizer.py — Visualise how RC weights evolve with gamma.

Produces a single 2×1 figure:
  Top:    RC curves for the three FracAssist models with endpoint RC values annotated.
  Bottom: Same three models + one imaginary model (F1=0.80 by default).

Usage
-----
    python gel/gel_concept_visualizer.py
    python gel/gel_concept_visualizer.py --gamma-min 1 --gamma-max 15
    python gel/gel_concept_visualizer.py --extra-f1 0.85
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gel.gel_config import GEL_CONFIG  # noqa: E402

PLOTS_DIR = Path("results/plots")


def _rc_curves(f1_scores, base_shift, gammas):
    anchors = np.array(f1_scores) + base_shift
    rc = np.zeros((len(f1_scores), len(gammas)))
    for k, g in enumerate(gammas):
        powered = anchors ** g
        rc[:, k] = powered / powered.sum()
    return rc


def _powered_curves(f1_scores, base_shift, gammas):
    anchors = np.array(f1_scores) + base_shift
    return np.array([anchors[i] ** gammas for i in range(len(f1_scores))])


def _draw_powered(ax, gammas, powered, labels, colors, title, annotate_endpoints=False):
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(gammas, powered[i], color=color, linewidth=2, label=label)
        if annotate_endpoints:
            val = powered[i, -1]
            ax.annotate(
                f"{val:.2f}",
                xy=(gammas[-1], val),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=9,
                color=color,
                va="center",
            )
    ax.set_xlabel("Gamma (γ)", fontsize=11)
    ax.set_ylabel("(F1 + base_shift)^γ", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def _draw_rc(ax, gammas, rc, labels, colors, title, annotate_endpoints=False):
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(gammas, rc[i], color=color, linewidth=2, label=label)
        if annotate_endpoints:
            val = rc[i, -1]
            ax.annotate(
                f"{val:.3f}",
                xy=(gammas[-1], val),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=9,
                color=color,
                va="center",
            )
    ax.set_xlabel("Gamma (γ)", fontsize=11)
    ax.set_ylabel("Relative Contribution (RC)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)


def main(gamma_min, gamma_max, extra_f1):
    base_shift = GEL_CONFIG["gel_base_shift"]
    f1_r = GEL_CONFIG["gel_f1_resnet"]
    f1_d = GEL_CONFIG["gel_f1_densenet"]
    f1_e = GEL_CONFIG["gel_f1_efficientnet"]

    gammas   = np.linspace(gamma_min, gamma_max, 400)
    subtitle = (f"RC$_i$ = (F1$_i$ + {base_shift})$^γ$ / Σ(F1$_j$ + {base_shift})$^γ$"
                f"  |  γ ∈ [{gamma_min:.0f}, {gamma_max:.0f}]")

    labels_3 = [
        f"ResNet (F1={f1_r:.3f})",
        f"DenseNet (F1={f1_d:.3f})",
        f"EfficientNet (F1={f1_e:.3f})",
    ]
    colors_3 = ["#4C72B0", "#DD8452", "#8172B2"]

    rc_3 = _rc_curves([f1_r, f1_d, f1_e], base_shift, gammas)
    rc_4 = _rc_curves([f1_r, f1_d, f1_e, extra_f1], base_shift, gammas)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    _draw_rc(
        axes[0], gammas, rc_3, labels_3, colors_3,
        f"GEL RC Weights — FracAssist Models\n{subtitle}",
        annotate_endpoints=True,
    )
    _draw_rc(
        axes[1], gammas, rc_4,
        labels_3 + [f"Imaginary Model (F1={extra_f1:.2f})"],
        colors_3  + ["#C44E52"],
        f"GEL RC Weights — With Imaginary High-Performer (F1={extra_f1:.2f})\n{subtitle}",
        annotate_endpoints=True,
    )

    fig.tight_layout()
    out = PLOTS_DIR / "gel_rc_concept.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out}")

    # Powered (pre-normalisation) figure
    powered_3 = _powered_curves([f1_r, f1_d, f1_e], base_shift, gammas)
    powered_4 = _powered_curves([f1_r, f1_d, f1_e, extra_f1], base_shift, gammas)
    pow_subtitle = (f"(F1$_i$ + {base_shift})$^γ$  — raw, before normalisation"
                    f"  |  γ ∈ [{gamma_min:.0f}, {gamma_max:.0f}]")

    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    _draw_powered(
        axes2[0], gammas, powered_3, labels_3, colors_3,
        f"GEL Pre-Normalisation Weights — FracAssist Models\n{pow_subtitle}",
        annotate_endpoints=True,
    )
    _draw_powered(
        axes2[1], gammas, powered_4,
        labels_3 + [f"Imaginary Model (F1={extra_f1:.2f})"],
        colors_3  + ["#C44E52"],
        f"GEL Pre-Normalisation Weights — With Imaginary High-Performer (F1={extra_f1:.2f})\n{pow_subtitle}",
        annotate_endpoints=True,
    )
    fig2.tight_layout()
    out2 = PLOTS_DIR / "gel_rc_concept_powered.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"  Plot saved → {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-min", type=float, default=1.0)
    parser.add_argument("--gamma-max", type=float, default=15.0)
    parser.add_argument("--extra-f1",  type=float, default=0.80,
                        help="F1 score for the imaginary 4th model (default: 0.80)")
    args = parser.parse_args()
    main(args.gamma_min, args.gamma_max, args.extra_f1)
