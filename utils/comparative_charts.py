#!/usr/bin/env python3
"""
Comparative bar charts for FracAssist experiment results.
Generates three summary charts saved to results/plots/:
  1. caalmix_architecture_selectivity.png  — augmentation effect across architectures
  2. all_models_val_comparison.png         — val F1 and val AUC for all models + GEL
  3. depth_scaling_caalmix_matrix.png      — depth x augmentation 3x2 ablation matrix
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Colour palette ────────────────────────────────────────────
BG    = "#0d1b2a"
PANEL = "#112233"
NAVY  = "#1a3a5c"
TEAL  = "#00b4d8"
WHITE = "#f0f0f0"
GREEN = "#57cc99"
RED   = "#e63946"
GRAY  = "#8ecae6"
AMBER = "#f4a261"

BAR_W = 0.35
OUTPUT = "results/plots"
os.makedirs(OUTPUT, exist_ok=True)


def dark(fig, axes):
    fig.patch.set_facecolor(BG)
    for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=WHITE, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#2a4a6a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)


# ════════════════════════════════════════════════════════════════
# Chart 1 — CAALMIX Architecture-Selectivity
# ════════════════════════════════════════════════════════════════
archs     = ["ResNet-18", "DenseNet-169", "EfficientNet-B3"]
arch_sub  = ["E4a → E6\n(+CLAHE+Albu)", "D1 → D3\n(+CLAHE only)", "F1 → F2\n(+CLAHE only)"]
base_f1   = np.array([0.6581, 0.7239, 0.6707])
aug_f1    = np.array([0.6892, 0.6622, 0.6310])
deltas    = aug_f1 - base_f1
d_strs    = ["+3.11pp", "−6.17pp", "−3.97pp"]
aug_cols  = [GREEN, RED, RED]

x = np.arange(len(archs))

fig, ax = plt.subplots(figsize=(10, 6))
dark(fig, ax)

b1 = ax.bar(x - BAR_W/2, base_f1, BAR_W, color=NAVY,
            edgecolor="#3a5a7a", linewidth=0.8, label="Baseline (Standard)", zorder=3)
b2 = ax.bar(x + BAR_W/2, aug_f1, BAR_W, color=aug_cols,
            edgecolor="#ffffff22", linewidth=0.8, alpha=0.92, zorder=3)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.018,
            f"{bar.get_height():.3f}", ha="center", va="top", fontsize=9, color=WHITE, alpha=0.85)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.018,
            f"{bar.get_height():.3f}", ha="center", va="top", fontsize=9, color=WHITE, alpha=0.85)
for bar, d, ds in zip(b2, deltas, d_strs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
            ds, ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=GREEN if d > 0 else RED)

ax.set_xticks(x)
ax.set_xticklabels([f"{a}\n{s}" for a, s in zip(archs, arch_sub)], fontsize=10)
ax.set_ylabel("Val F1 Score", fontsize=12, labelpad=10)
ax.set_ylim(0.58, 0.78)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.grid(axis="y", color="#2a4a6a", linewidth=0.6, alpha=0.7, zorder=0)
ax.set_title(
    "CAALMIX Architecture-Selectivity\n"
    "Augmentation helps shallow CNNs; hurts feature-reuse & compound-scaled architectures",
    fontsize=12, fontweight="bold", pad=14, color=WHITE)
ax.legend(handles=[
    mpatches.Patch(color=NAVY,  label="Baseline (Standard)"),
    mpatches.Patch(color=GREEN, label="Augmentation → GAIN  (+3.11pp)"),
    mpatches.Patch(color=RED,   label="Augmentation → LOSS  (−6.17pp / −3.97pp)"),
], loc="upper right", facecolor=PANEL, edgecolor="#2a4a6a", labelcolor=WHITE, fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, "caalmix_architecture_selectivity.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("OK caalmix_architecture_selectivity.png")


# ════════════════════════════════════════════════════════════════
# Chart 2 — All-Models Val Comparison (F1 + AUC side by side)
# ════════════════════════════════════════════════════════════════
# Sorted ascending by val F1 → GEL at top
model_names = [
    "EfficientNet-B3\n(F1, baseline)",
    "ResNet-18\n(E6, CAALMIX)",
    "DenseNet-169\n(D1, baseline)",
    "GEL v3  ★\n(γ=11.4, δ=0.21)",
]
val_f1  = [0.6707, 0.6892, 0.7239, 0.7394]
val_auc = [0.8825, 0.8888, 0.8439, 0.9060]

# DenseNet AUC in amber — notable calibration anomaly
cols_f1  = [GRAY, GRAY, GRAY,  TEAL]
cols_auc = [GRAY, GRAY, AMBER, TEAL]

y = np.arange(len(model_names))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
dark(fig, [ax1, ax2])

# F1 subplot
b = ax1.barh(y, val_f1, color=cols_f1, height=0.5, edgecolor="#ffffff15", linewidth=0.5)
for bar, val, col in zip(b, val_f1, cols_f1):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", ha="left", fontsize=10,
             color=TEAL if col == TEAL else WHITE,
             fontweight="bold" if col == TEAL else "normal")
ax1.set_xlim(0.625, 0.775)
ax1.set_yticks(y)
ax1.set_yticklabels(model_names, fontsize=10)
ax1.set_xlabel("Val F1", fontsize=11)
ax1.set_title("Val F1 Score", fontsize=12, fontweight="bold")
ax1.axvline(max(val_f1), color=TEAL, linestyle="--", alpha=0.2, linewidth=1)
ax1.grid(axis="x", color="#2a4a6a", linewidth=0.5, alpha=0.7)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

# AUC subplot
b = ax2.barh(y, val_auc, color=cols_auc, height=0.5, edgecolor="#ffffff15", linewidth=0.5)
for bar, val, col in zip(b, val_auc, cols_auc):
    ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", ha="left", fontsize=10,
             color=TEAL if col == TEAL else (AMBER if col == AMBER else WHITE),
             fontweight="bold" if col in (TEAL, AMBER) else "normal")
ax2.set_xlim(0.825, 0.925)
ax2.set_yticks(y)
ax2.set_yticklabels(model_names, fontsize=10)
ax2.set_xlabel("Val AUC", fontsize=11)
ax2.set_title("Val AUC", fontsize=12, fontweight="bold")
ax2.axvline(max(val_auc), color=TEAL, linestyle="--", alpha=0.2, linewidth=1)
ax2.grid(axis="x", color="#2a4a6a", linewidth=0.5, alpha=0.7)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))

# Annotate DenseNet AUC anomaly (y=2) — text placed right of bar end in empty space
ax2.annotate("Best F1 but\nworst AUC\n(calibration issue;\nGEL corrects this)",
             xy=(0.8439, 2.0),
             xytext=(0.874, 2.35),
             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2),
             fontsize=8, color=AMBER, ha="left")

fig.suptitle(
    "FracAssist — Val Metrics: All Models  ·  GEL v3 leads on both F1 and AUC",
    fontsize=13, fontweight="bold", color=WHITE, y=1.02)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, "all_models_val_comparison.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("OK all_models_val_comparison.png")


# ════════════════════════════════════════════════════════════════
# Chart 3 — Depth-Scaling 3×2 Matrix
# ════════════════════════════════════════════════════════════════
depth_labels = ["ResNet-18\n(E4a / E6)", "ResNet-34\n(E9 / E11)", "ResNet-50\n(E10 / E12)"]
std_f1_d = np.array([0.6581, 0.6709, 0.6316])
cal_f1_d = np.array([0.6892, 0.6879, 0.6228])
d_deltas = cal_f1_d - std_f1_d
d_strs   = ["+3.11pp", "+1.70pp", "−0.88pp"]
cal_cols = [GREEN, GREEN, RED]

x = np.arange(len(depth_labels))

fig, ax = plt.subplots(figsize=(10, 6))
dark(fig, ax)

b1 = ax.bar(x - BAR_W/2, std_f1_d, BAR_W, color=NAVY,
            edgecolor="#3a5a7a", linewidth=0.8, label="Standard (no augmentation)", zorder=3)
b2 = ax.bar(x + BAR_W/2, cal_f1_d, BAR_W, color=cal_cols,
            edgecolor="#ffffff22", linewidth=0.8, alpha=0.92, zorder=3)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.012,
            f"{bar.get_height():.3f}", ha="center", va="top", fontsize=9, color=WHITE, alpha=0.85)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.012,
            f"{bar.get_height():.3f}", ha="center", va="top", fontsize=9, color=WHITE, alpha=0.85)
for bar, d, ds in zip(b2, d_deltas, d_strs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            ds, ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=GREEN if d > 0 else RED)

# Arrow showing monotonic decline of CAALMIX benefit — kept below legend
ax.annotate("",
            xy=(x[2] + BAR_W/2, cal_f1_d[2] + 0.018),
            xytext=(x[0] + BAR_W/2, cal_f1_d[0] + 0.018),
            arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.8,
                            connectionstyle="arc3,rad=-0.15"))
ax.text(x[1] + BAR_W/2, cal_f1_d[1] + 0.023,
        "benefit declining with depth",
        ha="center", fontsize=9, color=AMBER, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(depth_labels, fontsize=11)
ax.set_ylabel("Val F1 Score", fontsize=12, labelpad=10)
ax.set_ylim(0.59, 0.74)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.grid(axis="y", color="#2a4a6a", linewidth=0.6, alpha=0.7, zorder=0)
ax.set_title(
    "Depth-Scaling × Augmentation — 3×2 Ablation Matrix\n"
    "CAALMIX benefit monotonically declines with depth; inverts at ResNet-50 (capacity ceiling)",
    fontsize=12, fontweight="bold", pad=14, color=WHITE)
ax.legend(handles=[
    mpatches.Patch(color=NAVY,  label="Standard (no augmentation)"),
    mpatches.Patch(color=GREEN, label="CAALMIX → GAIN"),
    mpatches.Patch(color=RED,   label="CAALMIX → LOSS  (capacity ceiling)"),
], loc="upper right", facecolor=PANEL, edgecolor="#2a4a6a", labelcolor=WHITE, fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT, "depth_scaling_caalmix_matrix.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("OK depth_scaling_caalmix_matrix.png")

print("\nDone -- all 3 charts written to results/plots/")
