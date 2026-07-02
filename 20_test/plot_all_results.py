"""
plot_all_results.py
===================
Generates ALL figures using the csv results .

Run AFTER scaled_exp.py has completed.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
COLORS  = ["#028090", "#ff7f0e", "#d62728", "#2ca02c"]   # D-DQN, Std, Greedy, RR
AGENTS  = ["Dueling DQN", "Standard DQN", "Greedy", "Round-Robin"]
STYLES  = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]

def save(fig, name):
    fig.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")

# ── Load N=20 results ─────────────────────────────────────────────────────────
print("Loading N=20 results...")
dfs_20 = {
    "Dueling DQN":  pd.read_csv("results_Dueling_DQN.csv"),
    "Standard DQN": pd.read_csv("results_Standard_DQN.csv"),
    "Greedy":       pd.read_csv("results_Greedy.csv"),
    "Round-Robin":  pd.read_csv("results_Round-Robin.csv"),
}

# ── Load convergence ──────────────────────────────────────────────────────────
print("Loading convergence data...")
cd = pd.read_csv("convergence_data.csv",        index_col="episode")
cs = pd.read_csv("convergence_data_stddqn.csv", index_col="episode")

# ── Load scalability summary ──────────────────────────────────────────────────
print("Loading scalability summary...")
if os.path.exists("scalability_summary.csv"):
    scale_df = pd.read_csv("scalability_summary.csv")
    has_scale = True
    print("  Scalability data found.")
else:
    has_scale = False
    print("  WARNING: scalability_summary.csv not found.")
    print("  Run scalability_experiment.py first.")
    print("  Figures 6 and 7 will be skipped.")

print()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Architecture 
# ══════════════════════════════════════════════════════════════════════════════
#with open("fig1_architecture_note.txt", "w") as f:
#    f.write(
       
#    )
#print("  Saved fig1_architecture_note.txt")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — CONVERGENCE ( smoothed, separated, adjusted scale)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Convergence ...")

def smooth(series, w=25):
    """Rolling mean to reveal the underlying trend."""
    return series.rolling(window=w, min_periods=1, center=True).mean()

# Skip episode 0 (initialisation spike)
cd_plot = cd.iloc[1:].copy()
cs_plot = cs.iloc[1:].copy()

# Smoothed means
cd_smooth = smooth(cd_plot["mean"])
cs_smooth = smooth(cs_plot["mean"])

fig, ax = plt.subplots(figsize=(8, 4.5))

# Dueling DQN
ax.plot(cd_plot.index, cd_smooth,
        color=COLORS[0], linewidth=2.2, label="Dueling DQN")
ax.fill_between(cd_plot.index,
                cd_smooth - cd_plot["std"] * 0.4,
                cd_smooth + cd_plot["std"] * 0.4,
                alpha=0.18, color=COLORS[0])

# Standard DQN
ax.plot(cs_plot.index, cs_smooth,
        color=COLORS[1], linewidth=2.2, linestyle="--", label="Standard DQN")
ax.fill_between(cs_plot.index,
                cs_smooth - cs_plot["std"] * 0.4,
                cs_smooth + cs_plot["std"] * 0.4,
                alpha=0.18, color=COLORS[1])

# Annotate converged values with horizontal reference lines
d_conv = float(cd_plot["mean"].iloc[-100:].mean())
s_conv = float(cs_plot["mean"].iloc[-100:].mean())
ax.axhline(d_conv, color=COLORS[0], linestyle=":", linewidth=1.0, alpha=0.6)
ax.axhline(s_conv, color=COLORS[1], linestyle=":", linewidth=1.0, alpha=0.6)
ax.text(490, d_conv + 8, f"{d_conv:.1f}", color=COLORS[0],
        fontsize=9, ha="right", va="bottom")
ax.text(490, s_conv - 15, f"{s_conv:.1f}", color=COLORS[1],
        fontsize=9, ha="right", va="top")

# Convergence annotation arrow at ~ep 20
ax.annotate("Rapid convergence\n(within ~20 episodes)",
            xy=(20, -100), xytext=(80, -280),
            fontsize=9, color="dimgray",
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.0))

# ──  adjust y-axis to separate curves clearly ──────────────────────
y_min = min(float(cd_plot["mean"].min()), float(cs_plot["mean"].min()))
y_max = 50
ax.set_ylim(y_min * 0.15, y_max)   # widen bottom, cap top above 0
ax.set_xlim(1, 500)

ax.set_xlabel("Training Episode", fontsize=12)
ax.set_ylabel("Cumulative Reward (smoothed, window=25)", fontsize=11)
ax.set_title("Figure 2: Training Convergence — Dueling DQN vs Standard DQN", fontsize=12, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
fig.tight_layout()
save(fig, "fig2_convergence")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ENERGY BAR CHART (N=20)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Energy consumption...")
means = [dfs_20[a]["energy"].mean() for a in AGENTS]
stds  = [dfs_20[a]["energy"].std()  for a in AGENTS]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(AGENTS, means, yerr=stds, color=COLORS,
              capsize=5, width=0.55, alpha=0.88,
              error_kw={"linewidth": 1.4, "ecolor": "black"})
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.06,
            f"{m:.3f} J", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold")
ax.set_ylabel("Average Energy per Task (Joules)", fontsize=12)
ax.set_title("Figure 3: Average Energy Consumption per Task (N = 20 devices)", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(means) + max(stds) * 1.8)
fig.tight_layout()
save(fig, "fig3_energy")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — LATENCY CDF (N=20)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Completion time CDF...")
fig, ax = plt.subplots(figsize=(8, 4.5))
for agent, color, style in zip(AGENTS, COLORS, STYLES):
    data = np.sort(dfs_20[agent]["time_ms"].values)
    cdf  = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, cdf, label=agent, color=color, linestyle=style, linewidth=1.8)

ax.axvline(50,  color="dimgray", linestyle="--", linewidth=1.1)
ax.axvline(100, color="black",   linestyle=":",  linewidth=1.1)
ax.text(53,  0.05, "Tier-1\n(50 ms)",  fontsize=9, color="dimgray")
ax.text(103, 0.05, "Tier-2\n(100 ms)", fontsize=9, color="black")
ax.set_xlim(0, 400)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Task Completion Time (ms)", fontsize=12)
ax.set_ylabel("Cumulative Probability", fontsize=12)
ax.set_title("Figure 4: CDF of Task Completion Times (N = 20 devices)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
fig.tight_layout()
save(fig, "fig4_latency_cdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — DEADLINE VIOLATIONS BY TIER (N=20)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Deadline violations by tier...")
tier_labels = ["Tier 1\n(Critical, 50 ms)",
               "Tier 2\n(Urgent, 100 ms)",
               "Tier 3\n(Routine, 5 s)"]
x     = np.arange(3)
width = 0.2

fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (agent, color) in enumerate(zip(AGENTS, COLORS)):
    df   = dfs_20[agent]
    viol = [df[df.priority == p]["violated"].mean() * 100 for p in [1, 2, 3]]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, viol, width, label=agent, color=color, alpha=0.88)
    for bar, v in zip(bars, viol):
        if v > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(tier_labels, fontsize=10)
ax.set_ylabel("Deadline Violation Rate (%)", fontsize=12)
ax.set_title("Figure 5: Deadline Violations by Priority Tier (N = 20 devices)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "fig5_tier_violations")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 6 & 7 — SCALABILITY (N=20, 40, 60)
# ══════════════════════════════════════════════════════════════════════════════
if has_scale:
    N_vals = sorted(scale_df["N_devices"].unique())

    # ── Figure 6: Energy vs N ────────────────────────────────────────────────
    print("Generating Figure 6: Scalability — energy vs N...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent, color, marker in zip(AGENTS, COLORS, MARKERS):
        sub = scale_df[scale_df["Algorithm"] == agent].sort_values("N_devices")
        ax.errorbar(sub["N_devices"],
                    sub["Avg_Energy_J"],
                    yerr=sub["Std_Energy_J"],
                    label=agent, color=color, marker=marker,
                    linewidth=2.0, markersize=7, capsize=4)

    ax.set_xticks(N_vals)
    ax.set_xticklabels([f"N = {n}" for n in N_vals], fontsize=11)
    ax.set_xlabel("Number of IoMT Devices", fontsize=12)
    ax.set_ylabel("Average Energy per Task (Joules)", fontsize=12)
    ax.set_title("Figure 6: Scalability — Energy Consumption vs Device Count", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "fig6_scalability_energy")

    # ── Figure 7: Tier-1 Violations vs N ────────────────────────────────────
    print("Generating Figure 7: Scalability — Tier-1 violations vs N...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent, color, marker in zip(AGENTS, COLORS, MARKERS):
        sub = scale_df[scale_df["Algorithm"] == agent].sort_values("N_devices")
        ax.plot(sub["N_devices"], sub["Viol_T1_pct"],
                label=agent, color=color, marker=marker,
                linewidth=2.0, markersize=7)

    ax.set_xticks(N_vals)
    ax.set_xticklabels([f"N = {n}" for n in N_vals], fontsize=11)
    ax.set_xlabel("Number of IoMT Devices", fontsize=12)
    ax.set_ylabel("Tier-1 Deadline Violation Rate (%)", fontsize=12)
    ax.set_title("Figure 7: Scalability — Tier-1 Violations vs Device Count", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(-2, 105)
    fig.tight_layout()
    save(fig, "fig7_scalability_viol")

else:
    print("  Skipping Figures 6 and 7 (scalability_summary.csv not found)")

# ══════════════════════════════════════════════════════════════════════════════
# PRINT DATA TABLES for each figure 
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  DATA TABLES ")
print("="*65)

print("\n--- Table for Figure 3 (Energy, N=20) ---")
print(f"{'Algorithm':<15} {'Mean (J)':>10} {'Std (J)':>10}")
print("-"*38)
for agent in AGENTS:
    df = dfs_20[agent]
    print(f"{agent:<15} {df['energy'].mean():>10.4f} {df['energy'].std():>10.4f}")

print("\n--- Table for Figure 5 (Violations by tier, N=20) ---")
print(f"{'Algorithm':<15} {'Overall%':>10} {'Tier-1%':>10} {'Tier-2%':>10} {'Tier-3%':>10}")
print("-"*58)
for agent in AGENTS:
    df = dfs_20[agent]
    print(f"{agent:<15} "
          f"{df['violated'].mean()*100:>10.2f} "
          f"{df[df.priority==1]['violated'].mean()*100:>10.2f} "
          f"{df[df.priority==2]['violated'].mean()*100:>10.2f} "
          f"{df[df.priority==3]['violated'].mean()*100:>10.2f}")

print("\n--- Table for Figure 2 (Convergence) ---")
print(f"{'Agent':<15} {'Ep-0 Reward':>12} {'Last-100 Mean':>14} {'Last-100 Std':>13}")
print("-"*57)
print(f"{'Dueling DQN':<15} {cd['mean'].iloc[0]:>12.1f} "
      f"{cd['mean'].iloc[-100:].mean():>14.1f} "
      f"{cd['mean'].iloc[-100:].std():>13.1f}")
print(f"{'Standard DQN':<15} {cs['mean'].iloc[0]:>12.1f} "
      f"{cs['mean'].iloc[-100:].mean():>14.1f} "
      f"{cs['mean'].iloc[-100:].std():>13.1f}")

if has_scale:
    print("\n--- Table for Figures 6 & 7 (Scalability) ---")
    cols = ["N_devices","Algorithm","Avg_Energy_J","Avg_Time_ms","Viol_Overall_pct","Viol_T1_pct"]
    print(scale_df[cols].to_string(index=False, float_format="%.3f"))

print("\nAll done. Check folder for fig2–fig7 PDF and PNG files.")
