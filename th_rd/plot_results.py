import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
AGENTS = ["Dueling DQN", "Standard DQN", "Greedy", "Round-Robin"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ── FIGURE 2: Convergence plot ───────────────────────────────────────────────
conv_d = pd.read_csv("convergence_data.csv",       index_col="episode")
conv_s = pd.read_csv("convergence_data_stddqn.csv", index_col="episode")

fig, ax = plt.subplots(figsize=(7, 4))

# Dueling DQN
ax.plot(conv_d["mean"], color="#1f77b4",
        label="Dueling DQN", linewidth=1.8)
ax.fill_between(conv_d.index,
                conv_d["mean"] - conv_d["std"],
                conv_d["mean"] + conv_d["std"],
                alpha=0.25, color="#1f77b4")

# Standard DQN
ax.plot(conv_s["mean"], color="#ff7f0e",
        label="Standard DQN", linestyle="--", linewidth=1.8)
ax.fill_between(conv_s.index,
                conv_s["mean"] - conv_s["std"],
                conv_s["mean"] + conv_s["std"],
                alpha=0.20, color="#ff7f0e")

ax.set_xlabel("Training Episode")
ax.set_ylabel("Cumulative Reward")
ax.set_title("Training Convergence")
ax.legend()
plt.tight_layout()
plt.savefig("fig2_convergence.pdf", dpi=300)
plt.close()
print("Figure 2 saved.")

# ── FIGURE 3: Energy bar chart ───────────────────────────────────────────────
means = []; stds = []
for name in AGENTS:
    df = pd.read_csv(f"results_{name.replace(' ','_')}.csv")
    means.append(df["energy"].mean())
    stds.append(df["energy"].std())

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(AGENTS, means, yerr=stds, color=COLORS, capsize=5,
              width=0.55, error_kw={"linewidth": 1.2})
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{m:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Avg. Energy per Task (J)"); ax.set_title("Energy Consumption")
plt.tight_layout(); plt.savefig("fig3_energy.pdf", dpi=300); plt.close()
print("Figure 3 saved.")

# ── FIGURE 4: Completion time CDF ───────────────────────────────────────────
STYLES = ["-", "--", "-.", ":"]
fig, ax = plt.subplots(figsize=(7, 4))
for name, color, style in zip(AGENTS, COLORS, STYLES):
    df   = pd.read_csv(f"results_{name.replace(' ','_')}.csv")
    data = np.sort(df["time_ms"].values)
    cdf  = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, cdf, label=name, color=color, linestyle=style, linewidth=1.5)
ax.axvline(50,  color="gray", linestyle="--", linewidth=0.9, label="Tier-1 (50ms)")
ax.axvline(100, color="black", linestyle=":",  linewidth=0.9, label="Tier-2 (100ms)")
ax.set_xlabel("Task Completion Time (ms)"); ax.set_ylabel("Cumulative Probability")
ax.set_title("Completion Time CDF"); ax.set_xlim(0, 200); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("fig4_latency_cdf.pdf", dpi=300); plt.close()
print("Figure 4 saved.")

# ── FIGURE 5: Deadline violation by priority tier ────────────────────────────
tier_labels = ["Tier 1 (Critical)", "Tier 2 (Urgent)", "Tier 3 (Routine)"]
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(3); width = 0.35

for i, name in enumerate(["Dueling DQN", "Standard DQN"]):
    df   = pd.read_csv(f"results_{name.replace(' ','_')}.csv")
    viol = [df[df["priority"]==p]["violated"].mean()*100 for p in [1,2,3]]
    ax.bar(x + i*width, viol, width, label=name,
           color=COLORS[i], capsize=4)

ax.set_xticks(x + width/2); ax.set_xticklabels(tier_labels)
ax.set_ylabel("Deadline Violation Rate (%)"); ax.set_title("Violations by Priority Tier")
ax.legend(); plt.tight_layout()
plt.savefig("fig5_tier_violations.pdf", dpi=300); plt.close()
print("Figure 5 saved. All figures complete.")
