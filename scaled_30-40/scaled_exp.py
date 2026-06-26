"""
scalability_experiment.py
=========================
Runs the IoMT task offloading simulation at N=40 and N=60 devices
to answer supervisor comment 7 (scalability analysis).

Place this file in the same folder as:
    simulator.py, agent.py, baselines.py

Run:
    python scalability_experiment.py

Outputs (saved to same folder):
    scalability_results_N20.csv   (copied from existing results)
    scalability_results_N40.csv
    scalability_results_N60.csv
    scalability_summary.csv       (one row per N per agent)
"""

import numpy as np
import pandas as pd
import torch
import os
import shutil
from simulator import build_environment
from agent import DuelingDQNAgent
from baselines import GreedyAgent, RoundRobinAgent, StandardDQNAgent

# ── Configuration ─────────────────────────────────────────────────────────────
N_MEC        = 3
N_EPISODES   = 500       # same as original training
TASKS_PER_EP = 100
N_TEST_TASKS = 1000
N_SEEDS      = 5
DEVICE_COUNTS = [30, 40]   # N=20 already exists; we add 40 and 60

# State size: 7 device+task features + N_MEC loads + N_MEC SNR = 7+3+3 = 13
# Action size: local + 3 MEC + cloud = 5
STATE_SIZE  = 13
ACTION_SIZE = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_reward(outcome, task, alpha=0.6, beta=0.4, lam=50.0):
    """Identical reward function to training."""
    priority_weight   = 4 - task["priority"]
    time_ratio        = min(outcome["time"] / task["deadline"], 3.0)
    energy_ratio      = min(outcome["energy"] / 0.5, 1.0)
    latency_penalty   = alpha * priority_weight * time_ratio
    energy_penalty    = beta  * energy_ratio
    violation_penalty = lam   * outcome["violated"]
    battery_penalty   = 0.1 * (1.0 - outcome.get("battery_remaining", 1.0))
    return -(latency_penalty + energy_penalty + violation_penalty + battery_penalty)


def train_dueling_dqn(n_devices, seed):
    """Train Dueling DQN for a given N and seed. Returns trained agent."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env, devices, mec_servers, controller = build_environment(n_devices, N_MEC)
    agent = DuelingDQNAgent(STATE_SIZE, ACTION_SIZE)

    for episode in range(N_EPISODES):
        for _ in range(TASKS_PER_EP):
            device_id = np.random.randint(n_devices)
            devices[device_id].queue_depth += 1
            task      = devices[device_id].generate_task()
            state     = controller.get_global_state(task)
            action    = agent.select_action(state)
            outcome   = controller.compute_outcome(task, action)
            reward    = compute_reward(outcome, task)
            next_task  = devices[device_id].generate_task()
            next_state = controller.get_global_state(next_task)
            agent.memory.push(state, action, reward, next_state)
            agent.train_step()
            devices[device_id].queue_depth = max(0, devices[device_id].queue_depth - 1)

        if (episode + 1) % 100 == 0:
            print(f"    [N={n_devices} seed={seed}] Episode {episode+1}/{N_EPISODES} "
                  f"| ε={agent.epsilon:.3f}")
    agent.epsilon = 0.0
    return agent, env, devices, mec_servers, controller


def train_standard_dqn(n_devices, seed):
    """Train Standard DQN for a given N and seed."""
    np.random.seed(seed + 50)
    torch.manual_seed(seed + 50)
    env, devices, mec_servers, controller = build_environment(n_devices, N_MEC)
    agent = StandardDQNAgent(STATE_SIZE, ACTION_SIZE)

    for episode in range(N_EPISODES):
        for _ in range(TASKS_PER_EP):
            device_id = np.random.randint(n_devices)
            devices[device_id].queue_depth += 1
            task      = devices[device_id].generate_task()
            state     = controller.get_global_state(task)
            action    = agent.select_action(state)
            outcome   = controller.compute_outcome(task, action)
            reward    = compute_reward(outcome, task)
            next_task  = devices[device_id].generate_task()
            next_state = controller.get_global_state(next_task)
            agent.push(state, action, reward, next_state)
            agent.train_step()
            devices[device_id].queue_depth = max(0, devices[device_id].queue_depth - 1)

        if (episode + 1) % 100 == 0:
            print(f"    [StdDQN N={n_devices} seed={seed}] Episode {episode+1}/{N_EPISODES}")
    agent.epsilon = 0.0
    return agent, env, devices, mec_servers, controller


def evaluate_agent(agent_name, agent, controller, devices, n_devices, n_tasks=1000):
    """Evaluate any agent over n_tasks. Returns a DataFrame of results."""
    results = []
    for _ in range(n_tasks):
        device_id = np.random.randint(n_devices)
        task      = devices[device_id].generate_task()
        state     = controller.get_global_state(task)
        action    = agent.select_action(state)
        outcome   = controller.compute_outcome(task, action)
        results.append({
            "energy":   outcome["energy"],
            "time_ms":  outcome["time"] * 1000,
            "violated": outcome["violated"],
            "priority": task["priority"],
        })
    return pd.DataFrame(results)


# ── Main loop ─────────────────────────────────────────────────────────────────

all_summary_rows = []

for N in DEVICE_COUNTS:
    print(f"\n{'='*60}")
    print(f"  SCALABILITY EXPERIMENT — N = {N} devices")
    print(f"{'='*60}")

    all_results = {"Dueling DQN": [], "Standard DQN": [], "Greedy": [], "Round-Robin": []}

    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed + 1}/{N_SEEDS}")

        # Train Dueling DQN
        print("    Training Dueling DQN...")
        d_agent, env, devices, mec_servers, controller = train_dueling_dqn(N, seed)

        # Train Standard DQN
        print("    Training Standard DQN...")
        s_agent, _, _, _, _ = train_standard_dqn(N, seed)

        # Re-use same environment for evaluation (reset seed for consistency)
        np.random.seed(seed + 100)
        env2, devices2, mec_servers2, controller2 = build_environment(N, N_MEC)

        # Reload trained Dueling DQN into fresh environment
        d_agent_eval = DuelingDQNAgent(STATE_SIZE, ACTION_SIZE)
        d_agent_eval.online_net.load_state_dict(d_agent.online_net.state_dict())
        d_agent_eval.epsilon = 0.0

        s_agent_eval = StandardDQNAgent(STATE_SIZE, ACTION_SIZE)
        s_agent_eval.online_net.load_state_dict(s_agent.online_net.state_dict())
        s_agent_eval.epsilon = 0.0

        greedy_agent = GreedyAgent(n_mec=N_MEC)
        rr_agent     = RoundRobinAgent(n_actions=ACTION_SIZE)

        agents = {
            "Dueling DQN":  d_agent_eval,
            "Standard DQN": s_agent_eval,
            "Greedy":       greedy_agent,
            "Round-Robin":  rr_agent,
        }

        print("    Evaluating all agents...")
        for name, agent in agents.items():
            df = evaluate_agent(name, agent, controller2, devices2, N, N_TEST_TASKS)
            all_results[name].append(df)
            print(f"      {name}: energy={df['energy'].mean():.4f}J  "
                  f"time={df['time_ms'].mean():.1f}ms  "
                  f"viol={df['violated'].mean()*100:.2f}%  "
                  f"T1={df[df.priority==1]['violated'].mean()*100:.2f}%")

    # Save per-N CSV files
    print(f"\n  Saving results for N={N}...")
    for name, dfs in all_results.items():
        combined = pd.concat(dfs, ignore_index=True)
        fname = f"scalability_results_N{N}_{name.replace(' ', '_')}.csv"
        combined.to_csv(fname, index=False)
        print(f"    Saved {fname}")

    # Build summary rows for this N
    for name, dfs in all_results.items():
        combined = pd.concat(dfs, ignore_index=True)
        t1 = combined[combined.priority == 1]
        all_summary_rows.append({
            "N_devices":       N,
            "Algorithm":       name,
            "Avg_Energy_J":    combined["energy"].mean(),
            "Std_Energy_J":    combined["energy"].std(),
            "Avg_Time_ms":     combined["time_ms"].mean(),
            "Std_Time_ms":     combined["time_ms"].std(),
            "Viol_Overall_pct":combined["violated"].mean() * 100,
            "Viol_T1_pct":     t1["violated"].mean() * 100 if len(t1) > 0 else 0.0,
            "Viol_T2_pct":     combined[combined.priority==2]["violated"].mean() * 100,
            "Viol_T3_pct":     combined[combined.priority==3]["violated"].mean() * 100,
        })

# Add N=20 from existing CSVs
print("\n  Loading existing N=20 results...")
existing = {
    "Dueling DQN":  pd.read_csv("results_Dueling_DQN.csv"),
    "Standard DQN": pd.read_csv("results_Standard_DQN.csv"),
    "Greedy":       pd.read_csv("results_Greedy.csv"),
    "Round-Robin":  pd.read_csv("results_Round-Robin.csv"),
}
for name, df in existing.items():
    t1 = df[df.priority == 1]
    all_summary_rows.append({
        "N_devices":       20,
        "Algorithm":       name,
        "Avg_Energy_J":    df["energy"].mean(),
        "Std_Energy_J":    df["energy"].std(),
        "Avg_Time_ms":     df["time_ms"].mean(),
        "Std_Time_ms":     df["time_ms"].std(),
        "Viol_Overall_pct":df["violated"].mean() * 100,
        "Viol_T1_pct":     t1["violated"].mean() * 100 if len(t1) > 0 else 0.0,
        "Viol_T2_pct":     df[df.priority==2]["violated"].mean() * 100,
        "Viol_T3_pct":     df[df.priority==3]["violated"].mean() * 100,
    })

# Save master summary
summary_df = pd.DataFrame(all_summary_rows)
summary_df = summary_df.sort_values(["N_devices", "Algorithm"]).reset_index(drop=True)
summary_df.to_csv("scalability_summary.csv", index=False)
print("\nScalability summary saved to scalability_summary.csv")
print(summary_df[["N_devices","Algorithm","Avg_Energy_J","Avg_Time_ms","Viol_Overall_pct","Viol_T1_pct"]].to_string())
print("\nDone. Now run: python plot_all_results.py")
