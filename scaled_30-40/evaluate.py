import numpy as np
import pandas as pd
import torch
from simulator import build_environment
from agent import DuelingDQNAgent
from baselines import GreedyAgent, RoundRobinAgent, StandardDQNAgent

# ── Configuration (matches train.py exactly) ──────────────────────────────
N_DEVICES    = 20
N_MEC        = 3
N_TEST_TASKS = 1000
STATE_SIZE   = 7 + N_MEC + N_MEC   # = 13
ACTION_SIZE  = 1 + N_MEC + 1       # local + 3 MEC + cloud = 5
N_SEEDS      = 5


def evaluate_agent(agent_name, agent, controller, devices, n_tasks=1000):
    """
    Runs an agent for n_tasks steps and records energy, completion time,
    deadline violations, and priority tier for every task.
    All agents receive only (state) as input — unified interface.
    """
    results = []

    for _ in range(n_tasks):
        # Pick a random device and generate a task
        device_id = np.random.randint(N_DEVICES)
        task      = devices[device_id].generate_task()
        state     = controller.get_global_state(task)

        # All agents share the same call signature: select_action(state)
        action = agent.select_action(state)

        # Execute the action and observe the outcome
        outcome = controller.compute_outcome(task, action)

        results.append({
            "energy":   outcome["energy"],
            "time_ms":  outcome["time"] * 1000,   # Convert to milliseconds
            "violated": outcome["violated"],
            "priority": task["priority"],
        })

    return pd.DataFrame(results)


def train_standard_dqn(seed, n_episodes=500, tasks_per_ep=100):
    """
    Trains the Standard DQN baseline so it gets a fair comparison.
    Uses the same training setup as the Dueling DQN in train.py.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env, devices, mec_servers, controller = build_environment(N_DEVICES, N_MEC)
    agent = StandardDQNAgent(STATE_SIZE, ACTION_SIZE)

    for episode in range(n_episodes):
        for _ in range(tasks_per_ep):
            device_id = np.random.randint(N_DEVICES)
            devices[device_id].queue_depth += 1

            task       = devices[device_id].generate_task()
            state      = controller.get_global_state(task)
            action     = agent.select_action(state)
            outcome    = controller.compute_outcome(task, action)

            # Same reward function as Dueling DQN for fair comparison
            priority_weight   = 4 - task["priority"]
            normalised_time   = outcome["time"] / task["deadline"]
            normalised_energy = outcome["energy"] / 0.5
            reward = -(0.6 * priority_weight * normalised_time
            	     + 0.4 * normalised_energy
         	     + 50.0 * outcome["violated"])

            next_task  = devices[device_id].generate_task()
            next_state = controller.get_global_state(next_task)

            agent.push(state, action, reward, next_state)
            agent.train_step()
            devices[device_id].queue_depth = max(0, devices[device_id].queue_depth - 1)

        if (episode + 1) % 100 == 0:
            print(f"  Standard DQN seed {seed} | Episode {episode+1}/{n_episodes} "
                  f"| ε: {agent.epsilon:.3f}")

    # Save trained weights for reuse
    torch.save(agent.online_net.state_dict(), f"std_dqn_seed{seed}.pt")
    return agent


# ── Main evaluation loop ──────────────────────────────────────────────────────
all_metrics = {name: [] for name in
               ["Dueling DQN", "Standard DQN", "Greedy", "Round-Robin"]}

for seed in range(N_SEEDS):
    print(f"\n── Evaluating seed {seed + 1}/{N_SEEDS} ──")
    np.random.seed(seed + 100)
    torch.manual_seed(seed + 100)

    # Fresh environment for each seed
    env, devices, mec_servers, controller = build_environment(N_DEVICES, N_MEC)

    # ── 1. Load trained Dueling DQN ──────────────────────────────────────────
    d_agent = DuelingDQNAgent(STATE_SIZE, ACTION_SIZE)
    d_agent.online_net.load_state_dict(
        torch.load(f"dueling_dqn_seed{seed}.pt", map_location="cpu"))
    d_agent.epsilon = 0.0   # Evaluation mode: no exploration
    print("  Loaded Dueling DQN.")

    # ── 2. Train & load Standard DQN ─────────────────────────────────────────
    print(f"  Training Standard DQN (seed {seed})...")
    std_agent = train_standard_dqn(seed, n_episodes=500, tasks_per_ep=100)
    std_agent.epsilon = 0.0
    print("  Standard DQN trained.")

    # ── 3. Instantiate rule-based baselines ───────────────────────────────────
    greedy_agent = GreedyAgent(n_mec=N_MEC)
    rr_agent     = RoundRobinAgent(n_actions=ACTION_SIZE)

    agents = {
        "Dueling DQN":  d_agent,
        "Standard DQN": std_agent,
        "Greedy":       greedy_agent,
        "Round-Robin":  rr_agent,
    }

    # ── 4. Evaluate every agent ───────────────────────────────────────────────
    for name, agent in agents.items():
        print(f"  Evaluating {name}...")
        df = evaluate_agent(name, agent, controller, devices, N_TEST_TASKS)
        all_metrics[name].append(df)
        print(f"    Avg energy: {df['energy'].mean():.4f} J | "
              f"Avg time: {df['time_ms'].mean():.1f} ms | "
              f"Violations: {df['violated'].mean()*100:.1f}%")

# ── Save results to CSV for plot_results.py ───────────────────────────────────
print("\nSaving results...")
for name, dfs in all_metrics.items():
    combined = pd.concat(dfs, ignore_index=True)
    filename = f"results_{name.replace(' ', '_')}.csv"
    combined.to_csv(filename, index=False)
    print(f"  Saved {filename} ({len(combined)} rows)")

print("\nEvaluation complete.")
