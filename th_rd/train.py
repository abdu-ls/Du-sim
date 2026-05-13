import numpy as np
import pandas as pd
import torch
from simulator import build_environment
from agent import DuelingDQNAgent

# ── Configuration ────────────────────────────────────────────────────────────
N_DEVICES    = 20
N_MEC        = 3
N_EPISODES   = 500
TASKS_PER_EP = 100   # Tasks generated per episode
N_SEEDS      = 5     # Run 5 times with different seeds for confidence intervals

# State size: 7 device+task features + N_MEC loads + N_MEC SNR values
STATE_SIZE  = 7 + N_MEC + N_MEC   # = 13
ACTION_SIZE = 1 + N_MEC + 1        # local + 3 MEC + cloud = 5

all_rewards = []   # Will store reward per episode per seed

for seed in range(N_SEEDS):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env, devices, mec_servers, controller = build_environment(N_DEVICES, N_MEC)
    agent = DuelingDQNAgent(STATE_SIZE, ACTION_SIZE)

    episode_rewards = []

    for episode in range(N_EPISODES):
        total_reward = 0.0

        for _ in range(TASKS_PER_EP):
            # Pick a random device to generate a task
            device_id = np.random.randint(N_DEVICES)
            devices[device_id].queue_depth += 1

            # Step 1: Generate task and build MDP state
            task  = devices[device_id].generate_task()
            state = controller.get_global_state(task)

            # Step 2: Agent selects offloading action
            action = agent.select_action(state)

            # Step 3: SDN controller executes action, observes outcome
            outcome = controller.compute_outcome(task, action)

            # Step 4: Compute reward
            reward = agent.compute_reward(outcome, task)
            total_reward += reward

            # Step 5: Get next state and store experience
            next_task  = devices[device_id].generate_task()
            next_state = controller.get_global_state(next_task)
            agent.memory.push(state, action, reward, next_state)

            # Step 6: Train the agent
            agent.train_step()
            devices[device_id].queue_depth = max(0, devices[device_id].queue_depth - 1)

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            print(f"Seed {seed} | Episode {episode+1}/{N_EPISODES} "
                  f"| Reward: {total_reward:.1f} | ε: {agent.epsilon:.3f}")
	
    all_rewards.append(episode_rewards)
    # Save trained model for evaluation
    torch.save(agent.online_net.state_dict(), f"dueling_dqn_seed{seed}.pt")

# Save convergence data for Figure 2 (convergence plot)
df = pd.DataFrame(all_rewards).T
df.columns = [f"seed_{i}" for i in range(N_SEEDS)]
df["mean"]  = df.mean(axis=1)
df["std"]   = df.std(axis=1)
df.to_csv("convergence_data.csv", index_label="episode")
print("Training complete. Convergence data saved.")


# ── Train Standard DQN and save convergence data ──────────────────────
from baselines import StandardDQNAgent

all_std_rewards = []

for seed in range(N_SEEDS):
    np.random.seed(seed + 50)
    torch.manual_seed(seed + 50)

    env, devices, mec_servers, controller = build_environment(N_DEVICES, N_MEC)
    agent = StandardDQNAgent(STATE_SIZE, ACTION_SIZE)

    episode_rewards = []

    for episode in range(N_EPISODES):
        total_reward = 0.0
        for _ in range(TASKS_PER_EP):
            device_id = np.random.randint(N_DEVICES)
            devices[device_id].queue_depth += 1
            task       = devices[device_id].generate_task()
            state      = controller.get_global_state(task)
            action     = agent.select_action(state)
            outcome    = controller.compute_outcome(task, action)

            priority_weight = 4 - task["priority"]
            time_ratio      = min(outcome["time"] / task["deadline"], 3.0)
            energy_ratio    = min(outcome["energy"] / 0.5, 1.0)
            reward = -(0.6 * priority_weight * time_ratio
                     + 0.4 * energy_ratio
                     + 50.0 * outcome["violated"])

            total_reward += reward
            next_task    = devices[device_id].generate_task()
            next_state   = controller.get_global_state(next_task)
            agent.push(state, action, reward, next_state)
            agent.train_step()
            devices[device_id].queue_depth = max(
                0, devices[device_id].queue_depth - 1)

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            print(f"Std DQN Seed {seed} | Episode {episode+1} "
                  f"| Reward: {total_reward:.1f}")

    all_std_rewards.append(episode_rewards)
    torch.save(agent.online_net.state_dict(), f"std_dqn_seed{seed}.pt")

# Save Standard DQN convergence
df_std = pd.DataFrame(all_std_rewards).T
df_std.columns = [f"seed_{i}" for i in range(N_SEEDS)]
df_std["mean"] = df_std.mean(axis=1)
df_std["std"]  = df_std.std(axis=1)
df_std.to_csv("convergence_data_stddqn.csv", index_label="episode")
print("Standard DQN training complete.")
