import simpy
import numpy as np

# ── Thresholds for priority classification (from the paper Table 1) ──
THRESHOLDS = {
    "heart_rate":     {"lo": 40,  "hi": 150, "wlo": 45,  "whi": 120},
    "spo2":           {"lo": 90,  "hi": 101, "wlo": 92,  "whi": 95},
    "blood_glucose":  {"lo": 54,  "hi": 250, "wlo": 70,  "whi": 180},
    "blood_pressure": {"lo": 0,   "hi": 180, "wlo": 130, "whi": 160},
}

VITAL_SIGNS = list(THRESHOLDS.keys())

def classify_priority(vital, value):
    """
    Returns 1 (critical), 2 (urgent), or 3 (routine).
    This is the MDP priority state P_i from Equation (2).
    """
    t = THRESHOLDS[vital]
    if value < t["lo"] or value > t["hi"]:
        return 1
    elif value < t["wlo"] or value > t["whi"]:
        return 2
    else:
        return 3

def generate_sensor_reading(vital, critical_prob=0.05, urgent_prob=0.15):
    """
    Simulates a sensor reading. Most readings are normal (routine),
    but occasionally generates warning or critical values to test
    the agent's priority-awareness.
    """
    t = THRESHOLDS[vital]
    r = np.random.random()
    if r < critical_prob:
        # Generate a reading outside critical threshold
        if np.random.random() < 0.5:
            return np.random.uniform(t["lo"] * 0.5, t["lo"] - 1)
        else:
            return np.random.uniform(t["hi"] + 1, t["hi"] * 1.3)
    elif r < critical_prob + urgent_prob:
        return np.random.uniform(t["wlo"], t["lo"])
    else:
        return np.random.uniform(t["lo"], t["hi"])

class IoMTDevice:
    """
    Represents one wearable IoMT device (e.g., ECG patch).
    Continuously generates tasks and reports its state to the SDN controller.
    """
    def __init__(self, device_id, env):
        self.id = device_id
        self.env = env
        # Battery starts between 60% and 100% — randomised per device
        self.battery = np.random.uniform(0.6, 1.0)
        # Local CPU speed in GHz (0.5 to 1.0 as per Table 2)
        self.cpu_freq = np.random.uniform(0.5e9, 1.0e9)
        # Transmission power in Watts
        self.tx_power = np.random.uniform(0.1, 0.5)
        # Uplink data rate in bits/second (10–20 MHz bandwidth → Shannon capacity)
        self.uplink_rate = np.random.uniform(10e6, 20e6)
        # Task queue depth
        self.queue_depth = 0

    def generate_task(self):
    	vital  = np.random.choice(VITAL_SIGNS)
    	value  = generate_sensor_reading(vital)
    	priority = classify_priority(vital, value)

    	# Data size in bits (0.1 MB to 5 MB) — same for all priorities
    	data_size = np.random.uniform(0.1e6 * 8, 5e6 * 8)

    	# CPU cycles and deadline scaled by priority
    	# Critical tasks are small urgent alerts — fast to compute, tight deadline
   	# Routine tasks are large analyses — slow to compute, relaxed deadline
    	if priority == 1:
            cpu_cycles = np.random.uniform(1e6, 5e6)   # Small
            deadline   = 0.050                          # 50 ms
    	elif priority == 2:
            cpu_cycles = np.random.uniform(5e6, 3e7)   # Medium
            deadline   = 0.100                          # 100 ms
    	else:
            cpu_cycles = np.random.uniform(3e7, 1e8)   # Large
            deadline   = 5.0                            # 5 seconds

    	task = {
            "device_id":  self.id,
            "priority":   priority,
            "data_size":  data_size,
            "cpu_cycles": cpu_cycles,
            "deadline":   deadline,
            "vital":      vital,
            "value":      value,
    	}
    	return task

    def get_state_features(self):
        """Returns the device-level features included in the MDP state vector."""
        return {
            "battery":     self.battery,
            "cpu_freq":    self.cpu_freq / 1e9,   # Normalise to GHz
            "queue_depth": self.queue_depth,
            "tx_power":    self.tx_power,
            "uplink_rate": self.uplink_rate / 1e6, # Normalise to Mbps
        }


class MECServer:
    """
    Represents one Mobile Edge Computing server.
    Tracks current CPU utilisation and computes processing time.
    """
    def __init__(self, server_id):
        self.id = server_id
        # Edge server CPU in GHz (5–10 GHz as per Table 2)
        self.cpu_freq = np.random.uniform(5e9, 10e9)
        # Load: fraction of CPU currently in use (0.0 to 1.0)
        self.load = np.random.uniform(0.1, 0.5)

    def available_freq(self):
        """Effective available CPU frequency after accounting for current load."""
        return self.cpu_freq * (1.0 - self.load)

    def get_state_features(self):
        return {"load": self.load, "cpu_freq": self.cpu_freq / 1e9}


class SDNController:
    """
    The SDN controller: the brain of the network.
    Assembles the global MDP state vector from all devices and servers.
    This is where the RL agent lives and where it observes the world.
    """
    def __init__(self, devices, mec_servers):
        self.devices = devices
        self.mec_servers = mec_servers
        # Cloud server parameters
        self.cloud_cpu = 50e9             # 50 GHz
        self.wan_delay = np.random.uniform(0.020, 0.080)  # 20–80 ms

    def get_global_state(self, task):
        """
        Builds the full MDP state vector s_t (Equation 9 in the paper).
        This is what the RL agent observes at every decision step.
        """
        dev = self.devices[task["device_id"]]
        dev_features = dev.get_state_features()

        # Collect MEC server loads and link SNR values
        mec_loads = [s.load for s in self.mec_servers]
        # SNR: simulate channel quality (higher = better link)
        snr_values = [np.random.uniform(10, 30) for _ in self.mec_servers]

        state = np.array([
            dev_features["battery"],
            dev_features["cpu_freq"],
            dev_features["queue_depth"] / 10.0,    # Normalise
            task["data_size"] / (5e6 * 8),         # Normalise to [0,1]
            task["cpu_cycles"] / 1e9,              # Normalise
            float(task["priority"]) / 3.0,         # Priority P_i: 1/3, 2/3, or 1.0
            dev_features["uplink_rate"] / 20.0,    # Normalise
        ] + [l for l in mec_loads]                 # One load value per MEC server
          + [s / 30.0 for s in snr_values],        # One SNR value per MEC server
        dtype=np.float32)
        return state

    def compute_outcome(self, task, action):
        """
        Given the agent's offloading action, compute energy consumed
        and task completion time. Returns the observed outcome.

        action = 0       → local execution
        action = 1,2,3   → MEC server index (1-based)
        action = 4       → cloud
        """
        dev = self.devices[task["device_id"]]
        D = task["data_size"]       # bits
        C = task["cpu_cycles"]      # cycles

        # ── Effective switched capacitance (standard IoMT value) ──
        kappa = 1e-28

        if action == 0:
            # Local execution (Equations 3 and 5)
            energy = kappa * (dev.cpu_freq ** 2) * C
            time   = C / dev.cpu_freq

        elif 1 <= action <= len(self.mec_servers):
            # MEC offloading (Equations 4 and 6)
            server = self.mec_servers[action - 1]
            energy = dev.tx_power * (D / dev.uplink_rate)
            tx_time = D / dev.uplink_rate
            proc_time = C / server.available_freq()
            time = tx_time + proc_time

        else:
            # Cloud offloading (Equations 4 and 7)
            wan_rate = np.random.uniform(5e6, 15e6)  # WAN is slower
            energy = dev.tx_power * (D / dev.uplink_rate)
            time = (D / wan_rate) + self.wan_delay + (C / self.cloud_cpu)

        # Deadline violation flag
        violated = int(time > task["deadline"])

        # Update device battery (proportional drain)
        dev.battery = max(0.0, dev.battery - energy * 1e-5)

        # Randomly fluctuate MEC loads to simulate dynamic conditions
        for server in self.mec_servers:
            server.load = np.clip(server.load + np.random.uniform(-0.03, 0.03), 0.05, 0.60)

        return {"energy": energy, "time": time, "violated": violated}


def build_environment(n_devices=20, n_mec=3):
    """Creates the full IoMT simulation environment."""
    env = simpy.Environment()
    devices = [IoMTDevice(i, env) for i in range(n_devices)]
    mec_servers = [MECServer(j) for j in range(n_mec)]
    controller = SDNController(devices, mec_servers)
    return env, devices, mec_servers, controller
