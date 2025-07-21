import numpy as np
import matplotlib.pyplot as plt
import os

data_dir = "/home/robotica/dnf_ros1/data_basic"

def get_nth_latest_file(prefix, n=1):
    """Get the nth latest file (n=1: latest, n=2: second latest, etc.)"""
    files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith('.npy')]
    if len(files) < n:
        raise FileNotFoundError(f"Not enough files with prefix '{prefix}' in {data_dir}")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    return os.path.join(data_dir, files[n-1])

# Load the second-to-last histories
u_act_hist = np.load(get_nth_latest_file("act_history_", n=2))
u_sim_hist = np.load(get_nth_latest_file("sim_history_", n=2))
u_wm_hist  = np.load(get_nth_latest_file("wm_history_", n=2))
u_f1_hist  = np.load(get_nth_latest_file("f1_history_", n=2))
u_f2_hist  = np.load(get_nth_latest_file("f2_history_", n=2))

# Set your dt (time step) as used in your node
dt = 1.0  # <-- Change this to your actual dt if different!
lengths = [
    u_act_hist.shape[0],
    u_sim_hist.shape[0],
    u_wm_hist.shape[0],
    u_f1_hist.shape[0],
    u_f2_hist.shape[0]
]
min_len = min(lengths)
time_steps = np.arange(min_len) * dt

print(f"LENGTH {min_len}.")

if len(set(lengths)) != 1:
    print(f"WARNING: Arrays have different lengths: {lengths}. Truncating to {min_len}.")

object_names = {
    -60: 'base',
    -20: 'load',
    20: 'bearing',
    40: 'motor'
}
input_positions = list(object_names.keys())

fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

for i, pos in enumerate(input_positions):
    axes[0].plot(time_steps, u_act_hist[:min_len, i], label=object_names[pos])
    axes[1].plot(time_steps, u_sim_hist[:min_len, i], label=object_names[pos])
    axes[2].plot(time_steps, u_wm_hist[:min_len, i], label=object_names[pos])
    axes[3].plot(time_steps, u_f1_hist[:min_len, i], label=object_names[pos])
    axes[4].plot(time_steps, u_f2_hist[:min_len, i], label=object_names[pos])

axes[0].set_title('u_act over time at input positions')
axes[1].set_title('u_sim over time at input positions')
axes[2].set_title('u_wm over time at input positions')
axes[3].set_title('u_f1 over time at input positions')
axes[4].set_title('u_f2 over time at input positions')

axes[0].set_ylim(-3, 5)  # For u_act
axes[1].set_ylim(-3, 5)  # For u_sim

for ax in axes:
    ax.set_ylabel('Activity')
    ax.legend()
    ax.grid(True)
axes[-1].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()