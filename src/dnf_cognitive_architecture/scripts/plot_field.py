
import numpy as np
import matplotlib.pyplot as plt

# Path to your .npy file
# file_path = '/home/robotica/dnf_ros1/data_basic/u_sm_20250717_173350.npy'
file_path = '/home/robotica/dnf_ros1/data_basic/h_amem_20250721_171431.npy'

# Load the vector
u_sm = np.load(file_path)

# Print shape for confirmation
print("Loaded field with shape:", u_sm.shape)

x_lim = 80
dx = 0.2
# Spatial grid
x = np.arange(-x_lim, x_lim + dx, dx)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x,u_sm)
plt.title('u_sm from file')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()