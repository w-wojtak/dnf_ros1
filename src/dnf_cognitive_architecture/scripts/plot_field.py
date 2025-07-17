
import numpy as np
import matplotlib.pyplot as plt

# Path to your .npy file
file_path = '/home/robotica/dnf_ros1/data_basic/u_sm_20250717_173350.npy'

# Load the vector
u_sm = np.load(file_path)

# Print shape for confirmation
print("Loaded field with shape:", u_sm.shape)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(u_sm)
plt.title('u_sm Vector')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()