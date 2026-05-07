import numpy as np
import matplotlib.pyplot as plt
from math import erf

# Input range
x = np.linspace(-5, 5, 1000)

# Activation functions
relu = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.1 * x)

# GELU using the exact erf form
gelu = 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# ReLU
axes[0].plot(x, relu)
axes[0].set_title("ReLU", fontsize=20)
axes[0].set_xlabel("Input", fontsize=18)
axes[0].set_ylabel("Output", fontsize=18)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-5, 5)
axes[0].set_ylim(-0.5, 5)

# Leaky ReLU
axes[1].plot(x, leaky_relu)
axes[1].set_title("Leaky ReLU", fontsize=20)
axes[1].set_xlabel("Input", fontsize=18)
axes[1].set_ylabel("Output", fontsize=18)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(-0.5, 5)

# GELU
axes[2].plot(x, gelu)
axes[2].set_title("GELU", fontsize=20)
axes[2].set_xlabel("Input", fontsize=18)
axes[2].set_ylabel("Output", fontsize=18)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(-5, 5)
axes[2].set_ylim(-0.5, 5)

for ax in axes:
    ax.tick_params(labelsize=16)

plt.tight_layout()
plt.show()