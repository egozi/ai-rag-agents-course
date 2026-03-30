import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ---------- Input ----------
x = np.linspace(-5, 5, 1000)

# ---------- Activations ----------
def relu(x): return np.maximum(0, x)
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def elu(x): return np.where(x > 0, x, np.exp(x) - 1)
def selu(x): return 1.0507 * np.where(x > 0, x, 1.6733 * (np.exp(x) - 1))

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def softplus(x): return np.log1p(np.exp(x))

def gelu(x): return 0.5 * x * (1 + erf(x / np.sqrt(2)))
def swish(x): return x * sigmoid(x)
def mish(x): return x * np.tanh(softplus(x))

# ---------- Grouped ----------
groups = {
    "ReLU Family": {
        "ReLU": relu(x),
        "Leaky ReLU": leaky_relu(x),
        "ELU": elu(x),
        "SELU": selu(x),
    },
    "Classical": {
        "Sigmoid": sigmoid(x),
        "Tanh": tanh(x),
        "Softplus": softplus(x),
    },
    "Modern Smooth": {
        "GELU": gelu(x),
        "Swish": swish(x),
        "Mish": mish(x),
    }
}

x_min, x_max = x.min(), x.max()
y_min = min(np.min(values) for category in groups.values() for values in category.values())
y_max = max(np.max(values) for category in groups.values() for values in category.values())

# ---------- Style ----------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 11
})

# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (title, funcs) in zip(axes, groups.items()):
    for name, y in funcs.items():
        ax.plot(x, y, linewidth=2, label=name)
    ax.set_title(title)
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

fig.suptitle("Activation Functions Overview", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# ---------- Export ----------
plt.savefig("activation_functions.png", dpi=300, bbox_inches="tight")

plt.show()

# Highlight ReLU vs. Leaky ReLU on identical axes for a fair comparison
relu_vals = relu(x)
leaky_vals = leaky_relu(x, alpha=0.1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
ax1.plot(x, relu_vals, label="ReLU")
ax1.set_title("ReLU")
ax1.set_xlabel("Input")
ax1.set_ylabel("Output")
ax1.grid(True, alpha=0.3)

ax2.plot(x, leaky_vals, label="Leaky ReLU")
ax2.set_title("Leaky ReLU")
ax2.set_xlabel("Input")
ax2.set_ylabel("Output")
ax2.grid(True, alpha=0.3)

shared_y_min = min(relu_vals.min(), leaky_vals.min())
shared_y_max = max(relu_vals.max(), leaky_vals.max())
for axis in (ax1, ax2):
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(shared_y_min, shared_y_max)
    axis.set_aspect('equal', adjustable='box')

plt.savefig("relu_leaky_relu.png", dpi=300, bbox_inches="tight")

plt.show()
