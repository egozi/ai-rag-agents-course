
"""
Generate two plots for slide 31 (Evaluate Predictions):
1. Predicted vs Actual scatter plot  
2. Residuals plot

Using California Housing dataset (MedInc → MedHouseVal)
Real dataset stats: slope≈0.42, intercept≈0.45, R²≈0.47, n=20640
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load California Housing dataset
data = fetch_california_housing()
X = data.data[:, 0].reshape(-1, 1)  # MedInc (feature 0)
y = data.target  # Median house value (in $100K)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X).flatten()
residuals = y - y_pred

# Compute metrics
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res / ss_tot

print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"N samples: {len(y)}")

# Style settings matching the deck
NAVY = '#1E223D'
GOLD = '#F5A623'
BLUE_DOT = '#4A90D9'
GRAY = '#AAAAAA'
GREEN = '#27AE60'
RED = '#E74C3C'
BG_COLOR = '#FFFFFF'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri']

# ==========================================
# Plot 1: Predicted vs Actual
# ==========================================
fig1, ax1 = plt.subplots(figsize=(6.5, 5.5), facecolor=BG_COLOR)
ax1.set_facecolor(BG_COLOR)

ax1.scatter(y, y_pred, c=BLUE_DOT, alpha=0.15, s=5, edgecolors='none',
            zorder=2, rasterized=True)

lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
ax1.plot(lims, lims, color=GOLD, linewidth=2.5, linestyle='--',
         label='Perfect Prediction', zorder=3)

ax1.set_xlabel('Actual House Value ($100K)', fontsize=15, color=NAVY, fontweight='bold')
ax1.set_ylabel('Predicted House Value ($100K)', fontsize=15, color=NAVY, fontweight='bold')

ax1.tick_params(colors=NAVY, labelsize=12)
for spine in ax1.spines.values():
    spine.set_color(GRAY)
    spine.set_linewidth(0.8)

ax1.legend(fontsize=12, loc='upper left', framealpha=0.9,
           edgecolor=GRAY, fancybox=True)

ax1.text(0.95, 0.08, f'R² = {r2:.2f}', transform=ax1.transAxes,
         fontsize=20, fontweight='bold', color=GOLD,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor=NAVY, alpha=0.95, edgecolor=GOLD))

plt.tight_layout()
fig1.savefig('pred_vs_actual.png', dpi=200, bbox_inches='tight',
             facecolor=BG_COLOR, edgecolor='none')
plt.close(fig1)

# ==========================================
# Plot 2: Residuals Plot
# ==========================================
fig2, ax2 = plt.subplots(figsize=(6.5, 5.5), facecolor=BG_COLOR)
ax2.set_facecolor(BG_COLOR)

ax2.scatter(y_pred, residuals, c=BLUE_DOT, alpha=0.15, s=5, edgecolors='none',
            zorder=2, rasterized=True)

ax2.axhline(y=0, color=GOLD, linewidth=2.5, linestyle='--', label='Zero Error', zorder=3)

ax2.set_xlabel('Predicted House Value ($100K)', fontsize=15, color=NAVY, fontweight='bold')
ax2.set_ylabel('Residual (Actual − Predicted)', fontsize=15, color=NAVY, fontweight='bold')

ax2.tick_params(colors=NAVY, labelsize=12)
for spine in ax2.spines.values():
    spine.set_color(GRAY)
    spine.set_linewidth(0.8)

ax2.legend(fontsize=12, loc='upper right', framealpha=0.9,
           edgecolor=GRAY, fancybox=True)

ax2.annotate('Pattern visible →\nmodel misses non-linear\nrelationships',
             xy=(y_pred.max() * 0.7, residuals[y_pred > y_pred.max()*0.6].mean() + 0.8),
             fontsize=11, color=RED, fontstyle='italic',
             ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF5F5',
                       alpha=0.95, edgecolor=RED, linewidth=1.5))

plt.tight_layout()
fig2.savefig('residuals_plot.png', dpi=200, bbox_inches='tight',
             facecolor=BG_COLOR, edgecolor='none')
plt.close(fig2)

print("\nPlots saved:")
print("  pred_vs_actual.png")
print("  residuals_plot.png")
# ```

# Copy-paste this into a `.py` file in your `bgu-ai\code` folder and run it. It will generate two PNG files in the same directory:
# - **pred_vs_actual.png** — Predicted vs Actual scatter with R² badge
# - **residuals_plot.png** — Residuals plot with pattern annotation

# Both are styled to match your deck colors (navy, gold, blue dots).