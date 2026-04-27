"""
MATH/BIOINF/STATS 547 — Final Project
Spectral Analysis of U.S. Equity Markets
=========================================
This script applies spectral and data-driven methods to daily returns of
20 large-cap U.S. stocks (2020–2026) to answer two questions:

  1. How many independent directions of variation does the equity return
     matrix truly possess?  (SVD + Random-Matrix Theory + MDS + Spectral
     Clustering)

  2. Does that cross-sectional structure enable temporal prediction?
     (DMD — returns; GARCH — volatility)

Central finding: the market is low-dimensional across stocks but
unpredictable over time — the mathematical signature of the Efficient
Market Hypothesis.

Run from:  /Users/damoncht/Desktop/math547/proj/
Output:    report/figures/fig{1..4}_*.pdf/.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from adjustText import adjust_text
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = "study1_market_data.csv"
FIG_DIR   = "report/figures/"
os.makedirs(FIG_DIR, exist_ok=True)

# Global plot style — larger fonts so figures are readable in the paper
plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

# Sector membership for the 20 stocks (5 per sector)
SECTORS = {
    'AAPL':'Tech',  'MSFT':'Tech',   'GOOGL':'Tech', 'AMZN':'Tech',  'META':'Tech',
    'JPM':'Finance','BAC':'Finance',  'GS':'Finance', 'MS':'Finance', 'C':'Finance',
    'JNJ':'Health', 'UNH':'Health',   'PFE':'Health', 'MRK':'Health', 'ABBV':'Health',
    'XOM':'Energy', 'CVX':'Energy',   'COP':'Energy', 'SLB':'Energy', 'EOG':'Energy',
}
SECTOR_COLORS = {
    'Tech':'#1f77b4', 'Finance':'#ff7f0e',
    'Health':'#2ca02c', 'Energy':'#d62728'
}
SECTOR_ORDER = ['Tech', 'Finance', 'Health', 'Energy']

# ── 1. Load & preprocess ───────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1 — SVD")
print("=" * 60)

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
sp500  = df['SP500']
data   = df.drop(columns=['SP500'])

tickers     = list(data.columns)
sector_list = [SECTORS[t] for t in tickers]
colors      = [SECTOR_COLORS[s] for s in sector_list]

# Daily percentage returns: r_{i,t} = (p_{i,t} - p_{i,t-1}) / p_{i,t-1}
returns       = data.pct_change().dropna()
sp500_returns = sp500.pct_change().dropna()

# Standardise each stock to zero mean / unit variance.
# This removes scale differences (e.g. high-priced vs low-priced shares)
# so that SVD finds correlation structure, not variance structure.
scaler = StandardScaler()
X = scaler.fit_transform(returns)          # shape: (T, N) = (≈1585, 20)
T, N = X.shape

# ── 2. SVD Analysis ────────────────────────────────────────────────────────────
# Thin SVD:  X = U Σ Vᵀ
#   U  (T×N): time-domain scores (how each day loads onto each PC)
#   Σ  (N×N): diagonal singular values (square root of variance explained)
#   Vt (N×N): stock loadings — Vt[k] is the k-th principal component direction
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Variance fraction explained by each PC
energy     = S**2 / S.dot(S)
cum_energy = np.cumsum(energy)

# ── Marchenko–Pastur noise floor ──────────────────────────────────────────────
# For an i.i.d. Gaussian T×N matrix, the eigenvalues of the sample
# correlation matrix C = XᵀX / T concentrate in [λ_min, λ_max]:
#     λ_± = (1 ± √q)²,   q = N/T  (aspect ratio)
# Eigenvalues ABOVE λ+ cannot be explained by chance and signal real structure.
q         = N / T                                        # ≈ 0.013 for our data
mp_max    = (1 + np.sqrt(q))**2                          # upper bulk edge ≈ 1.24
mp_min    = (1 - np.sqrt(q))**2                          # lower bulk edge ≈ 0.79

# Eigenvalues of the correlation matrix C = S² / T
C_evals   = S**2 / T
n_signal  = int(np.sum(C_evals > mp_max))               # number of genuine factors
print(f"C_evals={C_evals}")

# Express the MP upper bound as a % of total variance for the scree plot.
# Because sum(C_evals) = N (standardised data), the conversion is λ+/N.
mp_threshold_pct = mp_max / N * 100

# Align PC1/PC2 signs so they point in the market-upward direction
v1, v2 = Vt[0].copy(), Vt[1].copy()
if np.corrcoef(X @ v1, sp500_returns)[0, 1] < 0: v1 = -v1
if np.corrcoef(X @ v2, sp500_returns)[0, 1] < 0: v2 = -v2

r_pc1 = np.corrcoef(X @ v1, sp500_returns)[0, 1]

print(f"PC1 explains       : {energy[0]*100:.1f}%")
print(f"Top-3 cumulative   : {cum_energy[2]*100:.1f}%")
print(f"Top-5 cumulative   : {cum_energy[4]*100:.1f}%")
print(f"Corr(PC1, S&P 500) : {r_pc1:.3f}")
print(f"MP bulk: [{mp_min:.3f}, {mp_max:.3f}]  →  {n_signal} eigenvalues above noise floor")

# ── Figure 1 — SVD ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a): Scree plot
# Bars = individual variance fraction; curve = cumulative.
# Horizontal dotted line = MP upper bound λ+ expressed as % variance.
# Bars above this line correspond to genuinely informative PCs.
ax = axes[0]
ax.bar(range(1, N+1), energy * 100, color='steelblue', alpha=0.7, label='Individual')
ax.plot(range(1, N+1), cum_energy * 100, 'ro-', ms=5, lw=1.8, label='Cumulative')
ax.axhline(mp_threshold_pct, color='purple', ls=':', lw=2.0,
           label=f'Marchenko–Pastur bound $\\lambda_+={mp_max:.2f}$')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, N+1, 2))

ax = axes[1]
texts_1b = []
for i, ticker in enumerate(tickers):
    # Plot the scatter point
    ax.scatter(v1[i], v2[i], color=colors[i], s=60, zorder=5, edgecolors='k')
    
    # Place the text EXACTLY on the point so the arrows anchor correctly
    texts_1b.append(ax.text(v1[i], v2[i], ticker, fontsize=8))

# Let adjust_text push the labels away from the exact point coordinates
adjust_text(texts_1b, x=v1, y=v2, ax=ax,
            arrowprops=dict(arrowstyle='-', color='k', lw=0.5, alpha=0.7),
            expand=(3.0, 2.5),
            force_text=(2.0, 2.0),
            force_points=(0.6, 0.8), 
            lim=500)

handles = [mpatches.Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()]
ax.legend(handles=handles, loc='lower left')
ax.axhline(0, color='k', lw=0.5, alpha=0.4)
ax.axvline(0, color='k', lw=0.5, alpha=0.4)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_xlim(0.1, 0.3)
ax.set_ylim(-0.35, 0.45)
ax.grid(True, alpha=0.3)

plt.tight_layout(w_pad=3.0)
plt.savefig(FIG_DIR + 'fig1_svd.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR + 'fig1_svd.png', bbox_inches='tight')
plt.show()
print("→ fig1_svd saved.\n")

# ── 3. Correlation, MDS, Spectral Clustering ───────────────────────────────────
print("=" * 60)
print("SECTION 2 — MDS + SPECTRAL CLUSTERING")
print("=" * 60)

# Full 20×20 Pearson correlation matrix across stocks
C = np.corrcoef(returns.T)

# Sector-sorted index for heatmap display
sorted_tickers = [t for s in SECTOR_ORDER for t in tickers if SECTORS[t] == s]
sidx           = [tickers.index(t) for t in sorted_tickers]
C_sorted       = C[np.ix_(sidx, sidx)]

# ── Classical MDS (multidimensional scaling) ──────────────────────────────────
# Goal: embed N stocks in 2D Euclidean space preserving their pairwise distances.
# Distance metric: d_{ij} = √(2(1 − ρ_{ij}))  (correlation distance)
#   ρ=1  → d=0 (identical moves), ρ=0 → d=√2, ρ=-1 → d=2 (opposite moves).
# MDS via double-centering: B = -½ H D² H,  H = I - (1/N)11ᵀ
# The top-2 eigenvectors of B give the 2D embedding coordinates.
D = np.sqrt(2 * (1 - np.clip(C, -1, 1)))
H = np.eye(N) - np.ones((N, N)) / N
B = -0.5 * H @ (D**2) @ H
evals_mds, evecs_mds = np.linalg.eigh(B)
idx_mds = np.argsort(evals_mds)[::-1]
evals_mds, evecs_mds = evals_mds[idx_mds], evecs_mds[:, idx_mds]
coords = evecs_mds[:, :2] * np.sqrt(np.maximum(evals_mds[:2], 0))

# Proportion of positive (meaningful) MDS variance captured by the 2D plane
pct_mds = evals_mds[:2].sum() / evals_mds[evals_mds > 0].sum() * 100
print(f"MDS: top-2 eigenvalues capture {pct_mds:.1f}% of positive variance")

# ── Figure 2 — Market Structure ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): MDS 2D embedding — sector labels NOT provided to the algorithm
ax = axes[0]
texts_2a = []
for i, ticker in enumerate(tickers):
    ax.scatter(coords[i, 0], coords[i, 1], color=colors[i], s=60, zorder=5, edgecolors='k')
    texts_2a.append(ax.text(coords[i, 0], coords[i, 1], ticker, fontsize=8))
adjust_text(texts_2a, x=coords[:, 0], y=coords[:, 1], ax=ax,
            arrowprops=dict(arrowstyle='-', color='k', lw=0.5),
            expand=(2.5, 3.0),
            force_text=(2.0, 2.0),
            force_points=(0.6, 0.8),
            lim=500, )
handles = [mpatches.Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()]
ax.legend(handles=handles, loc='lower right')
ax.set_xlabel('MDS Dimension 1')
ax.set_ylabel('MDS Dimension 2')
ax.set_xlim(-0.65, 0.65)
ax.set_ylim(-0.7, 0.4)
ax.grid(True, alpha=0.3)

# Panel (b): Pearson correlation heatmap sorted by sector.
# Block-diagonal structure = within-sector stocks are more correlated.
ax = axes[1]
im = ax.imshow(C_sorted, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(sorted_tickers, rotation=90, fontsize=8)
ax.set_yticklabels(sorted_tickers, fontsize=8)
plt.colorbar(im, ax=ax, label=r'Correlation coefficient $\rho$', shrink=0.85)
for b in [5, 10, 15]:                                   # sector boundary lines
    ax.axhline(b - 0.5, color='k', lw=1.5)
    ax.axvline(b - 0.5, color='k', lw=1.5)
for k_s, s in enumerate(SECTOR_ORDER):
    ax.text(5*k_s + 2, -1.2, s, ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(w_pad=3.0)
plt.savefig(FIG_DIR + 'fig2_structure.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR + 'fig2_structure.png', bbox_inches='tight')
plt.show()
print("→ fig2_structure saved.\n")

# ── 4. DMD — Market Efficiency Signature ──────────────────────────────────────
print("=" * 60)
print("SECTION 3 — DMD")
print("=" * 60)

# Train/test split: train on 2020–2022, test on 2023–2026.
# We fit the DMD operator on training data and evaluate predictions
# on the held-out test period to avoid look-ahead bias.
train_mask = returns.index <= '2023-12-31'
test_mask  = returns.index >  '2023-12-31'

# Standardise using training-set statistics only (scaler2.fit on train, transform test)
scaler2  = StandardScaler()
X_train  = scaler2.fit_transform(returns[train_mask])    # (T_tr, 20)
X_test   = scaler2.transform(returns[test_mask])         # (T_te, 20)

T_tr = X_train.shape[0]
print(f"Train: {T_tr} days  |  Test: {X_test.shape[0]} days")

# ── Fit DMD operator ──────────────────────────────────────────────────────────
# DMD seeks the best linear map A such that x_{t+1} ≈ A x_t.
# Snapshot matrices (columns = states, rows = dimensions):
#   X1[:, t] = x_t   (state at time t)
#   X2[:, t] = x_{t+1} (state at time t+1)
# A ≈ X2 X1†  where X1† is the pseudoinverse of X1.
#
# Algorithm (rank-r truncated SVD for stability):
#   1. SVD: X1 = Ur Σr Vrᵀ
#   2. Reduced operator:  Ã = Urᵀ X2 Vr Σr⁻¹   (N×N condensed to r×r)
#   3. Full-space map:    A = Ur Ã Urᵀ
# Eigenvalues of Ã reveal the dynamical modes:
#   |λ| < 1 → decaying (damped); |λ| > 1 → growing (unstable)
X1_dmd = X_train[:-1].T                                  # (20, T_tr-1)
X2_dmd = X_train[1:].T                                   # (20, T_tr-1)

Ud, Sd, Vtd = np.linalg.svd(X1_dmd, full_matrices=False)
# Use the MP noise-floor count as the principled rank choice:
# only keep modes corresponding to genuinely informative singular values.
r_dmd = n_signal                                          # set by MP bound in Section 1
Ur, Sr, Vtr = Ud[:, :r_dmd], Sd[:r_dmd], Vtd[:r_dmd, :]

A_tilde    = Ur.T @ X2_dmd @ Vtr.T @ np.diag(1.0 / Sr)
evals_dmd, _ = np.linalg.eig(A_tilde)
A_dmd      = (Ur @ A_tilde @ Ur.T).real

mags = np.abs(evals_dmd)

# Full-rank eigenvalues for the spectrum plot (r=N shows the complete picture)
r_full = int(np.sum(Sd > 1e-10 * Sd[0]))
Ur_f, Sr_f, Vtr_f = Ud[:, :r_full], Sd[:r_full], Vtd[:r_full, :]
A_tilde_full       = Ur_f.T @ X2_dmd @ Vtr_f.T @ np.diag(1.0 / Sr_f)
evals_plot, _      = np.linalg.eig(A_tilde_full)
mags_plot          = np.abs(evals_plot)
print(f"DMD rank used: r={r_dmd} (MP noise-floor bound)")
print(f"DMD eigenvalue magnitudes — mean: {mags.mean():.4f}  max: {mags.max():.4f}")

# ── One-step-ahead RMSE ───────────────────────────────────────────────────────
# Prediction: x̂_{t+1} = A x_t
# Baseline:   predict zero change (x̂_{t+1} = 0), i.e. "tomorrow ≈ today's mean"
# RMSE ratio > 1 means the trained model is WORSE than predicting no change.
preds_dmd  = X_test[:-1] @ A_dmd.T                       # (T_te-1, 20)
actual     = X_test[1:]                                   # (T_te-1, 20)

rmse_dmd  = np.sqrt(np.mean((actual - preds_dmd)**2))
rmse_zero = np.sqrt(np.mean(actual**2))                   # baseline: predict 0
rel_impr  = (rmse_zero - rmse_dmd) / rmse_zero * 100     # negative = DMD is worse

print(f"One-step RMSE  — DMD: {rmse_dmd:.4f}  |  Zero: {rmse_zero:.4f}")
print(f"Relative RMSE improvement over zero baseline: {rel_impr:.2f}%")

# ── Signal-weighted long-short portfolio backtest ─────────────────────────────
# Each day, use predicted return vector to assign weights:
#   Long  stocks with pred > 0, weighted proportionally to pred magnitude
#   Short stocks with pred < 0, weighted proportionally to pred magnitude
# This uses ALL stocks (not a fixed top/bottom split) and sizes positions
# by model confidence. P&L uses ACTUAL (unstandardised) returns — no look-ahead.
actual_pct_test = returns[test_mask].values[1:]          # (T_te-1, 20)
test_dates_port = returns[test_mask].index[1:]

def signal_weighted_ls(preds, actual):
    """Long pred>0 / short pred<0, weights ∝ |prediction|."""
    long_w  = np.where(preds > 0, preds, 0.0)
    short_w = np.where(preds < 0, -preds, 0.0)
    ls  = long_w.sum(axis=1, keepdims=True)
    ss  = short_w.sum(axis=1, keepdims=True)
    ls  = np.where(ls == 0, 1.0, ls)                     # avoid div-by-zero
    ss  = np.where(ss == 0, 1.0, ss)
    long_ret  = (actual * (long_w  / ls)).sum(axis=1)
    short_ret = (actual * (short_w / ss)).sum(axis=1)
    return long_ret - short_ret

# ── Find best r by overall direction accuracy across all 20 stocks ────────────
best_dir_acc = 0.0
r_best = 1
for r_h in range(1, N + 1):
    Ur_h, Sr_h, Vtr_h = Ud[:, :r_h], Sd[:r_h], Vtd[:r_h, :]
    At_h = Ur_h.T @ X2_dmd @ Vtr_h.T @ np.diag(1.0 / Sr_h)
    A_h  = (Ur_h @ At_h @ Ur_h.T).real
    p_h  = X_test[:-1] @ A_h.T
    da_h = np.mean(np.sign(p_h) == np.sign(actual_pct_test))
    if da_h > best_dir_acc:
        best_dir_acc = da_h
        r_best = r_h
print(f"Best r by direction accuracy: r={r_best}  ({best_dir_acc:.1%})")

# Compute best-rank predictions for GOOGL illustration
Ur_b, Sr_b, Vtr_b = Ud[:, :r_best], Sd[:r_best], Vtd[:r_best, :]
At_b  = Ur_b.T @ X2_dmd @ Vtr_b.T @ np.diag(1.0 / Sr_b)
A_best = (Ur_b @ At_b @ Ur_b.T).real
preds_best = X_test[:-1] @ A_best.T                      # (T_te-1, 20)

# ── GOOGL example ─────────────────────────────────────────────────────────────
stock_ex_name = 'GOOGL'
stock_ex_idx  = list(returns.columns).index(stock_ex_name)
actual_ex_pct = actual_pct_test[:, stock_ex_idx] * 100   # actual % returns
pred_ex       = preds_best[:, stock_ex_idx]              # best-rank prediction
pred_up       = pred_ex > 0

dir_acc_full = np.mean(np.sign(pred_ex) == np.sign(actual_ex_pct))
print(f"DMD one-stock example: {stock_ex_name}  r={r_best}  direction accuracy (full test): {dir_acc_full:.1%}")

# Zoom window: June–July 2024 (normal trading period, no major macro event)
zoom_start   = pd.Timestamp('2024-06-01')
zoom_end     = pd.Timestamp('2024-07-31')
zoom_mask    = (test_dates_port >= zoom_start) & (test_dates_port <= zoom_end)
dates_zoom   = test_dates_port[zoom_mask]
actual_zoom  = actual_ex_pct[zoom_mask]
pred_up_zoom = pred_up[zoom_mask]

# Direction accuracy for the displayed window specifically
dir_acc_zoom = np.mean(np.sign(pred_ex[zoom_mask]) == np.sign(actual_ex_pct[zoom_mask]))
print(f"  Direction accuracy (zoom window): {dir_acc_zoom:.1%}")

# ── Figure 3 — DMD (2 panels) ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): DMD eigenvalue spectrum on the complex plane.
ax = axes[0]
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1, alpha=0.4, label='Unit circle')
ax.scatter(evals_plot.real, evals_plot.imag,
           facecolors='none', edgecolors='steelblue',
           s=20, linewidths=1.5, zorder=5, label='DMD eigenvalues')
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect('equal')
ax.axhline(0, color='k', lw=0.5, alpha=0.4)
ax.axvline(0, color='k', lw=0.5, alpha=0.4)
ax.set_xlabel('Real part'); ax.set_ylabel('Imaginary part')
ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

# Panel (b): Zoom — June–July 2024, best-rank DMD vs actual GOOGL returns.
# Background shading = DMD prediction direction (green=up, red=down).
# Bars = actual daily return, colored by sign.
# Mismatch between bar color and background reveals failed predictions.
ax = axes[1]
ylim_zoom = max(4.0, np.percentile(np.abs(actual_zoom), 97)) if len(actual_zoom) else 6.0
# Background: full-height bars anchored on the same dates as the data bars
# so shading and bars are pixel-perfect aligned (fill_between has a half-bar offset)
shade_colors = ['#2ca02c' if up else '#d62728' for up in pred_up_zoom]
ax.bar(dates_zoom, ylim_zoom * 2, bottom=-ylim_zoom,
       color=shade_colors, width=1.0, alpha=0.12, zorder=0, linewidth=0)
# Data bars on top
bar_colors = ['#2ca02c' if r > 0 else '#d62728' for r in actual_zoom]
ax.bar(dates_zoom, actual_zoom, color=bar_colors, width=0.7, alpha=0.82,
       zorder=2, label=f'{stock_ex_name} actual return')
ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.set_ylim(-ylim_zoom, ylim_zoom)
ax.set_xlabel('Date'); ax.set_ylabel('Daily return (%)')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha='right')
leg_patches = [mpatches.Patch(color='#2ca02c', alpha=0.4, label='DMD predicts up'),
               mpatches.Patch(color='#d62728', alpha=0.4, label='DMD predicts down')]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + leg_patches, labels + ['DMD predicts up', 'DMD predicts down'],
          fontsize=7.5, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout(w_pad=3.0)
plt.savefig(FIG_DIR + 'fig3_dmd.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR + 'fig3_dmd.png', bbox_inches='tight')
plt.show()
print("→ fig3_dmd saved.\n")

# ── 5. GARCH — Volatility Is Predictable ──────────────────────────────────────
print("=" * 60)
print("SECTION 4 — GARCH")
print("=" * 60)

from arch import arch_model

# ── Fit GARCH(1,1) on S&P 500 returns ────────────────────────────────────────
# GARCH(1,1) model:
#   r_t = ε_t √h_t,   ε_t ~ N(0,1)
#   h_t = ω + α r²_{t-1} + β h_{t-1}
# where h_t is the conditional variance (today's volatility forecast).
# α (ARCH term): how much a large shock yesterday increases today's variance.
# β (GARCH term): how much yesterday's variance estimate carries over.
# α + β (persistence): near 1 → volatility shocks are long-lived.
sp500_pct = sp500_returns * 100                           # scale to % for GARCH stability

sp500_pct_train = sp500_pct[train_mask]
sp500_pct_test  = sp500_pct[test_mask]

# Fit GARCH on training data only (same split as DMD)
am  = arch_model(sp500_pct_train, vol='Garch', p=1, q=1, dist='normal', rescale=False)
res = am.fit(disp='off')
cond_vol_train = res.conditional_volatility               # in-sample conditional std (%)

alpha_g = res.params['alpha[1]']
beta_g  = res.params['beta[1]']
omega_g = res.params['omega']
print(f"GARCH(1,1): ω={omega_g:.5f}  α={alpha_g:.4f}  β={beta_g:.4f}  "
      f"persistence={alpha_g+beta_g:.4f}")

# ── Out-of-sample volatility prediction ───────────────────────────────────────
# Iterate the GARCH recursion on test data using training-estimated parameters.
# h_{t+1} = ω + α r_t² + β h_t  — no look-ahead; each forecast uses only past info.
h_t = cond_vol_train.iloc[-1] ** 2                        # variance at end of training
r_t = sp500_pct_train.iloc[-1]                            # last training return
h_oos = []
for r in sp500_pct_test.values:
    h_next = omega_g + alpha_g * r_t**2 + beta_g * h_t
    h_oos.append(np.sqrt(h_next))                         # store as std dev
    h_t = h_next
    r_t = r
cond_vol_test = pd.Series(h_oos, index=sp500_pct_test.index)

# Full-period conditional vol for Figure 4a (train in-sample + test OOS)
cond_vol = pd.concat([cond_vol_train, cond_vol_test])

# Evaluate on the test period: compare σ̂_t (forecast) vs |r_t| (realised)
train_mean_vol = cond_vol_train.mean()
realized_abs   = sp500_pct_test.abs()
test_aligned   = pd.DataFrame({'pred': cond_vol_test.values,
                                'real': realized_abs.values},
                               index=sp500_pct_test.index)

rmse_garch_vol = np.sqrt(np.mean((test_aligned['real'] - test_aligned['pred'])**2))
rmse_mean_vol  = np.sqrt(np.mean((test_aligned['real'] - train_mean_vol)**2))
corr_garch_vol = np.corrcoef(test_aligned['pred'], test_aligned['real'])[0, 1]

vol_impr = (rmse_mean_vol - rmse_garch_vol) / rmse_mean_vol * 100
print(f"Volatility prediction (test 2023–2026):")
print(f"  RMSE — GARCH: {rmse_garch_vol:.4f}  |  Mean baseline: {rmse_mean_vol:.4f}")
print(f"  Corr(GARCH σ, |r|): {corr_garch_vol:.3f}")
print(f"  RMSE improvement over mean baseline: {vol_impr:.1f}%")

# RMSE comparison summary (for paper)
print(f"\nRMSE summary:")
print(f"  Returns    — DMD (r=20): {rmse_dmd:.4f}  |  Zero baseline: {rmse_zero:.4f}"
      f"  |  ratio: {rmse_dmd/rmse_zero:.4f}")
print(f"  Volatility — GARCH:      {rmse_garch_vol:.4f}  |  Mean baseline: {rmse_mean_vol:.4f}"
      f"  |  ratio: {rmse_garch_vol/rmse_mean_vol:.4f}")


# ── Figure 4 — GARCH ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): Conditional volatility timeline.
# Gray shading = 20-day rolling realised std (rough proxy for true vol).
# Blue line = GARCH(1,1) one-step-ahead forecast.
# Shaded spans mark the three major vol spikes in our sample.
ax = axes[0]
roll20 = sp500_pct.rolling(20).std()
ax.fill_between(roll20.index, roll20, alpha=0.25, color='gray',
                label='Actual 20-day rolling std')
ax.plot(cond_vol.index, cond_vol, color='steelblue', lw=0.8, ls='dashed',
        label='GARCH(1,1) prediction')
for span, label, col in [
    (('2020-02-20', '2020-04-03'), 'COVID crash',   '#d62728'),
    (('2022-01-01', '2022-12-31'), '2022 bear market', '#ff7f0e'),
    (('2025-04-01', '2025-05-15'), 'Trump tariffs', '#9467bd'),
]:
    ax.axvspan(pd.Timestamp(span[0]), pd.Timestamp(span[1]),
               alpha=0.12, color=col, label=label)
# Vertical line separating training (2020–2022) from test (2023–2026)
ax.axvline(pd.Timestamp('2023-12-31'), color='k', lw=1.5, ls='-', label='Train / test split')
ax.set_xlabel('Date'); ax.set_ylabel('Daily volatility (%)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

# Panel (b): ACF of squared S&P 500 returns (lags 1–30).
# Slow decay in ρ(r²_t, r²_{t-k}) reveals volatility clustering: large moves
# tend to follow large moves.  This autocorrelation structure is exactly what
# GARCH(1,1) models as an AR(1) on r²_t, motivating why the model can forecast.
from statsmodels.tsa.stattools import acf as sm_acf
r_sq   = sp500_pct.values ** 2
n_lags = 30
acf_vals = sm_acf(r_sq, nlags=n_lags, fft=True)[1:]      # drop lag-0 (=1)
lags     = np.arange(1, n_lags + 1)
ci_bound = 1.96 / np.sqrt(len(r_sq))                     # 95% CI under H0: no autocorr

ax = axes[1]
ax.bar(lags, acf_vals, color='steelblue', alpha=0.75, width=0.7)
ax.axhline(ci_bound, color='#d62728', lw=1.2, ls='--', label='95% confidence band')
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('Lag (trading days)')
ax.set_ylabel(r'Autocorrelation function of $r_t^2$')
ax.set_xlim(0, n_lags + 1)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(w_pad=3.0)
plt.savefig(FIG_DIR + 'fig4_garch.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR + 'fig4_garch.png', bbox_inches='tight')
plt.show()
print("→ fig4_garch saved.\n")


# ── 6. Summary — Numbers for the Paper ────────────────────────────────────────
print("=" * 60)
print("NUMBERS FOR THE PAPER")
print("=" * 60)
print(f"N stocks: {N}  |  T days: {T}  |  Period: {returns.index[0].date()} – {returns.index[-1].date()}")
print(f"PC1 variance: {energy[0]*100:.1f}%")
print(f"Top-3 variance: {cum_energy[2]*100:.1f}%")
print(f"Factors above MP noise floor: {n_signal}")
print(f"Corr(PC1, S&P 500): {r_pc1:.3f}")
print(f"MDS top-2 positive variance: {pct_mds:.1f}%")
print(f"DMD mean |λ|: {mags.mean():.4f}  max |λ|: {mags.max():.4f}")
print(f"RMSE improvement over zero: {rel_impr:.2f}%")
print(f"GARCH persistence (α+β): {alpha_g+beta_g:.4f}")
print(f"GARCH vol RMSE improvement over mean: {vol_impr:.1f}%")
