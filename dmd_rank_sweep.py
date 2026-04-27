"""
DMD rank sweep: compare r=1..20 on RMSE, eigenvalue stats, and actual portfolio returns.
Run from: /Users/damoncht/Desktop/math547/proj/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "study1_market_data.csv"
FIG_DIR   = "report/figures/"

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
data = df.drop(columns=['SP500'])
returns = data.pct_change().dropna()

train_mask = returns.index <= '2022-12-31'
test_mask  = returns.index >  '2022-12-31'

scaler = StandardScaler()
X_train = scaler.fit_transform(returns[train_mask])
X_test  = scaler.transform(returns[test_mask])

X1_dmd = X_train[:-1].T   # (20, T_tr-1)
X2_dmd = X_train[1:].T    # (20, T_tr-1)
actual_std = X_test[1:]    # standardised (T_te-1, 20)

# Actual pct returns for portfolio P&L
actual_pct = returns[test_mask].values[1:]    # (T_te-1, 20)
test_dates = returns[test_mask].index[1:]

Ud, Sd, Vtd = np.linalg.svd(X1_dmd, full_matrices=False)
rmse_zero = np.sqrt(np.mean(actual_std**2))

# Equal-weight benchmark
bench_daily = actual_pct.mean(axis=1)
cum_bench   = (1 + bench_daily).cumprod() - 1
total_bench = cum_bench[-1] * 100
sharpe_bench = bench_daily.mean() / bench_daily.std() * np.sqrt(252)

N      = X_train.shape[1]
n_side = N // 2   # 10 long, 10 short

print(f"Zero-baseline RMSE:  {rmse_zero:.4f}")
print(f"EW benchmark cumret: {total_bench:+.1f}%  Sharpe: {sharpe_bench:.3f}\n")
print(f"{'r':>3}  {'RMSE Δ%':>8}  {'cumret%':>8}  {'Sharpe':>7}  {'mean|λ|':>8}  {'max|λ|':>7}")

results = []
port_curves = {}

for r in range(1, N + 1):
    Ur, Sr, Vtr = Ud[:, :r], Sd[:r], Vtd[:r, :]
    A_tilde = Ur.T @ X2_dmd @ Vtr.T @ np.diag(1.0 / Sr)
    evals, _ = np.linalg.eig(A_tilde)
    A_dmd = (Ur @ A_tilde @ Ur.T).real

    preds = X_test[:-1] @ A_dmd.T              # standardised predictions
    rmse  = np.sqrt(np.mean((actual_std - preds)**2))
    rmse_rel = (rmse - rmse_zero) / rmse_zero * 100

    # Long top-10 / short bottom-10 by predicted return each day
    rank_idx  = np.argsort(preds, axis=1)
    rows      = np.arange(len(preds))[:, None]
    long_ret  = actual_pct[rows, rank_idx[:, -n_side:]].mean(axis=1)
    short_ret = actual_pct[rows, rank_idx[:,  :n_side]].mean(axis=1)
    port_daily = long_ret - short_ret

    cum_port   = (1 + port_daily).cumprod() - 1
    total_port = cum_port[-1] * 100
    sharpe     = port_daily.mean() / port_daily.std() * np.sqrt(252)

    mags = np.abs(evals)
    results.append({
        'r': r, 'rmse_rel': rmse_rel,
        'cumret': total_port, 'sharpe': sharpe,
        'mean_mag': mags.mean(), 'max_mag': mags.max(),
    })
    port_curves[r] = cum_port * 100

    print(f"{r:>3}  {rmse_rel:>+8.2f}%  {total_port:>+8.1f}%  {sharpe:>7.3f}  "
          f"{mags.mean():>8.4f}  {mags.max():>7.4f}")

df_res = pd.DataFrame(results)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: RMSE penalty vs zero baseline
ax = axes[0]
ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7, label='Zero baseline')
ax.plot(df_res['r'], df_res['rmse_rel'], 'o-', color='steelblue', ms=5, lw=1.8)
ax.set_xlabel('DMD rank  $r$')
ax.set_ylabel('RMSE vs zero baseline (%)')
ax.set_title('Prediction error')
ax.set_xticks(range(1, N+1))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Panel 2: L/S portfolio cumulative return
ax = axes[1]
cmap = plt.get_cmap('coolwarm', N)
for r in range(1, N+1):
    alpha = 0.4 if r not in (1, 3, 10, 20) else 1.0
    lw    = 0.8 if r not in (1, 3, 10, 20) else 1.8
    ax.plot(test_dates, port_curves[r], color=cmap(r-1), alpha=alpha, lw=lw,
            label=f'r={r}' if r in (1, 3, 10, 20) else None)
ax.plot(test_dates, cum_bench * 100, color='black', lw=2, ls='--',
        label=f'EW benchmark ({total_bench:+.0f}%)')
ax.axhline(0, color='k', lw=0.6, ls=':', alpha=0.4)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative return (%)')
ax.set_title('L/S portfolio actual returns')
ax.legend(fontsize=7.5, loc='lower left')
ax.grid(True, alpha=0.3)

# Panel 3: cumulative return and Sharpe vs r
ax = axes[2]
ax2 = ax.twinx()
ax.plot(df_res['r'], df_res['cumret'], 'o-', color='steelblue', ms=5, lw=1.8,
        label='Cumulative return (%)')
ax.axhline(total_bench, color='black', ls='--', lw=1.2,
           label=f'EW benchmark ({total_bench:+.0f}%)')
ax.axhline(0, color='gray', ls=':', lw=1, alpha=0.6)
ax2.plot(df_res['r'], df_res['sharpe'], 's--', color='#d62728', ms=5, lw=1.4,
         label='Sharpe (right)')
ax2.axhline(sharpe_bench, color='#d62728', ls=':', lw=1, alpha=0.6)
ax.set_xlabel('DMD rank  $r$')
ax.set_ylabel('Cumulative return (%)', color='steelblue')
ax2.set_ylabel('Annualised Sharpe', color='#d62728')
ax.set_title('Portfolio performance vs rank')
ax.set_xticks(range(1, N+1))
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7.5, loc='lower left')
ax.grid(True, alpha=0.3)

plt.suptitle('DMD rank sweep  (train 2020–2022, test 2023–)', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR + 'fig_dmd_rank_sweep.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR + 'fig_dmd_rank_sweep.png', bbox_inches='tight')
plt.show()

best_r_ret = df_res.loc[df_res['cumret'].idxmax(), 'r']
best_r_sr  = df_res.loc[df_res['sharpe'].idxmax(), 'r']
print(f"\nBest r by cumulative return: r={best_r_ret}  "
      f"({df_res.loc[df_res['cumret'].idxmax(), 'cumret']:+.1f}%)")
print(f"Best r by Sharpe:            r={best_r_sr}  "
      f"(Sharpe={df_res.loc[df_res['sharpe'].idxmax(), 'sharpe']:.3f})")
print(f"EW benchmark: {total_bench:+.1f}%  Sharpe={sharpe_bench:.3f}")
print("\n→ fig_dmd_rank_sweep saved.")
