"""
GARCH(1,1) fitted to each of the 20 stocks individually.
Compares out-of-sample volatility forecasting performance stock by stock,
then aggregates — a fairer analogue to the DMD comparison which also uses all 20.

Train: 2020-01-01 to 2022-12-31
Test:  2023-01-01 to 2026-04-24
"""

import pandas as pd
import numpy as np
from arch import arch_model

DATA_PATH = "study1_market_data.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
data   = df.drop(columns=["SP500"])
sp500  = df["SP500"]

returns = data.pct_change().dropna() * 100  # in percent, same scale arch expects

train_mask = returns.index <= "2022-12-31"
test_mask  = returns.index >  "2022-12-31"

print(f"Train: {train_mask.sum()} days  |  Test: {test_mask.sum()} days")
print(f"Stocks: {list(returns.columns)}\n")

results = []

for ticker in returns.columns:
    r = returns[ticker]
    r_train = r[train_mask]
    r_test  = r[test_mask]

    # Fit GARCH(1,1) on training data
    am  = arch_model(r_train, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")

    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta  = res.params["beta[1]"]
    persistence = alpha + beta

    # One-step-ahead volatility forecast on test set
    # Use rolling 1-step forecasts starting from end of training
    full_series = r[r.index <= r_test.index[-1]]
    am_full = arch_model(full_series, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res_full = am_full.fit(disp="off", starting_values=res.params.values)

    cond_vol = res_full.conditional_volatility  # σ_t for full period
    cond_vol_test = cond_vol[test_mask]

    # Realised volatility proxy: |r_t|
    realized_abs = r_test.abs()
    aligned = pd.DataFrame({"pred": cond_vol_test, "real": realized_abs}).dropna()

    train_mean_vol = r_train.abs().mean()  # historical mean baseline

    rmse_garch = np.sqrt(np.mean((aligned["pred"] - aligned["real"]) ** 2))
    rmse_mean  = np.sqrt(np.mean((train_mean_vol - aligned["real"]) ** 2))
    rmse_ratio = rmse_garch / rmse_mean   # < 1 means GARCH beats mean baseline

    corr = np.corrcoef(aligned["pred"], aligned["real"])[0, 1]

    results.append({
        "ticker":       ticker,
        "omega":        omega,
        "alpha":        alpha,
        "beta":         beta,
        "persistence":  persistence,
        "rmse_garch":   rmse_garch,
        "rmse_mean":    rmse_mean,
        "rmse_ratio":   rmse_ratio,
        "corr":         corr,
    })

    print(f"{ticker:6s}  α={alpha:.3f}  β={beta:.3f}  pers={persistence:.3f}  "
          f"RMSE ratio={rmse_ratio:.3f}  corr(σ,|r|)={corr:.3f}")

df_res = pd.DataFrame(results).set_index("ticker")

print("\n── Aggregate across 20 stocks ──────────────────────────────")
print(f"Mean persistence (α+β):  {df_res['persistence'].mean():.3f}  "
      f"(range {df_res['persistence'].min():.3f}–{df_res['persistence'].max():.3f})")
print(f"Mean RMSE ratio:         {df_res['rmse_ratio'].mean():.3f}  "
      f"(stocks where GARCH wins: {(df_res['rmse_ratio'] < 1).sum()}/20)")
print(f"Mean corr(σ_t, |r_t|):  {df_res['corr'].mean():.3f}")

# Compare with S&P 500 GARCH (from main analysis)
sp500_pct = sp500.pct_change().dropna() * 100
am_sp = arch_model(sp500_pct[train_mask], vol="Garch", p=1, q=1, dist="normal", rescale=False)
res_sp = am_sp.fit(disp="off")
am_sp_full = arch_model(sp500_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
res_sp_full = am_sp_full.fit(disp="off", starting_values=res_sp.params.values)
cond_vol_sp = res_sp_full.conditional_volatility[test_mask]
realized_sp = sp500_pct[test_mask].abs()
aligned_sp  = pd.DataFrame({"pred": cond_vol_sp, "real": realized_sp}).dropna()
mean_sp = sp500_pct[train_mask].abs().mean()
rmse_sp_garch = np.sqrt(np.mean((aligned_sp["pred"] - aligned_sp["real"]) ** 2))
rmse_sp_mean  = np.sqrt(np.mean((mean_sp - aligned_sp["real"]) ** 2))

print(f"\nS&P 500 GARCH RMSE ratio (for reference): {rmse_sp_garch/rmse_sp_mean:.3f}")
print(f"  α={res_sp.params['alpha[1]']:.3f}  β={res_sp.params['beta[1]']:.3f}  "
      f"pers={res_sp.params['alpha[1]']+res_sp.params['beta[1]']:.3f}")
