# Spectral Analysis of U.S. Equity Markets

Final project for MATH/BIOINF/STATS 547. We apply data-driven linear-algebraic methods — SVD, MDS, DMD, and GARCH — to understand the structure and predictability of U.S. equity returns.

## Repository Structure

```
.
├── analysis.py           # SVD and MDS analysis, figure generation
├── dmd_rank_sweep.py     # DMD rank sweep (r=1..20) and direction accuracy
├── garch_20stocks.py     # GARCH(1,1) volatility forecasting for all 20 stocks
├── data.ipynb            # Data download and preprocessing notebook
├── study1_market_data.csv
└── report/
    ├── main.tex          # Paper source
    ├── main.pdf          # Compiled paper
    ├── reference.bib
    └── figures/          # Generated figures (SVD, MDS, DMD, GARCH)
```

## Methods

- **SVD**: Identifies 4 latent factors explaining 74.6% of variance; confirms low-dimensional market structure via Marchenko-Pastur law
- **MDS**: Maps stock correlation distances into a 2-D embedding that recovers sector clusters
- **DMD**: Fits a linear dynamical system to return sequences; all eigenvalues decay within one trading day, yielding ~50% directional accuracy (coin-flip)
- **GARCH(1,1)**: Forecasts volatility clustering; achieves 15.8% RMSE improvement over a constant-volatility baseline

## Data

Daily adjusted closing prices for 20 large-cap U.S. stocks across four sectors (Technology, Finance, Healthcare, Energy), spanning January 2, 2020 – April 24, 2026. Sourced from Yahoo Finance.

## Requirements

```
numpy, pandas, scipy, matplotlib, arch, yfinance
```
