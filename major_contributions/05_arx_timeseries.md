# Two-Stage Orthogonalized ARX Futures Pricing

To isolate genuine physical climate risk pricing from global macroeconomic trends, energy price shocks, currency fluctuations, and seasonal crop cycles, I implement a **two-stage Frisch–Waugh–Lovell (FWL) residualization and distributed-lag ARX estimation** framework across daily agricultural futures returns.

---

## Stage 1: Macro-Financial & Calendar Orthogonalization

### 1. Log-Differenced Financial Series
All financial variables are transformed into daily percentage log-returns ($r_t = 100 \times \Delta \ln P_t$), verified for covariance stationarity via **ADF** and **KPSS** tests:
* **Target Futures:** Arabica coffee (`KC=F`) and Soybean (`ZS=F`) front-month continuous futures returns.
* **Macro-Financial Controls ($\mathbf{M}_t$):** Central South American foreign exchange rates against `USD` (capturing export competitiveness and local producer margins) and `Brent Crude` oil returns in `USD` (capturing global transport, fertilizer, and biofuel substitution channels).

### 2. Calendar Fixed-Effects Annihilation & $\text{ARX}(10)$ Filtering
Before introducing weather shocks, we strip out seasonal harvesting cycles, macroeconomic regimes, and short-term financial autocorrelation in two sequential projections:

$$\tilde{R}_{1t} = \sum_{\ell = 1}^{10} \sum_{j = 1}^{6} \beta_{j\ell}\,\tilde{R}_{j,t-\ell} + u_t, \qquad t = 11, \dots, T$$

The OLS estimator is:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$$

and the Stage 1 residuals are:

$$\hat{u}_t = \tilde{R}_{1t} - \sum_{\ell = 1}^{10} \sum_{j = 1}^{6} \hat{\beta}_{j\ell}\,\tilde{R}_{j,t-\ell}$$

The residual $\hat{u}_t$ isolates the component of daily futures returns unexplained by calendar year and month dummies, the 10-day lagged history of the commodity itself, and the macro-financial control contracts (FX rates and Brent crude). This series serves as the dependent variable in Stage 2, ensuring that regressing $\hat{u}_t$ on local weather anomalies identifies the causal climate response function net of macro-financial and seasonal dynamics.

The resulting innovation series $\hat{e}_t$ represents **pure, unanticipated daily futures return shocks** orthogonal to calendar seasonality, macro-financial spillovers, and short-horizon momentum.

<details>
<summary><b>View Pseudocode: Calendar FWL Projection & ARX(10) Return Residualization</b></summary>

```python
def residualize_futures_returns(core, var_list, target="KC=F", nlags=10):
    """Partial out Month/Year FEs and 10 lags of macro-financial controls (FX, Brent, cross-commodities)."""
    # 1. Build full-rank calendar dummy matrix (Month + Year FEs, Feb 29 excluded)
    D = build_calendar_dummies(core, drop_first=True)
    assert np.linalg.matrix_rank(D) == D.shape[1]

    # 2. FWL projection of 100x log-returns (percentage points) onto calendar FEs
    X_pp = core[var_list].to_numpy() * 100
    Z_tilde = X_pp - D @ np.linalg.lstsq(D, X_pp, rcond=None)[0]

    # 3. Build 10-lag macro-financial design matrix & extract orthogonal return innovations
    lagged_df = build_lag_matrix(Z_tilde, cols=var_list, max_lag=nlags).dropna()
    X_macro   = lagged_df[macro_lag_cols].to_numpy()
    y_target  = lagged_df[[target]].to_numpy()

    beta_macro = np.linalg.lstsq(X_macro, y_target, rcond=None)[0]
    lagged_df["return_res"] = y_target - (X_macro @ beta_macro)
    return lagged_df
</details>

The climate variables are measures of mean and anomalous vapour pressure deficit and precipitation flux.
Second, the climate time series is orthogonalised, on year and month dummies.

<details>
  ```python
      def array_projection_matrix(climate_ar,ndays,nyears,projection="both"):   
          csum_days_in_month=np.cumsum(np.array([0,31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
          day = np.arange(365)[:, None]                                   # (365, 1)
          month_dummies = ((day >= csum_days_in_month[:-1]) &
                           (day <  csum_days_in_month[1:])).astype(float)  # (365, 12)
          month_dummies = np.vstack([month_dummies]*nyears) ## (7665,12)
          print(f"{month_dummies.shape[0] == climate_ar.shape[0]}") ## True
          csum_days_in_year=np.arange(0,climate_ar.shape[0]+1,step=365)
          day=np.arange(climate_ar.shape[0])[:,None]
          year_dummies = ((day >= csum_days_in_year[:-1]) &
                           (day <  csum_days_in_year[1:])).astype(float) # (7665,21)
          print(f"{year_dummies.shape[0]==climate_ar.shape[0]}") #True
          if projection=="month": 
              all_dummies=month_dummies
          elif projection=="year":
              all_dummies=year_dummies
          elif projection=="month and year":
              None ## 12 *21 from month_dummies before vstack
          elif projection=="both":
              all_dummies = np.hstack([month_dummies,year_dummies]) ## (7665,33)
          coef = np.linalg.solve(all_dummies.T @ all_dummies, all_dummies.T @ climate_ar)  # 7665X12
          climate_ar_tilde = climate_ar - all_dummies @ coef
          return climate_ar_tilde
  ```
  
</details>

### Step 2: Distributed-Lag $\text{ARX}(L=30)$ Climate Impulse Response

Using the Stage 1 macro-financial return innovations $\hat{u}_t$ and the calendar-orthogonalized sub-regional climate anomaly matrix $\mathbf{X}_t$, we estimate a 30-trading-day autoregressive distributed-lag ($\text{ARX}, L=30$) specification:

$$\hat{u}_t = \mu + \sum_{i=1}^{30} \alpha_i \hat{u}_{t-i} + \sum_{j=1}^{30} \boldsymbol{\beta}_j' \mathbf{X}_{t-j} + \epsilon_t$$

To trace both immediate price discovery and persistent supply-shock accumulation over a six-week trading window, we compute the **cumulative impulse response** $\hat{\boldsymbol{\beta}}_c$ at every horizon $c$:

$$\hat{\boldsymbol{\beta}}_c = \sum_{j=1}^{c} \hat{\boldsymbol{\beta}}_j \qquad \text{for all } c = 1, \dots, 30$$

The model is estimated via OLS with **Newey–West (Bartlett kernel) Heteroskedasticity and Autocorrelation Consistent (HAC)** standard errors on $\hat{\epsilon}_t$, capturing serial dependence across overlapping trading windows.

<details>
  
full code: area_of_effects/code_blocks/arx30_timeseries.py

