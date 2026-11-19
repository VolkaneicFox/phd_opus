# Dynamic Rolling Baselines & Accelerated Warming

Evaluating modern weather shocks against static historical climatologies conflates **secular regional warming** with stationary tail extremes. Across both macro-regional panels and high-resolution commodity grids, I construct **adaptive, lagged rolling baselines** to isolate true unanticipated physical shocks.

| Dimension | Germany (County Panel) | Central South America ($0.1^\circ$ Grid) |
| :--- | :--- | :--- |
| **Target Window** | $2005\text{--}2019$ (Seasonal) | $2005\text{--}2025$ (Daily) |
| **Reference Window** | $30\text{-year}$ rolling window, $5\text{-year}$ lag | $20\text{-year}$ rolling window, $5\text{-year}$ lag ($\pm 5\text{ days}$) |
| **Compute Engine** | Vectorized `Pandas` | Out-of-core `Zarr` + `CuPy` (GPU VRAM) |
| **Headline Finding** | Up to **$0.24\text{pp}$** annual growth loss from heat intensity | **$+0.11\text{pp}$** Soybean & **$+0.14\text{pp}$** Arabica return spikes |

---

## 1. Seasonal Rolling Baselines — German Counties
> **Reference Design:** $\text{Season}_{t} \;\text{vs.}\; \big[t - 35,\; t - 5\big]$
* **Region-Secular:** Every county is benchmarked strictly against its own local climate history.
* **Trend-Agnostic:** Slowly evolving baseline shifts are absorbed into updated quantile thresholds.
* **Season-Sensitive:** Summer anomalies are evaluated strictly against historical summers.
* **Leakage Prevention:** A 5-year buffer prior to year $t$ prevents contemporaneous contamination.
* **Intensity (`how_high`) vs. Frequency (`is_high`):** Isolating exceedance magnitude reveals up to **$0.24\text{pp}$** annual regional growth loss driven by heat *intensity*.

<details>
<summary><b>View Python Implementation (Pandas County Threshold Engine)</b></summary>

```python
for year in range(main_start_year, main_end_year + 1):
    target_df = df[df["year"] == year].copy()
    hist_end = year - gap_years
    hist_df = df[df["year"].between(hist_end - 30, hist_end)]

    # Wet-day conditioning for precipitation
    if var == "prec":
        hist_df = hist_df[hist_df[var] > 0]

    # Season-specific 5th & 95th percentile thresholds
    bounds = hist_df.groupby(period)[var].quantile([0.05, 0.95]).unstack()
    bounds.columns = [f"{var}_lqtl", f"{var}_uqtl"]
    target_df = target_df.merge(bounds, on=period)

    # Flag tail events & compute exceedance intensity beyond threshold
    target_df[f"{var}_is_high"] = (target_df[var] >= target_df[f"{var}_uqtl"]).astype(int)
    target_df[f"{var}_how_high"] = (target_df[var] - target_df[f"{var}_uqtl"]).clip(lower=0)
```
**Main Code Files:**
1. [Constructing the Moving Thresholds for each County](../stranger_weather/make_anomalies.py)
2. [Compressing the Daily Anomaly Data to Annual Measures](../stranger_weather/make_periodly_anomalies.py)

   
## 2. Day-to-Day Rolling Baseline — Central South America
> **Reference Design:** $\text{Day}_{d,t} \;\text{vs.}\; \big[t - 25,\; t - 5\big] \times \big[d - 5,\; d + 5\big]$
* **Grid-Secular:** Pixel-level ($0.1^\circ$) baseline calibration across 13,000+ daily `AgERA5` reanalysis grids.
* **Trend Agnostic:** Circumvents parametric trend specification; residual breach drift provides evidence of accelerated regional dryness
* **Day-Sensitive:** Pooling a tight 11-day calendar window ($\pm 5\text{ days}$) eliminates arbitrary seasonal boundary jumps.
* **Intense vs. Dispersed Margins:** Arabica returns ($+0.08\text{pp}$ to $+0.14\text{pp}$) react to acute shock **intensity**, whereas Soybean returns ($+0.06\text{pp}$ to $+0.11\text{pp}$) respond to **spatially dispersed** dryness.
  
<details>
<summary><b> Python GPU-Accelerated Gridded Anomaly & Threshold Engine (Zarr + CuPy)</b></summary>

```python
def wet_day_quantile(grid_cube: cupy.ndarray, q: float) -> cupy.ndarray:
    """Compute pixel-wise quantiles strictly over wet days (precip > 0) on GPU."""
    wet_vals = cupy.where(grid_cube > 0, grid_cube, cupy.nan)
    n_wet    = cupy.sum(grid_cube > 0, axis=0)
    sorted_v = cupy.sort(wet_vals, axis=0)
    idx      = cupy.clip((n_wet * q).astype(int), 0, cupy.maximum(n_wet - 1, 0))
    return cupy.where(n_wet > 0, cupy.take_along_axis(sorted_v, idx[None, ...], axis=0)[0], 0.0)


def compute_rolling_grid_thresholds(zarr_store, target_years, window_days=5, lookback=20, gap=1):
    """Derive daily, pixel-level moving thresholds using +/- 5-day calendar pooling."""
    for year in target_years:
        # 1. Load lookback Zarr slice [year - gap - lookback, year - gap] onto VRAM
        hist_gpu = cupy.asarray(zarr_store.sel(year=slice(year - gap - lookback, year - gap)))

        # 2. Pool +/- 5 surrounding calendar days across all lookback years per MM-DD
        for mm_dd in unique_calendar_days:
            pooled_sample = hist_gpu[get_centered_window_indices(mm_dd, margin=window_days)]
            
            thresholds[year, mm_dd] = (
                wet_day_quantile(pooled_sample, q=0.95) if is_precip
                else cupy.quantile(pooled_sample, q=0.95, axis=0)
            )
        free_gpu_vram()


def extract_breach_events_batched(obs_zarr, thresh_zarr, batch_size=1000):
    """Stream 3D climate cubes in batches to flag binary breaches & exceedance intensity."""
    for batch in chunk_along_time(obs_zarr, thresh_zarr, batch_size):
        obs_gpu, thr_gpu = cupy.asarray(batch.obs), cupy.asarray(batch.thresh)

        is_breach  = obs_gpu > thr_gpu                                      # Extensive margin (0/1)
        breach_mag = cupy.where(is_breach, obs_gpu - thr_gpu, 0.0)          # Intensive margin (severity)

        write_compressed_netcdf(is_breach.get(), breach_mag.get())
        free_gpu_vram()
```
**Main Code Files:**
1. [Zarr and CuPy Anomaly Detection](../area_of_effects/code_blocks/grid_level_anomaly_making.py)
2. [Transforming Grid Anomalies to Sub-regional Time Series](../area_of_effects/code_blocks/grid_level_anomaly_shock.py)
   * [Helper Functions for Sub-regional Aggregation](../area_of_effects/code_blocks/grid_to_quadrants.py)


