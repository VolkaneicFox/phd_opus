# Crop-Area-Weighted Grid Aggregation & Spatial Partitioning

To translate high-resolution ($0.1^\circ$) gridded meteorological anomalies (`AgERA5`) into economically meaningful regional exposure indices across Central South America ($75^\circ\text{W}\text{--}35^\circ\text{W}$, $35^\circ\text{S}\text{--}5^\circ\text{S}$), I construct **production-area-weighted spatial aggregators** using `MAPSPAM` harvested cropland rasters.

---

## 1. Spatial Partitioning & Grid Alignment
* **Raster Harmonization:** Re-projected `MAPSPAM` crop-specific harvested area layers onto the exact spatial coordinate grid of Copernicus `AgERA5` reanalysis data ($0.1^\circ$ resolution), masking out oceans, rivers, and inland water bodies.
* **Four Equal-Span Quadrants:** Partitioned Central South America into four $20^\circ \times 15^\circ$ sub-regions to isolate core producing zones from peripheral regions:

| Quadrant | Longitude Bounds | Latitude Bounds | Agronomic Profile |
| :--- | :---: | :---: | :--- |
| **Quadrant 1 (SW)** | $75^\circ\text{W} \text{ to } 55^\circ\text{W}$ | $35^\circ\text{S} \text{ to } 20^\circ\text{S}$ | Sparse Arabica footprint *(high-volatility diagnostic)* |
| **Quadrant 2 (SE)** | $55^\circ\text{W} \text{ to } 35^\circ\text{W}$ | $35^\circ\text{S} \text{ to } 20^\circ\text{S}$ | **Core Brazilian production** (São Paulo, Paraná, Rio Grande do Sul) |
| **Quadrant 3 (NW)** | $75^\circ\text{W} \text{ to } 55^\circ\text{W}$ | $20^\circ\text{S} \text{ to } 5^\circ\text{S}$ | Amazonia / Western interior |
| **Quadrant 4 (NE)** | $55^\circ\text{W} \text{ to } 35^\circ\text{W}$ | $20^\circ\text{S} \text{ to } 5^\circ\text{S}$ | **Core Brazilian production** (Minas Gerais, Cerrado) |

---

## 2. Within-Quadrant Cropland Normalization
Within each quadrant $q \in \{1,2,3,4\}$, the weight $w_{g,q}$ assigned to grid cell $g$ is normalized strictly over **total active cropland area inside quadrant $q$** ($\sum_{g \in q} A_{g,q} = 1$), rather than total geographic surface area:

$$w_{g,q} = \frac{\text{Area}_{g,q}}{\sum_{g \in q} \text{Area}_{g,q}}, \qquad \text{Shock}_{q,t} = \sum_{g \in q} w_{g,q} \cdot \text{Anomaly}_{g,t}$$

### Why Normalize Over Cropland Instead of Total Land Area?
* **Prevents Micro-Weight Dilution:** Normalizing by total geographic area shrinks weights toward zero in quadrants with sparse agricultural land cover, artificially suppressing shock variance.
* **Cross-Quadrant Comparability & Volatility Diagnostics:** Preserving unit-sum weights ($\sum w_{g,q} = 1$) across every quadrant keeps regression coefficients on an identical scale—exposing small-sample volatility in low-cropland zones (such as **Quadrant 1** for Arabica coffee) so they can be cleanly isolated and excluded in ex-post robustness checks.

<details>
<summary><b>View Code: Quadrant Clipping & Cropland Weight Normalization</b></summary>

  ```python
      def partition_and_weights(self):        
        self.slice_1 = self.full_region.rio.clip_box(minx=-75, miny=-35, maxx=-55, maxy=-20).astype("float64")
        self.slice_2 = self.full_region.rio.clip_box(minx=-55, miny=-35, maxx=-35, maxy=-20).astype("float64")
        self.slice_3 = self.full_region.rio.clip_box(minx=-75, miny=-20, maxx=-55, maxy=-5).astype("float64")
        self.slice_4 = self.full_region.rio.clip_box(minx=-55, miny=-20, maxx=-35, maxy=-5).astype("float64")
        self.weights_1 = self.slice_1/self.slice_1.sum()
        self.weights_2 = self.slice_2/self.slice_2.sum()
        self.weights_3 = self.slice_3/self.slice_3.sum()
        self.weights_4 = self.slice_4/self.slice_4.sum()
        assert np.isclose(self.weights_1.sum().item(), 1.0)
        assert np.isclose(self.weights_2.sum().item(), 1.0)
        assert np.isclose(self.weights_3.sum().item(), 1.0)
        assert np.isclose(self.weights_4.sum().item(), 1.0)
  ```
  
</details>

## 3. Sub-Regional Climate Time Series Aggregation

Taking the Hadamard (element-wise) product of the normalized `MAPSPAM` cropland weights $W_{k,ij}$ and the daily pixel-level weather anomalies $C_{k,ij}(t)$ collapses the 3D climate cubes into daily, production-weighted sub-regional shock series $C_k(t)$ for each quadrant $k \in \{1,2,3,4\}$:

$$W_{k,ij} = \frac{Y_{k,ij}}{\sum_{i,j} Y_{k,ij}} \qquad \text{for quadrant } k \in \{1,2,3,4\}$$

$$C_k(t) = \sum_{i,j} W_{k,ij} \cdot C_{k,ij}(t)$$

Because $C_{k,ij}(t)$ is computed separately for **binary tail indicators** ($\mathbb{I}\{\text{breach}\}$) and **relative exceedance severity** ($%$ deviation beyond the 95th/5th percentile threshold), this aggregation yields five daily shock series per quadrant:

| Weather Variable | Physical Hazard | Intensive Margin (Severity) | Extensive Margin (Spatial Exposure) |
| :--- | :--- | :---: | :---: |
| **High VPD** (`> 95th pct`) | Atmospheric dryness / heat stress | Excess VPD ($%$ above threshold) | Cropland area share ($%$) in breach |
| **High PF**[^1] (`> 95th pct`) | Intense precipitation / flooding | Excess PF ($%$ above threshold) | Cropland area share ($%$) in breach |
| **Low PF** (`< 5th pct`) | Precipitation deficit | — | Cropland area share ($%$) in breach |

[^1]: PF events are derived from wet days.
---

## 4. Historical Disaster Backtests (Empirical Validation)

After Frisch–Waugh–Lovell (FWL) orthogonalization against year and month fixed effects, the resulting sub-regional anomaly series accurately isolate major historical meteorological extremes across Central South America:

### Atmospheric Dryness Shocks (`High VPD`)
* **2009 Argentine Drought:** Captured as an acute spike in excess VPD reaching **50% above** the upper 95th-percentile reference threshold.
* **2014 Amazon Drought:** Detected across nearly **95% of the regional cropland footprint** simultaneously exceeding its upper VPD threshold.
* **2019–2020 La Plata Basin Drought:** Registered as **38% -- 40% excess VPD** above threshold across multiple consecutive days.
* **January 2022 Heatwave & Drought:** Isolated a sharp VPD surge corresponding to regional temperatures exceeding **$45^\circ\text{C}$**.
* **2023 Pan-Regional Drought:** Captured widespread dryness across all Central South American quadrants, with sub-regional VPD rising **30% above** threshold baselines.

### Precipitation Flux Shocks (`High PF`)
* **2014–2017 El Niño–Southern Oscillation (ENSO) Cycle:**
  * Orthogonalized PF series flags a pronounced spike in **2014**, marking the onset of historic flooding across Paraguay, Bolivia, and Brazil—most sharply captured in **Quadrant 4** across Arabica-growing areas—before transitioning into the **2015–2016 El Niño** drought regime.
  * Across soybean-producing regions, the **January 2015** precipitation deficit emerges as the single most severe dry anomaly of the multi-year ENSO episode.
* **September 2023 Northern Brazil Floods:** Detected as sub-regional precipitation flux rising **12% above** historical upper thresholds.
* **2025 Western Amazon / Andean Floods (Quadrant 3 — Peru & Bolivia):** Captures nearly **40% excess PF** over upper reference thresholds in June 2025 following the severe regional flooding in April 2025.

---

<sub>**Full Implementation:** 1. [Sub-Regional Shock Aggregation Pipeline](area_of_effects/code_blocks/grid_to_quadrants.py)</sub>

---

<sub>*[Link to Landing Page](/README.md)*</sub>
