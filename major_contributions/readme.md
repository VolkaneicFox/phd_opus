<div style="max-width: 720px; margin: 0 auto; line-height: 1.65;">

### 1. Stranger Weather: Estimating the Growth Shock from Historic and Structural Deviations in the Local Weather of 400 German Counties, 1995–2019
**Dr. Oliver Holtemöller and Sukanya Mukherjee**  
*Keywords: Panel Econometrics, Causal Inference, Spatial Standard Errors, Heat Shocks, Weather Fluctuations, Physical Climate Risk*

<details style="text-align: justify;">
<summary><b>Read Abstract</b></summary>
<br>
Semantics underneath the definitions of weather anomalies can be exploited to answer questions such as: which regions are facing accelerated warming, is the weather deviating more from its seasonal or monthly expectations, and, importantly, *does it matter?* Based on evidence from four hundred German counties between 1995 and 2019, we find that these distinctions are non-trivial. Regional warming above respective upper thresholds has not been uniform across Germany, particularly by annually updating these thresholds, we see that southern counties have warmed faster than the north. 
The estimated loss in the growth-rate of GDP per-capita is 0.11 percentage points per annum, when heat events are identified against seasonal threshold. When we use monthly thresholds, the corresponding growth loss varies widely across constructions and is in the range 0.12--0.24. The highest growth loss follows including the days with ``normal'' temperatures in the calculation of average annual excess warming. 
We also find evidence of a negative association between growth rates and intra-annual fluctuations in temperature. The standard deviation of seasonal mean temperatures is greater than that of monthly mean temperatures by roughly $0.05^\circ C$. Subsequently, the growth loss associated with a 1 standard deviation increase in within-year variability is 0.50 (seasonal) and 0.33 (monthly) percentage points. 
The question - *``does it matter?''* - remains central to the interpretations presented in this paper. In the end, this paper attempts to contribute to existing climate discourse, the assessment that how weather or climate anomalies are defined, is critical for measuring climate risk, identifying geographical vulnerabilities and estimating potential economic losses, with the goal of informing diverse economic decisions.
</details>

---

### 2. Hide and (can't) Seek: Spatial Leakage in Climate-Based Prediction of Rice Yields in Indian Farms using XGBoost with k-Means Clustering
**Sukanya Mukherjee**  
*Keywords: Applied ML, Spatial Data, Over-fitting, Agricultural Economics, SHAP*

<details style="text-align: justify;">
<summary><b>Read Abstract</b></summary>
<br>
Using the 2018--19 NSSO farm (22,331 rice-growing farming households in 551 Indian districts) and a suite of climate features, we sought to predict farm level rice yields, and in doing so, investigated how much of yield is explained by farming methods and household characteristics, and how much depends on climate. Climate is measured at the district level, so when households are split random between test and training samples, the same district appear in both and climate variables become spatial identifiers. In this traditional design, adding climate features to the farm features, improves the out-of-sample $R^2$ of an XGboost model from 39.7% to 57.5%. However, when we segregate entire districts so that they never intersect between test and training samples, adding climate improves $R^2$ from 29.1% to 33.5%. The gain falls by roughly three-quarters, and we believe this is because the model can no longer recognise the district to which the farm belongs. Finally, what remains is a modest, transferable climate signal on top of farming practices, that delivers a more defensible benchmark for predicting yields when the model has never seen the district.
</details>

---

### 3. Area of Effects: Identifying Price Adjustments in Arabica Coffee and Soybean Futures Returns following Intense and Spatially Dispersed Local Weather Anomaly Shocks in Central South America
**Sukanya Mukherjee**  
*Keywords: Multi-Variate Auto-Regressive Time Series, Stationarity, Physical Climate Risk, Local Hazards, Commodities Returns*

<details style="text-align: justify;">
<summary><b>Read Abstract</b></summary>
<br>
Agricultural production in Central South America is vulnerable to erratic weather conditions. Brazil is the leading global exporter of critical agricultural commodities, and we find evidence that commodities futures markets are pricing localized physical climate risk into daily valuations. We perform a thirty-day multi-variate autoregressive (ARX, L=30) estimation of the relationship between the returns to arabica coffee and soybean, and regional weather anomalies across Central South America. Regional weather anomalies are identified against rolling twenty-year historical baselines to subsume any medium- to long-term trends in climate. Our findings indicate that financial markets are price adjusting against local shocks, but the adjustment is not uniform over the crops. The returns to arabica coffee show a positive and transient association with anomalous atmospheric dryness (vapour pressure deficit, VPD) in core Brazilian producing states (São Paulo, Paraná, Rio Grande do Sul, and Minas Gerais). High anomalous VPD is followed by 0.08pp to 0.14pp increase in futures returns, within two to three trading days. Intense precipitation (precipitation flux, PF), however, is associated with bearish movements in arabica returns: -0.09pp to -0.16pp. For soybeans, high VPD shocks in the same Brazilian growing areas, are followed by cumulative return gains of +0.06pp to +0.11pp that accumulate persistently across a thirty-day horizon. We distinguished shocks based on severity versus geographic coverage and found that changes in arabica returns were predominantly associated with shocks to the *intensive margin* (shock severity), whereas changes in soybean returns were predominantly associated with shocks to the *extensive margin* (spatial exposure).  Identifying how cropping area and crop physiology relate to market price adjustments, provides an empirical foundation for physical-risk hedging in agricultural commodity portfolios.
</details>
