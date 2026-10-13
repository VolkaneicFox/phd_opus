<h2 align="center">Hello :) </h2>

<p align="justify">
I started my PhD in the Leibniz Institute for Economic Research (IWH) in October 2022, under the supervisions of Dr. Oliver Holtemöller (IWH and MLU) and Dr. Martin Quaas (Leipzig Uni). The central agenda was to estimate the losses from climate shocks. In the course of my research, I have seen the capacity of satellite data to introduce novel measures of climate risk at granularity we have never seen before, to the best of my knowledge. 
  
However, the sheer meta density of satellite and high-resolution data is a significant hurdle in the path of meaningful modelling and inference. Therefore, a core component of my thesis is wrangling N-Dimensional data. From measuring granular weather shocks as deviations from their local historical references, to comparing the tacit distinctions across seasonal and monthly baselines, I have developed mastery of large data compression and its translation. 
  
This repository is still, as of 13 October 2026, a work in progress, and as it grows, so will my insights into the skills I have developed. 

### Research Projects

**Paper 1:** Stranger Weather: Estimating the Growth Shock from Historic and Structural Deviations in the Local Weather of 400 German Counties, 1995–2019. [*Read Paper 1 Overview & Code Pipeline*](https://github.com/VolkaneicFox/phd_opus/stranger_weather/readme.md)

The Projections used in the Panel Econometrics Model are built by me completely, as seen in [here](https://github.com/VolkaneicFox/phd_opus/stranger_weather/panel_estimation.py), so that robustness is confirmed under Conley Standard Errors.

For each county, the climate reference is derived from a baseline fixed in 1950-80, as well as from moving baselines that update each year, to account for recent climate pasts that make the measures of extremes more conservative, while evincing trends in accelerated climate change. The definition of anomalies is presented here and its application is seen here . 

<figure align="center">
  <img src="https://raw.githubusercontent.com/VolkaneicFox/phd_opus/main/stranger_weather/images/map_tas_surprise_heat_sn.png" alt="Regions of Germany warming map" width="80%">
  <figcaption><em>Regions of Germany with accelerated warming above regional 95th daily percentile thresholds, seasonally adjusted, showing particularly rapid warming in the east-to-south-eastern counties, and the south-western belt, along the Mainz and Rhine rivers.</em></figcaption>
</figure>
</p>
