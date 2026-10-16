<h1 align="center">Spoiled for Choice</h1>
<h3 align="center">Precision-Weighted Ensemble Yield Prediction for Heterogeneous Indian Farms</h3>

<p align="justify">
<b>Sukanya Mukherjee</b><br>
  <sub>IWH Halle (Saale) and MLU Halle (Saale)</sub>
</p>

---

## Abstract
# Localised Precision-Weighted Ensemble Framework for Agricultural Yield Prediction

## Abstract & Overview

Multi-model prediction tensors present the problem of **model-choice**. This project circumvents this issue using a **localised precision-weighting ensemble framework** that aggregates signals across diverse inductive biases and cross-validation folds to generate robust point predictions. 

The final point estimator predicts crop yields for rice and wheat farms across India with strong empirical performance:
* **Rice Yield Accuracy:** $R^2 = 0.54$
* **Wheat Yield Accuracy:** $R^2 = 0.59$

Given the substantial heterogeneity inherent in Indian agriculture, this framework prioritizes **robustness over model- or fold-level outperformance**. Candidate models with higher accuracy carry greater weight in the ensemble estimator, while models that overfit are explicitly penalized. 

---

## Key Methodology & Contributions

### 1. Robust Ensemble Weighting & Homoscedastic Risk
* **Candidate Models:** Random Forest (RF), XGBoost (XGB), and Support Vector Regression (SVR).
* Empirical results demonstrate that this candidate set exhibits **homoscedastic prediction risk**, making the precision-weighted ensemble collapse to a simple, highly robust **arithmetic mean** across estimators.

### 2. Within-Model vs. Between-Model Stability Check
* A custom **Stability Ratio** serves as a reliability threshold to ensure farm-level predictions are dominated by *between-model diversity* rather than *within-model sampling noise*.
* **98% of evaluated farms pass this qualification threshold.**

### 3. Tail-Risk Mitigation & Dispersion Weighting
* Residual analysis reveals heteroscedastic error distributions (a distinct U-shape at yield tails).
* We exploit this behavior using **conditional quantile prediction intervals** to flag farms with high tail-risk.
* Applying **dispersion weighting** to this high-uncertainty subsample yields a **10% improvement in point-prediction $R^2$**.

### 4. Farm-Level Risk & Policy Profiling
* Integrated **SHAP (SHapley Additive exPlanations)** values quantify the relative influence of:
  * **Climate & Weather Conditions**
  * **Input Costs**
  * **Farm Management Practices**
* Combines robust median forecasts, explicit risk exposure metrics, and feature importance into comprehensive, policy-relevant farm profiles.
---
## Policy & Risk Management Applications

> **Key Takeaway:** By integrating point-forecasts with explicit measures of predictive uncertainty and SHAP attribution, this framework provides an actionable tool for **agricultural risk management**, **crop insurance pricing**, and **targeted policy interventions** in regions vulnerable to climate shocks.
