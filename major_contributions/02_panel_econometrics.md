# Panel Econometrics & Spatially Robust Inference

To isolate the causal effect of local weather shocks on annual regional economic growth across **400 German counties (`Kreise`, $2005\text{--}2019$)** without relying on black-box econometric packages, I implemented a custom matrix-algebra estimation engine featuring **Frisch–Waugh–Lovell (FWL) annihilator projection** and **sandwich covariance estimators**.

---

## 1. Econometric Specification & FWL Projection

We model annual real GDP per capita growth $g_{i,t}$ in county $i$ and year $t$ as:

$$g_{i,t} = \mathbf{x}_{i,t}'\boldsymbol{\beta} + \alpha_i + \gamma_t + \theta_{1i}t + \theta_{2i}t^2 + \varepsilon_{i,t}$$

where $\alpha_i$ absorbs time-invariant county characteristics, $\gamma_t$ captures common national macroeconomic shocks, and $(\theta_{1i}t + \theta_{2i}t^2)$ removes county-specific quadratic structural trends.

### Frisch–Waugh–Lovell (FWL) Projection
Stack all nuisance controls into $\mathbf{D}$ (verifying full column rank $\text{rank}(\mathbf{D}) = k_D$) and construct the orthogonal annihilator matrix $\mathbf{M}_D$:

$$\mathbf{P}_D = \mathbf{D}(\mathbf{D}'\mathbf{D})^{-1}\mathbf{D}', \qquad \mathbf{M}_D = \mathbf{I}_N - \mathbf{P}_D$$

Projecting both outcomes and weather shock regressors into the orthogonal complement of $\mathbf{D}$ yields residualized series $\tilde{\mathbf{y}} = \mathbf{M}_D\mathbf{y}$ and $\tilde{\mathbf{X}} = \mathbf{M}_D\mathbf{X}$, recovering exact OLS estimates:

$$\hat{\boldsymbol{\beta}} = (\tilde{\mathbf{X}}'\tilde{\mathbf{X}})^{-1}\tilde{\mathbf{X}}'\tilde{\mathbf{y}}, \qquad \tilde{\mathbf{u}} = \tilde{\mathbf{y}} - \tilde{\mathbf{X}}\hat{\boldsymbol{\beta}}$$

<details>
<summary><b>View Pseudocode: FWL Annihilator & Quadratic Detrending</b></summary>
  
```python
def build_fwl_annihilator(df, absorb_fe="both", add_quad=True):
    """Construct full-rank nuisance matrix D (County/Year FEs + County Quadratic Trends)
    and return the orthogonal annihilator M_D = I - D(D'D)^(-1)D'."""
    dummies = create_fe_dummies(df, absorb=absorb_fe, drop_ref=True)          # alpha_i + gamma_t
    trends  = create_county_polynomials(df, degree=2 if add_quad else 1)   # theta_1i*t + theta_2i*t^2

    D = np.column_stack([dummies, trends])
    assert np.linalg.matrix_rank(D) == D.shape[1], "Collinear nuisance matrix D"

    P_D = D @ np.linalg.inv(D.T @ D) @ D.T
    M_D = np.eye(len(df)) - P_D

    X_tilde = M_D @ X
    y_tilde = M_D @ y
    beta_hat = np.linalg.inv(X_tilde.T @ X_tilde) @ (X_tilde.T @ y_tilde)
    u_tilde  = y_tilde - X_tilde @ beta_hat
    return beta_hat, X_tilde, u_tilde
```
</details>

## 2. Standard Errors with Clustering, controlling for within-county auto-correlation

*Assuming between county independence*

$$\Omega = sum_{i=1}^{400}[(X.T \times u ) \times (X.T \times u).T]_{X=X_i}$$
$$se = (X.T \cdot X)^{-1} \times \Omega \times (X.T \cdot X)^{-1}$$

## 3. Allowing spatial correlations between counties up to 250km apart (centroid distance)

<details> 
<summary><b> Calculating the Conley Weights</b></summary>
  
  ```python
  shp_kreise=gpd.read_file(f"{drive_path}/official_de_shapefiles.gk3.shape.ebenen/vg5000_ebenen_1231/VG5000_KRS.shp",crs=4326)
  shp_kreise=shp_kreise[['ARS','NUTS','geometry']]
  shp_kreise['centroid']=shp_kreise['geometry'].centroid
  shp_kreise['centroid_deg']=shp_kreise.centroid.to_crs(epsg=4326)
  shp_kreise['lat_deg']=shp_kreise.centroid_deg.y
  shp_kreise['lon_deg']=shp_kreise.centroid_deg.x
  distances=shp_kreise[['ARS','lon_deg','lat_deg']]
  distances=distances.rename(columns={'ARS':'Kreise_Code'})
  distances['Kreise_Code']=distances['Kreise_Code'].astype(int)
  x1_rows=distances['lon_deg'].values.reshape(1,-1)
  x2_rows=distances['lon_deg'].values.reshape(-1,1)
  x2_minus_x1=x2_rows-x1_rows
  y1_rows=distances['lat_deg'].values.reshape(1,-1)
  y2_rows=distances['lat_deg'].values.reshape(-1,1)
  y2_minus_y1=y2_rows-y1_rows
  euc_deg=np.sqrt((x2_minus_x1**2)+(y2_minus_y1**2))
  euc_dist=euc_deg*111
  
  bound=250
  euc_wts=np.maximum(1 - euc_dist / bound, 0)
  euc_wts=pd.DataFrame(euc_wts.copy())
  kreise_list=list(distances.Kreise_Code.astype(int))
  euc_wts.index=kreise_list
  euc_wts.columns=kreise_list
  ```
  
</details>

<details>
<summary><b> Implementing Conley Weights in the Sandwich Covariance Matrix</b></summary>

  ```python
    elif cluster=="conley":
              self.conley_wts=conley_wts
              self.conley_wts=self.conley_wts.set_index('Unnamed: 0')
              print("Standard Errors are Conley Standard Errors with bound of 250km")
              all_obs=self.regdf.index
              all_xu={}
              for obs,(kreise,year) in enumerate(all_obs):
                  xu_tilde=self.X_tilde[obs,:].reshape(-1,1) @ self.u_tilde[obs].reshape(1,1)
                  all_xu[(int(kreise),int(year))]=xu_tilde
              years=list(self.regdf.index.get_level_values('year'))
              for item in all_xu:
                  outer=all_xu[item]
                  k1=item[0]
                  temp_xu=list(self.conley_wts[(self.conley_wts[str(k1)]>0)].index)
                  pos_xu = [(county, year) for county, year in all_xu.keys() if county in temp_xu]
                  for jtem in pos_xu:
                      k2=jtem[0]
                      w_ij=self.conley_wts.loc[k1,str(k2)]
                      self.cluster_sum=self.cluster_sum+(w_ij * outer@all_xu[jtem].T)
          else:
              print("Cluster can only take kreise, year or conley as inputs")
          se_denominator = np.linalg.inv(self.X_tilde.T @ self.X_tilde)
          self.beta_cov = se_denominator @ self.cluster_sum @ se_denominator
          self.beta_se = np.sqrt(np.diag(self.beta_cov)).reshape(self.coef_n,1)
  ```
</details>

List of full code:
1. Complete Panel Estimation Pipeline: stranger_weather/panel_estimation.py
