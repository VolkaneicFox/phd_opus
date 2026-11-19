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
```python
  
```
</details>



    def _create_fixed_effect_dummies(self,absorb_fe="both",
                                 drop_unit=3257, drop_time=2012):
        drop_unit=int(drop_unit)
        drop_time=int(drop_time)
        drop_kreise=f"dummy_kr{drop_unit}"
        drop_year=f"dummy_yr{drop_time}"
        self.dummy_list=[]
        self.dummiesdf=pd.DataFrame()
        for kreise in self.kreise_list:
            temp=pd.DataFrame()
            for year in self.years_list:
                if absorb_fe=="both":
                    temp.loc[year,f"dummy_kr{int(kreise)}"]=1
                    temp.loc[year,f"dummy_yr{int(year)}"]=1
                    drop_cols=[drop_kreise,drop_year]
                elif absorb_fe=="time":
                    temp.loc[year,f"dummy_yr{int(year)}"]=1
                    drop_cols=[drop_year]
                elif absorb_fe=="unit":
                    temp.loc[year,f"dummy_kr{int(kreise)}"]=1
                    drop_cols=[drop_kreise]
            temp=temp.reset_index().rename(columns={'index':'year'})
            temp['Kreise_Code']=kreise
            self.dummy_list.append(temp)
        
        self.dummy_df=pd.concat(self.dummy_list,axis=0)
        self.dummy_df=self.dummy_df.drop(drop_cols,axis=1)
        self.dummy_df=self.dummy_df.reset_index(drop=True).fillna(0)
        
        self.dummy_cols=[i for i in self.dummy_df.columns if 'dummy' in i]
        dummy_len=len(self.dummy_cols)
        print(f"Created {dummy_len} dummy columns.") ## should be 423+2 columns in total
        self.dummy_df = self.dummy_df.set_index(['Kreise_Code', 'year'])
        
    def _create_county_trends(self, drop_unit=3257, add_quad=True, add_cube=False):
        base = self.regdf[['Kreise_Code', 'year']].copy()
    
        # Time variables
        t = base['year'] - base['year'].min() + 1
        t2 = t ** 2
        t3 = t ** 3
        self.trend_cols = {}
    
        for kr in base['Kreise_Code'].unique():
            if kr == drop_unit:
                continue
            col_t = f"{kr}_t"
            self.trend_cols[col_t] = (base['Kreise_Code'] == kr).astype(int) * t
            if add_quad:
                col_t2 = f"{kr}_t2"
                self.trend_cols[col_t2] = (base['Kreise_Code'] == kr).astype(int) * t2
            if add_cube:
                col_t3 = f"{kr}_t3"
                self.trend_cols[col_t3] = (base['Kreise_Code'] == kr).astype(int) * t3
            #self.trend_cols.extend([col_t, col_t2])

        self.trends_df = pd.concat([base, pd.DataFrame(self.trend_cols)], axis=1)
        
        self.trend_cols=list(self.trend_cols.keys())
        print(f"Created {len(self.trend_cols)} county time trend columns.")
        self.trends_df = self.trends_df.set_index(['Kreise_Code', 'year'])
        
    def _create_custom_trends(self,trend_cols):
        self.trend_cols=trend_cols
        self.trends_df=self.maindf[['Kreise_Code','year']+trend_cols].set_index(['Kreise_Code','year'])
        
    def _projection_matrix(self):
        self.y1,self.y2=self.regdf.year.min(),self.regdf.year.max()
        
        print(f'The years of observations = {int(self.y1)}-{int(self.y2)}')
        print(f'The total number of counties = {int(self.regdf.Kreise_Code.nunique())}')
        print(f'The total number of obervations = {int(self.regdf.shape[0])}')
        self.years_list=list(range(self.y1,self.y2+1))
        self.regdf = self.regdf.set_index(['Kreise_Code', 'year'])
        print(f'The number of regression variables = {self.regdf.shape[1]}')
        print(f'The number of trends = {self.trends_df.shape[1]}')
        print(f'The number of dummies = {self.dummy_df.shape[1]}')
    
        self.regdf = (
            self.regdf
            .join(self.trends_df, how="left")
            .join(self.dummy_df, how="left")
        )

        D_cols = self.trend_cols + self.dummy_cols
        y_cols = [self.y_var]
        const_col = np.ones((len(self.regdf), 1))
        
        self.D_mat = np.column_stack([self.regdf[col].to_numpy() for col in D_cols])
        temp = np.linalg.inv(self.D_mat.T @ self.D_mat)
        print("Checking if the D matrix is full rank ...")
        self.d_rank = np.linalg.matrix_rank(self.D_mat)
        self.N=self.regdf.shape[0]
        self.freedom=self.N-len(self.x_vars)-self.d_rank
        if self.d_rank == self.D_mat.shape[1]:
            print('D matrix is at full rank. Proceed. uWu.')
        else:
           warnings.warn("D matrix is collinear: Projection may be invalid. uWu.", UserWarning)
                    
        P_D = self.D_mat @ temp @ self.D_mat.T
        self.M_D = np.eye(P_D.shape[0]) - P_D
        
        self.X_mat = np.hstack([const_col, self.regdf[self.x_vars].to_numpy()])
        self.y_mat = self.regdf[y_cols].to_numpy()

        print(f'Shape of D Matrix = {self.D_mat.shape}')
        print(f'Shape of X Matrix = {self.X_mat.shape}')

Performing OLS with County-Clustered Standard Errors or Conley Standard Errors

 def _do_OLS(self):
        
        self.X_tilde = self.M_D @ self.X_mat
        self.y_tilde = self.M_D @ self.y_mat
        
        self.beta_tilde = np.linalg.inv(self.X_tilde.T @ self.X_tilde) @ (self.X_tilde.T @ self.y_tilde)
        
        self.coef_n=self.beta_tilde.shape[0]
        print(f"Estimated {self.coef_n} Coefficients from Regression.")
        self.y_hat_tilde=self.X_tilde @ self.beta_tilde
        
        self.u_tilde = self.y_tilde - self.y_hat_tilde
        
    def _do_anova(self,cluster="kreise",conley_wts=[]):
        self.cluster_sum = np.zeros((self.coef_n, self.coef_n))
        if cluster=="kreise":
            self.clusters=self.regdf.index.get_level_values('Kreise_Code')
            print("Standard Errors are clustered at the county-level")
            for kreise in self.kreise_list:
                kreise_mask = kreise == self.clusters
                x_cluster = self.X_tilde[kreise_mask,:] 
                u_cluster = self.u_tilde[kreise_mask,:]
                self.cluster_sum = self.cluster_sum + (x_cluster.T @ u_cluster)@((x_cluster.T @ u_cluster).T)
        elif cluster=="year":
            self.clusters=self.regdf.index.get_level_values('year')
            print("Standard Errors are clustered at the year-level")
            for year in self.years_list:
                year_mask = year == self.clusters
                x_cluster = self.X_tilde[year_mask,:] 
                u_cluster = self.u_tilde[year_mask,:]
                self.cluster_sum = self.cluster_sum + (x_cluster.T @ u_cluster)@((x_cluster.T @ u_cluster).T)
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
        
        self.t_stats = self.beta_tilde/self.beta_se
        print(f'The degrees of freedom = {self.freedom}')
        self.p_values = 2 * t.sf(np.abs(self.t_stats), self.freedom)
        
        self.y_fe = np.round(self.y_mat - self.y_tilde,6)
        self.SSR=np.sum(self.u_tilde**2)
        self.SST_tilde=np.sum(self.y_tilde**2)
        self.within_r2 = 1- (self.SSR / self.SST_tilde)
        self.SST=np.sum((self.y_mat - np.mean(self.y_mat))**2)
        temp=(self.N-1)/ self.freedom
        self.adjusted_r2 = 1- (temp*self.SSR/self.SST)
