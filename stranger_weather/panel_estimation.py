import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from mpl_toolkits.axes_grid1 import make_axes_locatable

class panel_regression:
    
    def __init__(self,givendf=[],years="1993-2019", ## c8 is May2026 vintage, c9 fixes the precipitation low measure
                 y_var="",x_vars=[],make_vars=True,
                 make_lags=True,lag_vars=[],lag_n=0,
                 sel_kreise=False,sel_sector='',sel_grp=0,drop_counties=False):
        self.maindf=givendf.copy()
        self.maindf['year']=self.maindf['year'].astype(int)
        self.x_vars=x_vars
        self.y_var=y_var
        self.regdf=self.maindf[['Kreise_Code','year']+[self.y_var]+self.x_vars].copy()
        self.regdf=self.regdf[(self.regdf['year']>=int(years.split("-")[0]))&
                                (self.regdf['year']<=int(years.split("-")[1]))]
        self.regdf=self.regdf[self.regdf['Kreise_Code'].isna()==False]
        
        print(f"Shape of the dataframe = {self.regdf.shape}")
       
        if make_lags:
            print(f'Making {lag_n} Lags: ...')
            for v in lag_vars:
                for n in range(lag_n):
                    n=n+1
                    self.regdf.loc[:,f'{v}.lag{n}']=self.regdf.groupby('Kreise_Code')[v].shift(n)
                    self.x_vars.append(f'{v}.lag{n}')
            self.regdf=self.regdf.dropna()
            print(f'First year = {self.regdf.year.min()}')
            print(f'Last year = {self.regdf.year.max()}')
        else:
            self.regdf=self.regdf.dropna()
        
        if drop_counties:
            self.regdf=self.regdf[self.regdf['Kreise_Code'].isin(drop_counties)==False]
        
        self.regdf=self.regdf.dropna()
        self.years_list=list(self.regdf['year'].unique())
        self.kreise_list=list(self.regdf['Kreise_Code'].unique())
        
        index = pd.MultiIndex.from_product([self.kreise_list, self.years_list], names=['Kreise_Code', 'year'])
        print('self.regdf.shape = ',self.regdf.shape)
        self.trend_cols=[]
        self.dummy_cols=[]
        self.dummy_df=pd.DataFrame(index=index)
        self.trends_df=pd.DataFrame(index=index)
        nyears=self.regdf.year.max() - self.regdf.year.min() +1
        print(f'Time Frame = {self.regdf.year.min()} - {self.regdf.year.max()}')
        print(f'Number of years = {nyears}')
        print(f'Number of counties = {len(self.kreise_list)}')
        nobs=nyears*self.regdf.Kreise_Code.nunique()
        self.counties=self.regdf.Kreise_Code.nunique()
        match=(nobs==self.regdf.shape[0])
        print('Is the panel balanced ? = ',match)
        
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
        
    def _correlation_heatmaps(self,x_var_name='',color_map='RdBu_r',title_txt='',n_ticks=15,save_figure=True,dir_prefix='tas'):
        self.x_var_name=x_var_name
        self.x_index=self.x_vars.index(self.x_var_name)+1
        self.tilde_df=pd.DataFrame(index=self.regdf.index)
        self.tilde_df[f'{self.x_var_name}_tilde']=self.X_tilde[:,self.x_index]
        self.tilde_df=self.tilde_df.reset_index()
        self.tilde_df.columns=['Kreise_Code','year',f'{self.x_var_name}_tilde']
        self.tilde_df=self.tilde_df.sort_values(['Kreise_Code','year'])
        self.tilde_df=self.tilde_df.pivot(index='year',columns='Kreise_Code',values=f'{self.x_var_name}_tilde')
        self.tilde_corr=self.tilde_df.corr(method='spearman')
        corr_values,corr_ticks=self.tilde_corr.values,self.tilde_corr.columns

        fig, ax = plt.subplots(figsize=(7.5,7.5))
        im = ax.imshow(corr_values, vmin=-1, vmax=1, cmap=color_map)
        plt.title(title_txt)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label='Spearman correlation')

        n = len(corr_ticks)
        step = max(1, n // n_ticks)
        tick_positions = np.arange(0, n, step)
        tick_labels = [int(corr_ticks[i]) for i in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=9.2)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=9.2)

        fig.tight_layout()
        os.makedirs(f'{self.path_prefix}/klima_tilde_spatial/{dir_prefix}/', exist_ok=True)
        fig.savefig(fname=f'{self.path_prefix}/klima_tilde_spatial/{dir_prefix}/{self.x_var_name}')
        plt.close(fig)

        
    def _model_summary(self,print_summary=True):
        self.summary_df=pd.DataFrame(index=["const"]+self.x_vars)
        self.summary_df["coefs"]=np.round(self.beta_tilde,5)
        self.summary_df["std er"]=np.round(self.beta_se,4)
        self.summary_df["t-stat"]=np.round(self.t_stats,4)
        self.summary_df["p_values"]=np.round(self.p_values,2)
        if print_summary:
            
            print("uWu ************* RESULTS SUMMARY ***************** uWu")
            print("=======================================================")
            print(f"The dependent variable = {self.y_var}")
            print("-------------------------------------------------------")
            print(f'Within R2 = {np.round(self.within_r2,6)}')
            print(f'Adjusted R2 = {np.round(self.adjusted_r2,6)}')
            print("-------------------------------------------------------")

        
            print(self.summary_df.sort_index())
            print("*******************************************************")
        else:
            a=1
            
class make_markdown_table:
    def __init__(self, models_list, within_r2, adjusted_r2):
        self.models_list = models_list
        self.within_r2 = within_r2
        self.adjusted_r2 = adjusted_r2
        self.var_labels = {
            'stdev_prec': 'Precip. SD (mm)',
            'stdev_tas':  'Temp. SD (°C)',
            'prec_stdev': 'Precip. SD (mm)',
            'tas_stdev':  'Temp. SD (°C)',
            'meprec':  'Precip. Mean (mm)',
            'metas':   'Temp. Mean (°C)',
            'tas_is_high': 'Extreme Heat Event', 
            'tas_is_low' : 'Extreme Cold Event',
            'prec_is_high': 'Extreme Prec Event', 
            'prec_is_low': 'Extreme Dry Event',
            'tas is high': 'Extreme Heat Event', 
            'tas is low' : 'Extreme Cold Event',
            'prec is high': 'Extreme Prec Event', 
            'prec is low': 'Extreme Dry Event',
            'tas_how_high': 'Surprise Heat (°C)', 
            'tas_how_low' : 'Surprise Cold (°C)',
            'prec_how_high': 'Surprise Prec. (mm)', 
            'prec_how_low': 'Surprise Dry (mm) ',
            'tas how high': 'Surprise Heat (°C)', 
            'tas how low' : 'Surprise Cold (°C)',
            'prec how high': 'Surprise Prec (mm)', 
            'prec how low': 'Surprise Dry (mm)',
            'const':      'Constant'
        }

    def stars(self, p):
        if p < 0.01:
            return '\∗\∗\∗'
        elif p < 0.05:
            return '\∗\∗'
        elif p < 0.10:
            return '\∗'
        return ''

    def clear_index(self, df):
        df = df.copy()
        index = [i.replace('sn_', '').replace('mn_', '').replace('an_','')
                  .replace('_type1', '').replace('_type2', '')
                  .replace('_f1','').replace('_f2','')
                  .replace('mn0_','').replace('sn0_','')
                  .replace('exmn','').replace('exsn','') for i in df.index]
        df.index = index
        return df

    def clear_models(self):
        self.clear_models_list = [self.clear_index(model) for model in self.models_list]

    def build_reg_table(self, model_names=None, digits=4):
        self.clear_models()
        n_models = len(self.clear_models_list)

        if model_names is None:
            model_names = [f"({i+1})" for i in range(n_models)]

        # build variable order: union across models, const pushed to bottom
        seen = []
        for m in self.clear_models_list:
            for v in m.index:
                if v not in seen:
                    seen.append(v)
        var_order = [v for v in seen if v != 'const'] + (['const'] if 'const' in seen else [])

        header = "| Variable | " + " | ".join(model_names) + " |"
        sep = "|---" * (n_models + 1) + "|"
        lines = [header, sep]

        for v in var_order:
            label = self.var_labels.get(v, v.replace('_', ' '))
            cells = []
            for m in self.clear_models_list:
                if v in m.index:
                    c = m.loc[v, 'coefs']
                    se = m.loc[v, 'std er']
                    p = m.loc[v, 'p_values']
                    cells.append(f"{c:.{digits}f}{self.stars(p)}<br>*({se:.{digits}f})*")
                else:
                    cells.append("")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        lines.append(sep)
        lines.append("| Within R² | " + " | ".join(f"{r:.3f}" for r in self.within_r2) + " |")
        lines.append("| Adj. R² | " + " | ".join(f"{r:.3f}" for r in self.adjusted_r2) + " |")
        lines.append(sep)

        footer = "\n*Significance: * p<0.10, ** p<0.05, *** p<0.01. Standard errors in italics/parentheses.*"
        table = "\n".join(lines) + footer
        return table