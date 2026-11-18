#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 15:12:57 2026

@author: darkmage

https://www.sciencedirect.com/science/article/pii/S0305750X25002517
https://www.sciencedirect.com/science/article/pii/S2405918817300405
https://journals.ametsoc.org/view/journals/clim/18/1/jcli-3256.1.xml
"""
import pandas as pd
import numpy as np
import re
from statsmodels.tsa.stattools import adfuller as adf
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
main_path="/media/darkmage/SSD-Storage1/"
drive_path="/media/darkmage/SSD-Storage1/timeseries_csv/"
#%%

commodities = pd.read_csv(f"{drive_path}/commodities.csv")
commodities["mm-dd"]=commodities["Unnamed: 0"].apply(lambda x: x[5:])
commodities=commodities[commodities["mm-dd"]!="02-29"]
commodities["year"]=commodities["Unnamed: 0"].apply(lambda x: x[:4])
commodities["month"]=commodities["Unnamed: 0"].apply(lambda x: int(x[5:7]))
commodities=commodities.set_index("Unnamed: 0")
print("Commodities shape = ",commodities.shape)
var_list= ["KC=F","USDBOB=X","ARSUSD=X","PYG=X","usdbrent","rial_usd_br"]
commodities=commodities.dropna(subset=var_list)
core = commodities[var_list + ["month", "year"]].copy()
print("Core trading days shape = ",core.shape)
"""
Commodities shape =  (6778, 13)
Core trading days shape =  (5003, 8)
"""

#%%

commodities = pd.read_csv(f"{drive_path}/commodities.csv")
commodities["mm-dd"]=commodities["Unnamed: 0"].apply(lambda x: x[5:])
commodities=commodities[commodities["mm-dd"]!="02-29"]
commodities["year"]=commodities["Unnamed: 0"].apply(lambda x: x[:4])
commodities["month"]=commodities["Unnamed: 0"].apply(lambda x: int(x[5:7]))
commodities=commodities.set_index("Unnamed: 0")
print("Commodities shape = ",commodities.shape)
var_list= ["ZS=F","USDBOB=X","ARSUSD=X","PYG=X","usdbrent","rial_usd_br"]
commodities=commodities.dropna(subset=var_list)
core = commodities[var_list + ["month", "year"]].copy()
print("Core trading days shape = ",core.shape)
"""
Commodities shape =  (6778, 13)
Core trading days shape =  (5007, 8)
"""
#%%

commodities = pd.read_csv(f"{drive_path}/commodities.csv")
commodities["mm-dd"]=commodities["Unnamed: 0"].apply(lambda x: x[5:])
commodities=commodities[commodities["mm-dd"]!="02-29"]
commodities["year"]=commodities["Unnamed: 0"].apply(lambda x: x[:4])
commodities["month"]=commodities["Unnamed: 0"].apply(lambda x: int(x[5:7]))
commodities=commodities.set_index("Unnamed: 0")
print("Commodities shape = ",commodities.shape)
var_list= ["ZS=F","KC=F"]
commodities=commodities.dropna(subset=var_list)
core = commodities[var_list + ["month", "year"]].copy()
print("Core trading days shape = ",core.shape)
"""
Commodities shape =  (6778, 13)
Core trading days shape =  (5007, 8)
"""


#%%

def projecting_derivatives_series(core,
                                  var_list):
    D = pd.concat([
        pd.get_dummies(core["month"], prefix="m", drop_first=True),
        pd.get_dummies(core["year"],  prefix="y", drop_first=True),
    ], axis=1).astype(float)
    D.insert(0, "const", 1.0)
    
    print(np.linalg.matrix_rank(D.to_numpy()), D.shape[1])  # should be equal 
    
    Dm = D.to_numpy()
    x = core[var_list].to_numpy()*100 #*** VERY IMPORTANT THE X 100 RESCALING !!!
    coef, *_ = np.linalg.lstsq(Dm, x, rcond=None)
    Z = x - Dm @ coef
    print(np.abs(Dm.T @ Z).max()) 
    return Z

#%% coffee futures KC=F

Z = projecting_derivatives_series(core=core,
                                  var_list=var_list)
tilde_df = pd.DataFrame(data=Z,columns=var_list,index=core.index)
nlags=10
lagged_list=[]
for i in range(1,nlags+1):
    temp=tilde_df.shift(i)
    temp.columns=[f"{col}.L{i}" for col in var_list]
    lagged_list.append(temp)
lagged_df=pd.concat(lagged_list,axis=1)
lagged_df[var_list[0]]=Z[:,0]
lagged_df.index=tilde_df.index
lagged_df = lagged_df.dropna()
"""
lagged_df.shape
Out[53]: (4993, 61)
"""
regressors=['KC=F.L1', 'USDBOB=X.L1', 'ARSUSD=X.L1', 'PYG=X.L1',
       'usdbrent.L1', 'rial_usd_br.L1', 'KC=F.L2', 'USDBOB=X.L2',
       'ARSUSD=X.L2', 'PYG=X.L2', 'usdbrent.L2', 'rial_usd_br.L2', 'KC=F.L3',
       'USDBOB=X.L3', 'ARSUSD=X.L3', 'PYG=X.L3', 'usdbrent.L3',
       'rial_usd_br.L3', 'KC=F.L4', 'USDBOB=X.L4', 'ARSUSD=X.L4', 'PYG=X.L4',
       'usdbrent.L4', 'rial_usd_br.L4', 'KC=F.L5', 'USDBOB=X.L5',
       'ARSUSD=X.L5', 'PYG=X.L5', 'usdbrent.L5', 'rial_usd_br.L5', 'KC=F.L6',
       'USDBOB=X.L6', 'ARSUSD=X.L6', 'PYG=X.L6', 'usdbrent.L6',
       'rial_usd_br.L6', 'KC=F.L7', 'USDBOB=X.L7', 'ARSUSD=X.L7', 'PYG=X.L7',
       'usdbrent.L7', 'rial_usd_br.L7', 'KC=F.L8', 'USDBOB=X.L8',
       'ARSUSD=X.L8', 'PYG=X.L8', 'usdbrent.L8', 'rial_usd_br.L8', 'KC=F.L9',
       'USDBOB=X.L9', 'ARSUSD=X.L9', 'PYG=X.L9', 'usdbrent.L9',
       'rial_usd_br.L9', 'KC=F.L10', 'USDBOB=X.L10', 'ARSUSD=X.L10',
       'PYG=X.L10', 'usdbrent.L10', 'rial_usd_br.L10']
x_mat = lagged_df[regressors].to_numpy() # (4993, 60)
y_mat = lagged_df[["KC=F"]].to_numpy().reshape(-1,1)
beta_tilde = np.linalg.lstsq(x_mat, y_mat, rcond=None)[0]

y_pred=x_mat @ beta_tilde
res = y_mat - y_pred
lagged_df["return_res"]=res
lagged_df.to_csv(f"{drive_path}/residuals_of_arx10.csv")

#%% soybean futures ZS=F
crop="soybean"
Z = projecting_derivatives_series(core=core,
                                  var_list=var_list)
tilde_df = pd.DataFrame(data=Z,columns=var_list,index=core.index)
nlags=10
lagged_list=[]
for i in range(1,nlags+1):
    temp=tilde_df.shift(i)
    temp.columns=[f"{col}.L{i}" for col in var_list]
    lagged_list.append(temp)
lagged_df=pd.concat(lagged_list,axis=1)
lagged_df[var_list[0]]=Z[:,0]
lagged_df.index=tilde_df.index
lagged_df = lagged_df.dropna()
"""
lagged_df.shape
Out[53]: (4997, 61)
"""
regressors=['ZS=F.L1', 'USDBOB=X.L1', 'ARSUSD=X.L1', 'PYG=X.L1', 'usdbrent.L1',
       'rial_usd_br.L1', 'ZS=F.L2', 'USDBOB=X.L2', 'ARSUSD=X.L2', 'PYG=X.L2',
       'usdbrent.L2', 'rial_usd_br.L2', 'ZS=F.L3', 'USDBOB=X.L3',
       'ARSUSD=X.L3', 'PYG=X.L3', 'usdbrent.L3', 'rial_usd_br.L3', 'ZS=F.L4',
       'USDBOB=X.L4', 'ARSUSD=X.L4', 'PYG=X.L4', 'usdbrent.L4',
       'rial_usd_br.L4', 'ZS=F.L5', 'USDBOB=X.L5', 'ARSUSD=X.L5', 'PYG=X.L5',
       'usdbrent.L5', 'rial_usd_br.L5', 'ZS=F.L6', 'USDBOB=X.L6',
       'ARSUSD=X.L6', 'PYG=X.L6', 'usdbrent.L6', 'rial_usd_br.L6', 'ZS=F.L7',
       'USDBOB=X.L7', 'ARSUSD=X.L7', 'PYG=X.L7', 'usdbrent.L7',
       'rial_usd_br.L7', 'ZS=F.L8', 'USDBOB=X.L8', 'ARSUSD=X.L8', 'PYG=X.L8',
       'usdbrent.L8', 'rial_usd_br.L8', 'ZS=F.L9', 'USDBOB=X.L9',
       'ARSUSD=X.L9', 'PYG=X.L9', 'usdbrent.L9', 'rial_usd_br.L9', 'ZS=F.L10',
       'USDBOB=X.L10', 'ARSUSD=X.L10', 'PYG=X.L10', 'usdbrent.L10',
       'rial_usd_br.L10']
x_mat = lagged_df[regressors].to_numpy() # (4997, 60)
y_mat = lagged_df[["ZS=F"]].to_numpy().reshape(-1,1)
beta_tilde = np.linalg.lstsq(x_mat, y_mat, rcond=None)[0]

y_pred=x_mat @ beta_tilde
res = y_mat - y_pred
lagged_df["return_res"]=res
lagged_df.to_csv(f"{drive_path}/{crop}residuals_of_arx10.csv")


#%% arx regression function with HAC and KPSS
class arx_model:
    
    def __init__(self,nlags,k,climate_ref,crop,
                 climate_tilde,lagged_df,comm_col,climate_name):
        self.nlags = nlags
        self.k = k
        self.crop = crop
        self.climate_name=climate_name
        self.climate_tilde=climate_tilde
        self.lagged_df=lagged_df
        self.climate_ref = climate_ref
        self.comm_col=comm_col
        self.tilde_df = pd.concat([self.climate_tilde,
                                   self.lagged_df[["return_res"]]],axis=1)
        self.tilde_df=self.tilde_df.loc["2005-01-01":"2025-12-31"] ## 7665, 29
        self.clim = self.tilde_df.drop(columns="return_res", errors="ignore").copy()
        self.clim.index = pd.to_datetime(self.clim.index)
        self.clim = self.clim.sort_index().asfreq("D").ffill(limit=1)  # (7670, 28)
        print(self.clim.isna().sum().sum())
        self._make_lags()
        
    
    def _make_lags(self):
        self.X_clim = pd.concat([self.clim.shift(i).add_suffix(f".L{i}")
                            for i in range(1, self.nlags + 1)], axis=1)
        
        # returns: trading days only, so AR lag k = k trading days
        self.y = self.lagged_df["return_res"].dropna().copy()
        self.y.index = pd.to_datetime(self.y.index)
        self.ar = pd.concat([self.y.shift(i).rename(f"return_res.L{i}")
                        for i in range(1, self.nlags + 1)], axis=1)
        self.data = pd.concat([self.y.rename("res"), self.ar, self.X_clim.reindex(self.y.index)], axis=1).dropna() ## final regression data
        self.dates_used = self.data.index
        self.obs_y = self.lagged_df[[self.comm_col]]
        self.obs_y.index = pd.to_datetime(self.obs_y.index)
        self.obs_y = self.obs_y.loc[self.obs_y.index.isin(self.dates_used),self.comm_col]
        print(self.data.shape)     # 4795,291
        
    def make_descriptive_statistics(self,dep_col,climate_data):
        """
        making the descriptive statistics on the original 
        """
        
        self.cols = [dep_col, "USDBOB=X", "ARSUSD=X", "PYG=X", "usdbrent", "rial_usd_br"]
        
        self.commodities = pd.read_csv(
            f"{drive_path}/commodities.csv",
            index_col="Unnamed: 0",
            parse_dates=True,        # makes the index datetime so it matches data.index
        )
        
        temp_sel = self.commodities.loc[self.commodities.index.isin(self.dates_used), self.cols]
        self.comm_desc_df= np.round(temp_sel.describe(),4)
        
        self.main_climate=pd.read_csv(f"{drive_path}/{climate_data}",
                            index_col = "Unnamed: 0",
                            parse_dates=True)
        self.main_climate=self.main_climate.sort_index()
        temp_sel = self.main_climate.loc[self.main_climate.index.isin(self.dates_used),:]
        self.clim_desc_df = np.round(temp_sel.describe(),4)
        
        #prec flux is 0 
    
        self.pfis0 = pd.read_csv(drive_path+'/timeseries_csv/pfis0.csv',
                            index_col = "Unnamed: 0",
                            parse_dates=True)
        temp_sel = self.pfis0.loc[self.pfis0.index.isin(self.dates_used)]
        self.pfis0_desc_df = np.round(temp_sel.describe(),4)
        
    def _print_descriptive_statistics(self,digits=4):
        if self.crop=="arabica":
            climate_og = pd.read_csv(f"{drive_path}/climate_series_d20.csv",
                                     parse_dates=True,
                                     index_col="Unnamed: 0")
        elif self.crop=="soybean":
            climate_og = pd.read_csv(f"{drive_path}/soybeanclimate_series_d20.csv",
                                     parse_dates=True,
                                     index_col="Unnamed: 0")
        subline_dict = {f"PfLWshd{self.climate_ref}": "Low PF threshold breach: share of area (PfLWsh)",
                        f"PfUPmgd{self.climate_ref}": "High PF threshold breach: magnitude (PfUPmg)",
                        f"PfUPshd{self.climate_ref}": "High PF threshold breach: share of area (PfUPsh)",
                        f"VdUPmgd{self.climate_ref}": "High VPD threshold breach: magnitude (VdUPmg)",
                        f"VdUPshd{self.climate_ref}": "High VPD threshold breach: share of area (VdUPsh)",
                        "Pfmn": "Mean precipitation flux (Pfmn) mm per day",
                        "Vdmn": "Mean vapor pressure deficit (Vdmn) hPa"}
        variable_desc = subline_dict[self.climate_name]
        desc_df = climate_og.loc[climate_og.index.isin(self.dates_used), :]

        lines = [rf"\multicolumn{{9}}{{l}}{{\textit{{{variable_desc}}}}} \\[2pt]"]
        for q in range(1, 5):
            s = desc_df[f"{self.climate_name}_ar{q}"].describe()
            vals = [s["mean"], s["std"], s["min"], s["25%"], s["50%"], s["75%"], s["max"]]
            cells = " & ".join(f"{x:.{digits}f}" for x in vals)
            lines.append(f"        Quadrant {q} & {int(s['count'])} & {cells} \\\\")

        block = "\n".join(lines)
        print(block)
        
    
    def _hac_newey_west(self,X,u,L):
        Xu = X * u[:,None]
        S = Xu.T @ Xu
        for l in range(1,L+1):
            G = Xu[l:].T @ Xu[:-l]
            S += (1 - l/(L+1)) * (G+G.T)   
        A = np.linalg.pinv(X.T @ X)
        return A @ S @ A
    
    def _add_cummulative(self):
        self.se_df["beta_c"] = np.nan
        self.se_df["se_c"]=np.nan
        for v,g in self.se_df.groupby("var"):
            g = g.sort_values(by="L")
            idx = g["pos"].to_numpy()
            Vg = self.V[np.ix_(idx, idx)]
            self.se_df.loc[g.index,"se_c"] = np.sqrt(Vg.cumsum(0).cumsum(1).diagonal())
            self.se_df.loc[g.index,"beta_c"] = g["beta"].cumsum().to_numpy()
        
    def do_the_regression(self,flag=""):
        self._print_descriptive_statistics()
        self.yv = self.data["res"].to_numpy()
        self.Xd = self.data.drop(columns="res")
        self.Xm = np.column_stack([np.ones(len(self.Xd)), self.Xd.to_numpy()])
        
        def _quadrant(name):
            if name.startswith("return_res") or "_ar" not in name:
                return 0                      # AR terms of the dependent variable, no quadrant
            tail = name.split("_ar", 1)[1]         # e.g. '1.L6'
            digits = ""
            for ch in tail:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            return int(digits)
        
        self.clim_var_sd = self.clim.loc[self.clim.index.isin(self.dates_used),:].std().reset_index()
        self.clim_var_sd["Quadrant"] = self.clim_var_sd["index"].apply(_quadrant)
        self.clim_var_sd= self.clim_var_sd[["Quadrant",0]].set_index("Quadrant")
        self.clim_var_sd_dict=self.clim_var_sd.to_dict()[0]
        self.clim_var_sd_dict[0]=0
        
        self.beta, *_ = np.linalg.lstsq(self.Xm, self.yv, rcond=None)
        self.yp=self.Xm @ self.beta
        self.u = self.yv - self.yp
        
        self.V = self._hac_newey_west(X=self.Xm, 
                                      u=self.u, 
                                      L=30)
            
        self.se = np.sqrt(np.diag(self.V))
        self.t = self.beta / self.se
        self.pred_df = pd.DataFrame({"uhat":self.yv,
                                     "eps":self.u,
                                     "pred_uhat":self.yp},index=self.obs_y.index)
        self.pred_df=pd.concat([self.pred_df,self.obs_y],axis=1)
        self.se_df=pd.DataFrame({"beta":self.beta[1:],
                            "t-val":self.t[1:],
                            "se":self.se[1:],
                            "var_name": self.Xd.columns}
                           )
        
        
        
        self.se_df["Quadrant"] = self.se_df["var_name"].apply(_quadrant)
        self.se_df["Q_sd"] = self.se_df["Quadrant"].apply(lambda x: self.clim_var_sd_dict[x])
        self.se_df["L"]=self.se_df["var_name"].apply(lambda x: int(x.split(".L")[1]))
        self.se_df["var"]=self.se_df["var_name"].apply(lambda x:x.split(".L")[0])
        self.se_df["lo"] = self.se_df["beta"] -self.k * self.se_df["se"]
        self.se_df["hi"] = self.se_df["beta"] + self.k * self.se_df["se"]
        self.se_df["sig3"] = (self.se_df["lo"] > 0) | (self.se_df["hi"] < 0)
        
        self.se_df = self.se_df.reset_index(drop=True)
        self.se_df["pos"] = np.arange(1,len(self.se_df)+1)
        self._add_cummulative()
        self.se_df.to_csv(f"{main_path}/plots_and_figures/{crop}_ardl/{self.climate_name}d20lag30_{flag}.csv")
        self.pred_df.to_csv(f"{main_path}/plots_and_figures/{crop}_ardl/{self.climate_name}{flag}hatd20lag30_{flag}.csv")
        
        
    def _plot_profile(self,v, beta_col,
                      ax=None, color="C0"):
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(5, 3))
        d = self.se_df[self.se_df["var"] == v].sort_values("L")
        se_col = "se_c" if beta_col=="beta_c" else "se"
        lo = d[beta_col] - self.k * d[se_col]
        hi = d[beta_col] + self.k * d[se_col]
        sig = (lo > 0) | (hi < 0)
    
        ax.fill_between(d["L"], lo, hi, color=color, alpha=0.25, lw=0)   # the shaded band
        ax.plot(d["L"], d[beta_col], "-", color=color, lw=1.2)
        ax.scatter(d["L"][sig], d[beta_col][sig], color=color, s=18, zorder=3)  # band excludes 0
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(v, fontsize=12)
        if own_fig:
            ax.set_xlabel("lag (calendar days)"); ax.set_ylabel(r"$\beta$")
            plt.tight_layout(); plt.show()
    
    def plot_all_profiles(self,beta_col,flag):
    # ---- grid: rows = variable type, columns = AR term ----
        vars_all = [v for v in self.se_df["var"].unique() if v != "return_res"]
        types = sorted({re.sub(r"_ar\d+$", "", v) for v in vars_all})
        ars = sorted({int(re.search(r"_ar(\d+)$", v).group(1)) for v in vars_all})
        
        fig, axes = plt.subplots(len(types), len(ars),
                                 figsize=(3.0 * len(ars), 1.9 * len(types)),
                                 sharex=True, sharey="row")
        for i, t_ in enumerate(types):
            for j, a in enumerate(ars):
                ax = axes[i, j]
                name = f"{t_}_ar{a}"
                if name in vars_all:
                    self._plot_profile(name,ax=ax, color=f"C{i}",beta_col=beta_col)
                ax.tick_params(labelsize=7)
                if j == 0: ax.set_ylabel(t_, fontsize=8)
        for ax in axes[-1]: ax.set_xlabel("lag (calendar days)", fontsize=8)
        if beta_col=="beta_c":
            fig.suptitle(rf"Cummulative Response functions ($\pm${self.k} s.e., hist ref = {self.climate_ref}yrs), crop = {self.crop}", fontsize=10)
        else:
            fig.suptitle(rf"Response functions ($\pm${self.k} s.e., hist ref = {self.climate_ref}yrs), crop = {self.crop}", fontsize=10)
        plt.tight_layout()
        plt.show()
        
        fig,ax=plt.subplots(figsize=(10,6))
        self._plot_profile(v="return_res",ax=ax,beta_col=beta_col)
        if beta_col=="beta_c":
            fig.suptitle(rf"AR ({self.nlags}) Cummulative Response Function ($\pm${self.k} s.e., hist ref = {self.climate_ref}yrs), crop = {self.crop}")
        else:
            fig.suptitle(rf"AR ({self.nlags}) Response Function ($\pm${self.k} s.e., hist ref = {self.climate_ref}yrs), crop = {self.crop}")
        ax.set_xlabel("lag (calendar days)",fontsize=8)
        plt.tight_layout()
        fig.savefig(f"{main_path}/plots_and_figures/{self.crop}_ardl/{self.climate_name}d20lag30_{flag}.png",dpi=150)
        plt.show()
        print(self.Xm.shape)
        
    def _split_name(self, v):
        base, ar = v.rsplit("_ar", 1)
        return base, int(ar)

    def plot_two_row_profiles(self, show=True, flag=""):
        """One figure per variable type: row 1 = beta_L, row 2 = cumulative beta, columns = AR term."""
        vars_all = [v for v in self.se_df["var"].unique()
                    if v != "return_res" and v.rsplit("_ar", 1)[-1].isdigit()]
        types = list(dict.fromkeys(self._split_name(v)[0] for v in vars_all))
        
        for i, t_ in enumerate(types):
            ars = sorted(self._split_name(v)[1] for v in vars_all
                         if self._split_name(v)[0] == t_)
            n_cols = len(ars)
            is_single = (n_cols == 1)
        
            # Give single-quadrant plots a proper wide canvas (like return_res)
            fig_w = 7.5 if is_single else 3.0 * n_cols
            fig_h = 5.5 if is_single else 4.2
        
            fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, fig_h),
                                     sharex=True, sharey="row", squeeze=False)
            for j, a in enumerate(ars):
                name = f"{t_}_ar{a}"
                self._plot_profile(name, beta_col="beta",   ax=axes[0, j], color=f"C{i}")
                self._plot_profile(name, beta_col="beta_c", ax=axes[1, j], color=f"C{i}")
                axes[1, j].set_title("")                       # title only on the top row
                axes[1, j].set_xlabel("lag (calendar days)", fontsize=10)
                for r in range(2):
                    axes[r, j].tick_params(labelsize=10)
        
            axes[0, 0].set_ylabel(r"$\beta_L$", fontsize=12)
            axes[1, 0].set_ylabel(r"$\sum_{l\leq L}\beta_l$", fontsize=12)
        
            if is_single:
                title_str = (rf"{t_}: response (top) and cumulative response (bottom)" + "\n"
                             rf"$\pm${self.k} s.e., hist ref = {self.climate_ref}yrs, crop = {self.crop}")
                title_fs = 12
            else:
                title_str = (rf"{t_}: response (top) and cumulative response (bottom), "
                             rf"$\pm${self.k} s.e., hist ref = {self.climate_ref}yrs, crop = {self.crop}")
                title_fs = 15
        
            fig.suptitle(title_str, fontsize=title_fs)
            fig.tight_layout(rect=[0, 0, 1, 0.93 if is_single else 0.95])
            save_dir=f"{main_path}/plots_and_figures/{self.crop}_ardl/{self.climate_name}{flag}d20lag30.png"
            if save_dir is not None:
                fig.savefig(save_dir, dpi=150)
            if show:
                plt.show()
            plt.close(fig)

        # AR(return_res) profile, same two-row layout
        if "return_res" in self.se_df["var"].values:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            self._plot_profile("return_res", beta_col="beta",   ax=axes[0])
            self._plot_profile("return_res", beta_col="beta_c", ax=axes[1])
            axes[1].set_title("")
            axes[0].set_ylabel(r"$\beta_L$", fontsize=9)
            axes[1].set_ylabel(r"$\sum_{l\leq L}\beta_l$", fontsize=9)
            axes[1].set_xlabel("lag (calendar days)", fontsize=8)
            fig.suptitle(rf"AR({self.nlags}) response and cumulative response, "
                         rf"$\pm${self.k} s.e., crop = {self.crop}", fontsize=10)
            plt.tight_layout()
            if save_dir is not None:
                fig.savefig(f"{main_path}/plots_and_figures/{self.crop}_ardl/{self.climate_name}{flag}d20lag30_res.png", dpi=150)
            if show:
                plt.show()
            plt.close(fig)
            
#%% printing the latex coefficient tables

def sig_summary_latex(se_df, names=None, t_crit=1.645, adjust_at="final",
                      digits=2, same_sign=True,
                      caption="Significant cumulative responses",
                      label="tab:sig_summary"):
    """
    One row per quadrant (ar suffix), SEs in a second row.
    Columns: first sig. lag, cum beta there, last sig. lag, cum beta there,
             no. of sig. horizons, longest unbroken run (start-end underneath),
             SD, cumulative movement x SD.
    A cumulative beta is significant if |beta_c / se_c| >= t_crit.
    Longest run = most consecutive lags that are significant (and, if same_sign,
    of the same sign). Ties: the earliest run is shown.
    adjust_at: 'final' -> cum beta at the last lag x SD; 'last' -> at last significant lag x SD.
    """
    df = se_df.copy()
    df["t_c"] = df["beta_c"] / df["se_c"]
    names = names or {}

    def stars(t):
        a = abs(t)
        return "***" if a >= 2.576 else "**" if a >= 1.96 else "*" if a >= 1.645 else ""

    def f(x):
        return f"{x:.{digits}f}".replace("-", "$-$")


    def longest_run(sig_df):
        """ 
        to return the length of the longest run
        the start of the longest run
        the end of the longest run
        a run is continuous if unbroken
        d = d[["L","beta_c","se_c","t_c"]]
        """
        run_df=sig_df.copy()
        L=run_df["L"].to_numpy()
        run_id = (np.diff(L,prepend=L[0])!=1).cumsum()
        run_df["run_id"] = run_id
        run_groups = run_df.groupby("run_id")
        gL = run_groups["L"].agg(size="size",start="min",end="max")
        gL=gL.sort_values(by=["size","end"],ascending=False).head(1)
        gB = run_groups["beta_c"].agg(first="first",last="last")
        gB = gB.loc[gL.index]
        gS = run_groups["se_c"].agg(first="first",last="last")
        gS = gS.loc[gL.index]
        return gL, np.round(gB,2), np.round(gS,2)


    #vars_ = [v for v in df["var"].unique()
    #         if "_ar" in v and v.rsplit("_ar", 1)[1].isdigit()]
    #bases = list(dict.fromkeys(v.rsplit("_ar", 1)[0] for v in vars_))

    out = [r"% Preamble: \usepackage{booktabs}",
           r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{llccccccc}", r"\toprule",
           r"Quadrant  & \shortstack{First\\sig. lag} & "
           r"\shortstack{$\sum\beta$\\(first)} & \shortstack{Last\\sig. lag} & "
           r"\shortstack{$\sum\beta$\\(last)} & \shortstack{No. sig.\\$\sum\beta$} & "
           r"\shortstack{Longest\\run} & SD & \shortstack{Cum. move\\($\times$ SD)} \\",
           r"\midrule"]
    s = "\n".join(out)
    print(s)
    clim_vars = [i for i in df["var"].unique() if "_ar" in i]
    #clim_vars = list(set(clim_vars))
    for k,clim_var in enumerate(clim_vars):
        print(r"\midrule  "+clim_var)
        #members = [f"{clim_var}_ar{i}" for i in range(1,5)]
        #members = sorted((v for v in vars_ if v.rsplit("_ar", 1)[0] == b),
        #                 key=lambda v: int(v.rsplit("_ar", 1)[1]))
        #title = names.get(b, b).replace("_", r"\_")
        #for k, v in enumerate(members):
        q = int(clim_var.rsplit("_ar", 1)[1])
        d = df[df["var"] == clim_var].sort_values("L")
        sig = d[d["t_c"].abs() >= t_crit]
        sd=d[d["Quadrant"]==q]["Q_sd"].mean().item() ## quadrant specific standard deviation of clim_var
        n_sig = len(sig)
        top = rf"Q{q}"
        bot =" "
        if n_sig>0:
            sig = sig[["L","beta_c","t_c","se_c"]]
            gL, gB, gS = longest_run(sig)
            start_run,end_run,size_run = gL["start"].item(),gL["end"].item(),gL["size"].item()
            #x,y = gB["first"].item(),gB["last"].item()
            x1,x2,y=sig.iloc[0]["beta_c"],sig.iloc[-1]["beta_c"],d.iloc[-1]["beta_c"]
            tx1,tx2,ty=sig.iloc[0]["t_c"],sig.iloc[-1]["t_c"],d.iloc[-1]["t_c"]
            start_beta_c,end_beta_c= f"{x1:.2f}"+stars(tx1),f"{x2:.2f}"+stars(tx2)
            top= top+" & "+str(int(sig.iloc[0]['L'])) + " & "+ start_beta_c + " & " + str(int(sig.iloc[-1]['L']))+ " & "+ end_beta_c 
            top= top +" & "+ str(int(n_sig)) + " & "+  str(int(size_run))+ f" & {sd:.2f} & "+ f"{y*sd:.2f}"+stars(ty) + r"\\"
            x1,x2 = sig.iloc[0]["se_c"],sig.iloc[-1]["se_c"]
            bot =bot+(rf" & & & ({x1:.2f}) &  & ({x2:.2f}) & & "rf"({start_run}--{end_run}) &  \\")
        else:
            top = top+" & "*8
            bot= bot+" & "*8
        print(top)
        print(bot)
        
    where = "at the final lag" if adjust_at == "final" else "at the last significant lag"
    runtxt = ("consecutive lags with a significant cumulative coefficient of the same sign")
    out = [r"\bottomrule", r"\end{tabular}",
            r"\caption{" + caption + r". A cumulative coefficient is counted as significant if "
            rf"$|t|\geq {t_crit}$, using HAC standard errors (in parentheses). "
            r"*, ** and *** denote significance at 10\%, 5\% and 1\%. The longest run is the largest "
            rf"number of {runtxt}, with its first and last lag below. Cumulative movement is the "
            rf"cumulative coefficient {where}, multiplied by the standard deviation of the climate "
            r"variable, in percentage points of the futures price change.}",
            rf"\label{{{label}}}", r"\end{table}"]
    s = "\n".join(out)
    print(s)
    return s
#%% arabica ardl, d20 and lags=30, with each climate variable one by one


climate_names=['PfLWshd20','PfUPmgd20','VdUPmgd20','PfUPshd20','VdUPshd20','Pfmn', 'Vdmn']
df2=pd.read_csv(f"{drive_path}/climate_orthogonalised_d20.csv")

crop="arabica"
df1=pd.read_csv(f"{drive_path}/residuals_of_arx10.csv")
lagged_df=df1.copy().set_index("Unnamed: 0")
for climate_name in climate_names:
    climate_tilde=df2.copy()
    climate_tilde=climate_tilde.set_index("Unnamed: 0")
    climate_tilde=climate_tilde[[f"{climate_name}_ar{i}" for i in range(1,5)]]
    
    arx_call_sh_ara = arx_model(nlags=30, k=1.96, climate_tilde=climate_tilde,climate_name=climate_name,
                                climate_ref=20, lagged_df=lagged_df,crop=crop,
                                comm_col="KC=F")
    arx_call_sh_ara.do_the_regression()
    arx_call_sh_ara.plot_two_row_profiles()
    
#%% making latex tables for individual climate variables arx30 runs
ardl_path=main_path+"/plots_and_figures/arabica_ardl/"
files=['VdUPmgd20d20lag30.csv','PfUPshd20d20lag30.csv','PfUPmgd20d20lag30.csv',
 'PfLWshd20d20lag30.csv','Vdmnd20lag30.csv','VdUPshd20d20lag30.csv','Pfmnd20lag30.csv']
for file in files:
    se_df=pd.read_csv(ardl_path+file)
    file=file.replace("d20d20lag30.csv","")
    x=sig_summary_latex(se_df=se_df)
    
#%% arabica ardl, d20 and lags=30, with each climate variable one by one, without Quad. 1


climate_names=['PfLWshd20','PfUPmgd20','VdUPmgd20','PfUPshd20','VdUPshd20','Pfmn', 'Vdmn']
df2=pd.read_csv(f"{drive_path}/climate_orthogonalised_d20.csv")

crop="arabica"
df1=pd.read_csv(f"{drive_path}/residuals_of_arx10.csv")
lagged_df=df1.copy().set_index("Unnamed: 0")
for climate_name in climate_names:
    climate_tilde=df2.copy()
    climate_tilde=climate_tilde.set_index("Unnamed: 0")
    climate_tilde=climate_tilde[[f"{climate_name}_ar{i}" for i in [2,3,4]]]
    
    arx_call_sh_ara = arx_model(nlags=30, k=1.96, climate_tilde=climate_tilde,climate_name=climate_name,
                                climate_ref=20, lagged_df=lagged_df,crop=crop,
                                comm_col="KC=F")
    arx_call_sh_ara.do_the_regression(flag="noQ1")
    arx_call_sh_ara.plot_two_row_profiles(flag="noQ1")
    
#%% making latex tables for individual climate variables arx30 runs
ardl_path=main_path+"/plots_and_figures/arabica_ardl/"
files=['VdUPmgd20d20lag30_noQ1.csv','PfUPshd20d20lag30_noQ1.csv','PfUPmgd20d20lag30_noQ1.csv',
 'PfLWshd20d20lag30_noQ1.csv','Vdmnd20lag30_noQ1.csv','VdUPshd20d20lag30_noQ1.csv','Pfmnd20lag30_noQ1.csv']
for file in files:
    se_df=pd.read_csv(ardl_path+file)
    file=file.replace("d20d20lag30.csv","")
    x=sig_summary_latex(se_df=se_df)
    
#%% arabica ardl, d20 and lags=30, with each climate variable one by one, without Quad. 2 vs Quad. 4


climate_names=['PfLWshd20','PfUPmgd20','VdUPmgd20','PfUPshd20','VdUPshd20','Pfmn', 'Vdmn']
df2=pd.read_csv(f"{drive_path}/climate_orthogonalised_d20.csv")

crop="arabica"
df1=pd.read_csv(f"{drive_path}/residuals_of_arx10.csv")
lagged_df=df1.copy().set_index("Unnamed: 0")
for climate_name in climate_names:
    climate_tilde=df2.copy()
    climate_tilde=climate_tilde.set_index("Unnamed: 0")
    climate_tilde=climate_tilde[[f"{climate_name}_ar{i}" for i in [2]]]
    
    arx_call_sh_ara = arx_model(nlags=30, k=1.96, climate_tilde=climate_tilde,climate_name=climate_name,
                                climate_ref=20, lagged_df=lagged_df,crop=crop,
                                comm_col="KC=F")
    arx_call_sh_ara.do_the_regression(flag="Q2")
    arx_call_sh_ara.plot_two_row_profiles(flag="Q2")
    
for climate_name in climate_names:
    climate_tilde=df2.copy()
    climate_tilde=climate_tilde.set_index("Unnamed: 0")
    climate_tilde=climate_tilde[[f"{climate_name}_ar{i}" for i in [4]]]
    
    arx_call_sh_ara = arx_model(nlags=30, k=1.96, climate_tilde=climate_tilde,climate_name=climate_name,
                                climate_ref=20, lagged_df=lagged_df,crop=crop,
                                comm_col="KC=F")
    arx_call_sh_ara.do_the_regression(flag="Q4")
    arx_call_sh_ara.plot_two_row_profiles(flag="Q4")
    
#%% making latex tables for individual climate variables arx30 runs
ardl_path=main_path+"/plots_and_figures/arabica_ardl/"
files=['VdUPmgd20d20lag30_noQ1.csv','PfUPshd20d20lag30_noQ1.csv','PfUPmgd20d20lag30_noQ1.csv',
 'PfLWshd20d20lag30_noQ1.csv','Vdmnd20lag30_noQ1.csv','VdUPshd20d20lag30_noQ1.csv','Pfmnd20lag30_noQ1.csv']
for file in files:
    se_df=pd.read_csv(ardl_path+file)
    file=file.replace("d20d20lag30.csv","")
    x=sig_summary_latex(se_df=se_df)
    
#%% making latex tables for individual climate variables arx30 runs
ardl_path=main_path+"/plots_and_figures/arabica_ardl/"
q=2
files=[f"VdUPmgd20d20lag30_Q{q}.csv",f"PfUPshd20d20lag30_Q{q}.csv",f"PfUPmgd20d20lag30_Q{q}.csv",
 f"PfLWshd20d20lag30_Q{q}.csv",f"Vdmnd20lag30_Q{q}.csv",f"VdUPshd20d20lag30_Q{q}.csv",f"Pfmnd20lag30_Q{q}.csv"]
for file in files:
    se_df=pd.read_csv(ardl_path+file)
    file=file.replace("d20d20lag30.csv","")
    x=sig_summary_latex(se_df=se_df)
    
q=4
files=[f"VdUPmgd20d20lag30_Q{q}.csv",f"PfUPshd20d20lag30_Q{q}.csv",f"PfUPmgd20d20lag30_Q{q}.csv",
 f"PfLWshd20d20lag30_Q{q}.csv",f"Vdmnd20lag30_Q{q}.csv",f"VdUPshd20d20lag30_Q{q}.csv",f"Pfmnd20lag30_Q{q}.csv"]
for file in files:
    se_df=pd.read_csv(ardl_path+file)
    file=file.replace("d20d20lag30.csv","")
    x=sig_summary_latex(se_df=se_df)


#%% soybean ardl, d20 and lags=30, with each climate variable one by one
climate_names=['PfLWshd20','PfUPmgd20','VdUPmgd20','PfUPshd20','VdUPshd20','Pfmn', 'Vdmn']
df2=pd.read_csv(f"{drive_path}/soybeanclimate_orthogonalised_d20.csv")
crop="soybean"
df1=pd.read_csv(f"{drive_path}/{crop}residuals_of_arx10.csv")
lagged_df=df1.copy().set_index("Unnamed: 0")
for climate_name in climate_names:
    climate_tilde=df2.copy()
    climate_tilde=climate_tilde.set_index("Unnamed: 0")
    climate_tilde=climate_tilde[[f"{climate_name}_ar{i}" for i in range(1,5)]]
    
    arx_call_sh_soy = arx_model(nlags=30, k=1.96, climate_tilde=climate_tilde,climate_name=climate_name,
                                climate_ref=20, lagged_df=lagged_df,crop=crop,
                                comm_col="ZS=F")
    arx_call_sh_soy.do_the_regression()
    arx_call_sh_soy.plot_two_row_profiles()
#%% making latex tables for individual climate variables arx30 runs
ardl_path=main_path+"/plots_and_figures/soybean_ardl/"
files=['VdUPmgd20d20lag30.csv','PfUPshd20d20lag30.csv','PfUPmgd20d20lag30.csv',
 'PfLWshd20d20lag30.csv','Vdmnd20lag30.csv','VdUPshd20d20lag30.csv','Pfmnd20lag30.csv']
for file in files:
    se_df=pd.read_csv(ardl_path+file)
    file=file.replace("d20d20lag30.csv","")
    x=sig_summary_latex(se_df=se_df)
    
#%% ADF test for the orthogonalised climate variables and commodities returns
"""
we need the hat dataframes for the uhat and the dates, then the 
climate_orthogonalised_d20.csv for arabica and soybean
"""

arabica=pd.read_csv(main_path+"/plots_and_figures/arabica_ardl/PfLWshd20hatd20lag30.csv",parse_dates=True,index_col="Unnamed: 0")
soybean=pd.read_csv(main_path+"/plots_and_figures/soybean_ardl/PfLWshd20hatd20lag30.csv",parse_dates=True,index_col="Unnamed: 0")

arabica_climate=pd.read_csv(drive_path+"/climate_orthogonalised_d20.csv",parse_dates=True, index_col="Unnamed: 0")
soybean_climate=pd.read_csv(drive_path+"/soybeanclimate_orthogonalised_d20.csv",parse_dates=True, index_col="Unnamed: 0")

def run_adf_and_format(df, col_name):
  """Runs ADF test and returns formatted test statistic with significance stars."""
  series = df[col_name].dropna()
  res = adf(series, autolag="AIC")
  stat, pval = res[0], res[1]

  if pval < 0.01:
    stars = "$^{***}$"
  elif pval < 0.05:
    stars = "$^{**}$"
  elif pval < 0.1:
    stars = "$^{*}$"
  else:
    stars = ""

  return f"{stat:.3f}{stars}", f"{pval:.3f}"


# 1. Test Orthogonalized Returns (uhat)
arb_uhat_stat, arb_uhat_pval = run_adf_and_format(arabica, "uhat")
soy_uhat_stat, soy_uhat_pval = run_adf_and_format(soybean, "uhat")

# 2. Climate variables base names and quadrants
base_vars = [
    "PfLWshd20",
    "PfUPmgd20",
    "PfUPshd20",
    "VdUPmgd20",
    "VdUPshd20",
    "Pfmn",
    "Vdmn",
]
quadrants = [1, 2, 3, 4]

rows_data = []

# Add uhat first
rows_data.append(
    ("Orthogonalized Returns ($\\hat{u}$)", arb_uhat_stat, arb_uhat_pval, soy_uhat_stat, soy_uhat_pval)
)

# Add climate variables by quadrant
for var in base_vars:
  for q in quadrants:
    col = f"{var}_ar{q}"
    label = f"{var} (Q{q})"
    arb_stat, arb_pval = run_adf_and_format(arabica_climate, col)
    soy_stat, soy_pval = run_adf_and_format(soybean_climate, col)
    rows_data.append((label, arb_stat, arb_pval, soy_stat, soy_pval))

# 3. Generate LaTeX Table String
latex_code = r"""
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{6pt}
\begin{tabular}{l c c c c}
\toprule
\textbf{Variable} & \multicolumn{2}{c}{\textbf{Arabica}} & \multicolumn{2}{c}{\textbf{Soybean}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& \textbf{ADF Stat.} & \textbf{$p$-value} & \textbf{ADF Stat.} & \textbf{$p$-value} \\
\midrule
"""

for row in rows_data:
  latex_code += f"  {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n"

latex_code += r"""
\bottomrule
\end{tabular}
\caption{Augmented Dickey-Fuller (ADF) Unit Root Test Results for Orthogonalized Returns and Climate Series. Null hypothesis: series has a unit root. *, **, and *** denote rejection of the null hypothesis at the 10\%, 5\%, and 1\% significance levels, respectively.}
\label{tab:appendix_adf_tests}
\end{table}
"""

print(latex_code)
