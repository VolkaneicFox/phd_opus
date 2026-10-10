import os
import pandas as pd
path_prefix=""
counties_list=os.listdir(f"{path_prefix}/kreise_daily_tas_full/")

from make_anomalies import anomalies

class make_periodly_anomalies:
    
    def __init__(self, period_flag, variables_flag,path_prefix=path_prefix,counties_list=counties_list):
        self.period_flag = period_flag
        self.period_shorthand={'month':'mn','season':'sn'}[self.period_flag]
        self.variables_flag = variables_flag
        self.counties_list = counties_list
        self.path_prefix=path_prefix
        self.outlist_an_tas = [] 
        self.outlist_sn_tas = [] 
        self.outlist_an_prec = [] 
        self.outlist_sn_prec=[]
        
    def make_extail_means(self):
        self.temp_c=[]
        for county in self.counties_list:
            try:
                county=int(county.split('.')[0])
            except:
                county=int(county)
            temp_v=[]
            for var in self.variables_flag:
                temp_f=[]
                for ref in ['f1','f2']:
                    if ref == 'f2': gap_years = 5
                    else: gap_years = 0
                    f=anomalies(county=county, path_prefix= self.path_prefix,
                                variable_flag=var, gap_years=gap_years,
                                reference_flag=ref,period_flag=self.period_flag)
                    f.apply_season_label()
                    f.identify_periodly_anomalies()
                    f.calculate_extail_means()

                    self.exmeans = f.nontail_mean
                    temp_f.append(self.exmeans)
                temp_f12=pd.concat(temp_f,axis=1)
                temp_v.append(temp_f12)
            temp_v=pd.concat(temp_v,axis=1)
            temp_v['Kreise_Code']=county
            self.temp_c.append(temp_v)
        
    def make_rolling_histories(self, county,variable_flag):
        f2=anomalies(county=county, path_prefix=self.path_prefix, 
                     variable_flag=variable_flag, gap_years=self.gap_years, 
                     reference_flag='f2', period_flag=self.period_flag)
        f2.apply_season_label()
        f2.identify_periodly_anomalies()
        f2.calculate_deviation()
        f2.periodly_time_series()
        
        self.qtls_f2=f2.fulldf[['year',self.period_flag,'hist_start_year','hist_end_year',f2.lqtlcol,f2.uqtlcol]] ## daily dataframe
        self.qtls_f2.columns=['year',self.period_flag,'hist_start_year','hist_end_year',
                         f'{variable_flag}_lqtl_f2',f'{variable_flag}_uqtl_f2']
        self.qtls_f2=self.qtls_f2.drop_duplicates(keep='first').set_index(['year',self.period_flag])
        self.period_f2 = f2.sndf
        
        return self
        
    def make_fixed_histories(self, county,variable_flag, make_means=True, make_stdevs=True):
        f1=anomalies(county=county,path_prefix=self.path_prefix,
                     variable_flag=variable_flag,gap_years=0,
                     reference_flag='f1',period_flag=self.period_flag)
        f1.apply_season_label()
        f1.make_twomoments()
        f1.identify_periodly_anomalies()
        f1.calculate_deviation()
        f1.periodly_time_series()

        self.qtls_f1=f1.fulldf[['year',self.period_flag,f1.lqtlcol,f1.uqtlcol]] ## daily dataframe
        self.qtls_f1.columns=['year',self.period_flag,f'{variable_flag}_lqtl_f1',f'{variable_flag}_uqtl_f1']
        self.qtls_f1=self.qtls_f1.drop_duplicates(keep='first').set_index(['year',self.period_flag])

        self.period_f1=f1.sndf
        
        if make_means:
            self.mns=f1.an_means.reset_index()
            self.mns=self.mns[self.mns['year']>=1990]
            self.mns.columns=['year',f'{self.period_shorthand}_mean_{variable_flag}']
            
        if make_stdevs:
            self.stdevs=f1.an_stdevs.reset_index()
            self.stdevs=self.stdevs[self.stdevs['year']>=1990]
            
        return self
           
    def make_the_lists(self,gap_years=5):
        self.gap_years=gap_years
        self.periodly_counties,self.annually_counties=[],[]
        for county in counties_list:
            county=int(county.split('.')[0])
            temp_periodly,temp_annually=[],[]
            for var in self.variables_flag:
                temp_rolling = self.make_rolling_histories(county=county,variable_flag=var)
                f2_df=temp_rolling.period_f2.reset_index().set_index(['year',self.period_flag])
                f2_qdf=temp_rolling.qtls_f2
                temp_fixed = self.make_fixed_histories(county=county,variable_flag=var)
                f1_df=temp_fixed.period_f1.reset_index().set_index(['year',self.period_flag])
                f1_qdf=temp_fixed.qtls_f1
                means_df=temp_fixed.mns.set_index('year')
                stdevs_df=temp_fixed.stdevs.set_index('year')
                temp_periodly.append(pd.concat([f1_df,f1_qdf,f2_df,f2_qdf],axis=1))
                temp_annually.append(pd.concat([means_df,stdevs_df],axis=1))
            periodly=pd.concat(temp_periodly,axis=1)
            periodly['Kreise_Code']=county
            annually=pd.concat(temp_annually,axis=1)
            annually['Kreise_Code']=county
            
            self.periodly_counties.append(periodly)
            self.annually_counties.append(annually)
            
    def make_standard_deviation(self):
        self.periodly_devs=[]
        for county in counties_list:
            try:
                county=int(county.split('.')[0])
            except:
                county=int(county)
            temp_var=[]
            for var in self.variables_flag:
                temp=anomalies(county=county,path_prefix=self.path_prefix,
                     variable_flag=var,gap_years=0,
                     reference_flag='f1',period_flag=self.period_flag)
                temp.make_period_to_period_variability()
                tempdf=temp.periodtoperiod
                tempdf['Kreise_Code']=county
                tempdf=tempdf.reset_index()[['Kreise_Code','year',self.period_flag,var]].rename(columns={var:f'{self.period_flag}_{var}_stdev'})
                temp_var.append(tempdf.set_index(['Kreise_Code','year',self.period_flag]))
            stdevs=pd.concat(temp_var,axis=1)
            self.periodly_devs.append(stdevs)
