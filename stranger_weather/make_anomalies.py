import pandas as pd
import numpy as np

class anomalies:
    def __init__(self,county,path_prefix="", variable_flag='tas',gap_years=0,reference_flag='f1',period_flag='season'):
        self.period_flag=period_flag
        self.period_shorthand={'season':'sn','month':'mn'}[self.period_flag]
        self.variable_flag=variable_flag
        self.gap_years=gap_years
        self.reference_flag=reference_flag
        if self.variable_flag=='tas':
            file_path=f"{path_prefix}/kreise_daily_{self.variable_flag}_full/{county}.txt"
        else:
            file_path=f"{path_prefix}/kreise_weekly_{self.variable_flag}_full_1/{county}.csv"
        self.df=pd.read_csv(file_path)
        self.main_df=self.df.copy()
        try:
            self.main_df=self.main_df.drop(['Unnamed: 0','time'],axis=1)
        except:
            None
        
    def apply_season_label(self):
        season_dict={1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
        season_name={1:'winter',2:'spring',3:'summer',4:'autumn'}
        #self.main_df['MESS_DATUM']=self.main_df['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        self.main_df['year']=(self.main_df['MESS_DATUM']//10000).astype(int)
        self.main_df['month']=((self.main_df['MESS_DATUM']%10000)/100).astype(int)
        self.main_df['season']=self.main_df['month'].apply(lambda x: season_dict[x])
        
    def make_period_to_period_variability(self):
        self.apply_season_label()
        self.periodtoperiod=self.main_df[['year',self.period_flag,self.variable_flag]].copy()
        self.periodtoperiod=self.periodtoperiod.groupby(['year',self.period_flag]).std()
        
    def identify_periodly_bounds(self):
        self.hist_df=self.hist_df[[self.period_flag,self.variable_flag]]
        if self.variable_flag=='prec':
            self.hist_df=self.hist_df[self.hist_df[self.variable_flag]>0]
        self.qtls=pd.concat([self.hist_df.groupby(self.period_flag).quantile(q=self.lqtl),
                   self.hist_df.groupby(self.period_flag).quantile(q=self.uqtl)],axis=1)
        self.uqtlcol=f'{self.variable_flag}_uqtl_{self.reference_flag}'
        self.lqtlcol=f'{self.variable_flag}_lqtl_{self.reference_flag}'
        self.qtls.columns=[self.lqtlcol,self.uqtlcol]
        self.qtls=self.qtls.reset_index()
        self.select_main_df=pd.merge(self.select_main_df,self.qtls,on=self.period_flag)
        
    def make_twomoments(self):
        self.sn_means=self.main_df.copy()
        self.sn_means=self.sn_means[['year',self.period_flag,self.variable_flag]].groupby(['year',self.period_flag]).mean().reset_index()
        self.an_means=self.sn_means[['year',self.variable_flag]].groupby('year').mean()
        
        ## type 1 standard deviation: we take the periodly mean, and then the standard deviation of periodly mean (Linsenmeier)
        self.sn_stdevs=self.main_df.copy()
        self.sn_stdevs=self.sn_stdevs[['year',self.period_flag,self.variable_flag]].groupby(['year',self.period_flag]).mean().reset_index()
        self.an_stdevs_type1=self.sn_stdevs[['year',self.variable_flag]].groupby('year').std()
        self.an_stdevs_type1.columns=[f'{self.period_shorthand}_stdev_{self.variable_flag}_type1']
        
        ## type 2 standard deviation: we take the periodly standard deviation, and then the mean of the periodly standard deviations (Kotz)
        self.sn_stdevs=self.main_df.copy()
        self.sn_stdevs=self.sn_stdevs[['year',self.period_flag,self.variable_flag]].groupby(['year',self.period_flag]).std().reset_index()
        self.an_stdevs_type2=self.sn_stdevs[['year',self.variable_flag]].groupby('year').mean()
        self.an_stdevs_type2.columns=[f'{self.period_shorthand}_stdev_{self.variable_flag}_type2']

        self.an_stdevs=pd.concat([self.an_stdevs_type1,self.an_stdevs_type2],axis=1)
        
    def identify_periodly_anomalies(self,lqtl=0.05,uqtl=0.95,main_start_year=1990,main_end_year=2020,
                                    f1_start_year=1950,f1_end_year=1980):
        self.lqtl,self.uqtl=lqtl,uqtl
        self.loop_list=[]
        if self.reference_flag=='f2':
            for year in range(main_start_year,main_end_year+1):
                self.select_main_df=self.main_df[self.main_df['year']==year].copy()
                hist_end_year=year-self.gap_years
                hist_start_year=hist_end_year-30
                self.hist_df=self.main_df[(self.main_df['year']>=hist_start_year)&
                                          (self.main_df['year']<=hist_end_year)]
                self.identify_periodly_bounds()
                self.select_main_df['hist_start_year']=hist_start_year
                self.select_main_df['hist_end_year']=hist_end_year
                self.loop_list.append(self.select_main_df)
            self.fulldf=pd.concat(self.loop_list,axis=0)
        else:
            self.hist_df=self.main_df[(self.main_df['year']>=f1_start_year)&
                                          (self.main_df['year']<=f1_end_year)]
            self.select_main_df=self.main_df[(self.main_df['year']>=main_start_year)&
                                             (self.main_df['year']<=main_end_year)]
            self.identify_periodly_bounds()
            self.fulldf=self.select_main_df.copy()
        self.fulldf[f'{self.variable_flag}_is_high_{self.reference_flag}']=(self.fulldf[self.variable_flag]>=self.fulldf[self.uqtlcol]).astype(int)
        self.fulldf[f'{self.variable_flag}_is_low_{self.reference_flag}']=(self.fulldf[self.variable_flag]<=self.fulldf[self.lqtlcol]).astype(int)
        self.sn=self.fulldf[['year',self.period_flag,self.variable_flag]].groupby(['year',self.period_flag]).mean()
        self.cn=self.fulldf[['year',self.period_flag,f'{self.variable_flag}_is_high_{self.reference_flag}',f'{self.variable_flag}_is_low_{self.reference_flag}']].groupby(['year',self.period_flag]).sum()
    
    def calculate_deviation(self):
        self.deviation=self.fulldf.copy()
        self.deviation_high=self.deviation[self.deviation[f'{self.variable_flag}_is_high_{self.reference_flag}']==1].copy()
        self.deviation_low=self.deviation[self.deviation[f'{self.variable_flag}_is_low_{self.reference_flag}']==1].copy()
        self.deviation_high[f'{self.variable_flag}_how_high_{self.reference_flag}']=self.deviation_high[self.variable_flag]-self.deviation_high[self.uqtlcol]
        if self.variable_flag=='prec':
            self.deviation_low[f'{self.variable_flag}_how_low_{self.reference_flag}']=self.deviation_low[self.variable_flag]
        else:
            self.deviation_low[f'{self.variable_flag}_how_low_{self.reference_flag}']=self.deviation_low[self.lqtlcol]-self.deviation_low[self.variable_flag]
        self.deviation_high=self.deviation_high[['year',self.period_flag,f'{self.variable_flag}_how_high_{self.reference_flag}']].groupby(['year',self.period_flag]).mean()
        self.deviation_low=self.deviation_low[['year',self.period_flag,f'{self.variable_flag}_how_low_{self.reference_flag}']].groupby(['year',self.period_flag]).mean()
        
        self.deviations_means=pd.concat([self.deviation_high,self.deviation_low],axis=1)
        
    def calculate_extail_means(self):
        self.nontail=self.fulldf.copy()
        self.nontail=self.nontail[(self.nontail[f'{self.variable_flag}_is_high_{self.reference_flag}']!=1) & (self.nontail[f'{self.variable_flag}_is_low_{self.reference_flag}']!=1)].copy()
        self.nontail_mean=self.nontail[['year',self.period_flag,self.variable_flag]].groupby(['year',self.period_flag]).mean()
        self.nontail_mean.columns=[f'{self.period_flag}_{self.variable_flag}_mean_{self.reference_flag}']
    def periodly_time_series(self):
        self.sndf=pd.concat([self.sn,self.cn,self.deviations_means],axis=1)