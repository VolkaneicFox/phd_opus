import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
from sklearn.model_selection import KFold, GridSearchCV, train_test_split,StratifiedGroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import seaborn as sns
sns.set_theme(style='darkgrid')
from sklearn.preprocessing import StandardScaler, PowerTransformer
state_space=3504
drive=''

def assign_unique_HHID(givendf,combine_col1='nss_district_code19'):
    df=givendf.copy()
    df['HHID']=df[['nss_district_code19','hh_cons_exp_pm']].groupby('nss_district_code19').transform(lambda x: x.rank(method='first',ascending=False))
    df['HHID']= df['HHID'].astype(int).astype(str)
    df['HHID']= df['nss_district_code19'].astype(int).astype(str)+'-'+df['HHID']
    print('the ranks are unique and give unique HHID: ',df.shape[0]==df['HHID'].nunique())
    print('the number of unique HHID = ',df['HHID'].nunique())
    print('the number of unique districts =',df[combine_col1].nunique())
    return df

def assign_climate_clusters(n_clusters,given_df,merge_cluster):
    df=given_df.copy()
    d = df.groupby('nss_district_code19')[climate].first() 
    hh = df.groupby('nss_district_code19').size()         

    X = StandardScaler().fit_transform(d.astype(float))   

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X, sample_weight=hh.loc[d.index])
    d['cluster'] = km.labels_

    df = df.merge(d[['cluster']], left_on='nss_district_code19', right_index=True)
      # households per cluster
    c = km.cluster_centers_
    dist = cdist(c[[merge_cluster]], np.delete(c, merge_cluster, axis=0)).argmin()
    target = np.delete(np.arange(len(c)), merge_cluster)[dist]
    df['cluster']=df['cluster'].replace({merge_cluster:target})
    
    print(df['cluster'].value_counts()) 
    return df

ricedf=pd.read_csv(f"{drive}/split_data/donot_touch_mainricedf.csv")
ricehhid=assign_unique_HHID(ricedf)
ricehhid.to_csv(f"{drive}/split_data/mainricedf_hhid.csv",index=False)

climate=['mean_speiSPEI-4MO',
       'median_speiSPEI-4MO', 'max_speiSPEI-4MO', 'min_speiSPEI-4MO',
       'std_speiSPEI-4MO', 'tp_mean', 'tp_std', 't2m_mean', 't2m_max',
       't2m_min', 't2m_std', 'slhf_mean', 'slhf_max', 'slhf_min', 'slhf_std',
       'ssr_mean', 'ssr_max', 'ssr_min', 'ssr_std']
for i in range(3,10):
    print('n_clusters =', i-1)
    rice_cluster=assign_climate_clusters(given_df=ricehhid,n_clusters=i,merge_cluster=i-1)

distribution=rice_cluster.copy()[['cluster','nss_district_code19','HHID']]
cluster_grouping=pd.concat([distribution[['cluster','HHID']].groupby('cluster').count(),
                            distribution[['cluster','nss_district_code19']].drop_duplicates().groupby('cluster').count()],
                           axis=1)

rice_cluster.to_csv(f"{drive}/split_data/mainricedfclustered.csv",index=False)
rice_cluster=pd.read_csv(f"{drive}/split_data/mainricedfclustered.csv")

