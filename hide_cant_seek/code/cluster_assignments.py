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

"""
make five random states, for each state space do the test-train split. 
from each cluster, there should be 80% as train and 20% as test.

Predicting for households in districts you’ve already seen: 
a household-level stratified split is fine.
Generalizing to new districts or regions (the usual claim for a climate paper): 
split by district, so no district appears in both sets.

the question is, after the clustering, do i need climate variables?
then the pipeline becomes:

split each cluster into 80-20            -- use climate, not use climate
split the sample so that no cluster appears in both train and test sets -- use climate, not use climate
"""

def split_and_save(given_df, crop_name, drive, random_states=(35, 4, 29, 92, 1)):
    df = given_df.copy()
    for state in random_states:
        # A: household-level, stratified by cluster (80/20)
        tr, te = train_test_split(df, test_size=0.2, stratify=df['cluster'], random_state=state)
        if (tr.index.nunique()==tr.shape[0]) & (tr.HHID.nunique()==tr.shape[0]):
            tr.to_csv(f"{drive}/split_data/insample_hh_{crop_name}_{state}.csv", index=False)
        else:
            print(f"for random state {state} the train ids are not unique")
        if (te.index.nunique()==te.shape[0]) & (te.HHID.nunique()==te.shape[0]):
            te.to_csv(f"{drive}/split_data/outsample_hh_{crop_name}_{state}.csv", index=False)
        else:
            print(f"for random state {state} the test ids are not unique")
        
        # B: district-grouped, stratified by cluster (~80/20)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=state)
        tr_idx, te_idx = next(sgkf.split(df, y=df['cluster'], groups=df['nss_district_code19']))
        sgkftr, sgkfte = df.iloc[tr_idx], df.iloc[te_idx]
        if (sgkftr.index.nunique()==sgkftr.shape[0]) & (sgkftr.HHID.nunique()==sgkftr.shape[0]):
            sgkftr.to_csv(f"{drive}/split_data/insample_sgkf_{crop_name}_{state}.csv", index=False)
        else:
            print(f"for random state {state} the train sgkf ids are not unique")
        if (sgkfte.index.nunique()==sgkfte.shape[0]) & (sgkfte.HHID.nunique()==sgkfte.shape[0]):
            sgkfte.to_csv(f"{drive}/split_data/outsample_sgkf_{crop_name}_{state}.csv", index=False)
        else:
            print(f"for random state {state} the test sgkf ids are not unique")

        print(pd.concat([sgkftr['cluster'].value_counts(normalize=True),
                 sgkfte['cluster'].value_counts(normalize=True)], axis=1))
        print(set(sgkftr['nss_district_code19']) & set(sgkfte['nss_district_code19']))  # empty set

split_and_save(given_df=rice_cluster,crop_name='rice',drive=drive)
y_climate=rice_cluster[['cluster','nss_district_code19','mean_speiSPEI-4MO',
       'median_speiSPEI-4MO', 'max_speiSPEI-4MO', 'min_speiSPEI-4MO',
       'std_speiSPEI-4MO', 'tp_mean', 'tp_std', 't2m_mean', 't2m_max',
       't2m_min', 't2m_std', 'slhf_mean', 'slhf_max', 'slhf_min', 'slhf_std',
       'ssr_mean', 'ssr_max', 'ssr_min', 'ssr_std']]
y=rice_cluster[['cluster','nss_district_code19']]
print('There is unique cluster - district relationship:')
print(y_climate.drop_duplicates().shape[0]==y.drop_duplicates().shape[0])

"""
There is unique cluster - district relationship:
True
"""

                                    
