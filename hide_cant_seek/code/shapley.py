
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
main_path = ''
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

def make_xgb_shapley(insample_df, dep_var, hyper_params):
    inX, inY = insample_df.copy().drop(dep_var, axis=1), insample_df[dep_var]
    model = XGBRegressor(**hyper_params)
    model.fit(inX, inY)
    y_hat = model.predict(inX)

    # Fixed: use np.sqrt to avoid scikit-learn squared=False errors
    rmse = np.sqrt(mean_squared_error(inY, y_hat))
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mean_absolute_error(inY, y_hat):.4f}")
    print(f"R²  : {r2_score(inY, y_hat):.4f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(inX).values
    shap_df = pd.DataFrame(shap_values, columns=[f'shap_{i}' for i in inX.columns])
    shap_df = pd.concat([shap_df, inY.reset_index(drop=True)], axis=1)
    return shap_df


def plot_shapley(shap_df,title_str):
    feature_names = {
        'farm_land_m2':'Farm Size ($m^2$)',
        'total_cost_sum': 'Aggregate Cost of Farming (Rs.)',
        'cost_total':'Reported Cost of Farming (Rs.)',
        'share_own_and_possessed':'Land Ownership Share', 
        'cost_energy':'Energy Cost (Rs.)',
        'cost_labour':'Labour Cost (Rs.)', 
        'cost_finance':'Finance Cost (Rs.)',
        'cost_fert':'Fertilizer Cost (Rs.)',
        'cost_pest':'Pesticide Cost (Rs.)',
        'cost_machine':'Machine Cost (Rs.)', 
        'cost_irrigation':'Irrigation Cost (Rs.)', 
        'cost_seeds':'Seeds Cost (Rs.)',
        'hh_cons_exp_pm':'Hhld. Cons. Exp. (Rs.per month)',
        'median_speiSPEI-4MO':'SPEI (median)', 
        'max_speiSPEI-4MO':'SPEI (max)',
        'min_speiSPEI-4MO':'SPEI (min)', 
        'std_speiSPEI-4MO':'SPEI (std)',
        'tp_mean':'Precipitation (mean)',
        'tp_std':'Precipitation (std)', 
        't2m_mean':'Surface Temperature (mean)',
        't2m_max':'Surface Temperature (max)', 
        't2m_min':'Surface Temperature (min)', 
        't2m_std':'Surface Temperature (std)', 
        'slhf_mean':'Surface Heat Flux (mean)', 
        'slhf_max':'Surface Heat Flux (J/$m^2$ max)',
        'slhf_min':'Surface Heat Flux (J/$m^2$ min)', 
        'slhf_std':'Surface Heat Flux (std)', 
        'ssr_mean':'Solar Radiation (mean)', 
        'ssr_max':'Solar Radiation (max)', 
        'ssr_min':'Solar Radiation (min)', 
        'ssr_std':'Solar Radiation (std)', 
        'if_land_irrigated':'If Land Irrigated', 
        'is_non_backward_caste':'Not-Backward Caste',
        'if_farmer_organisation':'Farmer Organisation', 
        'if_kisan_credit_card':'Access to Farmer Credit', 
        'if_soil_health_card':'Soil Health Card',
        'primary_irrigation_canal':'Irrigation Type: Canal',
        'primary_irrigation_gw':'Irrigation Type: GW', 
        'primary_irrigation_mixed':'Irrigation Type: Mixed',
        'primary_irrigation_msw':'Irrigation Type: Surface', 
        'primary_irrigation_others':'Irrigation Type: Others'
    }
    
    abs_shap = shap_df.drop('farm_crop_yield', axis=1).abs().mean(axis=0).sort_values(ascending=False)
    top_features = abs_shap[abs_shap > abs_shap.quantile(0.79)].index.tolist()
    top_features_labels = [feature_names.get(f.replace('shap_', ''), f) for f in top_features]
    mean_shap = abs_shap.loc[top_features]
    violin_shap = shap_df[top_features]
    
    x_pos = np.arange(len(top_features))
    fig, ax_left = plt.subplots(figsize=(12, 8))

    # ---- LEFT AXIS: violin plots ----
    vp = ax_left.violinplot(
        violin_shap.values,
        positions=x_pos,
        widths=0.7,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    # ---- RIGHT AXIS: mean |SHAP| ----
    ax_right = ax_left.twinx()

    ax_right.plot(
        x_pos,
        mean_shap.values,
        marker='o',
        linewidth=2,
        color='tab:red'
    )
    # ---- LEFT AXIS ----
    ax_left.set_ylabel("SHAP value", fontsize=16)
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(top_features_labels, rotation=45, ha='right', fontsize=16)
    ax_left.tick_params(axis='y', labelsize=16)  # Changes left y-tick numbers to 16

    # ---- RIGHT AXIS ----
    ax_right.set_ylabel("Mean |SHAP value|", fontsize=16)
    ax_right.tick_params(axis='y', labelsize=16)  # Changes right y-tick numbers to 16
    # ---- Aesthetics ----
    ax_left.grid(axis='y', alpha=0.3)
    plt.suptitle(title_str,fontsize=20)
    plt.tight_layout()
    plt.show()


def run_shapleys(split_level, random_state, hyper_params,title_str, crop_name='rice', 
                 print_cols=False, run_shapley=True, keep_climate=True):
    stem = f"{split_level}_{crop_name}_{random_state}.csv"
    train = pd.read_csv(main_path + f"split_data/insample_{stem}", index_col="HHID")
    
    DISTRICT_COL = ["nss_district_code19"]
    ID_COLS = ["cluster", "spei_district_code", "nss_district_code19"]
    STATE_DUMMIES = [f"state_{i}" for i in (*range(2, 31), 32, 33, 34, 36)]
    CLIMATE_COLS = [
        'mean_speiSPEI-4MO', 'median_speiSPEI-4MO', 'max_speiSPEI-4MO', 
        'min_speiSPEI-4MO', 'std_speiSPEI-4MO', 'tp_mean', 'tp_std', 
        't2m_mean', 't2m_max', 't2m_min', 't2m_std', 'slhf_mean', 
        'slhf_max', 'slhf_min', 'slhf_std', 'ssr_mean', 'ssr_max', 
        'ssr_min', 'ssr_std'
    ]
    
    drop_cols = DISTRICT_COL + ID_COLS + STATE_DUMMIES if keep_climate else DISTRICT_COL + ID_COLS + STATE_DUMMIES + CLIMATE_COLS
        
    train_shap_df = train.copy().drop(drop_cols, axis=1)
    
    if print_cols:
        print(train_shap_df.columns)
    else:
        print('Feeling brave, aren\'t we?')
        
    if run_shapley:
        out_shap_df = make_xgb_shapley(insample_df=train_shap_df, dep_var='farm_crop_yield', 
                                       hyper_params=hyper_params)
        return plot_shapley(out_shap_df,title_str=title_str)  # Fixed: Added return statement here
    else:
        print('No shapley')
        return None, None, None

## running Shapley on seeds selected for performance
## running Shapley without Climate and with Climate features
hyper_param_dict={'hh':{1: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8},
                        4: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8},
                        92: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}},
                  'sgkf':{1: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1},
                          4:{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8},
                         92:{'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}}}
for split_level in ['hh','sgkf']:
    for random_state in [1,4,92]:
        hyper_params = hyper_param_dict[split_level][random_state]
        hyper_params['objective']="reg:pseudohubererror"
        print(f"==== {split_level}, seed: {random_state} ====")
        title_str=f"Shapley (w/o climate features) at split={split_level}, seed={random_state}"
        run_shapleys(split_level=split_level,random_state=random_state,
                     print_cols=True,hyper_params=hyper_params,title_str=title_str,keep_climate=False)

  hyper_param_dict={'hh':{1: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8},
                        4: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8},
                        92:{'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}},
                  'sgkf':{1: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8},
                          4:{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8},
                         92:{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}}}
for split_level in ['hh','sgkf']:
    for random_state in [1,4,92]:
        hyper_params = hyper_param_dict[split_level][random_state]
        hyper_params['objective']="reg:pseudohubererror"
        print(f"==== {split_level}, seed: {random_state} ====")
        title_str=f"Shapley (with climate features) at split={split_level}, seed={random_state}"
        run_shapleys(split_level=split_level,random_state=random_state,
                     print_cols=True,hyper_params=hyper_params,title_str=title_str,keep_climate=True)
