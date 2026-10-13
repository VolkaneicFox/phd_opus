import shap
import pandas as pd
import matplotlib.pyplot as plt

def make_xgb_shapley(insample_df,dep_var):
        
        inX,inY=insample_df.copy().drop(dep_var,axis=1),insample_df[dep_var]
        model = XGBRegressor(objective="reg:pseudohubererror",learning_rate= 0.05,
                             n_estimators= 1000,max_depth= 6,min_child_weight= 6,                  
                             subsample= 0.7,colsample_bytree= 0.7,reg_lambda= 2.5,
                             reg_alpha= 2,gamma= 0.7)
        model.fit(inX,inY)
        y_hat = model.predict(inX)

        print(f"RMSE: {mean_squared_error(inY, y_hat, squared=False):.4f}")
        print(f"MAE : {mean_absolute_error(inY, y_hat,):.4f}")
        print(f"R²  : {r2_score(inY, y_hat,):.4f}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(inX).values
        shap_df=pd.DataFrame(shap_values,columns=[f'shap_{i}' for i in inX.columns])
        shap_df=pd.concat([shap_df,inY],axis=1)
        return shap_df

def plot_shapley(shap_df):
    feature_names={'shap_yeo_farm_land_m2':'Farm Size ($m^2$)', 
               'shap_yeo_share_own_and_possessed':'Land Ownership Share', 
               'shap_yeo_cost_energy_pm2':'Energy Cost (p$m^2$)',
               'shap_yeo_cost_labour_pm2':'Labour Cost (p$m^2$)', 
               'shap_yeo_cost_finance_pm2':'Finance Cost (p$m^2$)',
               'shap_yeo_cost_fert_pm2':'Fertilizer Cost (p$m^2$)',
               'shap_yeo_cost_pest_pm2':'Pesticide Cost (p$m^2$)',
               'shap_yeo_cost_machine_pm2':'Machine Cost (p$m^2$)', 
               'shap_yeo_cost_irrigation_pm2':'Irrigation Cost (p$m^2$)', 
               'shap_yeo_cost_seeds_pm2':'Seeds Cost (p$m^2$)',
               'shap_yeo_hh_cons_exp_pm_pm2':'Hhld. Cons. Exp. (per month and p$m^2$)',
               'shap_yeo_median_speiSPEI-4MO':'SPEI (median)', 
               'shap_yeo_max_speiSPEI-4MO':'SPEI (max)',
               'shap_yeo_min_speiSPEI-4MO':'SPEI (min)', 
               'shap_yeo_std_speiSPEI-4MO':'SPEI (std)',
               'shap_yeo_tp_mean':'Precipitation (mean)',
               'shap_yeo_tp_std':'Precipitation (std)', 
               'shap_ln_t2m_mean':'Surface Temperature (mean)',
               'shap_yeo_t2m_max':'Surface Temperature (max)', 
               'shap_yeo_t2m_min':'Surface Temperature (min)', 
               'shap_yeo_t2m_std':'Surface Temperature (std)', 
               'shap_yeo_slhf_mean':'Surface Heat Flux (mean)', 
               'shap_yeo_slhf_max':'Surface Heat Flux (max)', 
               'shap_yeo_slhf_std':'Surface Heat Flux (std)', 
               'shap_yeo_ssr_mean':'Solar Radiation (mean)', 
               'shap_ln_ssr_max':'Solar Radiation (max)', 
               'shap_ln_ssr_min':'Solar Radiation (min)', 
               'shap_yeo_ssr_std':'Solar Radiation (std)', 
               'shap_if_land_irrigated':'If Land Irrigated', 
               'shap_is_non_backward_caste':'Not-Backward Caste',
               'shap_if_farmer_organisation':'Farmer Organisation', 
               'shap_if_kisan_credit_card':'Access to Farmer Credit', 
               'shap_if_soil_health_card':'Soil Health Card',
               'shap_primary_irrigation_canal':'Irrigation Type: Canal',
               'shap_primary_irrigation_gw':'Irrigation Type: GW', 
               'shap_primary_irrigation_mixed':'Irrigation Type: Mixed',
               'shap_primary_irrigation_msw':'Irrigation Type: Surface', 
               'shap_primary_irrigation_others':'Irrigation Type: Others'}
    abs_shap = ( shap_df.drop('yeo_farm_crop_yield', axis=1).abs().mean(axis=0).sort_values(ascending=False))

    top_features = abs_shap[abs_shap > abs_shap.quantile(0.79)].index.tolist()

    mean_shap = abs_shap.loc[top_features]
    violin_shap = rice_shap[top_features]

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

    ax_left.set_ylabel("SHAP value")
    ax_left.set_xticks(x_pos)
    
    top_features_labels=[feature_names[i] for i in top_features]
    ax_left.set_xticklabels(top_features_labels, rotation=45, ha='right')
    # ---- RIGHT AXIS: mean |SHAP| ----
    ax_right = ax_left.twinx()

    ax_right.plot(
        x_pos,
        mean_shap.values,
        marker='o',
        linewidth=2
    )

    ax_right.set_ylabel("Mean |SHAP value|")

    # ---- Aesthetics ----
    #ax_left.set_title("Feature Importance: Distribution and Mean Absolute SHAP")
    ax_left.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
