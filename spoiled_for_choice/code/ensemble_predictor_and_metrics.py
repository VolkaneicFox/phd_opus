import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def make_xid(new_oos,crop):
    x_dict={'rice':['yeo_farm_land_m2', 'yeo_share_own_and_possessed',
            'yeo_cost_energy_pm2', 'yeo_cost_labour_pm2', 'yeo_cost_finance_pm2',
            'yeo_cost_pest_pm2','yeo_cost_fert_pm2', 'yeo_cost_machine_pm2', 'yeo_cost_irrigation_pm2',
            'yeo_cost_seeds_pm2', 'yeo_hh_cons_exp_pm_pm2',
            'yeo_median_speiSPEI-4MO', 'yeo_max_speiSPEI-4MO',
            'yeo_min_speiSPEI-4MO', 'yeo_std_speiSPEI-4MO', 'yeo_tp_mean',
            'yeo_tp_std', 'ln_t2m_mean', 'yeo_t2m_max', 'yeo_t2m_min',
            'yeo_t2m_std', 'yeo_slhf_mean', 'yeo_slhf_max', 'yeo_slhf_std',
            'yeo_ssr_mean', 'ln_ssr_max', 'ln_ssr_min', 'yeo_ssr_std',
            'if_land_irrigated', 'is_non_backward_caste',
            'if_farmer_organisation', 'if_kisan_credit_card', 'if_soil_health_card',
            'primary_irrigation_canal', 'primary_irrigation_gw',
            'primary_irrigation_mixed', 'primary_irrigation_msw','state_10', 'state_11',
            'state_12', 'state_13', 'state_14', 'state_15', 'state_16', 'state_17',
            'state_18', 'state_19', 'state_2', 'state_20', 'state_21', 'state_22',
            'state_23', 'state_24', 'state_25', 'state_26', 'state_27', 'state_28',
            'state_29', 'state_3', 'state_30', 'state_32', 'state_33', 'state_34',
            'state_36', 'state_4', 'state_5', 'state_6', 'state_7', 'state_8',
            'state_9'],
            'wheat':['yeo_farm_land_m2', 'yeo_share_own_and_possessed',
            'yeo_cost_energy_pm2','yeo_cost_labour_pm2', 'yeo_cost_finance_pm2', 
            'yeo_cost_fert_pm2','yeo_cost_pest_pm2',
            'yeo_cost_machine_pm2', 'yeo_cost_irrigation_pm2', 'yeo_cost_seeds_pm2',
            'yeo_hh_cons_exp_pm_pm2','yeo_median_speiSPEI-4MO', 'yeo_max_speiSPEI-4MO',
            'yeo_min_speiSPEI-4MO','yeo_std_speiSPEI-4MO','yeo_tp_mean','yeo_tp_std', 'ln_t2m_mean',
            'yeo_t2m_max', 'yeo_t2m_min','yeo_t2m_std', 'yeo_slhf_mean', 'yeo_slhf_max', 'yeo_slhf_std',
            'yeo_ssr_mean', 'ln_ssr_max', 'ln_ssr_min', 'yeo_ssr_std',
            'if_land_irrigated', 'is_non_backward_caste','if_farmer_organisation', 
            'if_kisan_credit_card', 'if_soil_health_card','primary_irrigation_canal',
            'primary_irrigation_gw', 'primary_irrigation_mixed',
            'primary_irrigation_msw', 'primary_irrigation_others',
            'state_10','state_11', 'state_12', 'state_13', 'state_14', 'state_15', 'state_16',
            'state_17', 'state_18', 'state_19', 'state_2', 'state_20', 'state_21',
            'state_22', 'state_23', 'state_24', 'state_25', 'state_26', 'state_27',
            'state_28', 'state_29', 'state_3', 'state_30', 'state_32', 'state_33',
            'state_34', 'state_36', 'state_4', 'state_5', 'state_6', 'state_7','state_8', 'state_9']}
    x_cols=x_dict[crop]
    new_oos['X_id']=new_oos.groupby(x_cols).ngroup()
    print('We have unique X-row identifiers : ',(new_oos['X_id'].nunique()==(new_oos.shape[0]/15))) ## True
    return new_oos

def ensemble_estimator(xid_df,r2_wts,flag='range'):
    model_list=['rfy_pred','svry_pred','xgb_abs_pred','xgb_q50_pred','xgb_huber_pred']
    
    xid_df=xid_df[model_list]
    y_hats=xid_df.values
    
    '''
    for each X_id, there are CV x M number of predictions - we want to combine them
    the weights are M x 1 => CV x 1 weighted predictions and the sum of these weig-
    hted predictions is the ensemble prediction.
    '''
    
    if flag=='range':
        dispersion=(xid_df.max(axis=0)-xid_df.min(axis=0)).values    
    elif flag=='var':
        dispersion=(xid_df.std(axis=0).values)
    else:
        dispersion=np.array([1.0,1.0,1.0,1.0,1.0])
    dispersion += 1e-9
    weights=r2_wts/dispersion
    weights=weights/sum(weights)
    y_hat_star=(y_hats @ weights).mean()
    
    most_stable, least_stable = weights.argmax(),weights.argmin()
    return [y_hat_star,model_list[most_stable],model_list[least_stable]]

def make_ensemble_predictions(new_oos,wts_star):

    farms_list=list(set(new_oos.X_id))
    range_model_rank=pd.DataFrame(columns=['y_star','y_obs','most_stable','least_stable'],index=farms_list)
    var_model_rank=pd.DataFrame(columns=['y_star','y_obs','most_stable','least_stable'],index=farms_list)
    mean_model_rank=pd.DataFrame(columns=['y_star','y_obs','most_stable','least_stable'],index=farms_list)
    r2_oos=pd.DataFrame()

    for farm in farms_list:
        temp=new_oos[new_oos['X_id']==farm].copy()
        y_obs=temp['yeo_farm_crop_yield'].mean()
        range_estimator=ensemble_estimator(xid_df=temp,
                                           r2_wts=wts_star.values)
        range_model_rank.loc[farm,'y_obs']=y_obs
        range_model_rank.loc[farm,'y_star']=range_estimator[0]
        range_model_rank.loc[farm,'most_stable']=range_estimator[1]
        range_model_rank.loc[farm,'least_stable']=range_estimator[2]

        variance_estimator=ensemble_estimator(xid_df=temp,
                                              r2_wts=wts_star.values,flag='var')
        var_model_rank.loc[farm,'y_star']=variance_estimator[0]
        var_model_rank.loc[farm,'y_obs']=y_obs
        var_model_rank.loc[farm,'most_stable']=variance_estimator[1]
        var_model_rank.loc[farm,'least_stable']=variance_estimator[2]

        mean_estimator=ensemble_estimator(xid_df=temp,
                                          r2_wts=wts_star.values,flag='mean')
        mean_model_rank.loc[farm,'y_star']=mean_estimator[0]
        mean_model_rank.loc[farm,'y_obs']=y_obs
        
    return [range_model_rank,var_model_rank,mean_model_rank]

def measure_performance(new_oos,models_list):
    oos_pred_df=new_oos[models_list+['yeo_farm_crop_yield','fold']].copy()
    r2_oos_df=pd.DataFrame(dtype=float)
    rmse_oos_df=pd.DataFrame(dtype=float)
    mae_oos_df=pd.DataFrame(dtype=float)
    for fold,temp in new_oos.groupby('fold'):
        y_obs=temp.yeo_farm_crop_yield.values
        for model in models_list:
            y_pred=temp[model].values
            r2_oos_df.loc[(fold+1),model]=r2_score(y_obs,y_pred)
            rmse_oos_df.loc[(fold+1),model]=mean_squared_error(y_obs,y_pred,squared=False)
            mae_oos_df.loc[(fold+1),model]=mean_absolute_error(y_obs,y_pred)     
    return {'r2':r2_oos_df, 'rmse':rmse_oos_df,'mae':mae_oos_df}

def candidate_performance(new_oos,models_list=['rfy_pred','svry_pred','xgb_abs_pred',
                                               'xgb_q50_pred','xgb_huber_pred']):
    temp=measure_performance(new_oos,models_list=['rfy_pred','svry_pred','xgb_abs_pred',
                                               'xgb_q50_pred','xgb_huber_pred'])
    r2_best=np.round(temp['r2'],4).max(axis=0).values
    rmse_best=np.round(temp['rmse'],4).min(axis=0).values
    mae_best=np.round(temp['mae'],4).min(axis=0).values
    
    return {'r2_best':r2_best, 'rmse_best':rmse_best,'mae_best':mae_best}
