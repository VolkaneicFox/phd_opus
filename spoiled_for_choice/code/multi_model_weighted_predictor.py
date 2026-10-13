import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
random_states=[3504,2992,3529,5042,4053,2492,9193,6843,3923,5935]

class finna_be_playin:
    def __init__(self,insample,oosample,
                 depvar='yeo_farm_crop_yield',
                 random_forest=True,rf_params={},
                 svr=True,svr_params={},
                 xgboost=True,xgb_params={},
                 n_folds=5,state_space=2935):
        self.insample_data=insample
        #print(f'~~~ Loaded in-sample data {insample.split()[-1]}')
        self.oosample_data=oosample
        #print(f'~~~ Loaded out-of-sample data {oosample.split('//')[-1]}')
        self.n_folds = n_folds
        self.depvar=depvar
        self.state_space=state_space
        self.random_forest=random_forest
        self.svr=svr
        self.xgboost=xgboost
        self.rf_params=rf_params
        self.svr_params=svr_params
        self.xgb_params=xgb_params
        
    def _run_kfold_pipeline(self):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.state_space)
        self.inX=self.insample_data.drop(self.depvar,axis=1).reset_index(drop=True)
        self.iny=self.insample_data[self.depvar].reset_index(drop=True)
        self.X_oos=self.oosample_data.drop(self.depvar,axis=1).reset_index(drop=True)
        self.y_oos=self.oosample_data[self.depvar].reset_index(drop=True)
        rf_pred_oos=pd.DataFrame()
        svr_pred_oos=pd.DataFrame()
        xgb_pred_oos=pd.DataFrame()
        self.oos=pd.DataFrame()
        self.r2_score=pd.DataFrame()
        fold=1
        for train_idx, test_idx in kf.split(self.inX):
            print(f"\n=== Fold {fold} ===")
            X_train, X_test = self.inX.loc[train_idx], self.inX.loc[test_idx]
            y_train, y_test = self.iny.loc[train_idx], self.iny.loc[test_idx]
            
            if self.random_forest:
                rf_model = RandomForestRegressor(**self.rf_params)
                rf_model.fit(X_train,y_train)
                rf_pred_train=rf_model.predict(X_train)
                rf_pred_test=rf_model.predict(X_test)
                rf_pred_oos=rf_model.predict(self.X_oos)
                print(" =============================================== ")
                print("Performance of random forest")
                print(f"Train RMSE: {mean_squared_error(y_train, rf_pred_train, squared=False):.4f}")
                print(f"Train MAE : {mean_absolute_error(y_train, rf_pred_train):.4f}")
                print(f"Train R²  : {r2_score(y_train, rf_pred_train):.4f}")
                print()
                print(f"Test  RMSE: {mean_squared_error(y_test, rf_pred_test, squared=False):.4f}")
                print(f"Test  MAE : {mean_absolute_error(y_test, rf_pred_test):.4f}")
                print(f"Test  R²  : {r2_score(y_test, rf_pred_test):.4f}")
                print()
                self.r2_score.loc[fold,'random-forest']=r2_score(y_test, rf_pred_test)
                print(f"OoS  RMSE: {mean_squared_error(self.y_oos, rf_pred_oos, squared=False):.4f}")
                print(f"OoS  MAE : {mean_absolute_error(self.y_oos, rf_pred_oos):.4f}")
                print(f"OoS  R²  : {r2_score(self.y_oos, rf_pred_oos):.4f}")
                
                rf_pred_oos=pd.DataFrame(data={'rfy_pred':rf_pred_oos})
            if self.svr:
                svr_model=SVR(**self.svr_params)
                svr_model.fit(X_train,y_train)
                svr_pred_train=svr_model.predict(X_train)
                svr_pred_test=svr_model.predict(X_test)
                svr_pred_oos=svr_model.predict(self.X_oos)
                print(" =============================================== ")
                print("Performance of SVR")
                print(f"Train RMSE: {mean_squared_error(y_train, svr_pred_train, squared=False):.4f}")
                print(f"Train MAE : {mean_absolute_error(y_train, svr_pred_train):.4f}")
                print(f"Train R²  : {r2_score(y_train, svr_pred_train):.4f}")
                print()
                print(f"Test  RMSE: {mean_squared_error(y_test, svr_pred_test, squared=False):.4f}")
                print(f"Test  MAE : {mean_absolute_error(y_test, svr_pred_test):.4f}")
                print(f"Test  R²  : {r2_score(y_test, svr_pred_test):.4f}")
                self.r2_score.loc[fold,'svr']=r2_score(y_test, svr_pred_test)
                print()
                print(f"OoS  RMSE: {mean_squared_error(self.y_oos, svr_pred_oos, squared=False):.4f}")
                print(f"OoS  MAE : {mean_absolute_error(self.y_oos, svr_pred_oos):.4f}")
                print(f"OoS  R²  : {r2_score(self.y_oos, svr_pred_oos):.4f}")
                
                svr_pred_oos=pd.DataFrame(data={'svry_pred':svr_pred_oos})
            if self.xgboost:
                print("Performance of XGBoost")
                xgb_objectives = [('abs', 'reg:absoluteerror', None),
                                  ('q05', 'reg:quantileerror', 0.05),
                                  ('q10', 'reg:quantileerror', 0.10),
                                  ('q25', 'reg:quantileerror', 0.25),
                                  ('q50', 'reg:quantileerror', 0.50),
                                  ('q75', 'reg:quantileerror', 0.75),
                                  ('q90', 'reg:quantileerror', 0.90),
                                  ('q95', 'reg:quantileerror', 0.95),
                                  ('huber', 'reg:pseudohubererror', None)]

                xgb_pred_oosdf = pd.DataFrame()
                for mod, obj, alpha in xgb_objectives:
                    print(f' ~~~~~~~ {mod}, {obj}, {alpha} ~~~~~~~')
                    params=self.xgb_params.copy()
                    params['objective'] = obj
                    if alpha is not None:
                        params['quantile_alpha'] = alpha
                    model = XGBRegressor(**params)
                    model.fit(X_train, y_train)

                    xgb_pred_train = model.predict(X_train)
                    xgb_pred_test = model.predict(X_test)
                    xgb_pred_oos = model.predict(self.X_oos)

                    self.r2_score.loc[fold, f'xgb_{mod}'] = r2_score(y_test, xgb_pred_test)

                    xgb_pred_oosdf[f'xgb_{mod}_pred'] = xgb_pred_oos
                    print(f"Train RMSE: {mean_squared_error(y_train, xgb_pred_train, squared=False):.4f}")
                    print(f"Train MAE : {mean_absolute_error(y_train, xgb_pred_train):.4f}")
                    print(f"Train R²  : {r2_score(y_train, xgb_pred_train):.4f}")
                    print()
                    print(f"Test  RMSE: {mean_squared_error(y_test, xgb_pred_test, squared=False):.4f}")
                    print(f"Test  MAE : {mean_absolute_error(y_test, xgb_pred_test):.4f}")
                    print(f"Test  R²  : {r2_score(y_test, xgb_pred_test):.4f}")
                    print()
                    print(f"OoS  RMSE: {mean_squared_error(self.y_oos, xgb_pred_oos, squared=False):.4f}")
                    print(f"OoS  MAE : {mean_absolute_error(self.y_oos, xgb_pred_oos):.4f}")
                    print(f"OoS  R²  : {r2_score(self.y_oos, xgb_pred_oos):.4f}")
                    print("  -----------------  ")
            temp=pd.concat([rf_pred_oos,svr_pred_oos,xgb_pred_oosdf,self.y_oos,self.X_oos.copy()],axis=1)
            temp['fold']=fold
            self.oos=pd.concat([self.oos,temp],axis=0)
            fold+=1
