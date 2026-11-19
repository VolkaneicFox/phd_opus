"""
XGBoost runner for ONE combination of (split_level, random_state, keep_climate).

Loads the insample_/outsample_ files written by split_and_save, tunes on the
training file only, fits, predicts, saves predictions and returns metrics.

Assumed file names (match these to whatever split_and_save wrote):
    insample_{split_level}_{crop_name}_{random_state}.csv
    outsample_{split_level}_{crop_name}_{random_state}.csv
with split_level in {"hh", "dist"}.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from xgboost import XGBRegressor
import shap
main_path = ''

class XGBRunner:
    DISTRICT_COL = "nss_district_code19"
    ID_COLS = ["cluster", "spei_district_code", "nss_district_code19"]
    STATE_DUMMIES = [f"state_{i}" for i in (*range(2, 31), 32, 33, 34, 36)]
    PARAM_GRID = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1],
    }

    def __init__(self, data_path, dep_var, crop_name, split_level, random_state,
                 keep_climate,shuffle_climate, climate_cols=None, drop_cols=None, n_jobs=1):
        if split_level not in ("hh", "sgkf"):
            raise ValueError("split_level must be 'hh' or 'sgkf'")
        if not keep_climate and not climate_cols:
            raise ValueError("pass climate_cols (same list used for clustering) so they can be dropped")

        self.data_path = Path(data_path)
        self.dep_var = dep_var
        self.crop_name = crop_name
        self.split_level = split_level
        self.random_state = random_state
        self.keep_climate = keep_climate
        self.shuffle_climate=shuffle_climate
        self.n_jobs = n_jobs  # parallelism for the grid search; each XGB model uses 1 thread
        self.climate_cols = climate_cols
        self.drop_cols = self.STATE_DUMMIES + self.ID_COLS + list(drop_cols or [])
        if not keep_climate:
            self.drop_cols += list(self.climate_cols)

        self._load_data()
    def _load_data(self):
        stem = f"{self.split_level}_{self.crop_name}_{self.random_state}.csv"
        train = pd.read_csv(self.data_path / f"insample_{stem}", index_col="HHID")
        test = pd.read_csv(self.data_path / f"outsample_{stem}", index_col="HHID")

        # leftover row-index columns from to_csv() without index=False
        train = train.loc[:, ~train.columns.str.startswith("Unnamed")]
        test = test.loc[:, ~test.columns.str.startswith("Unnamed")]
        if self.keep_climate and self.shuffle_climate:
            n = len(train)
            full = pd.concat([train, test])                 
            full = self._shuffle_climate_by_district(full)
            train, test = full.iloc[:n], full.iloc[n:]

        self.groups_train = train[self.DISTRICT_COL]  # needed for grouped CV
        
        train = train.drop(columns=self.drop_cols)
        test = test.drop(columns=self.drop_cols)
            
        self.X_train, self.y_train = train.drop(columns=self.dep_var), train[self.dep_var]
        self.X_test, self.y_test = test.drop(columns=self.dep_var), test[self.dep_var]
        self.features = list(self.X_train.columns)

        print(f"[{self.split_level} | seed {self.random_state} | climate={self.keep_climate}] "
              f"train {self.X_train.shape}, test {self.X_test.shape}")
        
    def _shuffle_climate_by_district(self, df):
        rng = np.random.default_rng(self.random_state)

        g = df.groupby('nss_district_code19')[self.climate_cols]
        assert (g.nunique(dropna=False) <= 1).all().all(), "climate not constant within district"

        clim = g.first()                                   # one row per district
        perm = rng.permutation(clim.index.to_numpy())      # district i receives climate of district perm[i]
        placebo = clim.loc[perm].set_axis(clim.index)      # relabel back to the original district ids

        out = df.copy()
        out[self.climate_cols] = placebo.loc[out['nss_district_code19']].to_numpy()
        return out
        
    def _tune_xgb(self):
        """Tune on the training file only. For the district split the inner CV is
        grouped by district too, otherwise districts leak between inner folds."""
        if self.split_level == "sgkf":
            cv = GroupKFold(n_splits=5)
            fit_kw = {"groups": self.groups_train}
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            fit_kw = {}

        grid = GridSearchCV(
            XGBRegressor(objective="reg:pseudohubererror", n_jobs=1,
                         random_state=self.random_state),
            self.PARAM_GRID,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=self.n_jobs,
            verbose=1,
        )
        grid.fit(self.X_train, self.y_train, **fit_kw)
        print(f"Best XGBoost parameters: {grid.best_params_}")
        return grid.best_params_

    @staticmethod
    def _metrics(y, p):
        return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
                "mae": float(mean_absolute_error(y, p)),
                "r2": float(r2_score(y, p))}

    def run(self, xgb_params="tune", shuffle_y=False):
        tuning = isinstance(xgb_params, str) and xgb_params == "tune"
        if tuning and shuffle_y:
            raise ValueError("Don't tune on a shuffled-y check; pass fixed xgb_params.")
        params = self._tune_xgb() if tuning else xgb_params

        y_fit = self.y_train.to_numpy()
        if shuffle_y:
            print(f"Shuffling the dependent variable {self.dep_var}")
            y_fit = np.random.default_rng(self.random_state).permutation(y_fit)

        model = XGBRegressor(objective="reg:pseudohubererror", n_jobs=1,
                             random_state=self.random_state, **params)
        model.fit(self.X_train, y_fit)
        pred_train = model.predict(self.X_train)
        pred_test = model.predict(self.X_test)
        self.model, self.xgb_params = model, params

        m_train = self._metrics(y_fit, pred_train)
        m_test = self._metrics(self.y_test, pred_test)
        for name, m in (("Train", m_train), ("Test ", m_test)):
            print(f"{name} RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f} | R2: {m['r2']:.4f}")

        # provision to shuffle climate or other features
        tag = f"{self.split_level}_{'clim' if self.keep_climate else 'noclim'}_{self.crop_name}"
        if shuffle_y:
            tag += "_shuffled"
        for name, X, y, p in (("insample", self.X_train, self.y_train, pred_train),
                              ("outsample", self.X_test, self.y_test, pred_test)):
            pd.DataFrame({"random_state": self.random_state,
                          "sample": name,
                          "y_obs": y.to_numpy(),
                          "y_pred": p}, index=X.index
                         ).to_csv(self.data_path / f"{name}_{tag}_preds_{self.random_state}.csv")

        return {"split_level": self.split_level, "keep_climate": self.keep_climate,
                "random_state": self.random_state, "shuffled": shuffle_y,
                **{f"train_{k}": v for k, v in m_train.items()},
                **{f"test_{k}": v for k, v in m_test.items()}}

climate_cols=['mean_speiSPEI-4MO',
       'median_speiSPEI-4MO', 'max_speiSPEI-4MO', 'min_speiSPEI-4MO',
       'std_speiSPEI-4MO', 'tp_mean', 'tp_std', 't2m_mean', 't2m_max',
       't2m_min', 't2m_std', 'slhf_mean', 'slhf_max', 'slhf_min', 'slhf_std',
       'ssr_mean', 'ssr_max', 'ssr_min', 'ssr_std']

results_frame=[]
for random_state in [29,92,35,4,1]:
    for split in ['hh','sgkf']:
        main_call_35=XGBRunner(dep_var='farm_crop_yield',
                              data_path = main_path+'split_data/',
                              crop_name='rice', split_level=split, random_state=random_state,
                         keep_climate=False,shuffle_climate=False,climate_cols=climate_cols) ## change keep_climate and/or shuffle_climate
        temp_frame = main_call_35.run()
        results_frame.append(temp_frame)
