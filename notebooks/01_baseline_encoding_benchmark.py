# %%
import sys, traceback
import warnings
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as _sp


from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)

import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# encoders
import category_encoders as ce

# boruta (pipeline-friendly wrapper defined below)
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import time

warnings.filterwarnings("ignore", category=UserWarning)

# ==== SPEED / SCOPE TOGGLES ====
FAST_DEV = False  

# Encoders used when Boruta is in the loop 
ENCODERS_BORUTA = ["TargetEncoder", "CatBoostEncoder", "JamesSteinEncoder", "GLMMEncoder"]

# All encoders
ENCODERS_ALL = [
    "TargetEncoder", "CatBoostEncoder",
    "JamesSteinEncoder", "GLMMEncoder", 
]

# Fast strong algos for quick runs
ALGOS_FAST = ["RandomForest", "XGBoost", "LGBM"]

# Full set for thesis run
ALGOS_ALL = ["Ridge", "RandomForest", "XGBoost", "CatBoost", "gbr", "ExtraTrees", "LGBM"]


# %%
def load_ames_data():
    try:
        train_data = pd.read_csv("../data/raw/ames-housing-dataset/AmesHousing.csv", index_col=0)

        # Fill missing
        cat_cols = train_data.select_dtypes(include="object").columns.tolist()
        num_cols = train_data.select_dtypes(include="number").columns.tolist()
        train_data[cat_cols] = train_data[cat_cols].fillna("missing")
        train_data[num_cols] = train_data[num_cols].fillna(-1)

        # Exclude
        X = train_data.drop(columns=["Order", "PID", "SalePrice"], errors="ignore")
        y = train_data["SalePrice"]

        cat_columns = X.select_dtypes(include="object").columns.tolist()
        num_columns = X.select_dtypes(include="number").columns.tolist()
        return X, y, cat_columns, num_columns
    except FileNotFoundError:
        print("Dataset was not found!")
        return None


# %%
DATASETS_TO_RUN = {"Ames_Housing": load_ames_data}


# %%
class BorutaSelector(BaseEstimator, TransformerMixin):
    """
    Boruta Selector class definition
    """
    def __init__(self, n_estimators=100, alpha=0.05, perc=100,
                 two_step=True, max_iter=20, random_state=42, n_jobs=-1, verbose=0,
                 base_estimator="extratrees"):
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.perc = perc
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.base_estimator = base_estimator
        self._boruta = None
        self.support_ = None
        self.feature_names_in_ = None

    def _as_2d(self, X):
        if _sp.issparse(X):
            return X.tocsr()
        A = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return A if A.ndim == 2 else A.reshape(-1, 1)

    def fit(self, X, y):
        if self.base_estimator == "extratrees":
            est = ExtraTreesRegressor(
                n_estimators=self.n_estimators, n_jobs=self.n_jobs, random_state=self.random_state
            )
        else:
            est = RandomForestRegressor(
                n_estimators=self.n_estimators, n_jobs=self.n_jobs, random_state=self.random_state
            )

        self._boruta = BorutaPy(
            estimator=est,
            n_estimators=100,       
            alpha=self.alpha,
            perc=self.perc,
            two_step=self.two_step,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=self.verbose
        )

        X_arr = self._as_2d(X)
        y_arr = np.asarray(y)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        self._boruta.fit(X_arr, y_arr)
        self.support_ = self._boruta.support_.astype(bool)
        return self

    def get_support(self, indices: bool = False):
        check_is_fitted(self, "support_")
        return np.flatnonzero(self.support_) if indices else self.support_

    def transform(self, X):
        check_is_fitted(self, "support_")
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != self.support_.size:
                raise ValueError("Column mismatch: making sure Boruta sees the same columns as the preprocessor output.")
            return X.iloc[:, np.flatnonzero(self.support_)].copy()

        X2 = self._as_2d(X)
        if X2.shape[1] != self.support_.size:
            raise ValueError("Feature mismatch: making sure Boruta sees the same columns as the preprocessor output.")
        return X2[:, np.flatnonzero(self.support_)]


# %%
class EncodingBenchmark:
    def __init__(self, random_state=42):
        self.random_state = random_state

        # Encoders (OHE sparse; others numeric outputs)
        self.encoders = {
            "TargetEncoder": ce.TargetEncoder(smoothing=25, min_samples_leaf=20),
            "CatBoostEncoder": ce.CatBoostEncoder(sigma=0.5, a=1),
            "JamesSteinEncoder": ce.JamesSteinEncoder(),
            "GLMMEncoder": ce.GLMMEncoder(),
        }

        if FAST_DEV:
            self.algorithms = {
                "Ridge": Ridge(alpha=1, random_state=self.random_state),
                "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=self.random_state),
                "XGBoost": xgb.XGBRegressor(
                    n_estimators=50, learning_rate=0.1, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    random_state=self.random_state, n_jobs=-1, verbosity=0
                ),
                "CatBoost": CatBoostRegressor(
                    iterations=100, learning_rate=0.09, depth=6,
                    random_state=self.random_state, verbose=False, thread_count=-1
                ),
                "gbr": GradientBoostingRegressor(        n_estimators=100, learning_rate=0.08, max_depth=3,
                random_state=42),
                "ExtraTrees": ExtraTreesRegressor(
                    n_estimators=50,max_depth=12, max_features="sqrt", random_state=self.random_state, n_jobs=-1
                ),
                "LGBM": LGBMRegressor(
                    n_estimators=50, learning_rate=0.1, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, random_state=self.random_state, n_jobs=-1, verbose=-1
                ),
            }


        else:
            self.algorithms = {
                "Ridge": Ridge(alpha=1, random_state=self.random_state),
                "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=self.random_state),
                "XGBoost": xgb.XGBRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=8,
                    subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                    random_state=self.random_state, n_jobs=-1, verbosity=0
                ),
                "CatBoost": CatBoostRegressor(
                    iterations=100, learning_rate=0.09, depth=8,
                    random_state=self.random_state, verbose=False, thread_count=-1, 
                ),
                "gbr": GradientBoostingRegressor(        n_estimators=100, learning_rate=0.08, max_depth=3,
                random_state=42),
                "ExtraTrees": ExtraTreesRegressor(
                    n_estimators=100, max_features="sqrt", random_state=self.random_state, n_jobs=-1
                ),
                "LGBM": LGBMRegressor(
                    n_estimators=100, learning_rate=0.1, num_leaves=31,
                    subsample=0.9, colsample_bytree=0.9, random_state=self.random_state, n_jobs=-1, verbose=-1
                ),
            }

    # ---------- helpers ----------

    # Fast run make preprocessor        
    def _make_preprocessor(self, encoder, cat_columns, num_columns):
        cat_steps = [("imp", SimpleImputer(strategy="most_frequent")),
                 ("enc", encoder)]

        categorical_pipe = Pipeline(cat_steps)
        numerical_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())])

        return ColumnTransformer([
        ("cat", categorical_pipe, cat_columns),
        ("num", numerical_pipe, num_columns),
    ])

    """    def _make_preprocessor(self, encoder, cat_columns, num_columns, densify_for_ridge=False):
        cat_steps = [("imp", SimpleImputer(strategy="most_frequent")), ("enc", encoder)]
        if densify_for_ridge and isinstance(encoder, OneHotEncoder):
            # densify after OHE for Ridge only
            cat_steps.append(("to_dense", FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)))

        categorical_pipe = Pipeline(cat_steps)
        numerical_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])

        return ColumnTransformer([
            ("cat", categorical_pipe, cat_columns),
            ("num", numerical_pipe, num_columns),
        ])"""
    
    ## fast run create pipeline
    def create_pipeline(self, encoder_name, algorithm_name, cat_columns, num_columns,
                    apply_encoding=True, apply_boruta=False):

        alg = self.algorithms[algorithm_name]

        if apply_encoding:
            enc = self.encoders.get(encoder_name)
            if enc is None:
                raise ValueError(f"Unknown encoder: {encoder_name}")
        else:
            enc = SklearnOrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


        preprocessor = self._make_preprocessor(enc, cat_columns, num_columns)

        steps: List[Tuple[str, Any]] = [("preprocessor", preprocessor)]
        if apply_boruta:
            steps.append(("boruta", BorutaSelector(
            n_estimators=100, alpha=0.05, perc=100, two_step=True,
            max_iter=20 if FAST_DEV else 100,
            random_state=self.random_state, n_jobs=-1, verbose=0
        )))
        steps.append(("algorithm", alg))
        return Pipeline(steps)

    # ---------- evaluations ----------
    def evaluate_combination(self, X, y, encoder_name, algorithm_name,
                             cat_columns, num_columns, apply_encoding=True, scenario=""):
        try:
            pipe = self.create_pipeline(encoder_name, algorithm_name, cat_columns, num_columns,
                                        apply_encoding=apply_encoding, apply_boruta=False)
            cv = getattr(self, "_cv_simple", RepeatedKFold(n_splits=3, n_repeats=1, random_state=self.random_state))

            rmse = np.sqrt(-cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_squared_error"))
            r2   = cross_val_score(pipe, X, y, cv=cv, scoring="r2")

            return {
                "scenario": scenario,
                "encoder": encoder_name,
                "algorithm": algorithm_name,
                "rmse_mean": rmse.mean(),
                "rmse_std": rmse.std(),
                "r2_mean": r2.mean(),
                "r2_std": r2.std(),
                "number_features": X.shape[1],
            }
        except Exception as e:
            print(f"❌ Failure in {scenario} | alg={algorithm_name} | enc={encoder_name} | apply_encoding={apply_encoding}")
            print("Reason:", e)
            traceback.print_exc(file=sys.stdout)
            return None

    def evaluate_combination_nested(self, X, y, encoder_name, algorithm_name,
                                    cat_columns, num_columns, apply_encoding=True, scenario=""):
        """
        Feature selection happens inside each foldvia Boruta step in the pipeline in order to avoid data leakage .
        """
        try:
            use_boruta = scenario.startswith("S3_With_Boruta") or scenario.startswith("S4_With_Boruta")
            cv = getattr(self, "_cv_nested", RepeatedKFold(n_splits=3, n_repeats=1, random_state=self.random_state))

            rmse_scores, r2_scores, feature_counts = [], [], []

            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipe = self.create_pipeline(
                    encoder_name, algorithm_name, cat_columns, num_columns,
                    apply_encoding=apply_encoding, apply_boruta=use_boruta
                )

                pipe.fit(X_train, y_train)

                # Count features after boruta if present
                pre_dim = pipe.named_steps["preprocessor"].transform(X_train.iloc[:1]).shape[1]
                bor = pipe.named_steps.get("boruta")
                kept = int(bor.support_.sum()) if (bor is not None and hasattr(bor, "support_")) else pre_dim
                feature_counts.append(kept)

                y_pred = pipe.predict(X_test)
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2_scores.append(r2_score(y_test, y_pred))

            return {
                "scenario": scenario,
                "encoder": encoder_name,
                "algorithm": algorithm_name,
                "rmse_mean": float(np.mean(rmse_scores)),
                "rmse_std": float(np.std(rmse_scores)),
                "r2_mean": float(np.mean(r2_scores)),
                "r2_std": float(np.std(r2_scores)),
                "number_features": int(np.mean(feature_counts)),
            }
        except Exception as e:
            print(f"❌ Failure in {scenario} | alg={algorithm_name} | enc={encoder_name} | apply_encoding={apply_encoding}")
            print("Reason:", e)
            traceback.print_exc(file=sys.stdout)
            return None
        

    ## fast run scenarios

    def run_scenarios(self, X, y, cat_columns, num_columns):
        results = []
        start_time = time.perf_counter()

    # CV + scope per mode
        if FAST_DEV:
            self._cv_simple = RepeatedKFold(n_splits=2, n_repeats=1, random_state=self.random_state)
            self._cv_nested = RepeatedKFold(n_splits=2, n_repeats=1, random_state=self.random_state)

        # Very few encoders and algorithms for speed
            algos_s2 = algos_s34 = ["RandomForest"]
            enc_s2 = ["JamesSteinEncoder", "GLMMEncoder"]
            enc_s34 = ["JamesSteinEncoder", "GLMMEncoder"]
        else:
            self._cv_simple = RepeatedKFold(n_splits=5, n_repeats=10, random_state=self.random_state)
            self._cv_nested = RepeatedKFold(n_splits=5, n_repeats=10, random_state=self.random_state)
            algos_s2 = algos_s34 = ALGOS_ALL
            enc_s2, enc_s34 = ENCODERS_ALL, ENCODERS_BORUTA

    # Scenario 1: No encoding, no Boruta
        print("Running Scenario 1: No feature selection and no encoding")
        for alg in algos_s2:
            print(f"  Testing: {alg} (OrdinalEncoder only)")
            r = self.evaluate_combination(
            X, y,
            encoder_name="OrdinalEncoder",  # <-- fixed
            algorithm_name=alg,
            cat_columns=cat_columns, num_columns=num_columns,
            apply_encoding=False, scenario="S1_No_Selection_No_Encoding"
        )
            if r:
                results.append(r)
        print(f"⏱ S1 done in {(time.perf_counter() - start_time)/60:.1f} min\n")
        start_time = time.perf_counter()

    # Scenario 2: Encoding, no Boruta
        print("Running Scenario 2: No feature selection, with encoding")
        for enc in enc_s2:
            for alg in algos_s2:
                print(f"  Testing: {alg} + {enc}")
                r = self.evaluate_combination(
                X, y, enc, alg, cat_columns, num_columns,
                apply_encoding=True, scenario="S2_No_Selection_With_Encoding"
            )
                if r:
                    results.append(r)
        print(f"⏱ S2 done in {(time.perf_counter() - start_time)/60:.1f} min\n")
        start_time = time.perf_counter()

    # Scenario 3: Boruta, no encoding
        print("Running Scenario 3: With Boruta, no encoding (OrdinalEncoder only)")
        for alg in algos_s34:
            print(f"  Testing: {alg} (Boruta + OrdinalEncoder)")
            r = self.evaluate_combination_nested(
            X, y, encoder_name="OrdinalEncoder", algorithm_name=alg,
            cat_columns=cat_columns, num_columns=num_columns,
            apply_encoding=False, scenario="S3_With_Boruta_No_Encoding"
        )
            if r:
                results.append(r)
        print(f"⏱ S3 done in {(time.perf_counter() - start_time)/60:.1f} min\n")
        start_time = time.perf_counter()

    # Scenario 4: Boruta + encoding
        print("Running Scenario 4: With Boruta and encoding")
        for enc in enc_s34:
            for alg in algos_s34:
                print(f"  Testing: {alg} + {enc} (Boruta)")
                r = self.evaluate_combination_nested(
                X, y, enc, alg, cat_columns, num_columns,
                apply_encoding=True, scenario="S4_With_Boruta_With_Encoding"
            )
                if r:
                    results.append(r)
        print(f"⏱ S4 done in {(time.perf_counter() - start_time)/60:.1f} min\n")

        return pd.DataFrame(results)
            

        
    
    def analyze_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        df = results_df.copy()
        for col in ["rmse_mean", "r2_mean"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["rmse_mean", "r2_mean"])

        print("\nBest Performance by Scenario (RMSE):")
        for scen in df["scenario"].unique():
            s = df[df["scenario"] == scen]
            best = s.loc[s["rmse_mean"].idxmin()]
            print(f"{scen}: {best['encoder']} + {best['algorithm']} → RMSE {best['rmse_mean']:,.0f} (R² {best['r2_mean']:.3f})")

        print("\nBest Performance by Scenario (R²):")
        for scen in df["scenario"].unique():
            s = df[df["scenario"] == scen]
            best = s.loc[s["r2_mean"].idxmax()]
            print(f"{scen}: {best['encoder']} + {best['algorithm']} → R² {best['r2_mean']:.3f} (RMSE {best['rmse_mean']:,.0f})")

        print("\nScenario Comparison (Average RMSE):")
        for scen, v in df.groupby("scenario")["rmse_mean"].mean().sort_values().items():
            print(f"  {scen}: {v:,.0f}")

        print("\nScenario Comparison (Average R²):")
        for scen, v in df.groupby("scenario")["r2_mean"].mean().sort_values(ascending=False).items():
            print(f"  {scen}: {v:.3f}")

        with_sel = df["scenario"].str.contains("With_Boruta", na=False)
        no_sel = df["scenario"].str.contains("No_Selection", na=False)

        print("\nFeature Selection Impact (Boruta):")
        print(f"  RMSE  — With: {df.loc[with_sel,'rmse_mean'].mean():,.0f} | Without: {df.loc[no_sel,'rmse_mean'].mean():,.0f}")
        print(f"  R²    — With: {df.loc[with_sel,'r2_mean'].mean():.3f} | Without: {df.loc[no_sel,'r2_mean'].mean():.3f}")

        with_enc = df["scenario"].str.contains("With_Encoding", na=False)
        no_enc   = df["scenario"].str.contains("No_Encoding", na=False)
        print("\nEncoding Impact:")
        print(f"  RMSE  — With: {df.loc[with_enc,'rmse_mean'].mean():,.0f} | Without: {df.loc[no_enc,'rmse_mean'].mean():,.0f}")
        print(f"  R²    — With: {df.loc[with_enc,'r2_mean'].mean():.3f} | Without: {df.loc[no_enc,'r2_mean'].mean():.3f}")

        best_rmse = df.loc[df["rmse_mean"].idxmin()]
        best_r2   = df.loc[df["r2_mean"].idxmax()]
        print("\nGlobal Best:")
        print(f"  RMSE → {best_rmse['encoder']} + {best_rmse['algorithm']} in {best_rmse['scenario']} "
              f"(RMSE {best_rmse['rmse_mean']:,.0f}, R² {best_rmse['r2_mean']:.3f})")
        print(f"  R²   → {best_r2['encoder']} + {best_r2['algorithm']} in {best_r2['scenario']} "
              f"(R² {best_r2['r2_mean']:.3f}, RMSE {best_r2['rmse_mean']:,.0f})")

        corr = df["rmse_mean"].corr(df["r2_mean"])
        print(f"\nRMSE–R² correlation: {corr:.3f} (expect strong negative)")
        return df

    

    def visualize_results(self, results_df, dataset_name='dataset'):
        scenario_list = sorted(results_df['scenario'].unique())
        color_map = {
            scen: c for scen, c in zip(
                scenario_list, sns.color_palette("Set2", n_colors=len(scenario_list))
            )
        }

        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle(
            f"4-Scenario Benchmark: {dataset_name}\nFeature Selection × Encoding Impact",
            fontsize=18, fontweight='bold'
        )

        # 1) Average RMSE by scenario
        scen_rmse = results_df.groupby('scenario')['rmse_mean'].mean().reindex(scenario_list)
        axes[0, 0].barh(scen_rmse.index, scen_rmse.values,
                        color=[color_map[s] for s in scen_rmse.index])
        axes[0, 0].set_xlabel('Average RMSE (lower = better)')
        axes[0, 0].set_title('Average RMSE by Scenario')
        axes[0, 0].invert_yaxis()

        # 2) Average R² by scenario
        scen_r2 = results_df.groupby('scenario')['r2_mean'].mean().reindex(scenario_list)
        axes[0, 1].barh(scen_r2.index, scen_r2.values,
                        color=[color_map[s] for s in scen_r2.index])
        axes[0, 1].set_xlabel('Average R² (higher = better)')
        axes[0, 1].set_title('Average R² by Scenario')
        axes[0, 1].invert_yaxis()

        # 3) Encoder RMSE across scenarios
        enc_rmse = results_df.groupby(['encoder', 'scenario'])['rmse_mean'].mean().unstack()
        enc_rmse.plot(kind='bar', ax=axes[1, 0], rot=45,
                      color=[color_map[s] for s in enc_rmse.columns])
        axes[1, 0].set_ylabel('RMSE'); axes[1, 0].set_title('Encoder RMSE Across Scenarios')
        axes[1, 0].legend(title='Scenario', fontsize=8, loc='best')

        # 4) Encoder R² across scenarios
        enc_r2 = results_df.groupby(['encoder', 'scenario'])['r2_mean'].mean().unstack()
        enc_r2.plot(kind='bar', ax=axes[1, 1], rot=45,
                    color=[color_map[s] for s in enc_r2.columns])
        axes[1, 1].set_ylabel('R²'); axes[1, 1].set_title('Encoder R² Across Scenarios')
        axes[1, 1].legend(title='Scenario', fontsize=8, loc='best')

        # 5) Feature count vs RMSE
        for scen in scenario_list:
            sd = results_df[results_df['scenario'] == scen]
            axes[2, 0].scatter(sd['number_features'], sd['rmse_mean'],
                               alpha=0.7, s=100, color=color_map[scen], label=scen)
        axes[2, 0].set_xlabel('Number of Features'); axes[2, 0].set_ylabel('RMSE')
        axes[2, 0].set_title('Feature Count vs RMSE'); axes[2, 0].legend(fontsize=8, ncol=2)

        # 6) Feature count vs R²
        for scen in scenario_list:
            sd = results_df[results_df['scenario'] == scen]
            axes[2, 1].scatter(sd['number_features'], sd['r2_mean'],
                               alpha=0.7, s=100, color=color_map[scen], label=scen)
        axes[2, 1].set_xlabel('Number of Features'); axes[2, 1].set_ylabel('R²')
        axes[2, 1].set_title('Feature Count vs R²'); axes[2, 1].legend(fontsize=8, ncol=2)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = f'../results/{dataset_name}_four_scenario_comparison_RMSE_R2.png'
        plt.savefig(out, dpi=300, bbox_inches='tight'); plt.show()
        print(f"✓ Combined RMSE & R² visualization saved to '{out}'")

# %%
print("\n" + "🔬 ENCODING BENCHMARK SUITE".center(70, "="))
print(f"Running benchmarks on {len(DATASETS_TO_RUN)} dataset(s)\n")

all_results = {}

for dataset_name, loader in DATASETS_TO_RUN.items():
    print("\n" + "▶️  " + f"DATASET: {dataset_name}".center(68, " "))
    print("="*70)

    data = loader()
    if data is None:
        print(f"⚠️  Skipping {dataset_name} due to loading error\n")
        continue

    X, y, cat_columns, num_columns = data

    bench = EncodingBenchmark(random_state=42)

    print("Starting 4-scenario benchmark...\n")
    results = bench.run_scenarios(X, y, cat_columns, num_columns)

    print("\n" + "📊 ANALYSIS".center(70, "-"))
    bench.analyze_results(results)

    print("\n" + "📈 VISUALIZATION".center(70, "-"))
    bench.visualize_results(results, dataset_name=dataset_name)

    out_csv = f'../results/{dataset_name}_benchmark_results.csv'
    results.to_csv(out_csv, index=False)
    print(f"✓ Results saved to: {out_csv}")

    all_results[dataset_name] = results
    print("\n" + "✅ BENCHMARK COMPLETE".center(70, "=") + "\n")

print("\n🎉 ALL BENCHMARKS COMPLETED!")


