import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_results(scores, models, X_train, y_train_log, oof_preds):

    plt.figure(figsize=(10, 6))
    score_list = list(scores.values())
    model_names = list(scores.keys())
    
    sns.barplot(x=model_names, y=score_list, palette='viridis')
    plt.title('Model Comparison (RMSLE - lower is better)')
    plt.ylabel('RMSLE')
    plt.savefig(FIGURES_DIR / "model_comparison.png")
    plt.close()
    #featyure importance for xgboost
    if 'XGBoost' in models:
        xgb_model = models['XGBoost']
        try:
            xgb_model.fit(X_train, y_train_log)
            importances = getattr(xgb_model, "feature_importances_", None)
            if importances is not None:
                feature_names = X_train.columns
                feature_importance_df = (
                    pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    .sort_values(by='Importance', ascending=False)
                    .head(15)
                )

                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
                plt.title('Top 15 Feature Importances (XGBoost)')
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / "feature_importance_xgb.png")
                plt.close()
        except Exception as e:
            print(f"[WARN] Could not plot XGBoost feature importances: {e}")

    #
    preferred = None
    for k in ["Stacking", "Weighted"]:
        if k in oof_preds:
            preferred = k
            break
    if preferred is None:
        # pick the lowest RMSLE among base models
        base_keys = [k for k in oof_preds.keys() if k in scores]
        preferred = min(base_keys, key=lambda k: scores.get(k, np.inf)) if base_keys else None

    if preferred is not None:
        y_true = np.expm1(np.asarray(y_train_log))
        y_pred = np.clip(np.expm1(np.asarray(oof_preds[preferred])), 0, None)
        df_scatter = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        plt.figure(figsize=(7, 7))
        sns.scatterplot(data=df_scatter.sample(min(len(df_scatter), 3000), random_state=RANDOM_STATE),
                        x="y_true", y="y_pred", alpha=0.25, s=12)
        max_v = float(np.nanmax([df_scatter["y_true"].max(), df_scatter["y_pred"].max()]))
        plt.plot([0, max_v], [0, max_v], linestyle="--", color="black", linewidth=1)
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"OOF Predicted vs True (log-log) — {preferred}")
        plt.xlabel("True price")
        plt.ylabel("Predicted price")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"oof_pred_vs_true_{preferred.lower()}.png")
        plt.close()

        # 4. Residual distribution (log space residuals)
        resid = np.asarray(y_train_log) - np.asarray(oof_preds[preferred])
        plt.figure(figsize=(10, 4))
        sns.histplot(resid, bins=50, kde=True)
        plt.title(f"OOF Residuals (log space) — {preferred}")
        plt.xlabel("log1p(y_true) - log1p(y_pred)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"oof_residuals_{preferred.lower()}.png")
        plt.close()

    print(f"Model visualizations saved to {FIGURES_DIR}")



RANDOM_STATE = 42
N_FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
FIGURES_DIR = BASE_DIR / "figures"



TRAIN_PATH = PROCESSED_DIR / "airbnb_train_last.csv"
TEST_PATH = PROCESSED_DIR / "airbnb_test_last.csv"
RAW_TEST_PATH = RAW_DIR / "test.csv"
OUTPUT_PATH = SUBMISSIONS_DIR / "submission_fnl.csv"


def rmsle(y_true, y_pred):
    """Calculates Root Mean Squared Logarithmic Error."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def evaluate(y_true_log, y_pred_log):
    """Evaluates model performance in both log and original scale."""
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    return {
        'RMSLE': rmsle(y_true, y_pred),
        'R2': r2_score(y_true_log, y_pred_log)
    }


def get_xgboost_model():
    return xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,         
        learning_rate=0.03,  
        subsample=0.7,        
        colsample_bytree=0.7,
        min_child_weight=5,    
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

def get_lightgbm_model():
    return lgb.LGBMRegressor(
        n_estimators=600,
        max_depth=10,         
        learning_rate=0.03,
        num_leaves=31,       
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,        #  L1 regularization
        reg_lambda=1.0,       # L2 regularization
        min_child_samples=50, 
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )


def load_processed_pipeline_data():
 
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Processed training file not found ")
    
    train_df = pd.read_csv(TRAIN_PATH)
    X_test = pd.read_csv(TEST_PATH)
    raw_test_df = pd.read_csv(RAW_TEST_PATH)
    
    submission_test_df = pd.DataFrame({'id': raw_test_df['id']})

    # Separate features and targets from train_df
    target_cols = ['price', 'log_price', 'log_price_cap']
    drop_cols = target_cols + ['id']
    
    y_train = train_df['log_price'].reset_index(drop=True)
    # Using the uncapped log_price for training as well to better capture the full range
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).reset_index(drop=True)
    
    X_test = X_test.drop(columns=['id'], errors='ignore').reset_index(drop=True)

    print(f"--- Data Loaded ---")
    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, submission_test_df

def train_and_evaluate(X_train, y_train):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    models = {
        'XGBoost': get_xgboost_model(),
        'LightGBM': get_lightgbm_model()
    }
    
    cv_predictions = {}
    scores = {}
    
    for name, model in models.items():
        print(f"Cross-validating {name}...")
        y_pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
        cv_predictions[name] = y_pred
        scores[name] = evaluate(y_train.values, y_pred)['RMSLE']
    
    return models, cv_predictions, scores

def build_ensemble(cv_predictions, y_train, scores):
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # weighted Ensemble 
    weights = {name: 1/score for name, score in scores.items()}
    weight_sum = sum(weights.values())
    weights = {name: w/weight_sum for name, w in weights.items()}
    
    pred_weighted = np.zeros(len(y_train))
    for name, pred in cv_predictions.items():
        pred_weighted += weights[name] * pred
    weighted_score = evaluate(y_train.values, pred_weighted)['RMSLE']
    
    # 2. Stacking (Meta-learner)
    meta_train = pd.DataFrame(cv_predictions)
    meta_model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    stack_pred = cross_val_predict(meta_model, meta_train, y_train, cv=cv, n_jobs=-1)
    stack_score = evaluate(y_train.values, stack_pred)['RMSLE']
    
    return weights, meta_model, weighted_score, stack_score, pred_weighted, stack_pred

def run_final_predictions(models, meta_model, X_train, y_train, X_test, cv_predictions):
    test_predictions = {}
    for name, model in models.items():
        print(f"Final training for {name}...")
            # train on full y_train
        model.fit(X_train, y_train)
        test_predictions[name] = model.predict(X_test)
    
    # fit meta learner on full CV predictions
    meta_train = pd.DataFrame(cv_predictions)
    meta_model.fit(meta_train, y_train)
    
    # predict on test
    meta_test = pd.DataFrame(test_predictions)
    final_pred_log = meta_model.predict(meta_test)
    
    # Back-transform from log scale
    final_pred = np.clip(np.expm1(final_pred_log), 0, None)
    
    return final_pred

def main():
    print("Starting Final Model Pipeline...")
    
    X_train, X_test, y_train, sub_df = load_processed_pipeline_data()
    
    # cv evaluation
    models, cv_predictions, scores = train_and_evaluate(X_train, y_train)
    
    # ensemble building
    weights, meta_model, w_score, s_score, oof_weighted, oof_stacking = build_ensemble(cv_predictions, y_train, scores)
    
    # final prediction
    final_predictions = run_final_predictions(
        models, meta_model, X_train, y_train, X_test, cv_predictions
    )
    
    submission_final = pd.DataFrame({
        'ID': sub_df['id'].astype(int),
        'TARGET': final_predictions
    })
    submission_final.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "="*30)
    print("FINAL RESULTS (RMSLE)")
    print("="*30)
    for name, score in scores.items():
        print(f"{name:12}: {score:.5f}")
    print(f"{'Weighted':12}: {w_score:.5f}")
    print(f"{'Stacking':12}: {s_score:.5f}")
    print("="*30)
    print(f"Submission saved to: {OUTPUT_PATH}")

    # visualizations for presentation
    all_scores = scores.copy()
    all_scores['Weighted'] = w_score
    all_scores['Stacking'] = s_score
    # merge oof predictions for diagnostics
    oof_preds = dict(cv_predictions)
    oof_preds["Weighted"] = oof_weighted
    oof_preds["Stacking"] = oof_stacking
    plot_model_results(all_scores, models, X_train, y_train, oof_preds)

if __name__ == "__main__":
    main()

