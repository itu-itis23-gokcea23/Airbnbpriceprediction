from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
TRAIN_PATH = PROCESSED_DIR / "airbnb_train_last.csv"
TEST_PATH = PROCESSED_DIR / "airbnb_test_last.csv"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_baseline_rf.csv"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading train data from {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Loading test data from {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separates features and target, ensuring multiple target columns from 
    preprocess_v1.py are handled correctly.
    """
    # Columns to exclude from features
    target_cols = ['price', 'log_price', 'log_price_cap']
    
    # Target for training (using log_price directly from preprocess_v1)
    if 'log_price' in train_df.columns:
        y = train_df['log_price']
    else:
        y = np.log1p(train_df['price'])

    # Features: drop targets and any ID/Index columns
    X = train_df.drop(columns=[c for c in target_cols if c in train_df.columns])
    X = X.select_dtypes(include=[np.number, 'bool'])
    
    if "id" in X.columns:
        X = X.drop(columns=["id"])
    
    # Test features: ensure alignment
    test_features = test_df.select_dtypes(include=[np.number, 'bool'])
    if "id" in test_features.columns:
        test_features = test_features.drop(columns=["id"])
    
    test_ids = test_df["id"] if "id" in test_df.columns else test_df.index
    
    # Perfect alignment
    test_features = test_features.reindex(columns=X.columns, fill_value=0)
    
    print(f"Prepared features — train rows: {len(X)}, cols: {X.shape[1]}")
    return X, y, test_features, test_ids


def train_models(
    X: pd.DataFrame, y_log: pd.Series
) -> tuple[dict, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Trains Ridge and Random Forest models on log-transformed target."""
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print(f"Split data: train {X_train.shape[0]} rows, val {X_val.shape[0]} rows, {X_train.shape[1]} features")

    # Ridge Baseline with Scaling to prevent ill-conditioned matrix warning
    ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    print("Training Ridge(alpha=1.0) with StandardScaler")
    ridge.fit(X_train, y_train_log)
    pred_ridge_log = ridge.predict(X_val)
    rmsle_ridge = np.sqrt(mean_squared_error(y_val_log, pred_ridge_log))

    # Random Forest Baseline - Regularized to prevent overfitting
    rf = RandomForestRegressor(
        n_estimators=300, 
        max_depth=10,         # Reduced from 15 to 10
        min_samples_leaf=10,  # Added min samples per leaf
        n_jobs=-1, 
        random_state=42
    )
    print("Training RandomForest(n_estimators=300, max_depth=15)")
    rf.fit(X_train, y_train_log)
    pred_rf_log = rf.predict(X_val)
    rmsle_rf = np.sqrt(mean_squared_error(y_val_log, pred_rf_log))

    metrics = {
        "ridge": {"model": ridge, "rmsle": rmsle_ridge},
        "rf": {"model": rf, "rmsle": rmsle_rf},
    }
    splits = (X_train, X_val, y_train_log, y_val_log)
    return metrics, splits


def fit_full_and_predict(
    model, X: pd.DataFrame, y_log: pd.Series, test_features: pd.DataFrame, test_ids: pd.Series
) -> pd.DataFrame:
    model.fit(X, y_log)
    preds_log = model.predict(test_features)
    preds = np.maximum(np.expm1(preds_log), 0)  # back-transform and clip
    print(f"Generated predictions for {len(preds)} test rows")
    submission = pd.DataFrame({"ID": test_ids.astype(int), "TARGET": preds})
    return submission


def main() -> None:
    train_df, test_df = load_data()
    X, y_log, test_features, test_ids = prepare_features(train_df, test_df)
    metrics, _ = train_models(X, y_log)
    print("--- Validation RMSLE ---")
    print(f"Ridge: {metrics['ridge']['rmsle']:.5f}")
    print(f"RandomForest: {metrics['rf']['rmsle']:.5f}")
    best_name = min(metrics, key=lambda k: metrics[k]["rmsle"])
    best_model = metrics[best_name]["model"]
    print(f"Best model: {best_name}")

    submission = fit_full_and_predict(best_model, X, y_log, test_features, test_ids)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()

