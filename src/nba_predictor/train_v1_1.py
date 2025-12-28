"""Multi-model training for NBA Predictor v1.1.

Trains and compares:
- Logistic Regression (baseline with new features)
- Random Forest (non-linear interactions)
- XGBoost (gradient boosting)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from typing import Dict, Tuple
from . import config
from .evaluate import compute_metrics, compute_baseline_metrics


def create_feature_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline for v1.1 features.

    One-hot encodes categorical features, imputes missing values in numerical
    features (median strategy), and scales numerical features.

    Returns:
        ColumnTransformer ready to fit and transform features
    """
    categorical = [config.HOME_TEAM_COL, config.VISITOR_TEAM_COL, config.SEASON_COL]
    numerical = [f for f in config.ENGINEERED_FEATURE_COLS if f not in categorical]

    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical)
    ])


def create_logistic_regression_pipeline() -> Pipeline:
    """Create Logistic Regression pipeline.

    Uses L2 regularization with default C=1.0.

    Returns:
        Pipeline with preprocessing and Logistic Regression classifier
    """
    return Pipeline([
        ('preprocessor', create_feature_preprocessor()),
        ('classifier', LogisticRegression(
            random_state=config.RANDOM_SEED,
            max_iter=1000,
            C=1.0,
            solver='lbfgs'
        ))
    ])


def create_random_forest_pipeline() -> Pipeline:
    """Create Random Forest pipeline.

    Uses conservative parameters to avoid overfitting.

    Returns:
        Pipeline with preprocessing and Random Forest classifier
    """
    return Pipeline([
        ('preprocessor', create_feature_preprocessor()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        ))
    ])


def create_xgboost_pipeline() -> Pipeline:
    """Create XGBoost pipeline.

    Uses moderate learning rate and tree depth.

    Returns:
        Pipeline with preprocessing and XGBoost classifier
    """
    return Pipeline([
        ('preprocessor', create_feature_preprocessor()),
        ('classifier', XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=config.RANDOM_SEED,
            eval_metric='logloss',
            verbosity=0
        ))
    ])


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict:
    """Train all models and evaluate on test set.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary mapping model name to results dict containing
        'pipeline' and 'metrics'
    """
    models = {
        'Logistic Regression': create_logistic_regression_pipeline(),
        'Random Forest': create_random_forest_pipeline(),
        'XGBoost': create_xgboost_pipeline()
    }

    results = {}

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred_proba)

        results[name] = {
            'pipeline': pipeline,
            'metrics': metrics
        }

        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

    return results


def print_comparison_table(results: Dict, baseline_metrics: Dict) -> Tuple[str, float]:
    """Print model comparison table.

    Args:
        results: Dictionary of model results
        baseline_metrics: Baseline metrics dictionary

    Returns:
        Tuple of (best_model_name, best_log_loss)
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - v1.1")
    print("=" * 80)

    comparison = {
        'Baseline (Constant)': baseline_metrics['constant_prob'],
        **{name: res['metrics'] for name, res in results.items()}
    }

    df = pd.DataFrame(comparison).T[['log_loss', 'roc_auc', 'accuracy']]
    print("\n" + df.to_string())

    # Find best model
    best_name = min(results.keys(), key=lambda k: results[k]['metrics']['log_loss'])
    best_ll = results[best_name]['metrics']['log_loss']
    baseline_ll = baseline_metrics['constant_prob']['log_loss']
    improvement = (baseline_ll - best_ll) / baseline_ll * 100

    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_name}")
    print(f"  Log Loss: {best_ll:.4f}")
    print(f"  Improvement over baseline: {improvement:.2f}%")

    if best_ll < baseline_ll:
        print("  [SUCCESS] Model beats baseline!")
    else:
        print("  [FAIL] Model does not beat baseline")

    print("=" * 80)

    return best_name, best_ll


def main():
    """Main training pipeline for v1.1."""
    print("=" * 60)
    print("NBA Predictor v1.1 - Multi-Model Training")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)

    # Load feature-engineered data
    print("\nLoading feature-engineered data...")

    try:
        train_df = pd.read_parquet(config.TRAIN_FEATURES_FILE)
        test_df = pd.read_parquet(config.TEST_FEATURES_FILE)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Feature files not found: {e}")
        print("Please run preprocessing first:")
        print("  uv run python -m nba_predictor.data_prep")
        return

    X_train = train_df[config.ENGINEERED_FEATURE_COLS]
    y_train = train_df[config.LABEL_COL].values
    X_test = test_df[config.ENGINEERED_FEATURE_COLS]
    y_test = test_df[config.LABEL_COL].values

    print(f"  Train: {len(X_train):,} games, {len(config.ENGINEERED_FEATURE_COLS)} features")
    print(f"  Test:  {len(X_test):,} games")
    print(f"\nFeatures: {config.ENGINEERED_FEATURE_COLS}")

    # Compute baselines
    print("\nComputing baseline metrics...")
    baseline_metrics = compute_baseline_metrics(y_test)
    print(f"  Baseline (Constant Prob): Log Loss = {baseline_metrics['constant_prob']['log_loss']:.4f}")

    # Train all models
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Print comparison
    best_name, best_ll = print_comparison_table(results, baseline_metrics)

    # Save best model
    best_pipeline = results[best_name]['pipeline']
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_V1_1_PATH

    joblib.dump(best_pipeline, model_path)
    print(f"\n[OK] Best model ({best_name}) saved to: {model_path}")

    print("\nNext steps:")
    print("  - Analyze feature importance")
    print("  - Create visualizations in notebooks")
    print("  - Experiment with hyperparameter tuning")


if __name__ == "__main__":
    main()
