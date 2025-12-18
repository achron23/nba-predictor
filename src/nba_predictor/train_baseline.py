"""Training module for NBA Predictor baseline model.

This module implements the training pipeline for the Logistic Regression
baseline model with one-hot encoded team IDs.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from pathlib import Path
from typing import Tuple, Dict
from . import config
from .evaluate import compute_metrics, compute_baseline_metrics, print_evaluation_report


def create_baseline_pipeline() -> Pipeline:
    """Create sklearn pipeline: OneHotEncoder -> LogisticRegression.

    Returns:
        Configured sklearn Pipeline
    """
    # One-hot encode both team ID columns
    # handle_unknown='ignore' ensures new teams in production don't cause errors
    preprocessor = ColumnTransformer(
        transformers=[
            ('team_encoder',
             OneHotEncoder(handle_unknown='ignore', sparse_output=True),
             config.FEATURE_COLS)
        ],
        remainder='drop'  # Drop all other columns
    )

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS))
    ])

    return pipeline


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train and test datasets.

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        FileNotFoundError: If processed data files don't exist
    """
    if not config.TRAIN_FILE.exists() or not config.TEST_FILE.exists():
        raise FileNotFoundError(
            "Processed data files not found. Run preprocessing first:\n"
            "  uv run python -m nba_predictor.data_prep"
        )

    train_df = pd.read_parquet(config.TRAIN_FILE)
    test_df = pd.read_parquet(config.TEST_FILE)

    return train_df, test_df


def prepare_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract features (X) and labels (y) from dataframe.

    Args:
        df: DataFrame with feature columns and label column

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is labels array
    """
    X = df[config.FEATURE_COLS].copy()
    y = df[config.LABEL_COL].values

    return X, y


def train_model(X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    """Train baseline model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained pipeline
    """
    # Create pipeline
    pipeline = create_baseline_pipeline()

    # Fit on training data
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict[str, float]:
    """Evaluate trained model on test set.

    Args:
        pipeline: Trained model pipeline
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    # Get predicted probabilities for home win (class 1)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_proba)

    return metrics


def save_model(pipeline: Pipeline, model_path: Path = config.BASELINE_MODEL_PATH) -> None:
    """Save trained pipeline to disk.

    Args:
        pipeline: Trained model pipeline
        model_path: Path to save model
    """
    # Create models directory if needed
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with joblib
    joblib.dump(pipeline, model_path)

    print(f"✓ Model saved to: {model_path}")


def main() -> None:
    """Main training pipeline.

    Loads data, trains model, evaluates performance, and saves model artifact.

    Can be run as: uv run python -m nba_predictor.train_baseline
    """
    print("=" * 60)
    print("NBA Predictor - Training Baseline Model")
    print("=" * 60)

    # Set seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)

    try:
        # Load data
        print("\nLoading processed data...")
        train_df, test_df = load_processed_data()
        print(f"✓ Train: {len(train_df):,} games")
        print(f"✓ Test:  {len(test_df):,} games")

        # Prepare features and labels
        print("\nPreparing features and labels...")
        X_train, y_train = prepare_features_labels(train_df)
        X_test, y_test = prepare_features_labels(test_df)
        print(f"✓ Features: {config.FEATURE_COLS}")
        print(f"✓ Number of unique home teams: {X_train[config.HOME_TEAM_COL].nunique()}")
        print(f"✓ Number of unique visitor teams: {X_train[config.VISITOR_TEAM_COL].nunique()}")

        # Train model
        print("\nTraining baseline model (Logistic Regression)...")
        print(f"  Hyperparameters: {config.LOGISTIC_REGRESSION_PARAMS}")
        pipeline = train_model(X_train, y_train)
        print("✓ Training complete!")

        # Evaluate model on test set
        print("\nEvaluating on test set...")
        model_metrics = evaluate_model(pipeline, X_test, y_test)

        # Compute baseline metrics for comparison
        baseline_metrics = compute_baseline_metrics(y_test)

        # Print comprehensive evaluation report
        print_evaluation_report(model_metrics, baseline_metrics)

        # Save model
        print("\nSaving model artifact...")
        save_model(pipeline)

        # Print completion message
        print("\n" + "=" * 60)
        print("Training Pipeline Complete!")
        print("=" * 60)
        print("\nDefinition of Done:")
        print("  ✓ Model trained with time-based split")
        print("  ✓ Metrics computed and compared to baselines")
        print("  ✓ Model artifact saved")
        print("\nNext steps:")
        print("  - Test predictions: see predict.py")
        print("  - Explore notebooks: notebooks/02_train_baseline.ipynb")
        print("  - Plan v1.1 improvements: add season, rolling features")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
