"""Evaluation utilities for NBA Predictor.

This module provides functions for computing metrics and comparing
model performance against naive baselines.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from typing import Dict, Any


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities for class 1 (home win)

    Returns:
        Dictionary with log_loss, roc_auc, and accuracy
    """
    # Binary predictions using 0.5 threshold
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    return {
        "log_loss": log_loss(y_true, y_pred_proba),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred_binary)
    }


def compute_baseline_metrics(y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute naive baseline model metrics.

    Baseline 1: Always predict home win (prob = 1.0)
    Baseline 2: Constant probability = empirical home win rate

    Args:
        y_true: True labels (0 or 1)

    Returns:
        Nested dictionary with metrics for each baseline
    """
    home_win_rate = y_true.mean()

    # Baseline 1: Always predict home win
    always_home_proba = np.ones_like(y_true, dtype=float)
    baseline1_metrics = compute_metrics(y_true, always_home_proba)

    # Baseline 2: Constant probability = training home win rate
    constant_proba = np.full_like(y_true, fill_value=home_win_rate, dtype=float)
    baseline2_metrics = compute_metrics(y_true, constant_proba)

    return {
        "always_home": baseline1_metrics,
        "constant_prob": baseline2_metrics
    }


def compare_to_baselines(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Create comparison table of model vs baselines.

    Args:
        model_metrics: Model performance metrics
        baseline_metrics: Baseline model metrics

    Returns:
        DataFrame with models as rows, metrics as columns
    """
    # Combine all metrics into a single dictionary
    all_metrics = {
        "Model (Logistic Regression)": model_metrics,
        "Baseline 1 (Always Home)": baseline_metrics["always_home"],
        "Baseline 2 (Constant Prob)": baseline_metrics["constant_prob"]
    }

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics).T

    # Reorder columns for better readability
    df = df[["log_loss", "roc_auc", "accuracy"]]

    return df


def print_evaluation_report(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]]
) -> None:
    """Pretty-print evaluation results with baseline comparisons.

    Args:
        model_metrics: Model performance metrics
        baseline_metrics: Baseline model metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Create comparison table
    comparison_df = compare_to_baselines(model_metrics, baseline_metrics)

    print("\nMetrics Comparison:")
    print("-" * 60)
    print(comparison_df.to_string())
    print("-" * 60)

    # Highlight model vs baselines
    print("\nModel Performance vs Baselines:")
    print("-" * 60)

    model_log_loss = model_metrics["log_loss"]
    baseline1_log_loss = baseline_metrics["always_home"]["log_loss"]
    baseline2_log_loss = baseline_metrics["constant_prob"]["log_loss"]

    # Check if model beats baselines (lower log loss is better)
    beats_baseline1 = model_log_loss < baseline1_log_loss
    beats_baseline2 = model_log_loss < baseline2_log_loss

    print(f"Log Loss (PRIMARY METRIC - lower is better):")
    print(f"  Model:             {model_log_loss:.4f}")
    print(f"  Always Home:       {baseline1_log_loss:.4f} {'✓' if beats_baseline1 else '✗'}")
    print(f"  Constant Prob:     {baseline2_log_loss:.4f} {'✓' if beats_baseline2 else '✗'}")

    print(f"\nROC AUC (higher is better):")
    print(f"  Model:             {model_metrics['roc_auc']:.4f}")
    print(f"  Always Home:       {baseline_metrics['always_home']['roc_auc']:.4f}")
    print(f"  Constant Prob:     {baseline_metrics['constant_prob']['roc_auc']:.4f}")

    print(f"\nAccuracy (higher is better):")
    print(f"  Model:             {model_metrics['accuracy']:.4f}")
    print(f"  Always Home:       {baseline_metrics['always_home']['accuracy']:.4f}")
    print(f"  Constant Prob:     {baseline_metrics['constant_prob']['accuracy']:.4f}")

    print("-" * 60)

    # Overall assessment
    if beats_baseline1 and beats_baseline2:
        print("\n✓ SUCCESS: Model beats both baselines!")
        print("  The model learned useful patterns from the data.")
    elif beats_baseline1 or beats_baseline2:
        print("\n⚠ PARTIAL SUCCESS: Model beats some baselines")
        print("  Consider adding more features or tuning hyperparameters.")
    else:
        print("\n✗ FAILURE: Model does not beat baselines")
        print("  Check for data leakage, bugs, or consider a different approach.")

    print("=" * 60)
