"""NBA Game Outcome Predictor - Baseline Model (v1)."""

__version__ = "0.1.0"

# Import will be available after modules are created
# Expose public functions for easy imports
try:
    from .data_prep import preprocess_data, load_raw_data, create_label, split_data
    from .train_baseline import train_model, main as train_main
    from .predict import load_model, predict_proba, predict_winner

    __all__ = [
        # Data preprocessing
        "preprocess_data",
        "load_raw_data",
        "create_label",
        "split_data",
        # Training
        "train_model",
        "train_main",
        # Inference
        "load_model",
        "predict_proba",
        "predict_winner",
    ]
except ImportError:
    # Modules not yet created, skip imports
    __all__ = []
