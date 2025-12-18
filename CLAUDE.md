# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA prediction project using Python machine learning libraries. Currently in early development stage with minimal codebase.

## Development Environment

- Python version: 3.11 (specified in `.python-version`)
- Package manager: `uv` (modern Python package manager)
- Virtual environment: `.venv` (auto-managed by uv)

## Key Commands

### Environment Setup
```bash
# Install dependencies (uv automatically creates/uses .venv)
uv sync

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>
```

### Running Code
```bash
# Run the main script
python main.py

# Or using uv
uv run main.py
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Or using uv
uv run jupyter notebook
```

## Technology Stack

Core dependencies (from `pyproject.toml`):
- `pandas` (2.3.3+) - Data manipulation and analysis
- `numpy` (2.3.5+) - Numerical computing
- `scikit-learn` (1.8.0+) - Machine learning models
- `matplotlib` (3.10.8+) - Data visualization
- `jupyter` (1.1.1+) - Interactive notebooks
- `ipykernel` (7.1.0+) - Jupyter kernel support
- `joblib` (1.5.3+) - Model persistence and caching
- `pyarrow` (22.0.0+) - Fast data serialization

## Architecture Notes

The project implements a baseline NBA game predictor using Logistic Regression:

- **Package structure**: `src/nba_predictor/` contains all core modules
- **Data pipeline**: Time-based train/test split to prevent data leakage
- **Model**: Logistic Regression with one-hot encoded team IDs
- **Evaluation**: Log Loss (primary), ROC AUC, Accuracy (secondary)
- **Notebooks**: Interactive EDA and training demonstrations in `notebooks/`

## Implementation Notes

### Data Pipeline

**Preprocessing command**:
```bash
uv run python -m nba_predictor.data_prep
```

**What it does**:
- Loads raw CSV from `data/raw/games.csv`
- Creates `home_win` label (1 if home points > away points, else 0)
- Sorts by date and splits: earliest 80% train, most recent 20% test
- Validates: `max(train dates) < min(test dates)` to prevent leakage
- Saves processed data to `data/processed/` as parquet files

**Critical**: Date sorting happens BEFORE split to prevent data leakage

### Model Training

**Training command**:
```bash
uv run python -m nba_predictor.train_baseline
```

**What it does**:
- Loads processed train/test data
- Creates sklearn Pipeline: OneHotEncoder → LogisticRegression
- Trains on team IDs only (baseline approach)
- Evaluates against naive baselines (always home, constant prob)
- Prints metrics comparison
- Saves model to `models/baseline.joblib`

**Success criteria**: Model log loss < baseline log losses

### Making Predictions

```python
from nba_predictor import load_model, predict_proba

model = load_model()
prob = predict_proba(model, home_team_id=1610612747, visitor_team_id=1610612738)
print(f"Home win probability: {prob:.3f}")
```

### Key Files and Their Roles

- **`src/nba_predictor/config.py`** - All constants and hyperparameters
  - Change `RANDOM_SEED` for different random splits
  - Modify `TRAIN_SPLIT_RATIO` to adjust train/test proportion
  - Update `LOGISTIC_REGRESSION_PARAMS` for hyperparameter tuning

- **`src/nba_predictor/data_prep.py`** - Time-based split implementation
  - Critical function: `split_data()` - MUST sort by date first
  - Validation: Asserts no date overlap between train/test

- **`src/nba_predictor/train_baseline.py`** - Main training pipeline
  - Creates sklearn Pipeline (handles one-hot encoding + training)
  - Uses `handle_unknown='ignore'` for unseen teams in production

- **`src/nba_predictor/evaluate.py`** - Metrics and baseline comparisons
  - Computes Log Loss (primary), ROC AUC, Accuracy
  - Implements two naive baselines for comparison

- **`src/nba_predictor/predict.py`** - Inference interface
  - `predict_proba()` - Returns probability
  - `predict_winner()` - Returns 'home' or 'away'
  - `predict_batch()` - Batch predictions for multiple games

### Testing Approach

**Manual validation checklist**:
1. Check train/test date ranges don't overlap (printed during preprocessing)
2. Verify home win rate is ~55-60% (typical NBA home advantage)
3. Ensure model log loss < both baselines (printed during training)
4. Re-run training and verify identical metrics (reproducibility)

**Expected metrics**:
- Home win rate: 55-60%
- Baseline log loss: 0.65-0.68
- Model log loss: 0.60-0.63 (should beat baselines)
- Model accuracy: 62-66%
- ROC AUC: 0.65-0.70

### Common Issues and Solutions

**Issue**: Dataset missing
```
FileNotFoundError: Dataset not found: data/raw/games.csv
```
**Solution**: Download from Kaggle: https://www.kaggle.com/datasets/nathanlauga/nba-games

**Issue**: Import errors in notebooks
```
ModuleNotFoundError: No module named 'nba_predictor'
```
**Solution**: Add `sys.path.append('../src')` at top of notebook

**Issue**: Data leakage detected
```
AssertionError: Data leakage detected! Train max date >= Test min date
```
**Solution**: This is a bug - check `split_data()` implementation, ensure sorting by date

**Issue**: Model doesn't beat baselines
**Possible causes**:
1. Data leakage (check date split validation)
2. One-hot encoding bug (verify feature names)
3. Label creation error (verify home_win logic)
4. Random seed not set (check config.RANDOM_SEED usage)

### Development Workflow

1. **Data exploration**: Use `notebooks/01_eda.ipynb`
2. **Preprocessing**: Run `data_prep.py` to create train/test splits
3. **Training**: Run `train_baseline.py` to train and evaluate
4. **Experimentation**: Use `notebooks/02_train_baseline.ipynb` for interactive analysis
5. **Prediction**: Use `predict.py` for inference

### Future Improvements (v1.1+)

**Feature engineering**:
- Add `season` as categorical feature
- Rolling statistics (must compute from past games only):
  - Last-10 win percentage
  - Last-10 point differential
  - Rest days between games
- Advanced features: injuries, player stats, lineups

**Model improvements**:
- Random Forest, XGBoost, LightGBM
- Neural networks (requires more features)
- Ensemble methods
- Hyperparameter tuning with GridSearchCV

**Deployment (v2)**:
- FastAPI REST endpoint
- Web UI (Streamlit or React)
- Docker containerization
- Automated retraining pipeline
