"""
models/trainer.py — XGBoost training with walk-forward CV + Optuna hyperparameter tuning

Walk-forward cross-validation prevents data leakage in time-series data.
Optuna replaces GridSearchCV for faster, smarter hyperparameter search.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report, confusion_matrix,
)
from config.settings import (
    MODEL_PATH, TEST_SIZE, RANDOM_STATE,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Walk-Forward Cross-Validation ───────────────────────────────────────────

class WalkForwardCV:
    """
    Rolling / expanding walk-forward cross-validation for time-series data.
    Ensures NO data leakage between train and validation splits.
    """
    def __init__(self, n_splits: int = 5, gap: int = 0, expanding: bool = True):
        self.n_splits  = n_splits
        self.gap       = gap          # bars between train end and val start
        self.expanding = expanding    # True = expanding window, False = rolling

    def split(self, X):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        splits = []
        for i in range(1, self.n_splits + 1):
            if self.expanding:
                train_end = fold_size * i
                train_idx = np.arange(0, train_end)
            else:
                train_start = max(0, fold_size * (i - 1))
                train_end   = fold_size * i
                train_idx   = np.arange(train_start, train_end)

            val_start = train_end + self.gap
            val_end   = min(val_start + fold_size, n)
            val_idx   = np.arange(val_start, val_end)

            if len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits


# ─── Model Trainer ───────────────────────────────────────────────────────────

class ModelTrainer:
    def __init__(self, model_path=MODEL_PATH):
        self.model: xgb.XGBClassifier | None = None
        self.model_path = model_path
        self.feature_cols: list[str] | None = None

    # ── Optuna objective ─────────────────────────────────────────────
    def _objective(self, trial: optuna.Trial, X, y) -> float:
        """Single Optuna trial: train with sampled params, score via walk-forward CV."""
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "objective":         "binary:logistic",
            "eval_metric":       "auc",
            "random_state":      RANDOM_STATE,
            "verbosity":         0,
        }

        cv = WalkForwardCV(n_splits=3, gap=5)
        auc_scores = []

        X_arr = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        for train_idx, val_idx in cv.split(X_arr):
            X_tr  = X_arr.iloc[train_idx]
            X_val = X_arr.iloc[val_idx]
            y_tr  = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_val = y.iloc[val_idx]   if hasattr(y, 'iloc') else y[val_idx]

            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            prob = clf.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, prob))

        return float(np.mean(auc_scores))

    # ── Optuna tuning ────────────────────────────────────────────────
    def tune(self, X, y, n_trials: int = 50) -> dict:
        """Run Optuna hyperparameter search. Returns best params dict."""
        print(f"\nStarting Optuna tuning: {n_trials} trials …")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: self._objective(t, X, y),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        print(f"Best CV AUC: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        return study.best_params

    # ── Main training entry point ────────────────────────────────────
    def train(self, X, y, feature_cols, n_tune_trials: int = 50):
        """
        Full pipeline: Optuna tune → train final model → evaluate on holdout.

        Args:
            X:              Feature matrix (pd.DataFrame or np.ndarray)
            y:              Target vector
            feature_cols:   List of feature column names
            n_tune_trials:  Number of Optuna trials (default 50)

        Returns:
            float: Test accuracy
        """
        self.feature_cols = feature_cols

        # Chronological train / test split
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size:     {len(X_test)}")
        print(f"Target → Accuracy ≥ {TARGET_ACCURACY:.0%}, "
              f"Sharpe ≥ {TARGET_SHARPE}, Max DD ≤ {TARGET_MAX_DRAWDOWN:.0%}")

        # ── Optuna hyperparameter search on training data only ─────
        best_params = self.tune(X_train, y_train, n_trials=n_tune_trials)

        # ── Train final model on full training set ─────────────────
        print("\nTraining final model with best params …")
        best_params.update({
            "objective":    "binary:logistic",
            "eval_metric":  "auc",
            "random_state":  RANDOM_STATE,
            "verbosity":     0,
        })
        self.model = xgb.XGBClassifier(**best_params)
        self.model.fit(X_train, y_train)

        # ── Evaluate on held-out test set ──────────────────────────
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc      = roc_auc_score(y_test, y_prob)
        f1       = f1_score(y_test, y_pred)

        acc_status = "✓ MET" if accuracy >= TARGET_ACCURACY else "✗ MISSED"
        print(f"\nTest Accuracy: {accuracy:.4f}  ({acc_status} target {TARGET_ACCURACY})")
        print(f"Test AUC:      {auc:.4f}")
        print(f"Test F1:       {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return accuracy

    # ── Prediction ───────────────────────────────────────────────────
    def predict(self, X):
        """
        Return class probabilities [[prob_down, prob_up], …].
        Compatible with SignalGenerator.decide_trade().
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict_proba(X)

    def predict_proba(self, X):
        """Return probability of the positive class (1-d array)."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        if self.feature_cols is not None and isinstance(X, pd.DataFrame):
            X = X[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]

    # ── Persistence ──────────────────────────────────────────────────
    def save(self):
        """Save trained model + feature list to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols,
            }, f)
        print(f"Model saved to {self.model_path}")

    def load(self):
        """Load trained model + feature list from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_cols = data['feature_cols']
            print(f"Model loaded from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file not found at {self.model_path}")
            return False

    # ── Feature importance ────────────────────────────────────────────
    def get_feature_importance(self, top_n: int = 10):
        """
        Return top-N features by importance.

        Returns:
            dict: {feature_name: importance_score}
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        importance = self.model.feature_importances_
        if self.feature_cols:
            pairs = list(zip(self.feature_cols, importance))
        else:
            pairs = [(f"f{i}", v) for i, v in enumerate(importance)]

        pairs.sort(key=lambda x: x[1], reverse=True)
        return dict(pairs[:top_n])

    def feature_importance_df(self, top_n: int = 20) -> pd.DataFrame:
        """Return top-N features as a DataFrame (useful for plotting)."""
        imp = self.get_feature_importance(top_n)
        return pd.DataFrame(
            list(imp.items()), columns=["feature", "importance"]
        ).sort_values("importance", ascending=False)
