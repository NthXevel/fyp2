"""
XGBoost model training module

Training targets:
    • Accuracy  ≥ 55 %
    • Sharpe    ≥ 0.5
    • Max DD    ≤ 10 %
"""
import xgboost as xgb
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.config import (
    XGB_PARAMS, TEST_SIZE, RANDOM_STATE, MODEL_PATH,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
)


class ModelTrainer:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        self.model_path = model_path
        self.feature_cols = None
        self.scaler = None
    
    def train(self, X, y, feature_cols):
        """
        Train XGBoost model with early stopping and time-series CV
        hyperparameter search.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_cols: List of feature column names
            
        Returns:
            float: Test accuracy
        """
        self.feature_cols = feature_cols
        
        # Split data (chronological)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Target -> Accuracy ≥ {TARGET_ACCURACY:.0%}, "
              f"Sharpe ≥ {TARGET_SHARPE}, Max DD ≤ {TARGET_MAX_DRAWDOWN:.0%}")
        
        # ── Hyperparameter search via TimeSeriesSplit ────────────────
        print("\nRunning time-series cross-validation grid search ...")
        base_estimator = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=RANDOM_STATE,
        )
        param_grid = {
            'n_estimators':     [200, 300, 400],
            'max_depth':        [3, 4, 5],
            'learning_rate':    [0.01, 0.03, 0.05],
            'subsample':        [0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'reg_alpha':        [0, 0.1],
            'reg_lambda':       [1.0, 2.0],
        }
        tscv = TimeSeriesSplit(n_splits=4)
        search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            scoring='accuracy',
            cv=tscv,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        
        self.model = search.best_estimator_
        best_cv_acc = search.best_score_
        
        print(f"Best CV accuracy: {best_cv_acc:.4f}")
        print(f"Best params: {search.best_params_}")
        
        # ── Evaluate on held-out test set ────────────────────────────
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        status = "✓ MET" if accuracy >= TARGET_ACCURACY else "✗ MISSED"
        print(f"\nTest Accuracy: {accuracy:.4f}  ({status} target {TARGET_ACCURACY})")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict(self, X):
        """
        Make predictions with trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            np.array: Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def save(self):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols
            }, f)
        print(f"Model saved to {self.model_path}")
    
    def load(self):
        """Load trained model from disk"""
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
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance from trained model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance = self.model.get_booster().get_score(importance_type='weight')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance[:top_n])
