"""
Income classification model implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import joblib
import warnings

warnings.filterwarnings("ignore")

from utils import (
    create_directories,
    evaluate_classification_model,
    save_model,
    plot_feature_importance,
)


class IncomeClassifier:
    """
    Income classification model with multiple algorithms and comprehensive evaluation.
    """

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}

    def define_models(self):
        """Define the models to be evaluated."""
        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                solver="liblinear",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                max_depth=10,
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric="logloss",
                scale_pos_weight=15,
            ),
        }

    def handle_class_imbalance(self, X_train, y_train, method="smote"):
        """
        Handle class imbalance using various techniques.

        Args:
            X_train: Training features
            y_train: Training labels
            method: Resampling method ('smote', 'undersample', 'none')

        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print(f"\nHandling class imbalance using {method.upper()}")

        if method == "smote":
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif method == "undersample":
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train

        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")

        return X_resampled, y_resampled

    def train_models(self, X_train, y_train, X_test, y_test, handle_imbalance=True):
        """
        Train all models and evaluate their performance.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            handle_imbalance: Whether to handle class imbalance
        """
        print("=" * 60)
        print("TRAINING AND EVALUATING MODELS")
        print("=" * 60)

        # Define models
        self.define_models()

        # Handle class imbalance if requested
        if handle_imbalance:
            X_train_resampled, y_train_resampled = self.handle_class_imbalance(
                X_train, y_train
            )
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            try:
                # Train model
                model.fit(X_train_resampled, y_train_resampled)

                # Make predictions
                y_pred = model.predict(X_test)
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Evaluate model
                results = evaluate_classification_model(y_test, y_pred, y_prob, name)

                # Store results
                self.results[name] = {
                    "model": model,
                    "predictions": y_pred,
                    "probabilities": y_prob,
                    "metrics": results,
                }

                # Plot feature importance for tree-based models
                if hasattr(model, "feature_importances_"):
                    plot_feature_importance(model, X_train.columns.tolist(), name)

            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue

        # Find best model based on F1-score
        self.find_best_model()

    def find_best_model(self):
        """Find the best performing model based on F1-score."""
        best_f1 = 0
        best_model_name = None

        for name, result in self.results.items():
            if result["metrics"]["classification_report"]:
                f1_score = result["metrics"]["classification_report"]["weighted avg"][
                    "f1-score"
                ]
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model_name = name

        if best_model_name:
            self.best_model = self.results[best_model_name]["model"]
            self.best_model_name = best_model_name
            print(f"\nBest model: {best_model_name} (F1-score: {best_f1:.4f})")

    def cross_validation_analysis(self, X, y, cv_folds=5):
        """
        Perform cross-validation analysis for all models.

        Args:
            X: Features
            y: Labels
            cv_folds: Number of CV folds
        """
        print(f"\nCross-validation analysis ({cv_folds}-fold)")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, model in self.models.items():
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
                print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

            except Exception as e:
                print(f"Error in CV for {name}: {str(e)}")

    def compare_models(self):
        """Compare all models and create summary visualizations."""
        if not self.results:
            print("No models to compare. Train models first.")
            return

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        # Extract metrics for comparison
        model_names = []
        accuracy_scores = []
        f1_scores = []
        roc_auc_scores = []

        for name, result in self.results.items():
            if result["metrics"]["classification_report"]:
                model_names.append(name)
                accuracy_scores.append(
                    result["metrics"]["classification_report"]["accuracy"]
                )
                f1_scores.append(
                    result["metrics"]["classification_report"]["weighted avg"][
                        "f1-score"
                    ]
                )
                roc_auc_scores.append(
                    result["metrics"]["roc_auc"] if result["metrics"]["roc_auc"] else 0
                )

        # Create comparison dataframe
        comparison_df = pd.DataFrame(
            {
                "Model": model_names,
                "Accuracy": accuracy_scores,
                "F1-Score": f1_scores,
                "ROC-AUC": roc_auc_scores,
            }
        )

        print("\nModel Performance Summary:")
        print(comparison_df.round(4))

        # Visualize comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Accuracy comparison
        axes[0].bar(comparison_df["Model"], comparison_df["Accuracy"])
        axes[0].set_title("Accuracy Comparison")
        axes[0].set_ylabel("Accuracy")
        axes[0].tick_params(axis="x", rotation=45)

        # F1-score comparison
        axes[1].bar(comparison_df["Model"], comparison_df["F1-Score"])
        axes[1].set_title("F1-Score Comparison")
        axes[1].set_ylabel("F1-Score")
        axes[1].tick_params(axis="x", rotation=45)

        # ROC-AUC comparison
        axes[2].bar(comparison_df["Model"], comparison_df["ROC-AUC"])
        axes[2].set_title("ROC-AUC Comparison")
        axes[2].set_ylabel("ROC-AUC")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            "results/classification/model_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        return comparison_df

    def save_best_model(
        self, preprocessor, filepath="models/best_classification_model.pkl"
    ):
        """Save the best model and preprocessor."""
        if self.best_model is not None:
            save_model(self.best_model, preprocessor, self.best_model_name, filepath)
        else:
            print("No best model available. Train models first.")


def main():
    """Main function to run the classification pipeline."""
    # Create directories
    create_directories()

    # Load processed data
    try:
        processed_data = joblib.load("data/processed_data.pkl")
        X_train = processed_data["X_train"]
        X_test = processed_data["X_test"]
        y_train = processed_data["y_train"]
        y_test = processed_data["y_test"]
        preprocessor = processed_data["preprocessor"]
        print("Loaded processed data successfully.")
    except FileNotFoundError:
        print("Processed data not found. Please run data_processing.py first.")
        return

    # Initialize classifier
    classifier = IncomeClassifier()

    # Train and evaluate models
    classifier.train_models(X_train, y_train, X_test, y_test)

    # Cross-validation analysis
    classifier.cross_validation_analysis(X_train, y_train)

    # Compare models
    comparison_df = classifier.compare_models()

    # Save best model
    classifier.save_best_model(preprocessor)

    # Save comparison results
    comparison_df.to_csv("results/classification/model_comparison.csv", index=False)
    print(f"\nModel comparison saved to results/classification/model_comparison.csv")

    print(f"\nClassification modeling complete!")
    print(f"Best model: {classifier.best_model_name}")
    print(f"Next step: Run segmentation_model.py to create customer segments")


if __name__ == "__main__":
    main()
