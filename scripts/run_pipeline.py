import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.load import load_data  # Data loading with error handling


def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.

    """

    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"  # Local file-based tracking

    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # Creates experiment if doesn't exist

    # Start MLflow run - all subsequent logging will be tracked under this run
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", "xgboost")  # Model type for comparison
        mlflow.log_param("threshold", args.threshold)  # Classification threshold (default: 0.35)
        mlflow.log_param("test_size", args.test_size)  # Train/test split ratio

        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")


        # === STAGE 4: Train/Test Split ===
        print("📊 Splitting data...")
        X = df.drop(columns=['churn', 'Customer ID'])
        y = df['churn']

        # Stratified split to maintain class distribution in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,  # Default: 20% for testing
            stratify=y,  # Maintain class balance
            random_state=42  # Reproducible splits
        )
        print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")


        # === CRITICAL: Handle Class Imbalance ===
        # Calculate scale_pos_weight to handle imbalanced dataset
        # This tells XGBoost to give more weight to the minority class (churners)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # === STAGE 5: Model Training with Optimized Hyperparameters ===
        print("🤖 Training XGBoost model...")

        # IMPORTANT: These hyperparameters were optimized through hyperparameter tuning
        # In production, consider using hyperparameter optimization tools like Optuna
        model = XGBClassifier(
            # Tree structure parameters
            n_estimators=600,  # Number of trees (OPTIMIZED)
            learning_rate=0.010,  # Step size shrinkage (OPTIMIZED)
            max_depth=8,  # Maximum tree depth (OPTIMIZED)

            # Regularization parameters
            subsample=0.542,  # Sample ratio of training instances
            colsample_bytree=0.810,  # Sample ratio of features for each tree

            # Performance parameters
            n_jobs=-1,  # Use all CPU cores
            random_state=42,  # Reproducible results
            eval_metric="logloss",  # Evaluation metric

            # ESSENTIAL: Handle class imbalance
            scale_pos_weight=scale_pos_weight  # Weight for positive class (churners)
        )

        # === Train Model and Track Training Time ===
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)  # Track training performance
        print(f"✅ Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Model Evaluation ===
        print("📊 Evaluating model performance...")

        # Generate predictions and track inference time
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]  # Get probability of churn (class 1)

        # Apply classification threshold (default: 0.35, optimized for churn detection)
        # Lower threshold = more sensitive to churn (higher recall, lower precision)
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)  # Track inference performance

        # === CRITICAL: Log Evaluation Metrics to MLflow ===
        # These metrics are essential for model comparison and monitoring
        precision = precision_score(y_test, y_pred)  # Of predicted churners, how many actually churned?
        recall = recall_score(y_test, y_pred)  # Of actual churners, how many did we catch?
        f1 = f1_score(y_test, y_pred)  # Harmonic mean of precision and recall
        roc_auc = roc_auc_score(y_test, proba)  # Area under ROC curve (threshold-independent)

        # Log all metrics for experiment tracking
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"🎯 Model Performance:")
        print(f"   Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # === STAGE 7: Model Serialization and Logging ===
        print("💾 Saving model to MLflow...")
        # ESSENTIAL: Log model in MLflow's standard format for serving
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"  # This creates a 'model/' folder in MLflow run artifacts
        )
        print("✅ Model saved to MLflow for serving pipeline")

        # === Final Performance Summary ===
        print(f"\n⏱️  Performance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test) / pred_time:.0f}")

        print(f"\n📈 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)