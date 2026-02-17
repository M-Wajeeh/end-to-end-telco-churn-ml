import mlflow
import mlflow.xgboost
import pandas as pd
import logging

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


logger = logging.getLogger(__name__)


def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains an XGBoost model and logs everything to MLflow.
    """

    logger.info("Starting model training...")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratify is VERY important in churn dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    params = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)

    # Set experiment name
    mlflow.set_experiment("Telco_Churn_XGBoost")

    with mlflow.start_run() as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)

        # Log params + metrics
        mlflow.log_params(params)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log model
        mlflow.xgboost.log_model(model, artifact_path="model")

        # Log datasets separately
        train_ds = mlflow.data.from_pandas(X_train, source="X_train")
        test_ds = mlflow.data.from_pandas(X_test, source="X_test")

        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(test_ds, context="testing")

        logger.info(
            f"Training complete. Accuracy={acc:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )

        return model, run.info.run_id
