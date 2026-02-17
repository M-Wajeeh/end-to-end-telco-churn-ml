import os
import sys
import logging
import pandas as pd

# ==============================
# Logging Configuration
# ==============================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================
# Make src importable
# ==============================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
logger.debug(f"Project root added to path: {project_root}")

from src.data.preprocessing import preprocess_data
from src.features.build_features import build_features

RAW = "D:\\Github\\end-to-end-telco-churn-ml\\data\\raw\\Dataset.csv"
OUT = "D:\\Github\\end-to-end-telco-churn-ml\\data\\processed\\Dataset_processed.csv"

try:
    # ==============================
    # Step 1: Load raw dataset
    # ==============================
    logger.info(f"Loading raw dataset from: {RAW}")
    df = pd.read_csv(RAW)
    logger.debug(f"Raw dataset shape: {df.shape}")

    # ==============================
    # Step 2: Preprocessing
    # ==============================
    logger.info("Starting preprocessing step")
    df = preprocess_data(df, target_col="Churn")
    logger.debug(f"Shape after preprocessing: {df.shape}")

    # ==============================
    # Step 3: Target encoding check
    # ==============================
    if "Churn" in df.columns and df["Churn"].dtype == "object":
        logger.debug("Encoding target column 'Churn' to binary")
        df["Churn"] = (
            df["Churn"]
            .str.strip()
            .map({"No": 0, "Yes": 1})
            .astype("Int64")
        )

    logger.debug("Running sanity checks on target column")

    assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocess"
    assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

    logger.info("Target column validation passed")

    # ==============================
    # Step 4: Feature engineering
    # ==============================
    logger.info("Starting feature engineering")
    df_processed = build_features(df, target_col="Churn")
    logger.debug(f"Processed dataset shape: {df_processed.shape}")

    # ==============================
    # Step 5: Save processed dataset
    # ==============================
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df_processed.to_csv(OUT, index=False)

    logger.info(
        f"Processed dataset saved to {OUT} | Shape: {df_processed.shape}"
    )

except Exception as e:
    logger.exception("Pipeline execution failed")
    raise
