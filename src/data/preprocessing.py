import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Preprocess Telco churn dataset with logging and safe transformations.
    """

    df = df.copy()
    logger.info("Starting preprocessing...")

    # tidy headers
    df.columns = df.columns.str.strip()
    logger.info("Column headers stripped of whitespace.")

    # drop ID columns
    id_cols = {"customerID", "CustomerID", "customer_id"}
    dropped_cols = [c for c in id_cols if c in df.columns]
    df = df.drop(columns=dropped_cols)
    if dropped_cols:
        logger.info(f"Dropped ID columns: {dropped_cols}")

    # normalize and map target
    if target_col in df.columns:
        if df[target_col].dtype == "object":
            df[target_col] = (
                df[target_col]
                .str.strip()
                .str.lower()
                .map({"no": 0, "yes": 1})
            )
            logger.info(f"Mapped target column '{target_col}' to 0/1.")
        if df[target_col].isna().any():
            logger.warning(f"Target column '{target_col}' contains unmapped values.")
    else:
        logger.warning(f"Target column '{target_col}' not found in dataframe.")

    # fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        logger.info("Converted 'TotalCharges' to numeric.")

    # SeniorCitizen normalization
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)
        logger.info("Normalized 'SeniorCitizen' to 0/1 integers.")

    # numeric NA handling
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled missing values in '{col}' with median ({median_val}).")

    if df.empty:
        logger.error("Preprocessed dataframe is empty!")
        raise ValueError("Preprocessed dataframe is empty.")

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df