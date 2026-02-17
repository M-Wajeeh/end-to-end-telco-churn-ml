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


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    """
    logger.debug("Starting binary mapping for series")

    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    logger.debug(f"Unique values detected: {valset}")

    # Yes/No mapping
    if valset == {"Yes", "No"}:
        logger.debug("Applying Yes/No deterministic mapping")
        return s.map({"No": 0, "Yes": 1}).astype("Int64")

    # Gender mapping
    if valset == {"Male", "Female"}:
        logger.debug("Applying Gender deterministic mapping")
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # Generic binary mapping
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        logger.debug(f"Applying generic binary mapping: {mapping}")
        return s.astype(str).map(mapping).astype("Int64")

    logger.debug("Non-binary feature detected — returning unchanged")
    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    """
    logger.info("Starting feature engineering pipeline")

    df = df.copy()
    logger.debug(f"Initial dataframe shape: {df.shape}")

    # ==============================
    # Step 1: Identify feature types
    # ==============================
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    logger.info(f"Detected {len(obj_cols)} categorical columns")
    logger.info(f"Detected {len(numeric_cols)} numeric columns")
    logger.debug(f"Categorical columns: {obj_cols}")
    logger.debug(f"Numeric columns: {numeric_cols}")

    # ==============================
    # Step 2: Split categorical columns
    # ==============================
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    logger.info(f"Binary columns: {len(binary_cols)}")
    logger.info(f"Multi-category columns: {len(multi_cols)}")
    logger.debug(f"Binary column names: {binary_cols}")
    logger.debug(f"Multi-category column names: {multi_cols}")

    # ==============================
    # Step 3: Binary encoding
    # ==============================
    for c in binary_cols:
        original_dtype = df[c].dtype
        logger.debug(f"Encoding binary column '{c}' (dtype: {original_dtype})")

        df[c] = _map_binary_series(df[c].astype(str))

        logger.debug(f"Column '{c}' encoded successfully")

    # ==============================
    # Step 4: Convert boolean columns
    # ==============================
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    if bool_cols:
        logger.debug(f"Converting boolean columns to int: {bool_cols}")
        df[bool_cols] = df[bool_cols].astype(int)

    # ==============================
    # Step 5: One-hot encoding
    # ==============================
    if multi_cols:
        logger.info(f"Applying one-hot encoding to {len(multi_cols)} columns")

        original_shape = df.shape
        logger.debug(f"Shape before encoding: {original_shape}")

        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

        new_shape = df.shape
        logger.debug(f"Shape after encoding: {new_shape}")

        new_features = new_shape[1] - original_shape[1] + len(multi_cols)
        logger.info(f"Created {new_features} new one-hot encoded features")

    # ==============================
    # Step 6: Data type cleanup
    # ==============================
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            logger.debug(f"Cleaning binary column '{c}' (filling NaNs and casting to int)")
            df[c] = df[c].fillna(0).astype(int)

    logger.info(f"Feature engineering complete — final shape: {df.shape}")

    return df
