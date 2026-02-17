import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validate Telco Customer Churn dataset using Great Expectations.

    Returns:
        Tuple[bool, List[str]]: (is_valid, failed_expectations)
    """

    logger.info("Starting data validation...")

    # =======================
    # EARLY COLUMN CHECKS
    # =======================
    required_cols = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False, ["missing_required_columns"]

    logger.info("All required columns are present.")

    # =======================
    # CHECKS
    # =======================
    logger.info("Running schema + business validation checks...")
    failed_expectations: List[str] = []

    # Ensure stable numeric checks
    total_charges_num = df["TotalCharges"]
    if total_charges_num.dtype == "object":
        total_charges_num = total_charges_num.replace(" ", None)
    total_charges_num = total_charges_num.astype("float64")

    # Non-null checks
    if df["customerID"].isna().any():
        failed_expectations.append("expect_customer_id_not_null")
    if df["tenure"].isna().any():
        failed_expectations.append("expect_tenure_not_null")
    if df["MonthlyCharges"].isna().any():
        failed_expectations.append("expect_monthly_charges_not_null")

    # Set membership checks
    if not df["gender"].isin(["Male", "Female"]).all():
        failed_expectations.append("expect_gender_in_set")
    if not df["Partner"].isin(["Yes", "No"]).all():
        failed_expectations.append("expect_partner_in_set")
    if not df["Dependents"].isin(["Yes", "No"]).all():
        failed_expectations.append("expect_dependents_in_set")
    if not df["PhoneService"].isin(["Yes", "No"]).all():
        failed_expectations.append("expect_phone_service_in_set")
    if not df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all():
        failed_expectations.append("expect_contract_in_set")
    if not df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all():
        failed_expectations.append("expect_internet_service_in_set")

    # Numeric range checks
    if not df["tenure"].between(0, 120).all():
        failed_expectations.append("expect_tenure_between_0_120")
    if not df["MonthlyCharges"].between(0, 200).all():
        failed_expectations.append("expect_monthly_charges_between_0_200")
    total_charges_non_null = total_charges_num.dropna()
    if not (total_charges_non_null >= 0).all():
        failed_expectations.append("expect_total_charges_gte_0")

    # Consistency check: mostly 95% rows should have TotalCharges >= MonthlyCharges
    consistency_mask = total_charges_num.notna()
    consistency_ratio = (total_charges_num[consistency_mask] >= df.loc[consistency_mask, "MonthlyCharges"]).mean()
    if consistency_ratio < 0.95:
        failed_expectations.append("expect_total_charges_gte_monthly_charges_mostly")

    total_checks = 13
    failed_checks = len(failed_expectations)
    passed_checks = total_checks - failed_checks
    success = failed_checks == 0

    if success:
        logger.info(f"Validation PASSED: {passed_checks}/{total_checks} checks passed.")
    else:
        logger.error(f"Validation FAILED: {failed_checks}/{total_checks} checks failed.")
        logger.error(f"Failed expectations: {failed_expectations}")

    return success, failed_expectations
