import great_expectations as ge
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def validate_telco_data(df) -> Dict[str, Any]:
    """
    Validate Telco Customer Churn dataset using Great Expectations.

    Returns:
        Dict[str, Any]: Structured validation report
    """

    logger.info("Starting data validation with Great Expectations...")

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

        return {
            "success": False,
            "missing_columns": missing_cols,
            "failed_expectations": ["missing_required_columns"],
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "details": "Schema validation failed due to missing required columns."
        }

    logger.info("All required columns are present.")

    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)

    # =======================
    # EXPECTATIONS
    # =======================
    logger.info("Running schema + business validation expectations...")

    # Schema checks
    ge_df.expect_column_values_to_not_be_null("customerID")

    # Business logic checks
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])

    ge_df.expect_column_values_to_be_in_set(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "InternetService", ["DSL", "Fiber optic", "No"]
    )

    # Numeric constraints
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    # Consistency check
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )

    # =======================
    # RUN VALIDATION
    # =======================
    logger.info("Executing validation suite...")
    results = ge_df.validate()

    # =======================
    # PROCESS RESULTS
    # =======================
    failed_expectations: List[str] = []

    for r in results["results"]:
        if not r["success"]:
            failed_expectations.append(r["expectation_config"]["expectation_type"])

    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    success = results["success"]

    report = {
        "success": success,
        "missing_columns": [],
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "failed_expectations": failed_expectations,
        "details": "Validation passed successfully." if success else "Validation failed.",
    }

    if success:
        logger.info(f"Validation PASSED: {passed_checks}/{total_checks} checks passed.")
    else:
        logger.error(f"Validation FAILED: {failed_checks}/{total_checks} checks failed.")
        logger.error(f"Failed expectations: {failed_expectations}")

    return report
