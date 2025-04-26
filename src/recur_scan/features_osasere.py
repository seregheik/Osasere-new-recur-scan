import re

import numpy as np  # type: ignore

from recur_scan.transactions import Transaction
from recur_scan.utils import get_day, parse_date


def has_min_recurrence_period(
    transaction: Transaction,
    all_transactions: list[Transaction],
    min_days: int = 60,
) -> bool:
    """Check if transactions from the same vendor span at least `min_days`."""
    vendor_txs = [t for t in all_transactions if t.name.lower() == transaction.name.lower()]
    if len(vendor_txs) < 2:
        return False
    dates = sorted([parse_date(t.date) for t in vendor_txs])
    return (dates[-1] - dates[0]).days >= min_days


def get_day_of_month_consistency(
    transaction: Transaction,
    all_transactions: list[Transaction],
    tolerance_days: int = 7,
) -> float:
    """Calculate the fraction of transactions within `tolerance_days` of the target day."""
    vendor_txs = [t for t in all_transactions if t.name.lower() == transaction.name.lower()]
    if len(vendor_txs) < 2:
        return 0.0
    target_day = get_day(transaction.date)
    matches = 0
    for t in vendor_txs:
        day_diff = abs(get_day(t.date) - target_day)
        if day_diff <= tolerance_days or day_diff >= 28 - tolerance_days:  # Handle month-end
            matches += 1
    return matches / len(vendor_txs)


def get_day_of_month_variability(
    transaction: Transaction,
    all_transactions: list[Transaction],
) -> float:
    """Measure consistency of day-of-month (lower = more consistent)."""
    vendor_txs = [t for t in all_transactions if t.name.lower() == transaction.name.lower()]
    if len(vendor_txs) < 2:
        return 31.0  # Max possible variability

    days = [get_day(t.date) for t in vendor_txs]
    # Handle month-end transitions (e.g., 28th vs 1st)
    adjusted_days = []
    for day in days:
        if day > 28 and day < 31:  # Treat 28+, 1, 2, 3 as close
            adjusted_days.extend([day, day - 31])
        else:
            adjusted_days.append(day)
    return np.std(adjusted_days)  # type: ignore


def get_recurrence_confidence(
    transaction: Transaction,
    all_transactions: list[Transaction],
    decay_rate: float = 2,  # Higher = recent transactions matter more
) -> float:
    """Calculate a confidence score (0-1) based on weighted historical recurrences."""
    vendor_txs = sorted(
        [t for t in all_transactions if t.name.lower() == transaction.name.lower()],
        key=lambda x: x.date,
    )
    if len(vendor_txs) < 2:
        return 0.0

    confidence = 0.0
    for i in range(1, len(vendor_txs)):
        days_diff = (parse_date(vendor_txs[i].date) - parse_date(vendor_txs[i - 1].date)).days
        # Weight by decay_rate^(time ago) and normalize
        confidence += (decay_rate**i) * (1.0 if days_diff <= 35 else 0.0)

    return confidence / sum(decay_rate**i for i in range(1, len(vendor_txs)))


def is_weekday_consistent(transaction: Transaction, all_transactions: list[Transaction]) -> bool:
    vendor_txs = [t for t in all_transactions if t.name.lower() == transaction.name.lower()]
    weekdays = [parse_date(t.date).weekday() for t in vendor_txs]  # Monday=0, Sunday=6
    return len(set(weekdays)) <= 2  # Allow minor drift (e.g., weekend vs. Monday)


def get_median_period(transaction: Transaction, all_transactions: list[Transaction]) -> float:
    vendor_txs = [t for t in all_transactions if t.name.lower() == transaction.name.lower()]
    dates = sorted([parse_date(t.date) for t in vendor_txs])
    if len(dates) < 2:
        return 0.0
    day_diffs = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    return float(np.median(day_diffs))  # Median is robust to outliers


# New Features


def get_fixed_recurring(name: str, transaction: Transaction) -> bool:
    """Check if the transaction is a fixed recurring payment."""
    # Check if the transaction name contains the specified name (case-insensitive)
    return name.lower() in transaction.name.lower()


# def is_att_recurring_pattern(
#     transaction: Transaction,
#     all_transactions: list[Transaction],
# ) -> bool:
#     """
#     Identifies AT&T transactions as recurring by examining the pattern of all AT&T transactions.

#     This function first identifies if the current transaction is from AT&T, then checks if there
#     are multiple AT&T transactions to confirm a recurring pattern.

#     Args:
#         transaction: The transaction to check
#         all_transactions: List of all transactions to analyze for patterns

#     Returns:
#         True if the transaction is from AT&T and shows recurring patterns, False otherwise
#     """
#     # AT&T identification patterns (case insensitive)
#     att_patterns = ["at&t", "att ", "at and t", "at & t", "at-t"]

#     # Check if this transaction is from AT&T
#     transaction_name = transaction.name.lower()
#     is_att = any(pattern in transaction_name for pattern in att_patterns)

#     if not is_att:
#         return False

#     # Find all AT&T transactions
#     att_transactions = []
#     for t in all_transactions:
#         t_name = t.name.lower()
#         if any(pattern in t_name for pattern in att_patterns):
#             att_transactions.append(t)

#     # Always mark as recurring if we have at least 2 AT&T transactions
#     # This ensures all AT&T transactions are marked as recurring if multiple exist
#     return len(att_transactions) >= 2


def detect_installment_payments(
    transaction: Transaction,
    all_transactions: list[Transaction],
) -> bool:
    """
    Detects installment payment services like AfterPay and similar services.

    Args:
        transaction: The transaction to check
        all_transactions: List of all transactions to analyze for patterns

    Returns:
        True if the transaction appears to be an installment payment, False otherwise
    """
    # List of common installment payment services
    installment_services = [
        "afterpay",
        "klarna",
        "affirm",
        "splitit",
        "sezzle",
        "quadpay",
        "zip",
    ]

    # Check if transaction is from an installment service
    if not any(service in transaction.name.lower() for service in installment_services):
        return False

    # Get all transactions from the same vendor
    vendor_txs = [t for t in all_transactions if t.name.lower() == transaction.name.lower()]

    # Installment payments typically have at least 2 payments
    if len(vendor_txs) < 2:
        return False

    # Analyze date patterns - installments often happen every 2-4 weeks
    dates = sorted([parse_date(t.date) for t in vendor_txs])
    if len(dates) < 2:
        return False

    day_diffs = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]

    # Check if any consecutive transactions are 10-20 days apart (typical for biweekly payments)
    has_biweekly = any(10 <= diff <= 20 for diff in day_diffs)

    # If we have any reasonably close payments and it's an installment service, mark as recurring
    return has_biweekly


def detect_financial_service_fees(
    transaction: Transaction,
    all_transactions: list[Transaction],
) -> bool:
    """
    Detects recurring financial service fees from apps like Albert, FloatMe, etc.

    Args:
        transaction: The transaction to check
        all_transactions: List of all transactions to analyze for patterns

    Returns:
        True if the transaction appears to be a financial service fee, False otherwise
    """
    # Common financial service apps
    financial_services = [
        "albert",
        "floatme",
        "earnin",
        "dave",
        "brigit",
        "empower",
        "moneyLion",
        "vola",
        "chime",
    ]

    # Check if transaction is from a financial service
    if not any(service in transaction.name.lower() for service in financial_services):
        return False

    # Get all transactions from the same vendor
    vendor_txs = [t for t in all_transactions if t.name.lower() == transaction.name.lower()]

    # If we have 3+ transactions from the same financial service with the same amount,
    # it's likely a recurring service fee
    if len(vendor_txs) >= 3:
        # Check if at least 3 transactions have the same amount
        amounts: dict[float, int] = {}
        for t in vendor_txs:
            amount = t.amount
            amounts[amount] = amounts.get(amount, 0) + 1

        return any(count >= 3 for count in amounts.values())

    return False


# def detect_variable_amount_recurring(
#     transaction: Transaction,
#     all_transactions: list[Transaction],
#     min_transactions: int = 3,
#     max_amount_variation: float = 0.5,
# ) -> bool:
#     """
#     Detects recurring payments that have varying amounts (like utilities).

#     Args:
#         transaction: The transaction to check
#         all_transactions: List of all transactions to analyze for patterns
#         min_transactions: Minimum number of transactions to establish a pattern
#         max_amount_variation: Maximum allowed variation in amount (as a ratio)

#     Returns:
#         True if the transaction appears to be recurring with variable amounts, False otherwise
#     """
#     # Get all transactions from the same vendor
#     vendor_txs = [
#         t for t in all_transactions if t.name.lower() == transaction.name.lower()
#     ]

#     if len(vendor_txs) < min_transactions:
#         return False

#     # Sort by date
#     vendor_txs.sort(key=lambda x: x.date)

#     # Check if there's a pattern in transaction dates
#     dates = [parse_date(t.date) for t in vendor_txs]

#     # Calculate intervals between transactions
#     intervals = []
#     for i in range(1, len(dates)):
#         interval = (dates[i] - dates[i - 1]).days
#         intervals.append(interval)

#     # Check if intervals are consistent (within 7 days variation)
#     if len(intervals) >= 2:
#         median_interval = np.median(intervals)
#         is_regular_timing = all(abs(i - median_interval) <= 7 for i in intervals)

#         if is_regular_timing:
#             # Check amount variation
#             amounts = [float(t.amount) for t in vendor_txs]
#             mean_amount = np.mean(amounts)

#             # Calculate coefficient of variation to measure amount consistency
#             if mean_amount > 0:
#                 cv = np.std(amounts) / mean_amount

#                 # If amount variation is within acceptable range, consider it recurring
#                 if cv <= max_amount_variation:
#                     return True

#     return False


def normalize_vendor_name(name: str) -> str:
    """
    Normalize vendor names by removing common suffixes and extra spaces.

    Args:
        name: The vendor name to normalize

    Returns:
        Normalized vendor name
    """
    # Convert to lowercase
    name = name.lower()

    # Remove common suffixes
    suffixes = [" apa", " llc", " inc", " corp", " co", " ltd"]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Remove phone numbers and location details in common formats
    name = re.sub(r"\s+\d{3}-\d{3}-\d{4}\s+.*", "", name)
    name = re.sub(r"\s+\d{3}-\d{7}\s+.*", "", name)
    name = re.sub(r"\s+\d{10}\s+.*", "", name)
    name = re.sub(r"\s+\d{3}\-\d{4}\s+.*", "", name)

    # Remove URLs and common transaction details
    name = re.sub(r"\s+[a-z0-9]+\.[a-z]{2,3}(/[a-z]+)?\s+.*", "", name)

    # Remove city and state patterns
    name = re.sub(r"\s+[A-Za-z]+\s+[A-Z]{2}.*", "", name)

    # Remove extra spaces
    name = " ".join(name.split())

    return name


def detect_housing_payments(
    transaction: Transaction,
    all_transactions: list[Transaction],
) -> bool:
    """
    Detects housing/rent payments which may vary in amount but occur monthly.

    Args:
        transaction: The transaction to check
        all_transactions: List of all transactions to analyze for patterns

    Returns:
        True if the transaction appears to be housing/rent related, False otherwise
    """
    # Housing-related keywords
    housing_keywords = [
        "rent",
        "lease",
        "apartment",
        "apt",
        "condo",
        "townhome",
        "housing",
        "mortgage",
        "property",
        "realty",
        "real estate",
        "waterford",
        "grove",
        "residence",
        "home",
    ]

    # Check if transaction name contains housing keywords
    if not any(keyword in transaction.name.lower() for keyword in housing_keywords):
        return False

    # Normalize the vendor name
    norm_name = normalize_vendor_name(transaction.name)

    # Find all transactions from similar vendors
    similar_txs = []
    for t in all_transactions:
        if norm_name in normalize_vendor_name(t.name):
            similar_txs.append(t)

    # Need at least 2 transactions to establish a pattern
    if len(similar_txs) < 2:
        return False

    # Sort by date
    similar_txs.sort(key=lambda x: x.date)
    dates = [parse_date(t.date) for t in similar_txs]

    # Check if transactions occur roughly monthly (15-45 days apart)
    if len(dates) >= 2:
        intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
        monthly_intervals = [i for i in intervals if 15 <= i <= 45]

        # If at least half of the intervals are monthly, consider it a housing payment
        return len(monthly_intervals) >= len(intervals) // 2

    return False


def get_new_features(
    transaction: Transaction,
    all_transactions: list[Transaction],
) -> dict[str, float]:
    """Generate new features for the transaction."""
    return {
        "is_AT&T_": get_fixed_recurring("AT&T", transaction),
        "is_water_utility": get_fixed_recurring("Water", transaction),
        "is_installment_payment": detect_installment_payments(transaction, all_transactions),
        "is_financial_service_fee": detect_financial_service_fees(transaction, all_transactions),
        # "is_variable_recurring": detect_variable_amount_recurring(
        #     transaction, all_transactions
        # ),
        "is_housing_payment": detect_housing_payments(transaction, all_transactions),
        # "is_insurance_payment_O": detect_insurance_payments(
        #     transaction, all_transactions
        # ),
    }
