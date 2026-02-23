"""
Data Validator

Purpose:
    Validates fetched OHLCV data for quality and integrity.
    Detects anomalies, missing data, and corrupt values.

Dependencies:
    - pandas for data manipulation

Logging:
    - Validation issues at WARNING
    - Clean data at DEBUG

Fallbacks:
    Flags bad data but doesn't discard it (let caller decide).
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger
from src.utils.config_loader import load_config
from src.utils.validators import validate_ohlcv_data

logger = get_logger(__name__)


class DataValidator:
    """Validates and cleans fetched market data."""

    def __init__(self):
        """Initialize with system config thresholds."""
        config = load_config("system")
        data_config = config.get("data", {})
        self.max_price_change_pct = data_config.get(
            "max_price_change_percent", 20
        )
        self.min_volume = data_config.get(
            "min_volume_threshold", 1000
        )

    def validate_records(
        self,
        records: List[Dict],
        symbol: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate a list of OHLCV records.

        Args:
            records: List of OHLCV dictionaries.
            symbol: Stock symbol for logging.

        Returns:
            Tuple of (valid_records, invalid_records).
        """
        valid = []
        invalid = []

        for i, record in enumerate(records):
            issues = self._validate_single_record(record)

            if issues:
                record["_validation_issues"] = issues
                invalid.append(record)
                logger.debug(
                    f"Invalid record for {symbol} "
                    f"on {record.get('date')}: {issues}"
                )
            else:
                valid.append(record)

        if invalid:
            logger.warning(
                f"Validation: {len(invalid)}/{len(records)} "
                f"invalid records for {symbol}",
                extra={
                    "symbol": symbol,
                    "total": len(records),
                    "valid": len(valid),
                    "invalid": len(invalid),
                },
            )
        else:
            logger.debug(
                f"All {len(records)} records valid for {symbol}"
            )

        return valid, invalid

    def _validate_single_record(
        self, record: Dict
    ) -> List[str]:
        """
        Validate a single OHLCV record.

        Args:
            record: OHLCV dictionary.

        Returns:
            List of issue descriptions (empty if valid).
        """
        issues = []

        # Check required fields
        required = ["date", "open", "high", "low", "close", "volume"]
        for field in required:
            if field not in record or record[field] is None:
                issues.append(f"Missing field: {field}")

        if issues:
            return issues

        o, h, l, c = (
            record["open"],
            record["high"],
            record["low"],
            record["close"],
        )
        v = record["volume"]

        # Price sanity checks
        if any(p <= 0 for p in [o, h, l, c]):
            issues.append("Negative or zero price")

        if h < l:
            issues.append(f"High ({h}) < Low ({l})")

        if h < max(o, c):
            issues.append("High is not the highest value")

        if l > min(o, c):
            issues.append("Low is not the lowest value")

        # Volume check
        if v < 0:
            issues.append(f"Negative volume: {v}")

        if v == 0:
            issues.append("Zero volume")

        # Check for NaN/Inf
        for field, value in [
            ("open", o), ("high", h), ("low", l),
            ("close", c), ("volume", v),
        ]:
            if isinstance(value, float) and (
                np.isnan(value) or np.isinf(value)
            ):
                issues.append(f"NaN/Inf in {field}")

        return issues

    def validate_price_continuity(
        self,
        records: List[Dict],
        symbol: str,
    ) -> List[Dict]:
        """
        Check for abnormal price jumps between consecutive days.

        Args:
            records: Sorted list of OHLCV records.
            symbol: Stock symbol.

        Returns:
            List of records with abnormal jumps.
        """
        anomalies = []

        for i in range(1, len(records)):
            prev_close = records[i - 1]["close"]
            curr_open = records[i]["open"]

            if prev_close <= 0:
                continue

            gap_pct = abs(
                (curr_open - prev_close) / prev_close * 100
            )
            if gap_pct > self.max_price_change_pct:
                anomalies.append(
                    {
                        "date": records[i]["date"],
                        "prev_close": prev_close,
                        "curr_open": curr_open,
                        "gap_percent": gap_pct,
                    }
                )
                logger.warning(
                    f"Price gap anomaly for {symbol} on "
                    f"{records[i]['date']}: "
                    f"{gap_pct:.1f}% gap "
                    f"({prev_close} -> {curr_open})",
                    extra={
                        "symbol": symbol,
                        "gap_percent": gap_pct,
                    },
                )

        return anomalies

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Validate a DataFrame of OHLCV data.

        Args:
            df: OHLCV DataFrame.
            symbol: Stock symbol.

        Returns:
            Validation results dictionary.
        """
        result = validate_ohlcv_data(df)

        # Additional checks
        if not df.empty:
            # Check for duplicates
            if hasattr(df.index, "duplicated"):
                dup_count = df.index.duplicated().sum()
                if dup_count > 0:
                    result["issues"].append(
                        f"Duplicate dates: {dup_count}"
                    )
                    result["valid"] = False

            # Check for gaps (missing trading days)
            if len(df) > 1 and hasattr(df.index, "to_series"):
                date_diffs = df.index.to_series().diff()
                # Flag gaps > 5 business days
                large_gaps = date_diffs[
                    date_diffs > pd.Timedelta(days=7)
                ]
                if len(large_gaps) > 0:
                    result["issues"].append(
                        f"Large date gaps: {len(large_gaps)}"
                    )

        if result["issues"]:
            logger.warning(
                f"DataFrame validation issues for {symbol}: "
                f"{result['issues']}",
                extra={"symbol": symbol},
            )

        return result

    def clean_records(
        self,
        records: List[Dict],
        symbol: str,
    ) -> List[Dict]:
        """
        Clean and sanitize OHLCV records.

        Args:
            records: Raw OHLCV records.
            symbol: Stock symbol.

        Returns:
            Cleaned records list.
        """
        valid_records, _ = self.validate_records(records, symbol)

        # Sort by date
        valid_records.sort(key=lambda x: x["date"])

        # Remove duplicates (keep last)
        seen_dates = {}
        for record in valid_records:
            date_key = str(record["date"])
            seen_dates[date_key] = record

        cleaned = list(seen_dates.values())

        # Forward-fill short gaps (weekends, holidays) so strategies
        # don't see sudden jumps caused by missing trading days.
        cleaned = self._forward_fill_gaps(cleaned, symbol)

        logger.debug(
            f"Cleaned {symbol}: {len(records)} -> "
            f"{len(cleaned)} records"
        )

        return cleaned

    def _forward_fill_gaps(
        self,
        records: List[Dict],
        symbol: str,
        max_fill_days: int = 5,
    ) -> List[Dict]:
        """
        Forward-fill gaps of up to *max_fill_days* consecutive missing
        trading days by copying the last known record.

        The date field is parsed to a ``datetime`` object when it arrives
        as a string (some data sources return ISO-format strings instead
        of ``datetime`` instances).

        Args:
            records: Sorted, deduplicated OHLCV records.
            symbol: Stock symbol (used only for logging).
            max_fill_days: Skip filling if the gap is larger than this
                (e.g. long market closures / listing gaps).

        Returns:
            Records with gaps filled in-place.
        """
        if len(records) <= 1:
            return records

        def _to_dt(value: Any) -> datetime:
            """Return a datetime from a datetime or ISO string."""
            if isinstance(value, datetime):
                return value
            # Accept "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS …"
            return datetime.strptime(str(value)[:10], "%Y-%m-%d")

        filled: List[Dict] = [records[0]]
        for record in records[1:]:
            curr_date = _to_dt(record["date"])
            prev_date = _to_dt(filled[-1]["date"])
            gap_days = (curr_date - prev_date).days

            if 1 < gap_days <= max_fill_days:
                for offset in range(1, gap_days):
                    synthetic = dict(filled[-1])
                    synthetic["date"] = prev_date + timedelta(days=offset)
                    synthetic["_filled"] = True
                    filled.append(synthetic)

            filled.append(record)

        return filled
