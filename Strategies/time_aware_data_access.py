from datetime import datetime
from typing import Optional, List

from Config.trading_config import TimeFrame
from Database.models import PriceBar
from Logger.logger import DBLogger


class TimeAwareDataAccess:
    """
    Provides time-aware data access to prevent look-ahead bias.

    This class enforces that strategies can only access data that would have been
    available at a particular point in time.
    """

    def __init__(self, data_fetcher, logger: DBLogger = None):
        """
        Initialize the time-aware data access.

        Args:
            data_fetcher: Data fetcher instance
            logger: Logger instance
        """
        self.data_fetcher = data_fetcher
        self.logger = logger
        self.current_timestamp = None  # Current reference time

    def set_reference_time(self, timestamp: datetime):
        """
        Set the reference time for data access.

        Args:
            timestamp: The reference timestamp
        """
        self.current_timestamp = timestamp

        if self.logger:
            self.logger.log_event(
                level="DEBUG",
                message=f"Set reference time to {str(timestamp)}",
                event_type="TIME_AWARE_DATA",
                component="time_aware_data",
                action="set_reference_time",
                status="success"
            )

    def get_bars(self, symbol: str, timeframe: TimeFrame, count: int = 100) -> Optional[List[PriceBar]]:
        """
        Get bars for a specific symbol and timeframe, respecting the reference time.

        Args:
            symbol: The symbol to get bars for
            timeframe: The timeframe to get bars for
            count: Number of bars to return

        Returns:
            List of price bars or None if not available
        """
        if self.current_timestamp is None:
            # No reference time set, use current time
            self.set_reference_time(datetime.now())

        # Get all available bars
        all_bars = self.data_fetcher.get_latest_bars(symbol, timeframe, count)

        if not all_bars:
            return None

        # Filter bars to include only those that would have been available
        # at the reference time
        available_bars = [bar for bar in all_bars if bar.timestamp < self.current_timestamp]

        # Log what we're doing
        if self.logger:
            self.logger.log_event(
                level="DEBUG",
                message=f"Retrieved {len(available_bars)} bars for {symbol}/{timeframe.name} before {str(self.current_timestamp)}",
                event_type="TIME_AWARE_DATA",
                component="time_aware_data",
                action="get_bars",
                status="success",
                details={
                    "symbol": symbol,
                    "timeframe": timeframe.name,
                    "reference_time": str(self.current_timestamp),
                    "requested": count,
                    "all_available": len(all_bars) if all_bars else 0,
                    "time_filtered": len(available_bars)
                }
            )

        return available_bars

    def check_timestamp_valid(self, timestamp: datetime) -> bool:
        """
        Check if a timestamp is valid (not in the future relative to reference time).

        Args:
            timestamp: The timestamp to check

        Returns:
            True if valid, False if in the future
        """
        if self.current_timestamp is None:
            # No reference time set, use current time
            self.set_reference_time(datetime.now())

        is_valid = timestamp <= self.current_timestamp

        if not is_valid and self.logger:
            self.logger.log_event(
                level="WARNING",
                message=f"Attempted to access future data: {str(timestamp)} > {str(self.current_timestamp)}",
                event_type="TIME_AWARE_DATA",
                component="time_aware_data",
                action="check_timestamp_valid",
                status="future_data_detected",
                details={
                    "timestamp": str(timestamp),
                    "reference_time": str(self.current_timestamp),
                    "time_difference": str(timestamp - self.current_timestamp)
                }
            )

        return is_valid