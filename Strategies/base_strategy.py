# Strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import numpy as np
import logging

from Events.events import SignalEvent
from Database.models import PriceBar
from Config.trading_config import TimeFrame
from Logger.logger import DBLogger


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.

    This class enforces the use of completed candles for signal generation
    and provides proper multi-timeframe synchronization.
    """

    def __init__(self, name: str, symbol: str, timeframes: Set[TimeFrame], logger: DBLogger = None):
        """
        Initialize the strategy with required parameters.

        Args:
            name: Strategy name
            symbol: Trading instrument symbol
            timeframes: Set of timeframes this strategy will use
            logger: Logger instance
        """
        self.name = name
        self.symbol = symbol
        self.timeframes = timeframes
        self.logger = logger

        # Will be set during registration
        self.instrument_id = None
        self.timeframe_ids = {}  # {TimeFrame: id}

        # Track the last processed candle timestamp for each timeframe
        self.last_processed_timestamps = {tf: None for tf in timeframes}

        # Store the latest completed (closed) bars for each timeframe
        self.completed_bars = {tf: [] for tf in timeframes}

        # Store cached indicator values for each timeframe
        self.indicators = {tf: {} for tf in timeframes}

        # Signal state
        self.last_signal_time = {tf: None for tf in timeframes}
        self.enabled = True

    def set_logger(self, logger: DBLogger):
        """Set the logger instance"""
        self.logger = logger

    def set_instrument_info(self, instrument_id: int, symbol: str):
        """Set instrument information"""
        self.instrument_id = instrument_id
        self.symbol = symbol

    def set_timeframe_ids(self, timeframe_ids: Dict[TimeFrame, int]):
        """Set timeframe IDs mapping"""
        self.timeframe_ids = timeframe_ids

    def is_enabled(self) -> bool:
        """Check if the strategy is enabled"""
        return self.enabled

    def enable(self, enabled: bool = True):
        """Enable or disable the strategy"""
        self.enabled = enabled

    def on_bar(self, timeframe: TimeFrame, bars: List[PriceBar]) -> Optional[SignalEvent]:
        """
        Process new bars for a specific timeframe.

        This method is called by the trading system when new bars are available.
        It checks if a bar has completed since the last check and triggers
        signal calculation if appropriate.

        Args:
            timeframe: The timeframe of the bars
            bars: List of price bars (includes current forming bar)

        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        if not self.is_enabled() or len(bars) < 2:
            return None

        # Check if we have a newly completed candle
        # bars[-1] is the current forming candle, bars[-2] is the latest completed candle
        completed_candle = bars[-2]

        # If we've already processed this candle, skip
        if (self.last_processed_timestamps[timeframe] is not None and
                completed_candle.timestamp <= self.last_processed_timestamps[timeframe]):
            return None

        # Log the new completed candle
        self.log_strategy_event(
            level="DEBUG",
            message=f"New completed {timeframe.name} candle detected for {self.symbol}",
            action="on_bar",
            status="processing",
            details={
                "timeframe": timeframe.name,
                "timestamp": completed_candle.timestamp.isoformat(),
                "open": completed_candle.open,
                "high": completed_candle.high,
                "low": completed_candle.low,
                "close": completed_candle.close,
                "volume": completed_candle.volume
            }
        )

        # Update the timestamp of the last processed candle
        self.last_processed_timestamps[timeframe] = completed_candle.timestamp

        # Store the completed bars (excluding the current forming bar)
        self.completed_bars[timeframe] = bars[:-1]

        # Calculate and cache indicators for this timeframe
        self.update_indicators(timeframe)

        # Calculate signal based on completed candles
        return self.calculate_signals(timeframe)

    def update_indicators(self, timeframe: TimeFrame):
        """
        Calculate and cache indicators for the specified timeframe.

        This method should be overridden by subclasses to implement
        specific indicator calculations.

        Args:
            timeframe: The timeframe to update indicators for
        """
        pass

    @abstractmethod
    def calculate_signals(self, timeframe: TimeFrame) -> Optional[SignalEvent]:
        """
        Calculate trading signals for the given timeframe.

        This method must be implemented by all strategy subclasses.

        Args:
            timeframe: The timeframe to calculate signals for

        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        pass

    def log_strategy_event(self, level: str, message: str, action: str,
                           status: str, details: Dict[str, Any] = None):
        """
        Log a strategy-related event.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Main log message
            action: Action being performed
            status: Status of the action (starting, success, failed, etc.)
            details: Additional details to log
        """
        if self.logger:
            self.logger.log_event(
                level=level,
                message=message,
                event_type="STRATEGY",
                component=self.name,
                action=action,
                status=status,
                details=details or {}
            )

    def get_completed_bars(self, timeframe: TimeFrame, lookback: int = None) -> List[PriceBar]:
        """
        Get the completed bars for a specific timeframe.

        Args:
            timeframe: The timeframe to get bars for
            lookback: Number of bars to return (None for all available)

        Returns:
            List of completed price bars
        """
        bars = self.completed_bars.get(timeframe, [])
        if lookback is not None and lookback > 0 and len(bars) > lookback:
            return bars[-lookback:]
        return bars

    def get_indicator(self, timeframe: TimeFrame, indicator_name: str) -> Any:
        """
        Get a cached indicator value for a specific timeframe.

        Args:
            timeframe: The timeframe to get indicator for
            indicator_name: Name of the indicator

        Returns:
            Cached indicator value or None if not available
        """
        return self.indicators.get(timeframe, {}).get(indicator_name)

    def set_indicator(self, timeframe: TimeFrame, indicator_name: str, value: Any):
        """
        Cache an indicator value for a specific timeframe.

        Args:
            timeframe: The timeframe to set indicator for
            indicator_name: Name of the indicator
            value: Indicator value to cache
        """
        if timeframe not in self.indicators:
            self.indicators[timeframe] = {}
        self.indicators[timeframe][indicator_name] = value

    def clear_indicators(self, timeframe: TimeFrame = None):
        """
        Clear cached indicators.

        Args:
            timeframe: Specific timeframe to clear (None for all)
        """
        if timeframe is None:
            self.indicators = {tf: {} for tf in self.timeframes}
        elif timeframe in self.indicators:
            self.indicators[timeframe] = {}

    def check_signal_throttling(self, timeframe: TimeFrame) -> bool:
        """
        Check if signal generation should be throttled.

        Ensures only one signal per candle is generated.

        Args:
            timeframe: The timeframe to check

        Returns:
            True if a signal can be generated, False if throttled
        """
        if timeframe not in self.last_signal_time:
            return True

        if (self.last_signal_time[timeframe] is None or
                self.last_processed_timestamps[timeframe] != self.last_signal_time[timeframe]):
            return True

        return False

    def mark_signal_generated(self, timeframe: TimeFrame):
        """
        Mark that a signal has been generated for the current candle.

        Args:
            timeframe: The timeframe the signal was generated for
        """
        self.last_signal_time[timeframe] = self.last_processed_timestamps[timeframe]

    def get_bars_as_arrays(self, timeframe: TimeFrame, lookback: int = None) -> Dict[str, np.ndarray]:
        """
        Convert price bars to numpy arrays for indicator calculations.

        Args:
            timeframe: The timeframe to get data for
            lookback: Number of bars to include (None for all available)

        Returns:
            Dictionary of numpy arrays for open, high, low, close, volume
        """
        bars = self.get_completed_bars(timeframe, lookback)

        if not bars:
            return {
                'open': np.array([]),
                'high': np.array([]),
                'low': np.array([]),
                'close': np.array([]),
                'volume': np.array([])
            }

        return {
            'open': np.array([bar.open for bar in bars]),
            'high': np.array([bar.high for bar in bars]),
            'low': np.array([bar.low for bar in bars]),
            'close': np.array([bar.close for bar in bars]),
            'volume': np.array([bar.volume for bar in bars])
        }

    def validate_bars(self, timeframe: TimeFrame, min_bars: int) -> bool:
        """
        Validate that we have enough bars for analysis.

        Args:
            timeframe: The timeframe to validate
            min_bars: Minimum number of bars required

        Returns:
            True if validation passes, False otherwise
        """
        bars = self.get_completed_bars(timeframe)

        if len(bars) < min_bars:
            self.log_strategy_event(
                level="WARNING",
                message=f"Not enough bars for {self.symbol} on {timeframe.name}. Need {min_bars}, got {len(bars)}",
                action="validate_bars",
                status="failed"
            )
            return False

        return True

    def create_signal(self, timeframe: TimeFrame, direction: str, reason: str,
                      strength: float = 1.0, entry_price: Optional[float] = None,
                      stop_loss: Optional[float] = None,
                      take_profit: Optional[float] = None) -> SignalEvent:
        """
        Create a signal event with proper validation.

        Args:
            timeframe: The timeframe that generated the signal
            direction: Trade direction ('BUY' or 'SELL')
            reason: Reason for the signal
            strength: Signal strength (0.0-1.0)
            entry_price: Optional entry price
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price

        Returns:
            SignalEvent with the signal details
        """
        # Check if signal throttling is active
        if not self.check_signal_throttling(timeframe):
            self.log_strategy_event(
                level="INFO",
                message=f"Signal throttled for {self.symbol} on {timeframe.name}",
                action="create_signal",
                status="throttled"
            )
            return None

        # Mark that a signal has been generated
        self.mark_signal_generated(timeframe)

        # Create the signal event
        signal = SignalEvent(
            instrument_id=self.instrument_id,
            symbol=self.symbol,
            timeframe_id=self.timeframe_ids.get(timeframe),
            timeframe_name=timeframe.name,
            direction=direction,
            strength=strength,
            strategy_name=self.name,
            reason=reason,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now()
        )

        # Log the signal generation
        self.log_strategy_event(
            level="INFO",
            message=f"Generated {direction} signal for {self.symbol} on {timeframe.name}: {reason}",
            action="create_signal",
            status="success",
            details={
                "timeframe": timeframe.name,
                "direction": direction,
                "reason": reason,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
        )

        return signal