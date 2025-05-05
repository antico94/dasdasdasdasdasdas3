# Strategies/base_strategy.py
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple

import numpy as np

from Config.trading_config import TimeFrame
from Database.models import PriceBar
from Events.events import SignalEvent
from Logger.logger import DBLogger


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.

    This class enforces the use of completed candles for signal generation
    and provides proper multi-timeframe synchronization.
    """

    def __init__(self, name: str, symbol: str, timeframes: Set[TimeFrame], logger: DBLogger = None, **kwargs):
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

    def on_bar(self, timeframe: TimeFrame, bars: List[PriceBar]) -> Optional[SignalEvent]:
        """
        Process new bars for a specific timeframe, ensuring only completed bars are used.
        This method is called by the trading system when new bars are available.
        It strictly enforces using only completed bars (not the current forming bar).

        Args:
            timeframe: The timeframe of the bars
            bars: List of price bars (includes current forming bar)

        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        if not self.is_enabled():
            return None

        # Validate we have enough bars (at least one completed bar and one forming bar)
        if len(bars) < 2:
            self.log_strategy_event(
                level="WARNING",
                message=f"Insufficient bars for {self.symbol} on {timeframe.name}. Need at least 2, got {len(bars)}",
                action="on_bar",
                status="validation_failed",
                details={"timeframe": timeframe.name, "bars_count": len(bars)}
            )
            return None

        # Separate the completed bars from the current forming bar
        # IMPORTANT: Create full copies of all bars to avoid session binding issues
        completed_bars = []
        for bar in bars[:-1]:  # All bars except the last one
            completed_bar = PriceBar(
                instrument_id=bar.instrument_id,
                timeframe_id=bar.timeframe_id,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                spread=bar.spread
            )
            completed_bars.append(completed_bar)

        # Deep copy the forming bar too
        forming_bar = PriceBar(
            instrument_id=bars[-1].instrument_id,
            timeframe_id=bars[-1].timeframe_id,
            timestamp=bars[-1].timestamp,
            open=bars[-1].open,
            high=bars[-1].high,
            low=bars[-1].low,
            close=bars[-1].close,
            volume=bars[-1].volume,
            spread=bars[-1].spread
        )

        # Get the latest completed candle
        latest_completed_candle = completed_bars[-1]

        # Check if we've already processed this candle
        if (self.last_processed_timestamps[timeframe] is not None and
                latest_completed_candle.timestamp <= self.last_processed_timestamps[timeframe]):
            # We've already processed this candle, no need to do it again
            return None

        # Format timestamp safely for logging
        timestamp_str = str(latest_completed_candle.timestamp)

        # Log the new completed candle we're about to process
        self.log_strategy_event(
            level="DEBUG",
            message=f"Processing new completed {timeframe.name} candle for {self.symbol}",
            action="on_bar",
            status="processing",
            details={
                "timeframe": timeframe.name,
                "timestamp": timestamp_str,
                "open": latest_completed_candle.open,
                "high": latest_completed_candle.high,
                "low": latest_completed_candle.low,
                "close": latest_completed_candle.close,
                "volume": latest_completed_candle.volume,
                "is_completed": True,
                "forming_bar_time": str(forming_bar.timestamp)  # Using the forming bar for context
            }
        )

        # Update the timestamp of the last processed candle
        self.last_processed_timestamps[timeframe] = latest_completed_candle.timestamp

        # Store the completed bars for strategy access (excluding the forming bar)
        self.completed_bars[timeframe] = completed_bars

        # Calculate and cache indicators for this timeframe (using only completed bars)
        self.update_indicators(timeframe)

        # Check if signal throttling is active for this timeframe/bar
        if not self.check_signal_throttling(timeframe):
            # Safely format the last signal time for logging
            last_signal_time_str = "None"
            if timeframe in self.last_signal_time and self.last_signal_time[timeframe] is not None:
                last_signal_time_str = str(self.last_signal_time[timeframe])

            self.log_strategy_event(
                level="INFO",
                message=f"Signal throttling active for {self.symbol} on {timeframe.name}",
                action="on_bar",
                status="throttled",
                details={
                    "timeframe": timeframe.name,
                    "last_signal_time": last_signal_time_str
                }
            )
            return None

        # Calculate signal based on completed candles only
        signal = self.calculate_signals(timeframe)

        # If a signal was generated, mark this timestamp to prevent multiple signals
        if signal:
            self.mark_signal_generated(timeframe)

        return signal

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
        Ensures only one signal per completed candle is generated.

        Args:
            timeframe: The timeframe to check

        Returns:
            True if a signal can be generated, False if throttled
        """
        if timeframe not in self.last_signal_time:
            return True

        # Get the timestamp of the last processed candle
        last_processed = self.last_processed_timestamps.get(timeframe)
        if last_processed is None:
            return True

        # Get the timestamp of the last signal for this timeframe
        last_signal = self.last_signal_time.get(timeframe)
        if last_signal is None:
            return True

        # If the last signal was from this same candle, throttle it
        if last_signal == last_processed:
            return False

        # Signal is allowed if it's for a new candle
        return True

    def mark_signal_generated(self, timeframe: TimeFrame):
        """
        Mark that a signal has been generated for the current completed candle.
        This prevents multiple signals from the same candle.

        Args:
            timeframe: The timeframe the signal was generated for
        """
        # First check if the timeframe exists in our timestamps dictionary
        if timeframe not in self.last_processed_timestamps:
            # Nothing to mark, just return
            return

        # Get the timestamp, being careful with types
        last_processed = self.last_processed_timestamps[timeframe]

        # Only proceed if we have a valid timestamp
        if last_processed is None:
            return

        # Store the timestamp
        self.last_signal_time[timeframe] = last_processed

        # Format for logging only if it's a datetime object
        if isinstance(last_processed, datetime):
            timestamp_str = last_processed.isoformat()
        else:
            timestamp_str = str(last_processed)

        self.log_strategy_event(
            level="DEBUG",
            message=f"Signal marked for {self.symbol} on {timeframe.name} at {timestamp_str}",
            action="mark_signal_generated",
            status="success",
            details={"timeframe": timeframe.name, "timestamp": timestamp_str}
        )

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

        # Create arrays directly using data from detached bars
        open_array = np.array([float(bar.open) for bar in bars])  # Explicitly convert to float
        high_array = np.array([float(bar.high) for bar in bars])
        low_array = np.array([float(bar.low) for bar in bars])
        close_array = np.array([float(bar.close) for bar in bars])
        volume_array = np.array([float(bar.volume) for bar in bars])

        return {
            'open': open_array,
            'high': high_array,
            'low': low_array,
            'close': close_array,
            'volume': volume_array
        }

    def validate_bars(self, timeframe: TimeFrame, min_bars: int) -> bool:
        """
        Validate that we have enough completed bars for analysis.

        Args:
            timeframe: The timeframe to validate
            min_bars: Minimum number of completed bars required

        Returns:
            True if validation passes, False otherwise
        """
        bars = self.get_completed_bars(timeframe)

        if len(bars) < min_bars:
            self.log_strategy_event(
                level="WARNING",
                message=f"Not enough completed bars for {self.symbol} on {timeframe.name}. Need {min_bars}, got {len(bars)}",
                action="validate_bars",
                status="failed",
                details={"timeframe": timeframe.name, "required": min_bars, "available": len(bars)}
            )
            return False

        # Get server time from MT5 manager (should be injected or accessible)
        # This is safer than using local machine time
        if hasattr(self, 'mt5_manager') and self.mt5_manager:
            current_time = self.mt5_manager.get_server_time()
        else:
            # Fallback to UTC time if MT5 manager not available
            current_time = datetime.now(timezone.utc).replace(tzinfo=None)
            self.log_strategy_event(
                level="WARNING",
                message="MT5 server time not available, using local UTC time for validation",
                action="validate_bars",
                status="fallback_time",
                details={"timeframe": timeframe.name}
            )

        # Validate no bars from the future are included
        future_bars = [bar for bar in bars if bar.timestamp > current_time]

        if future_bars:
            self.log_strategy_event(
                level="ERROR",
                message=f"Found {len(future_bars)} bars from the future for {self.symbol} on {timeframe.name}",
                action="validate_bars",
                status="future_data_detected",
                details={
                    "timeframe": timeframe.name,
                    "future_bars_count": len(future_bars),
                    "first_future_timestamp": future_bars[0].timestamp.isoformat() if future_bars else None
                }
            )
            return False

        return True

    def get_completed_bars(self, timeframe: TimeFrame, lookback: int = None) -> List[PriceBar]:
        """
        Get the completed bars for a specific timeframe with validation.

        This method ensures only completed bars are used for strategy decisions.

        Args:
            timeframe: The timeframe to get bars for
            lookback: Number of bars to return (None for all available)

        Returns:
            List of completed price bars
        """
        if timeframe not in self.completed_bars:
            self.log_strategy_event(
                level="WARNING",
                message=f"No completed bars available for {self.symbol} on {timeframe.name}",
                action="get_completed_bars",
                status="no_data",
                details={"timeframe": timeframe.name}
            )
            return []

        bars = self.completed_bars.get(timeframe, [])

        if not bars:
            return []

        # Apply lookback filter if specified
        if lookback is not None and 0 < lookback < len(bars):
            return bars[-lookback:]

        return bars

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

    # Add to Strategies/base_strategy.py

    # Add to Strategies/base_strategy.py

    def print_strategy_conditions(self, timeframe: TimeFrame,
                                  condition_groups: Dict[str, List[Tuple[str, bool, str]]]) -> None:
        """
        Print the current values of strategy-specific indicators and whether they meet conditions.

        Args:
            timeframe: The timeframe being analyzed
            condition_groups: Dictionary of group_name -> list of conditions [(name, passed, details)]
        """
        output_lines = []

        # Process each condition group
        for group_name, conditions in condition_groups.items():
            output_lines.append(f"=== {group_name} for {self.symbol} on {timeframe.name} ===")

            # Add each condition with its status and details
            for i, (name, passed, details) in enumerate(conditions, 1):
                status = "✅" if passed else "❌"
                output_lines.append(f"{i}. {name}: {status} ({details})")

            # Add a blank line between groups
            output_lines.append("")

        # Remove the last empty line if there is one
        if output_lines and output_lines[-1] == "":
            output_lines.pop()

        # Combine and print
        output = "\n".join(output_lines)
        print(output)

        # Also log the output
        self.log_strategy_event(
            level="INFO",
            message=f"Strategy conditions for {self.symbol} on {timeframe.name}",
            action="print_strategy_conditions",
            status="analysis",
            details={"conditions": output}
        )