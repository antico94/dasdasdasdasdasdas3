# Strategies/triple_ma_strategy.py
from typing import Optional, Set

from Config.trading_config import TimeFrame
from Events.events import SignalEvent
from Strategies.base_strategy import BaseStrategy
from Strategies.indicator_utils import IndicatorUtils, MAType


class TripleMAStrategy(BaseStrategy):
    """
    Triple Moving Average strategy.

    This strategy uses three moving averages of different periods to generate signals:
    - Short, medium, and long-term moving averages
    - Signals are generated when the short MA crosses the medium MA in the direction of the trend
    - The long MA is used to determine the overall trend direction
    - Optional ADX filter for trend strength
    """

    def __init__(self, name: str, symbol: str, timeframes: Set[TimeFrame],
                 short_period: int = 8,
                 medium_period: int = 21,
                 long_period: int = 55,
                 ma_type: str = "EMA",
                 adx_period: int = 14,
                 adx_threshold: float = 25.0,
                 require_macd_confirmation: bool = False,
                 stop_loss_atr_multiplier: float = 2.0,
                 take_profit_atr_multiplier: float = 3.0,
                 logger=None):
        """
        Initialize the Triple MA strategy.

        Args:
            name: Strategy name
            symbol: Trading instrument symbol
            timeframes: Set of timeframes this strategy will use
            short_period: Period for short-term MA
            medium_period: Period for medium-term MA
            long_period: Period for long-term MA
            ma_type: Type of moving average (SMA, EMA, WMA, HULL, TEMA)
            adx_period: Period for ADX calculation
            adx_threshold: ADX threshold for valid signals
            require_macd_confirmation: Require MACD confirmation for signals
            stop_loss_atr_multiplier: ATR multiplier for stop loss
            take_profit_atr_multiplier: ATR multiplier for take profit
            logger: Logger instance
        """
        super().__init__(name, symbol, timeframes, logger)

        # MA parameters
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period

        # Convert ma_type string to enum
        self.ma_type = next((t for t in MAType if t.name == ma_type), MAType.EMA)

        # Trend filter
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

        # MACD confirmation
        self.require_macd_confirmation = require_macd_confirmation

        # Risk management
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier

        # Track previous signals to avoid duplicates
        self.last_signal_type = {tf: None for tf in timeframes}

    def update_indicators(self, timeframe: TimeFrame):
        """
        Calculate and cache indicators for the Triple MA strategy.

        Args:
            timeframe: The timeframe to update indicators for
        """
        # Get completed bars
        bars = self.get_completed_bars(timeframe)

        # Ensure we have enough bars
        required_bars = self.long_period + 10  # Add some buffer
        if len(bars) < required_bars:
            return

        # Convert to numpy arrays
        arrays = self.get_bars_as_arrays(timeframe)

        # Calculate moving averages
        short_ma = IndicatorUtils.moving_average(arrays['close'], self.short_period, self.ma_type)
        medium_ma = IndicatorUtils.moving_average(arrays['close'], self.medium_period, self.ma_type)
        long_ma = IndicatorUtils.moving_average(arrays['close'], self.long_period, self.ma_type)

        # Store moving averages
        self.set_indicator(timeframe, 'short_ma', short_ma)
        self.set_indicator(timeframe, 'medium_ma', medium_ma)
        self.set_indicator(timeframe, 'long_ma', long_ma)

        # Calculate MA crossovers
        crosses_above, crosses_below = IndicatorUtils.detect_ma_cross(
            short_ma, medium_ma, self.short_period, self.medium_period
        )
        self.set_indicator(timeframe, 'crosses_above', crosses_above)
        self.set_indicator(timeframe, 'crosses_below', crosses_below)

        # Calculate trend alignment
        bullish_setup, bearish_setup = IndicatorUtils.detect_triple_ma_setup(
            arrays['close'], self.short_period, self.medium_period, self.long_period, self.ma_type
        )
        self.set_indicator(timeframe, 'bullish_setup', bullish_setup)
        self.set_indicator(timeframe, 'bearish_setup', bearish_setup)

        # Calculate ADX for trend strength
        adx, plus_di, minus_di = IndicatorUtils.adx(
            arrays['high'], arrays['low'], arrays['close'], self.adx_period
        )
        self.set_indicator(timeframe, 'adx', adx)
        self.set_indicator(timeframe, 'plus_di', plus_di)
        self.set_indicator(timeframe, 'minus_di', minus_di)

        # Calculate ATR for stop loss and take profit
        atr = IndicatorUtils.atr(
            arrays['high'], arrays['low'], arrays['close'], self.adx_period
        )
        self.set_indicator(timeframe, 'atr', atr)

        # Calculate MACD if confirmation required
        if self.require_macd_confirmation:
            macd_line, signal_line, histogram = IndicatorUtils.macd(arrays['close'])
            self.set_indicator(timeframe, 'macd_line', macd_line)
            self.set_indicator(timeframe, 'macd_signal', signal_line)
            self.set_indicator(timeframe, 'macd_histogram', histogram)

    def calculate_signals(self, timeframe: TimeFrame) -> Optional[SignalEvent]:
        """
        Calculate trading signals for the given timeframe.

        Args:
            timeframe: Timeframe to calculate signals for

        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        # Get completed bars
        bars = self.get_completed_bars(timeframe)

        # Ensure we have enough bars
        min_bars = self.long_period + 10  # Add some buffer
        if not self.validate_bars(timeframe, min_bars):
            return None

        # Get the latest completed bar
        current_bar = bars[-1]

        # Get indicators
        short_ma = self.get_indicator(timeframe, 'short_ma')
        medium_ma = self.get_indicator(timeframe, 'medium_ma')
        long_ma = self.get_indicator(timeframe, 'long_ma')
        crosses_above = self.get_indicator(timeframe, 'crosses_above')
        crosses_below = self.get_indicator(timeframe, 'crosses_below')
        bullish_setup = self.get_indicator(timeframe, 'bullish_setup')
        bearish_setup = self.get_indicator(timeframe, 'bearish_setup')
        adx = self.get_indicator(timeframe, 'adx')
        plus_di = self.get_indicator(timeframe, 'plus_di')
        minus_di = self.get_indicator(timeframe, 'minus_di')
        atr = self.get_indicator(timeframe, 'atr')

        # MACD values if confirmation required
        macd_line = None
        macd_signal = None
        macd_histogram = None
        if self.require_macd_confirmation:
            macd_line = self.get_indicator(timeframe, 'macd_line')
            macd_signal = self.get_indicator(timeframe, 'macd_signal')
            macd_histogram = self.get_indicator(timeframe, 'macd_histogram')

        # Ensure we have all necessary indicators
        if (short_ma is None or medium_ma is None or long_ma is None or
                crosses_above is None or crosses_below is None or
                bullish_setup is None or bearish_setup is None or
                adx is None or atr is None):
            return None

        if self.require_macd_confirmation and (macd_line is None or macd_signal is None or macd_histogram is None):
            return None

        # Calculate conditions - use .item() to get scalar values from arrays
        # For boolean arrays, use indexing to get specific values
        fresh_bullish_cross = bool(crosses_above[-1])
        fresh_bearish_cross = bool(crosses_below[-1])

        current_bullish_setup = bool(bullish_setup[-1])
        current_bearish_setup = bool(bearish_setup[-1])

        strong_trend = adx[-1] >= self.adx_threshold

        # Handle potential array comparison issues
        bullish_di = False
        bearish_di = False
        if plus_di is not None and minus_di is not None:
            bullish_di = plus_di[-1] > minus_di[-1]
            bearish_di = minus_di[-1] > plus_di[-1]

        macd_bullish = False
        macd_bearish = False
        if self.require_macd_confirmation and macd_line is not None and macd_signal is not None and macd_histogram is not None:
            macd_bullish = macd_line[-1] > macd_signal[-1] and macd_histogram[-1] > 0
            macd_bearish = macd_line[-1] < macd_signal[-1] and macd_histogram[-1] < 0

        # Prepare condition groups for display
        condition_groups = {
            "Bullish Triple MA Conditions": [
                ("Short MA crossed above Medium MA", fresh_bullish_cross,
                 f"Cross detected at index [-1]"),

                ("Bullish Setup (Short > Medium > Long)", current_bullish_setup,
                 f"Short: {short_ma[-1]:.5f}, Medium: {medium_ma[-1]:.5f}, Long: {long_ma[-1]:.5f}"),

                ("Strong Trend (ADX)", strong_trend,
                 f"ADX: {adx[-1]:.2f} >= {self.adx_threshold:.2f}"),

                ("Bullish DI", bullish_di,
                 f"+DI: {plus_di[-1]:.2f}, -DI: {minus_di[-1]:.2f}" if plus_di is not None and minus_di is not None else "N/A")
            ],

            "Bearish Triple MA Conditions": [
                ("Short MA crossed below Medium MA", fresh_bearish_cross,
                 f"Cross detected at index [-1]"),

                ("Bearish Setup (Short < Medium < Long)", current_bearish_setup,
                 f"Short: {short_ma[-1]:.5f}, Medium: {medium_ma[-1]:.5f}, Long: {long_ma[-1]:.5f}"),

                ("Strong Trend (ADX)", strong_trend,
                 f"ADX: {adx[-1]:.2f} >= {self.adx_threshold:.2f}"),

                ("Bearish DI", bearish_di,
                 f"+DI: {plus_di[-1]:.2f}, -DI: {minus_di[-1]:.2f}" if plus_di is not None and minus_di is not None else "N/A")
            ]
        }

        # Add MACD conditions if required
        if self.require_macd_confirmation:
            condition_groups["Bullish Triple MA Conditions"].append(
                ("MACD Confirmation", macd_bullish,
                 f"MACD: {macd_line[-1]:.5f}, Signal: {macd_signal[-1]:.5f}, Hist: {macd_histogram[-1]:.5f}")
            )
            condition_groups["Bearish Triple MA Conditions"].append(
                ("MACD Confirmation", macd_bearish,
                 f"MACD: {macd_line[-1]:.5f}, Signal: {macd_signal[-1]:.5f}, Hist: {macd_histogram[-1]:.5f}")
            )

        # Print the conditions
        self.print_strategy_conditions(timeframe, condition_groups)

        # Check for bullish signal (short MA crosses above medium MA with bullish setup)
        if crosses_above[-1] and bullish_setup[-1] and self.last_signal_type[timeframe] != "BUY":
            # Additional confirmation checks
            if self._check_bullish_confirmations(timeframe):
                # Calculate entry, stop loss, and take profit
                entry_price = current_bar.close
                stop_loss = entry_price - (atr[-1] * self.stop_loss_atr_multiplier)
                take_profit = entry_price + (atr[-1] * self.take_profit_atr_multiplier)

                # Log the signal
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bullish Triple MA signal detected for {self.symbol} on {timeframe.name}",
                    action="calculate_signals",
                    status="signal_detected",
                    details={
                        "short_ma": short_ma[-1],
                        "medium_ma": medium_ma[-1],
                        "long_ma": long_ma[-1],
                        "price": current_bar.close,
                        "adx": adx[-1]
                    }
                )

                # Update last signal type
                self.last_signal_type[timeframe] = "BUY"

                # Create and return signal
                return self.create_signal(
                    timeframe=timeframe,
                    direction="BUY",
                    reason="Triple MA Bullish Crossover",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

        # Check for bearish signal (short MA crosses below medium MA with bearish setup)
        elif crosses_below[-1] and bearish_setup[-1] and self.last_signal_type[timeframe] != "SELL":
            # Additional confirmation checks
            if self._check_bearish_confirmations(timeframe):
                # Calculate entry, stop loss, and take profit
                entry_price = current_bar.close
                stop_loss = entry_price + (atr[-1] * self.stop_loss_atr_multiplier)
                take_profit = entry_price - (atr[-1] * self.take_profit_atr_multiplier)

                # Log the signal
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bearish Triple MA signal detected for {self.symbol} on {timeframe.name}",
                    action="calculate_signals",
                    status="signal_detected",
                    details={
                        "short_ma": short_ma[-1],
                        "medium_ma": medium_ma[-1],
                        "long_ma": long_ma[-1],
                        "price": current_bar.close,
                        "adx": adx[-1]
                    }
                )

                # Update last signal type
                self.last_signal_type[timeframe] = "SELL"

                # Create and return signal
                return self.create_signal(
                    timeframe=timeframe,
                    direction="SELL",
                    reason="Triple MA Bearish Crossover",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

        # Check for exit signals
        elif self.last_signal_type[timeframe] == "BUY":
            # Exit long position when short crosses below medium
            if crosses_below[-1]:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Triple MA Exit Signal (Short MA crossed below Medium MA)"
                )

        elif self.last_signal_type[timeframe] == "SELL":
            # Exit short position when short crosses above medium
            if crosses_above[-1]:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Triple MA Exit Signal (Short MA crossed above Medium MA)"
                )

        return None

    def _check_bullish_confirmations(self, timeframe: TimeFrame) -> bool:
        """
        Check additional bullish confirmations.

        Args:
            timeframe: The timeframe to check

        Returns:
            True if confirmations pass, False otherwise
        """
        # Get ADX for trend strength
        adx = self.get_indicator(timeframe, 'adx')
        plus_di = self.get_indicator(timeframe, 'plus_di')
        minus_di = self.get_indicator(timeframe, 'minus_di')

        # Check ADX threshold
        if adx is not None and adx[-1] < self.adx_threshold:
            self.log_strategy_event(
                level="INFO",
                message=f"Bullish signal rejected due to weak trend for {self.symbol}",
                action="_check_bullish_confirmations",
                status="rejected",
                details={"adx": adx[-1], "threshold": self.adx_threshold}
            )
            return False

        # Check directional indicators
        if plus_di is not None and minus_di is not None:
            if plus_di[-1] <= minus_di[-1]:
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bullish signal rejected due to negative directional indicator for {self.symbol}",
                    action="_check_bullish_confirmations",
                    status="rejected",
                    details={"plus_di": plus_di[-1], "minus_di": minus_di[-1]}
                )
                return False

        # Check MACD confirmation if required
        if self.require_macd_confirmation:
            macd_line = self.get_indicator(timeframe, 'macd_line')
            macd_signal = self.get_indicator(timeframe, 'macd_signal')
            macd_histogram = self.get_indicator(timeframe, 'macd_histogram')

            if macd_line is not None and macd_signal is not None and macd_histogram is not None:
                # MACD should be positive or crossing up
                if macd_line[-1] <= macd_signal[-1] or macd_histogram[-1] <= 0:
                    self.log_strategy_event(
                        level="INFO",
                        message=f"Bullish signal rejected due to MACD not confirming for {self.symbol}",
                        action="_check_bullish_confirmations",
                        status="rejected",
                        details={
                            "macd_line": macd_line[-1],
                            "macd_signal": macd_signal[-1],
                            "macd_histogram": macd_histogram[-1]
                        }
                    )
                    return False

        return True

    def _check_bearish_confirmations(self, timeframe: TimeFrame) -> bool:
        """
        Check additional bearish confirmations.

        Args:
            timeframe: The timeframe to check

        Returns:
            True if confirmations pass, False otherwise
        """
        # Get ADX for trend strength
        adx = self.get_indicator(timeframe, 'adx')
        plus_di = self.get_indicator(timeframe, 'plus_di')
        minus_di = self.get_indicator(timeframe, 'minus_di')

        # Check ADX threshold
        if adx is not None and adx[-1] < self.adx_threshold:
            self.log_strategy_event(
                level="INFO",
                message=f"Bearish signal rejected due to weak trend for {self.symbol}",
                action="_check_bearish_confirmations",
                status="rejected",
                details={"adx": adx[-1], "threshold": self.adx_threshold}
            )
            return False

        # Check directional indicators
        if plus_di is not None and minus_di is not None:
            if minus_di[-1] <= plus_di[-1]:
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bearish signal rejected due to positive directional indicator for {self.symbol}",
                    action="_check_bearish_confirmations",
                    status="rejected",
                    details={"minus_di": minus_di[-1], "plus_di": plus_di[-1]}
                )
                return False

        # Check MACD confirmation if required
        if self.require_macd_confirmation:
            macd_line = self.get_indicator(timeframe, 'macd_line')
            macd_signal = self.get_indicator(timeframe, 'macd_signal')
            macd_histogram = self.get_indicator(timeframe, 'macd_histogram')

            if macd_line is not None and macd_signal is not None and macd_histogram is not None:
                # MACD should be negative or crossing down
                if macd_line[-1] >= macd_signal[-1] or macd_histogram[-1] >= 0:
                    self.log_strategy_event(
                        level="INFO",
                        message=f"Bearish signal rejected due to MACD not confirming for {self.symbol}",
                        action="_check_bearish_confirmations",
                        status="rejected",
                        details={
                            "macd_line": macd_line[-1],
                            "macd_signal": macd_signal[-1],
                            "macd_histogram": macd_histogram[-1]
                        }
                    )
                    return False

        return True