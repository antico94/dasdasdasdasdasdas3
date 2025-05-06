# Adding to Strategies/scalping_strategy.py

from typing import Optional, Set

import numpy as np

from Config.trading_config import TimeFrame
from Database.models import PriceBar
from Events.events import SignalEvent
from Strategies.base_strategy import BaseStrategy
from Strategies.indicator_utils import IndicatorUtils, MAType


class ScalpingStrategy(BaseStrategy):
    """
    Fast MACD and RSI scalping strategy for short timeframes.

    This strategy looks for quick price movements:
    - MACD crossing signal line with fast settings
    - RSI confirming momentum direction
    - Price above/below moving average
    - Optional volume spike confirmation
    """

    def __init__(self, name: str, symbol: str, timeframes: Set[TimeFrame],
                 macd_fast_period: int = 3,
                 macd_slow_period: int = 10,
                 macd_signal_period: int = 16,
                 rsi_period: int = 5,
                 ma_period: int = 20,
                 ma_type: str = "EMA",
                 require_volume_confirmation: bool = True,
                 volume_threshold: float = 1.5,
                 ema_values: Optional[list] = None,
                 rapid_exit: bool = False,
                 max_spread_pips: float = 1.0,
                 alternative_macd_settings: Optional[dict] = None,
                 atr_period: int = 14,
                 stop_loss_atr_multiplier: float = 1.0,
                 take_profit_atr_multiplier: float = 1.5,
                 logger=None):
        """
        Initialize the Scalping strategy.

        Args:
            name: Strategy name
            symbol: Trading instrument symbol
            timeframes: Set of timeframes this strategy will use
            macd_fast_period: Fast period for MACD
            macd_slow_period: Slow period for MACD
            macd_signal_period: Signal period for MACD
            rsi_period: Period for RSI calculation
            ma_period: Period for moving average
            ma_type: Type of moving average (SMA, EMA, WMA, HULL, TEMA)
            require_volume_confirmation: Whether to require volume confirmation
            volume_threshold: Volume threshold multiplier
            ema_values: Optional list of EMA periods to use [short, medium, long]
            rapid_exit: Whether to use rapid exit strategy
            max_spread_pips: Maximum allowed spread in pips
            alternative_macd_settings: Optional alternative MACD settings
            atr_period: Period for ATR calculation
            stop_loss_atr_multiplier: ATR multiplier for stop loss
            take_profit_atr_multiplier: ATR multiplier for take profit
            logger: Logger instance
        """
        super().__init__(name, symbol, timeframes, logger)

        # MACD parameters - use alternative settings if provided
        if alternative_macd_settings:
            self.macd_fast_period = alternative_macd_settings.get('fast_period', macd_fast_period)
            self.macd_slow_period = alternative_macd_settings.get('slow_period', macd_slow_period)
            self.macd_signal_period = alternative_macd_settings.get('signal_period', macd_signal_period)
        else:
            self.macd_fast_period = macd_fast_period
            self.macd_slow_period = macd_slow_period
            self.macd_signal_period = macd_signal_period

        # RSI parameters
        self.rsi_period = rsi_period

        # MA parameters
        self.ma_period = ma_period
        self.ema_values = ema_values or [5, 20, 55]  # Default EMA values if none provided

        # Convert ma_type string to enum
        self.ma_type = next((t for t in MAType if t.name == ma_type), MAType.EMA)

        # Volume confirmation
        self.require_volume_confirmation = require_volume_confirmation
        self.volume_threshold = volume_threshold

        # Scalping specific parameters
        self.rapid_exit = rapid_exit
        self.max_spread_pips = max_spread_pips

        # Risk management
        self.atr_period = atr_period
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier

        # Track previous signals to avoid duplicates
        self.last_signal_type = {tf: None for tf in timeframes}

    def update_indicators(self, timeframe: TimeFrame):
        """
        Calculate and cache indicators for the Scalping strategy.

        Args:
            timeframe: The timeframe to update indicators for
        """
        # Get completed bars
        bars = self.get_completed_bars(timeframe)

        # Ensure we have enough bars
        required_bars = max(self.macd_slow_period + self.macd_signal_period,
                            self.rsi_period, self.ma_period, self.atr_period) + 10
        if len(bars) < required_bars:
            return

        # Convert to numpy arrays
        arrays = self.get_bars_as_arrays(timeframe)

        # Calculate MACD
        macd_line, macd_signal, macd_histogram = IndicatorUtils.macd(
            arrays['close'],
            self.macd_fast_period,
            self.macd_slow_period,
            self.macd_signal_period
        )
        self.set_indicator(timeframe, 'macd_line', macd_line)
        self.set_indicator(timeframe, 'macd_signal', macd_signal)
        self.set_indicator(timeframe, 'macd_histogram', macd_histogram)

        # Calculate RSI
        rsi = IndicatorUtils.rsi(arrays['close'], self.rsi_period)
        self.set_indicator(timeframe, 'rsi', rsi)

        # Calculate Moving Average
        ma = IndicatorUtils.moving_average(arrays['close'], self.ma_period, self.ma_type)
        self.set_indicator(timeframe, 'ma', ma)

        # Calculate EMAs if provided
        for i, period in enumerate(self.ema_values):
            ema = IndicatorUtils.moving_average(arrays['close'], period, MAType.EMA)
            self.set_indicator(timeframe, f'ema_{period}', ema)

        # Calculate ATR for stop loss and take profit
        atr = IndicatorUtils.atr(
            arrays['high'], arrays['low'], arrays['close'], self.atr_period
        )
        self.set_indicator(timeframe, 'atr', atr)

        # Calculate volume indicators if needed
        if self.require_volume_confirmation:
            # Make sure we're properly calculating the average volume
            avg_volume = IndicatorUtils.average_volume(arrays['volume'], 20)

            # Calculate volume spike - ensure we're using the correct parameters
            volume_spike = IndicatorUtils.volume_spike(
                arrays['volume'], 20, self.volume_threshold
            )

            # Store both indicators
            self.set_indicator(timeframe, 'avg_volume', avg_volume)
            self.set_indicator(timeframe, 'volume_spike', volume_spike)

            # Log to help with debugging
            self.log_strategy_event(
                level="DEBUG",
                message=f"Volume spike calculated for {self.symbol} on {timeframe.name}",
                action="update_indicators",
                status="calculated",
                details={
                    "volume_threshold": self.volume_threshold,
                    "current_volume": float(arrays['volume'][-1]) if len(arrays['volume']) > 0 else 0,
                    "avg_volume": float(avg_volume[-1]) if len(avg_volume) > 0 else 0,
                    "spike_detected": bool(volume_spike[-1]) if len(volume_spike) > 0 else False
                }
            )

        # Calculate MACD crossovers
        macd_crosses_above = np.zeros(len(macd_line), dtype=bool)
        macd_crosses_below = np.zeros(len(macd_line), dtype=bool)

        for i in range(1, len(macd_line)):
            # MACD crosses above signal
            if macd_line[i - 1] <= macd_signal[i - 1] and macd_line[i] > macd_signal[i]:
                macd_crosses_above[i] = True

            # MACD crosses below signal
            if macd_line[i - 1] >= macd_signal[i - 1] and macd_line[i] < macd_signal[i]:
                macd_crosses_below[i] = True

        self.set_indicator(timeframe, 'macd_crosses_above', macd_crosses_above)
        self.set_indicator(timeframe, 'macd_crosses_below', macd_crosses_below)

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
        min_bars = max(self.macd_slow_period + self.macd_signal_period,
                       self.rsi_period, self.ma_period, self.atr_period) + 10
        if not self.validate_bars(timeframe, min_bars):
            return None

        # Get the latest bar
        current_bar = bars[-1]

        # Check if current spread exceeds maximum allowed spread
        if hasattr(current_bar, 'spread') and current_bar.spread > self.max_spread_pips:
            self.log_strategy_event(
                level="INFO",
                message=f"Spread too high for {self.symbol} on {timeframe.name}: {current_bar.spread} > {self.max_spread_pips}",
                action="calculate_signals",
                status="high_spread",
                details={"spread": current_bar.spread, "max_spread": self.max_spread_pips}
            )
            return None

        # Get indicators
        macd_line = self.get_indicator(timeframe, 'macd_line')
        macd_signal = self.get_indicator(timeframe, 'macd_signal')
        macd_histogram = self.get_indicator(timeframe, 'macd_histogram')
        macd_crosses_above = self.get_indicator(timeframe, 'macd_crosses_above')
        macd_crosses_below = self.get_indicator(timeframe, 'macd_crosses_below')
        rsi = self.get_indicator(timeframe, 'rsi')
        ma = self.get_indicator(timeframe, 'ma')
        atr = self.get_indicator(timeframe, 'atr')

        # Get EMAs if configured
        ema_indicators = {}
        for period in self.ema_values:
            ema_indicators[period] = self.get_indicator(timeframe, f'ema_{period}')

        # Get volume indicators if required
        volume_spike = None
        if self.require_volume_confirmation:
            volume_spike = self.get_indicator(timeframe, 'volume_spike')

            # Add debug logging for volume indicators
            avg_volume = self.get_indicator(timeframe, 'avg_volume')
            self.log_strategy_event(
                level="DEBUG",
                message=f"Volume indicator status for {self.symbol} on {timeframe.name}",
                action="calculate_signals",
                status="checking",
                details={
                    "volume_spike_available": volume_spike is not None,
                    "avg_volume_available": avg_volume is not None,
                    "require_volume_confirmation": self.require_volume_confirmation,
                    "volume_threshold": self.volume_threshold
                }
            )

        # Ensure we have all necessary indicators
        if (macd_line is None or macd_signal is None or macd_histogram is None or
                macd_crosses_above is None or macd_crosses_below is None or
                rsi is None or ma is None or atr is None or
                (self.require_volume_confirmation and volume_spike is None) or
                any(v is None for v in ema_indicators.values())):
            self.log_strategy_event(
                level="WARNING",
                message=f"Missing indicators for {self.symbol} on {timeframe.name}",
                action="calculate_signals",
                status="missing_indicators",
                details={
                    "macd_line": macd_line is not None,
                    "macd_signal": macd_signal is not None,
                    "macd_histogram": macd_histogram is not None,
                    "macd_crosses_above": macd_crosses_above is not None,
                    "macd_crosses_below": macd_crosses_below is not None,
                    "rsi": rsi is not None,
                    "ma": ma is not None,
                    "atr": atr is not None,
                    "volume_spike": volume_spike is not None if self.require_volume_confirmation else "Not Required",
                    "ema_indicators": {k: v is not None for k, v in ema_indicators.items()}
                }
            )
            return None

        # Get short, medium, and long EMAs
        short_ema = ema_indicators[self.ema_values[0]]
        medium_ema = ema_indicators[self.ema_values[1]]
        long_ema = None if len(self.ema_values) < 3 else ema_indicators[self.ema_values[2]]

        # Calculate conditions
        fresh_bullish_cross = bool(macd_crosses_above[-1])
        fresh_bearish_cross = bool(macd_crosses_below[-1])

        macd_above_signal = macd_line[-1] > macd_signal[-1]
        macd_below_signal = macd_line[-1] < macd_signal[-1]

        positive_histogram = macd_histogram[-1] > 0
        negative_histogram = macd_histogram[-1] < 0

        price_above_ma = current_bar.close > ma[-1]
        price_below_ma = current_bar.close < ma[-1]

        # Properly check volume spike
        if self.require_volume_confirmation and volume_spike is not None and len(volume_spike) > 0:
            volume_confirmed = bool(volume_spike[-1])
            volume_status = f"Volume spike detected: {volume_confirmed}"
        else:
            volume_confirmed = not self.require_volume_confirmation
            volume_status = "Not required" if not self.require_volume_confirmation else "Volume spike data missing"

        # EMA alignment
        ema_bullish = short_ema[-1] > medium_ema[-1]
        ema_bearish = short_ema[-1] < medium_ema[-1]

        above_long_ema = True if long_ema is None else current_bar.close > long_ema[-1]
        below_long_ema = True if long_ema is None else current_bar.close < long_ema[-1]

        # Prepare condition groups for display
        condition_groups = {
            "Bullish Scalping Conditions": [
                ("MACD crossed above Signal", fresh_bullish_cross,
                 f"Cross detected at index [-1]"),

                ("MACD above Signal", macd_above_signal,
                 f"MACD: {macd_line[-1]:.5f}, Signal: {macd_signal[-1]:.5f}"),

                ("Positive MACD Histogram", positive_histogram,
                 f"Histogram: {macd_histogram[-1]:.5f}"),

                ("Short EMA > Medium EMA", ema_bullish,
                 f"EMA{self.ema_values[0]}: {short_ema[-1]:.5f}, EMA{self.ema_values[1]}: {medium_ema[-1]:.5f}"),

                ("Price above Long EMA", above_long_ema,
                 f"Price: {current_bar.close:.5f}, EMA{self.ema_values[2] if len(self.ema_values) > 2 else 'N/A'}: "
                 f"{long_ema[-1] if long_ema is not None else 'N/A'}"),

                ("Volume Confirmation", volume_confirmed,
                 volume_status),

                ("Spread Below Max",
                 current_bar.spread <= self.max_spread_pips if hasattr(current_bar, 'spread') else True,
                 f"Spread: {current_bar.spread if hasattr(current_bar, 'spread') else 'N/A'}, Max: {self.max_spread_pips}")
            ],

            "Bearish Scalping Conditions": [
                ("MACD crossed below Signal", fresh_bearish_cross,
                 f"Cross detected at index [-1]"),

                ("MACD below Signal", macd_below_signal,
                 f"MACD: {macd_line[-1]:.5f}, Signal: {macd_signal[-1]:.5f}"),

                ("Negative MACD Histogram", negative_histogram,
                 f"Histogram: {macd_histogram[-1]:.5f}"),

                ("Short EMA < Medium EMA", ema_bearish,
                 f"EMA{self.ema_values[0]}: {short_ema[-1]:.5f}, EMA{self.ema_values[1]}: {medium_ema[-1]:.5f}"),

                ("Price below Long EMA", below_long_ema,
                 f"Price: {current_bar.close:.5f}, EMA{self.ema_values[2] if len(self.ema_values) > 2 else 'N/A'}: "
                 f"{long_ema[-1] if long_ema is not None else 'N/A'}"),

                ("Volume Confirmation", volume_confirmed,
                 volume_status),

                ("Spread Below Max",
                 current_bar.spread <= self.max_spread_pips if hasattr(current_bar, 'spread') else True,
                 f"Spread: {current_bar.spread if hasattr(current_bar, 'spread') else 'N/A'}, Max: {self.max_spread_pips}")
            ]
        }

        # Print the conditions
        self.print_strategy_conditions(timeframe, condition_groups)

        # Check for bullish scalping signal
        if (fresh_bullish_cross and ema_bullish and
                above_long_ema and volume_confirmed and
                self.last_signal_type[timeframe] != "BUY"):

            # Calculate entry, stop loss, and take profit
            entry_price = current_bar.close
            stop_loss = entry_price - (atr[-1] * self.stop_loss_atr_multiplier)
            take_profit = entry_price + (atr[-1] * self.take_profit_atr_multiplier)

            # Log the signal
            self.log_strategy_event(
                level="INFO",
                message=f"Bullish scalping signal detected for {self.symbol} on {timeframe.name}",
                action="calculate_signals",
                status="signal_detected",
                details={
                    "macd_line": macd_line[-1],
                    "macd_signal": macd_signal[-1],
                    "macd_histogram": macd_histogram[-1],
                    "price": current_bar.close,
                    "short_ema": short_ema[-1],
                    "medium_ema": medium_ema[-1],
                    "long_ema": long_ema[-1] if long_ema is not None else None,
                    "volume_confirmed": volume_confirmed
                }
            )

            # Update last signal type
            self.last_signal_type[timeframe] = "BUY"

            # Create and return signal
            return self.create_signal(
                timeframe=timeframe,
                direction="BUY",
                reason="Scalping Bullish Signal",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        # Check for bearish scalping signal
        elif (fresh_bearish_cross and ema_bearish and
              below_long_ema and volume_confirmed and
              self.last_signal_type[timeframe] != "SELL"):

            # Calculate entry, stop loss, and take profit
            entry_price = current_bar.close
            stop_loss = entry_price + (atr[-1] * self.stop_loss_atr_multiplier)
            take_profit = entry_price - (atr[-1] * self.take_profit_atr_multiplier)

            # Log the signal
            self.log_strategy_event(
                level="INFO",
                message=f"Bearish scalping signal detected for {self.symbol} on {timeframe.name}",
                action="calculate_signals",
                status="signal_detected",
                details={
                    "macd_line": macd_line[-1],
                    "macd_signal": macd_signal[-1],
                    "macd_histogram": macd_histogram[-1],
                    "price": current_bar.close,
                    "short_ema": short_ema[-1],
                    "medium_ema": medium_ema[-1],
                    "long_ema": long_ema[-1] if long_ema is not None else None,
                    "volume_confirmed": volume_confirmed
                }
            )

            # Update last signal type
            self.last_signal_type[timeframe] = "SELL"

            # Create and return signal
            return self.create_signal(
                timeframe=timeframe,
                direction="SELL",
                reason="Scalping Bearish Signal",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        # Check for rapid exit if enabled
        elif self.rapid_exit and self.last_signal_type[timeframe] == "BUY":
            if macd_crosses_below[-1] or short_ema[-1] < medium_ema[-1]:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Rapid Exit Signal (MACD crossed below or EMA cross)"
                )

        elif self.rapid_exit and self.last_signal_type[timeframe] == "SELL":
            if macd_crosses_above[-1] or short_ema[-1] > medium_ema[-1]:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Rapid Exit Signal (MACD crossed above or EMA cross)"
                )

        # Check for standard exit signals
        elif not self.rapid_exit and self.last_signal_type[timeframe] == "BUY":
            # Exit long position when MACD crosses below signal line
            if macd_crosses_below[-1]:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Scalping Exit Signal (MACD crossed below signal line)"
                )

        elif not self.rapid_exit and self.last_signal_type[timeframe] == "SELL":
            # Exit short position when MACD crosses above signal line
            if macd_crosses_above[-1]:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Scalping Exit Signal (MACD crossed above signal line)"
                )

        return None