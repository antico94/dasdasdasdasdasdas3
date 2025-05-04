# Strategies/momentum_strategy.py
from typing import Dict, Any, List, Optional, Set

import numpy as np

from Config.trading_config import TimeFrame
from Database.models import PriceBar
from Events.events import SignalEvent
from Strategies.base_strategy import BaseStrategy
from Strategies.indicator_utils import IndicatorUtils, MAType


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using MACD, RSI, and price action.

    This strategy looks for strong momentum:
    - MACD crossing signal line
    - RSI confirming momentum direction
    - Price above/below moving average
    - Optional volume confirmation
    """

    def __init__(self, name: str, symbol: str, timeframes: Set[TimeFrame],
                 macd_fast_period: int = 12,
                 macd_slow_period: int = 26,
                 macd_signal_period: int = 9,
                 rsi_period: int = 14,
                 rsi_threshold_low: float = 40.0,
                 rsi_threshold_high: float = 60.0,
                 ma_period: int = 50,
                 ma_type: str = "EMA",
                 require_volume_confirmation: bool = True,
                 volume_threshold: float = 1.5,
                 atr_period: int = 14,
                 stop_loss_atr_multiplier: float = 2.0,
                 take_profit_atr_multiplier: float = 3.0,
                 logger=None):
        """
        Initialize the Momentum strategy.

        Args:
            name: Strategy name
            symbol: Trading instrument symbol
            timeframes: Set of timeframes this strategy will use
            macd_fast_period: Fast period for MACD
            macd_slow_period: Slow period for MACD
            macd_signal_period: Signal period for MACD
            rsi_period: Period for RSI calculation
            rsi_threshold_low: RSI threshold for downward momentum
            rsi_threshold_high: RSI threshold for upward momentum
            ma_period: Period for moving average
            ma_type: Type of moving average (SMA, EMA, WMA, HULL, TEMA)
            require_volume_confirmation: Whether to require volume confirmation
            volume_threshold: Volume threshold multiplier
            atr_period: Period for ATR calculation
            stop_loss_atr_multiplier: ATR multiplier for stop loss
            take_profit_atr_multiplier: ATR multiplier for take profit
            logger: Logger instance
        """
        super().__init__(name, symbol, timeframes, logger)

        # MACD parameters
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period

        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_threshold_low = rsi_threshold_low
        self.rsi_threshold_high = rsi_threshold_high

        # MA parameters
        self.ma_period = ma_period
        # Convert ma_type string to enum
        self.ma_type = next((t for t in MAType if t.name == ma_type), MAType.EMA)

        # Volume confirmation
        self.require_volume_confirmation = require_volume_confirmation
        self.volume_threshold = volume_threshold

        # Risk management
        self.atr_period = atr_period
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier

        # Track previous signals to avoid duplicates
        self.last_signal_type = {tf: None for tf in timeframes}

    def update_indicators(self, timeframe: TimeFrame):
        """
        Calculate and cache indicators for the Momentum strategy.

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

        # Calculate ATR for stop loss and take profit
        atr = IndicatorUtils.atr(
            arrays['high'], arrays['low'], arrays['close'], self.atr_period
        )
        self.set_indicator(timeframe, 'atr', atr)

        # Calculate volume indicators if needed
        if self.require_volume_confirmation:
            avg_volume = IndicatorUtils.average_volume(arrays['volume'], 20)
            volume_spike = IndicatorUtils.volume_spike(
                arrays['volume'], 20, self.volume_threshold
            )
            self.set_indicator(timeframe, 'avg_volume', avg_volume)
            self.set_indicator(timeframe, 'volume_spike', volume_spike)

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

        # Get indicators
        macd_line = self.get_indicator(timeframe, 'macd_line')
        macd_signal = self.get_indicator(timeframe, 'macd_signal')
        macd_histogram = self.get_indicator(timeframe, 'macd_histogram')
        macd_crosses_above = self.get_indicator(timeframe, 'macd_crosses_above')
        macd_crosses_below = self.get_indicator(timeframe, 'macd_crosses_below')
        rsi = self.get_indicator(timeframe, 'rsi')
        ma = self.get_indicator(timeframe, 'ma')
        atr = self.get_indicator(timeframe, 'atr')

        # Get volume indicators if required
        volume_spike = None
        if self.require_volume_confirmation:
            volume_spike = self.get_indicator(timeframe, 'volume_spike')

        # Ensure we have all necessary indicators
        if (macd_line is None or macd_signal is None or macd_histogram is None or
                macd_crosses_above is None or macd_crosses_below is None or
                rsi is None or ma is None or atr is None or
                (self.require_volume_confirmation and volume_spike is None)):
            return None

        # Check for bullish momentum signal
        if macd_crosses_above[-1] and self.last_signal_type[timeframe] != "BUY":
            # Check additional confirmations
            if self._check_bullish_confirmations(timeframe, current_bar):
                # Calculate entry, stop loss, and take profit
                entry_price = current_bar.close
                stop_loss = entry_price - (atr[-1] * self.stop_loss_atr_multiplier)
                take_profit = entry_price + (atr[-1] * self.take_profit_atr_multiplier)

                # Log the signal
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bullish momentum signal detected for {self.symbol} on {timeframe.name}",
                    action="calculate_signals",
                    status="signal_detected",
                    details={
                        "macd_line": macd_line[-1],
                        "macd_signal": macd_signal[-1],
                        "macd_histogram": macd_histogram[-1],
                        "rsi": rsi[-1],
                        "price": current_bar.close,
                        "ma": ma[-1]
                    }
                )

                # Update last signal type
                self.last_signal_type[timeframe] = "BUY"

                # Create and return signal
                return self.create_signal(
                    timeframe=timeframe,
                    direction="BUY",
                    reason="Momentum Bullish Signal",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

        # Check for bearish momentum signal
        elif macd_crosses_below[-1] and self.last_signal_type[timeframe] != "SELL":
            # Check additional confirmations
            if self._check_bearish_confirmations(timeframe, current_bar):
                # Calculate entry, stop loss, and take profit
                entry_price = current_bar.close
                stop_loss = entry_price + (atr[-1] * self.stop_loss_atr_multiplier)
                take_profit = entry_price - (atr[-1] * self.take_profit_atr_multiplier)

                # Log the signal
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bearish momentum signal detected for {self.symbol} on {timeframe.name}",
                    action="calculate_signals",
                    status="signal_detected",
                    details={
                        "macd_line": macd_line[-1],
                        "macd_signal": macd_signal[-1],
                        "macd_histogram": macd_histogram[-1],
                        "rsi": rsi[-1],
                        "price": current_bar.close,
                        "ma": ma[-1]
                    }
                )

                # Update last signal type
                self.last_signal_type[timeframe] = "SELL"

                # Create and return signal
                return self.create_signal(
                    timeframe=timeframe,
                    direction="SELL",
                    reason="Momentum Bearish Signal",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

        # Check for exit signals
        elif self.last_signal_type[timeframe] == "BUY":
            # Exit long position when MACD crosses below signal line
            if macd_crosses_below[-1]:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Momentum Exit Signal (MACD crossed below signal line)"
                )

        elif self.last_signal_type[timeframe] == "SELL":
            # Exit short position when MACD crosses above signal line
            if macd_crosses_above[-1]:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Momentum Exit Signal (MACD crossed above signal line)"
                )

        return None

    def _check_bullish_confirmations(self, timeframe: TimeFrame, current_bar: PriceBar) -> bool:
        """
        Check additional bullish confirmations.

        Args:
            timeframe: The timeframe to check
            current_bar: The current price bar

        Returns:
            True if confirmations pass, False otherwise
        """
        # Get indicators
        rsi = self.get_indicator(timeframe, 'rsi')
        ma = self.get_indicator(timeframe, 'ma')
        volume_spike = self.get_indicator(timeframe, 'volume_spike') if self.require_volume_confirmation else None

        # RSI should be above threshold for bullish momentum
        if rsi is not None and rsi[-1] < self.rsi_threshold_high:
            self.log_strategy_event(
                level="INFO",
                message=f"Bullish signal rejected due to low RSI for {self.symbol}",
                action="_check_bullish_confirmations",
                status="rejected",
                details={"rsi": rsi[-1], "threshold": self.rsi_threshold_high}
            )
            return False

        # Price should be above moving average
        if ma is not None and current_bar.close <= ma[-1]:
            self.log_strategy_event(
                level="INFO",
                message=f"Bullish signal rejected due to price below MA for {self.symbol}",
                action="_check_bullish_confirmations",
                status="rejected",
                details={"price": current_bar.close, "ma": ma[-1]}
            )
            return False

        # Check volume confirmation if required
        if self.require_volume_confirmation and volume_spike is not None:
            if not volume_spike[-1]:
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bullish signal rejected due to insufficient volume for {self.symbol}",
                    action="_check_bullish_confirmations",
                    status="rejected",
                    details={"volume": current_bar.volume}
                )
                return False

        return True

    def _check_bearish_confirmations(self, timeframe: TimeFrame, current_bar: PriceBar) -> bool:
        """
        Check additional bearish confirmations.

        Args:
            timeframe: The timeframe to check
            current_bar: The current price bar

        Returns:
            True if confirmations pass, False otherwise
        """
        # Get indicators
        rsi = self.get_indicator(timeframe, 'rsi')
        ma = self.get_indicator(timeframe, 'ma')
        volume_spike = self.get_indicator(timeframe, 'volume_spike') if self.require_volume_confirmation else None

        # RSI should be below threshold for bearish momentum
        if rsi is not None and rsi[-1] > self.rsi_threshold_low:
            self.log_strategy_event(
                level="INFO",
                message=f"Bearish signal rejected due to high RSI for {self.symbol}",
                action="_check_bearish_confirmations",
                status="rejected",
                details={"rsi": rsi[-1], "threshold": self.rsi_threshold_low}
            )
            return False

        # Price should be below moving average
        if ma is not None and current_bar.close >= ma[-1]:
            self.log_strategy_event(
                level="INFO",
                message=f"Bearish signal rejected due to price above MA for {self.symbol}",
                action="_check_bearish_confirmations",
                status="rejected",
                details={"price": current_bar.close, "ma": ma[-1]}
            )
            return False

        # Check volume confirmation if required
        if self.require_volume_confirmation and volume_spike is not None:
            if not volume_spike[-1]:
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bearish signal rejected due to insufficient volume for {self.symbol}",
                    action="_check_bearish_confirmations",
                    status="rejected",
                    details={"volume": current_bar.volume}
                )
                return False

        return True