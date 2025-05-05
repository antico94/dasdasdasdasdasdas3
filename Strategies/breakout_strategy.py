# Strategies/breakout_strategy.py
from typing import Optional
import numpy as np
from Config.trading_config import TimeFrame
from Events.events import SignalEvent
from Strategies.base_strategy import BaseStrategy
from Strategies.indicator_utils import IndicatorUtils


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy for EUR/USD and other currency pairs.

    This strategy identifies and trades breakouts using Donchian channels,
    Bollinger Bands, and ATR for volatility filtering.
    """

    def __init__(self, name: str, symbol: str, timeframes: set,
                 donchian_period: int = 20,
                 bollinger_period: int = 20,
                 bollinger_deviation: float = 2.0,
                 atr_period: int = 14,
                 min_volatility_trigger: float = 1.2,
                 session_filter: str = None,
                 stop_loss_atr_multiplier: float = 2.0,
                 take_profit_atr_multiplier: float = 3.0,
                 adx_period: int = 14,
                 adx_threshold: float = 25.0,
                 logger=None):
        """
        Initialize the breakout strategy.

        Args:
            name: Strategy name
            symbol: Trading instrument symbol
            timeframes: Set of timeframes this strategy will use
            donchian_period: Period for Donchian channel calculation
            bollinger_period: Period for Bollinger Bands calculation
            bollinger_deviation: Deviation for Bollinger Bands
            atr_period: Period for ATR calculation
            min_volatility_trigger: Minimum volatility trigger as ATR multiplier
            session_filter: Session filter ('london', 'new_york', 'london_new_york', etc.)
            stop_loss_atr_multiplier: ATR multiplier for stop loss
            take_profit_atr_multiplier: ATR multiplier for take profit
            adx_period: Period for ADX calculation
            adx_threshold: Threshold for ADX trend confirmation
            logger: Logger instance
        """
        super().__init__(name, symbol, timeframes, logger)

        # Indicator parameters
        self.donchian_period = donchian_period
        self.bollinger_period = bollinger_period
        self.bollinger_deviation = bollinger_deviation
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

        # Strategy parameters
        self.min_volatility_trigger = min_volatility_trigger
        self.session_filter = session_filter

        # Risk management
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier

    def update_indicators(self, timeframe: TimeFrame):
        """
        Calculate and cache indicators for the breakout strategy.

        Args:
            timeframe: Timeframe to calculate indicators for
        """
        # Get bars for this timeframe
        bars = self.get_completed_bars(timeframe)

        # Ensure we have enough bars
        required_bars = max(self.donchian_period, self.bollinger_period, self.atr_period, self.adx_period) + 10
        if len(bars) < required_bars:
            return

        # Convert to numpy arrays
        arrays = self.get_bars_as_arrays(timeframe)

        # Calculate Donchian Channels
        upper, middle, lower = IndicatorUtils.donchian_channel(
            arrays['high'], arrays['low'], self.donchian_period
        )
        self.set_indicator(timeframe, 'donchian_upper', upper)
        self.set_indicator(timeframe, 'donchian_lower', lower)
        self.set_indicator(timeframe, 'donchian_middle', middle)

        # Calculate Bollinger Bands
        upper, middle, lower = IndicatorUtils.bollinger_bands(
            arrays['close'], self.bollinger_period, self.bollinger_deviation
        )
        self.set_indicator(timeframe, 'bollinger_upper', upper)
        self.set_indicator(timeframe, 'bollinger_ma', middle)  # Use bollinger_ma for consistency with test
        self.set_indicator(timeframe, 'bollinger_lower', lower)

        # Calculate ATR
        atr = IndicatorUtils.atr(
            arrays['high'], arrays['low'], arrays['close'], self.atr_period
        )
        self.set_indicator(timeframe, 'atr', atr)

        # Calculate ADX
        adx, plus_di, minus_di = IndicatorUtils.adx(
            arrays['high'], arrays['low'], arrays['close'], self.adx_period
        )
        self.set_indicator(timeframe, 'adx', adx)
        self.set_indicator(timeframe, 'plus_di', plus_di)
        self.set_indicator(timeframe, 'minus_di', minus_di)

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
        min_bars = max(self.donchian_period, self.bollinger_period, self.atr_period, self.adx_period) + 10
        if not self.validate_bars(timeframe, min_bars):
            return None

        # Check if in valid trading session
        if self.session_filter and not IndicatorUtils.is_valid_session(bars[-1].timestamp, self.session_filter):
            return None

        # Check for breakout signal
        signal_direction = self._check_breakout(timeframe)

        if signal_direction == 0:
            return None

        # Apply additional filters
        if not self._check_filters(timeframe, signal_direction):
            return None

        # Calculate entry, stop loss, and take profit levels
        entry_price, stop_loss, take_profit = self._calculate_levels(timeframe, signal_direction)

        # Create and return signal
        return self.create_signal(
            timeframe=timeframe,
            direction="BUY" if signal_direction > 0 else "SELL",
            reason=f"Breakout {'Bullish' if signal_direction > 0 else 'Bearish'}",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _check_breakout(self, timeframe: TimeFrame) -> int:
        """
        Check for a breakout condition.

        Args:
            timeframe: Timeframe to check

        Returns:
            1 for bullish breakout, -1 for bearish breakout, 0 for no breakout
        """
        # Get the latest bars
        bars = self.get_completed_bars(timeframe)
        if len(bars) < 2:
            return 0

        # Get the current and previous candle
        current_bar = bars[-1]
        previous_bar = bars[-2]

        # Check Donchian channel breakout
        donchian_upper = self.get_indicator(timeframe, 'donchian_upper')
        donchian_lower = self.get_indicator(timeframe, 'donchian_lower')

        if donchian_upper is not None and donchian_lower is not None:
            # Use values from two bars ago to avoid the current bar in the calculation
            upper = donchian_upper[-2]
            lower = donchian_lower[-2]

            # Bullish breakout: close above upper Donchian
            if previous_bar.close <= upper < current_bar.close:
                return 1

            # Bearish breakout: close below lower Donchian
            if previous_bar.close >= lower > current_bar.close:
                return -1

        # Check Bollinger Band breakout
        bollinger_upper = self.get_indicator(timeframe, 'bollinger_upper')
        bollinger_lower = self.get_indicator(timeframe, 'bollinger_lower')

        if bollinger_upper is not None and bollinger_lower is not None:
            upper = bollinger_upper[-2]
            lower = bollinger_lower[-2]

            # Bullish breakout: close above upper Bollinger Band
            if previous_bar.close <= upper < current_bar.close:
                return 1

            # Bearish breakout: close below lower Bollinger Band
            if previous_bar.close >= lower > current_bar.close:
                return -1

        return 0

    def _check_filters(self, timeframe: TimeFrame, direction: int) -> bool:
        """
        Apply additional filters to confirm the breakout.

        Args:
            timeframe: Timeframe to check
            direction: Breakout direction (1 for bullish, -1 for bearish)

        Returns:
            True if filters pass, False otherwise
        """
        # Check minimum volatility trigger
        atr = self.get_indicator(timeframe, 'atr')
        if atr is not None and self.min_volatility_trigger > 0:
            current_atr = atr[-1]
            avg_atr = np.mean(atr[-20:])  # Average of last 20 ATR values

            if current_atr < avg_atr * self.min_volatility_trigger:
                self.log_strategy_event(
                    level="INFO",
                    message=f"Breakout rejected due to low volatility for {self.symbol}",
                    action="check_filters",
                    status="rejected",
                    details={
                        "current_atr": current_atr,
                        "avg_atr": avg_atr,
                        "min_trigger": self.min_volatility_trigger
                    }
                )
                return False

        # Check ADX filter for trend strength
        adx = self.get_indicator(timeframe, 'adx')
        if adx is not None and self.adx_threshold > 0:
            current_adx = adx[-1]

            if current_adx < self.adx_threshold:
                self.log_strategy_event(
                    level="INFO",
                    message=f"Breakout rejected due to weak trend for {self.symbol}",
                    action="check_filters",
                    status="rejected",
                    details={
                        "current_adx": current_adx,
                        "threshold": self.adx_threshold
                    }
                )
                return False

            # Check directional movement for trend direction
            plus_di = self.get_indicator(timeframe, 'plus_di')
            minus_di = self.get_indicator(timeframe, 'minus_di')

            if plus_di is not None and minus_di is not None:
                # For bullish breakout, +DI should be above -DI
                if direction > 0 and plus_di[-1] <= minus_di[-1]:
                    self.log_strategy_event(
                        level="INFO",
                        message=f"Bullish breakout rejected due to bearish trend for {self.symbol}",
                        action="check_filters",
                        status="rejected",
                        details={
                            "plus_di": plus_di[-1],
                            "minus_di": minus_di[-1]
                        }
                    )
                    return False

                # For bearish breakout, -DI should be above +DI
                if direction < 0 and minus_di[-1] <= plus_di[-1]:
                    self.log_strategy_event(
                        level="INFO",
                        message=f"Bearish breakout rejected due to bullish trend for {self.symbol}",
                        action="check_filters",
                        status="rejected",
                        details={
                            "plus_di": plus_di[-1],
                            "minus_di": minus_di[-1]
                        }
                    )
                    return False

        return True

    def _calculate_levels(self, timeframe: TimeFrame, direction: int) -> tuple:
        """
        Calculate entry, stop loss, and take profit levels.

        Args:
            timeframe: Timeframe to use
            direction: Breakout direction (1 for bullish, -1 for bearish)

        Returns:
            Tuple of (entry_price, stop_loss, take_profit)
        """
        # Get the latest bar
        bars = self.get_completed_bars(timeframe)
        current_bar = bars[-1]

        # Use current close as entry price
        entry_price = current_bar.close

        # Use ATR for stop loss and take profit calculations
        atr = self.get_indicator(timeframe, 'atr')
        if atr is not None:
            current_atr = atr[-1]

            if direction > 0:  # Bullish
                stop_loss = entry_price - (current_atr * self.stop_loss_atr_multiplier)
                take_profit = entry_price + (current_atr * self.take_profit_atr_multiplier)
            else:  # Bearish
                stop_loss = entry_price + (current_atr * self.stop_loss_atr_multiplier)
                take_profit = entry_price - (current_atr * self.take_profit_atr_multiplier)
        else:
            # Fallback if ATR not available
            if direction > 0:  # Bullish
                stop_loss = current_bar.low
                take_profit = entry_price + (entry_price - stop_loss) * 2
            else:  # Bearish
                stop_loss = current_bar.high
                take_profit = entry_price - (stop_loss - entry_price) * 2

        return entry_price, stop_loss, take_profit