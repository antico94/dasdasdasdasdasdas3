# Strategies/breakout_strategy.py
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, time

from Events.events import SignalEvent
from Database.models import PriceBar
from Config.trading_config import TimeFrame
from Strategies.base_strategy import BaseStrategy


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
        if len(bars) < max(self.donchian_period, self.bollinger_period, self.atr_period, self.adx_period) + 10:
            return

        # Convert to numpy arrays
        arrays = self.get_bars_as_arrays(timeframe)

        # Calculate Donchian channel
        if len(arrays['high']) >= self.donchian_period:
            donchian_upper = np.zeros(len(arrays['high']))
            donchian_lower = np.zeros(len(arrays['high']))

            for i in range(self.donchian_period - 1, len(arrays['high'])):
                donchian_upper[i] = np.max(arrays['high'][i - self.donchian_period + 1:i + 1])
                donchian_lower[i] = np.min(arrays['low'][i - self.donchian_period + 1:i + 1])

            self.set_indicator(timeframe, 'donchian_upper', donchian_upper)
            self.set_indicator(timeframe, 'donchian_lower', donchian_lower)

        # Calculate Bollinger Bands
        if len(arrays['close']) >= self.bollinger_period:
            # Calculate rolling mean
            ma = np.zeros(len(arrays['close']))
            for i in range(self.bollinger_period - 1, len(arrays['close'])):
                ma[i] = np.mean(arrays['close'][i - self.bollinger_period + 1:i + 1])

            # Calculate rolling standard deviation
            std = np.zeros(len(arrays['close']))
            for i in range(self.bollinger_period - 1, len(arrays['close'])):
                std[i] = np.std(arrays['close'][i - self.bollinger_period + 1:i + 1], ddof=1)

            # Calculate bands
            upper = ma + (std * self.bollinger_deviation)
            lower = ma - (std * self.bollinger_deviation)

            self.set_indicator(timeframe, 'bollinger_ma', ma)
            self.set_indicator(timeframe, 'bollinger_upper', upper)
            self.set_indicator(timeframe, 'bollinger_lower', lower)

        # Calculate ATR
        if len(arrays['high']) >= self.atr_period:
            tr = np.zeros(len(arrays['high']))

            # First TR value is simply high - low
            tr[0] = arrays['high'][0] - arrays['low'][0]

            # Calculate TR for remaining periods
            for i in range(1, len(arrays['high'])):
                hl = arrays['high'][i] - arrays['low'][i]
                hc = abs(arrays['high'][i] - arrays['close'][i - 1])
                lc = abs(arrays['low'][i] - arrays['close'][i - 1])
                tr[i] = max(hl, hc, lc)

            # Calculate ATR
            atr = np.zeros(len(arrays['high']))
            atr[self.atr_period - 1] = np.mean(tr[:self.atr_period])

            # Calculate smoothed ATR
            for i in range(self.atr_period, len(arrays['high'])):
                atr[i] = (atr[i - 1] * (self.atr_period - 1) + tr[i]) / self.atr_period

            self.set_indicator(timeframe, 'atr', atr)

        # Calculate ADX (simplified version)
        if len(arrays['high']) >= self.adx_period * 2:
            # This is a simplified ADX calculation that reflects the general strength
            # For a complete ADX calculation, a more comprehensive implementation would be needed
            plus_dm = np.zeros(len(arrays['high']))
            minus_dm = np.zeros(len(arrays['high']))

            for i in range(1, len(arrays['high'])):
                up_move = arrays['high'][i] - arrays['high'][i - 1]
                down_move = arrays['low'][i - 1] - arrays['low'][i]

                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                else:
                    plus_dm[i] = 0

                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
                else:
                    minus_dm[i] = 0

            # Smooth DM values
            smoothed_plus_dm = np.zeros(len(arrays['high']))
            smoothed_minus_dm = np.zeros(len(arrays['high']))

            smoothed_plus_dm[self.adx_period - 1] = np.sum(plus_dm[:self.adx_period])
            smoothed_minus_dm[self.adx_period - 1] = np.sum(minus_dm[:self.adx_period])

            for i in range(self.adx_period, len(arrays['high'])):
                smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / self.adx_period) + plus_dm[i]
                smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / self.adx_period) + \
                                       minus_dm[i]

            # Calculate DI values
            tr_sum = np.zeros(len(arrays['high']))
            tr_sum[self.adx_period - 1] = np.sum(tr[:self.adx_period])

            for i in range(self.adx_period, len(arrays['high'])):
                tr_sum[i] = tr_sum[i - 1] - (tr_sum[i - 1] / self.adx_period) + tr[i]

            plus_di = np.zeros(len(arrays['high']))
            minus_di = np.zeros(len(arrays['high']))

            for i in range(self.adx_period - 1, len(arrays['high'])):
                if tr_sum[i] > 0:
                    plus_di[i] = 100 * smoothed_plus_dm[i] / tr_sum[i]
                    minus_di[i] = 100 * smoothed_minus_dm[i] / tr_sum[i]

            # Calculate DX
            dx = np.zeros(len(arrays['high']))

            for i in range(self.adx_period - 1, len(arrays['high'])):
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

            # Calculate ADX
            adx = np.zeros(len(arrays['high']))
            adx[2 * self.adx_period - 1] = np.mean(dx[self.adx_period - 1:2 * self.adx_period])

            for i in range(2 * self.adx_period, len(arrays['high'])):
                adx[i] = (adx[i - 1] * (self.adx_period - 1) + dx[i]) / self.adx_period

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
        if not self._is_valid_session(bars[-1].timestamp):
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

    def _is_valid_session(self, timestamp: datetime) -> bool:
        """
        Check if the current time is within the valid trading session.

        Args:
            timestamp: The timestamp to check

        Returns:
            True if within valid session, False otherwise
        """
        if not self.session_filter:
            return True

        # Extract hour for session filtering
        hour = timestamp.hour

        if self.session_filter == "london":
            # London session: 7:00-16:00 UTC
            return 7 <= hour < 16
        elif self.session_filter == "new_york":
            # New York session: 12:00-21:00 UTC
            return 12 <= hour < 21
        elif self.session_filter == "london_new_york":
            # London + New York overlap: 12:00-16:00 UTC
            return 12 <= hour < 16
        elif self.session_filter == "tokyo":
            # Tokyo session: 0:00-9:00 UTC
            return 0 <= hour < 9
        elif self.session_filter == "tokyo_ny_overlap":
            # Tokyo + New York overlap: 8:00-12:00 UTC
            return 8 <= hour < 12

        return True

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
            if previous_bar.close <= upper and current_bar.close > upper:
                return 1

            # Bearish breakout: close below lower Donchian
            if previous_bar.close >= lower and current_bar.close < lower:
                return -1

        # Check Bollinger Band breakout
        bollinger_upper = self.get_indicator(timeframe, 'bollinger_upper')
        bollinger_lower = self.get_indicator(timeframe, 'bollinger_lower')

        if bollinger_upper is not None and bollinger_lower is not None:
            upper = bollinger_upper[-2]
            lower = bollinger_lower[-2]

            # Bullish breakout: close above upper Bollinger Band
            if previous_bar.close <= upper and current_bar.close > upper:
                return 1

            # Bearish breakout: close below lower Bollinger Band
            if previous_bar.close >= lower and current_bar.close < lower:
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