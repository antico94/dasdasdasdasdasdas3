# Adding to Strategies/hybrid_strategy.py

from typing import Optional, Set, Dict, Any
from datetime import datetime, timezone

import numpy as np

from Config.trading_config import TimeFrame
from Database.models import PriceBar
from Events.events import SignalEvent
from Strategies.base_strategy import BaseStrategy
from Strategies.indicator_utils import IndicatorUtils, MAType


class HybridStrategy(BaseStrategy):
    """
    Hybrid strategy combining technical and fundamental factors.

    This strategy integrates:
    - Technical indicators (MACD, MA, etc.)
    - Fundamental bias based on yield differentials, central bank stance, etc.
    - Optional filters for macro conditions
    """

    def __init__(self, name: str, symbol: str, timeframes: Set[TimeFrame],
                 ma_period: int = 100,
                 ma_type: str = "EMA",
                 yield_differential_threshold: float = 0.5,
                 boj_dovish_bias: bool = True,
                 us_bond_yield_rising: bool = True,
                 real_yield_filter: bool = False,
                 max_real_yield_long: float = 0.0,
                 min_real_yield_short: float = 0.5,
                 atr_period: int = 14,
                 stop_loss_atr_multiplier: float = 1.5,
                 logger=None):
        """
        Initialize the Hybrid strategy.

        Args:
            name: Strategy name
            symbol: Trading instrument symbol
            timeframes: Set of timeframes this strategy will use
            ma_period: Period for moving average
            ma_type: Type of moving average (SMA, EMA, WMA, HULL, TEMA)
            yield_differential_threshold: Minimum yield differential in %
            boj_dovish_bias: Flag indicating if BOJ has a dovish bias
            us_bond_yield_rising: Flag indicating if US bond yields are rising
            real_yield_filter: Whether to use real yield filter
            max_real_yield_long: Maximum real yield for long positions
            min_real_yield_short: Minimum real yield for short positions
            atr_period: Period for ATR calculation
            stop_loss_atr_multiplier: ATR multiplier for stop loss
            logger: Logger instance
        """
        super().__init__(name, symbol, timeframes, logger)

        # Moving average parameters
        self.ma_period = ma_period
        # Convert ma_type string to enum
        self.ma_type = next((t for t in MAType if t.name == ma_type), MAType.EMA)

        # Fundamental parameters
        self.yield_differential_threshold = yield_differential_threshold
        self.boj_dovish_bias = boj_dovish_bias
        self.us_bond_yield_rising = us_bond_yield_rising

        # Real yield filter parameters
        self.real_yield_filter = real_yield_filter
        self.max_real_yield_long = max_real_yield_long
        self.min_real_yield_short = min_real_yield_short

        # Risk management
        self.atr_period = atr_period
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier

        # Mock fundamental data - in a real system this would come from external data
        # These would be updated periodically with real data
        self._mock_fundamental_data = {
            "yield_differential": self.yield_differential_threshold + 0.1,  # Mock > threshold
            "boj_stance": "dovish" if self.boj_dovish_bias else "hawkish",
            "us_bond_yield_trend": "rising" if self.us_bond_yield_rising else "falling",
            "real_yield": -0.2 if self.symbol in ["XAUUSD", "EURUSD"] else 0.7  # Mock data
        }
        self._last_fundamental_update = datetime.now()

        # Track previous signals to avoid duplicates
        self.last_signal_type = {tf: None for tf in timeframes}

    def update_indicators(self, timeframe: TimeFrame):
        """
        Calculate and cache indicators for the Hybrid strategy.

        Args:
            timeframe: The timeframe to update indicators for
        """
        # Get completed bars
        bars = self.get_completed_bars(timeframe)

        # Ensure we have enough bars
        required_bars = max(self.ma_period, self.atr_period) + 10
        if len(bars) < required_bars:
            return

        # Convert to numpy arrays
        arrays = self.get_bars_as_arrays(timeframe)

        # Calculate Moving Average
        ma = IndicatorUtils.moving_average(arrays['close'], self.ma_period, self.ma_type)
        self.set_indicator(timeframe, 'ma', ma)

        # Calculate ATR for stop loss and take profit
        atr = IndicatorUtils.atr(
            arrays['high'], arrays['low'], arrays['close'], self.atr_period
        )
        self.set_indicator(timeframe, 'atr', atr)

        # Calculate Momentum (for trend direction)
        momentum = IndicatorUtils.momentum(arrays['close'], 10)
        self.set_indicator(timeframe, 'momentum', momentum)

        # Check if we need to update mock fundamental data (in real system would call API)
        current_time = datetime.now()
        if (current_time - self._last_fundamental_update).total_seconds() > 3600:  # Hourly update
            self._update_mock_fundamental_data()

    def _update_mock_fundamental_data(self):
        """Update mock fundamental data - in a real system would fetch from API"""
        # This would be replaced with real API calls to get latest yield data, central bank stance, etc.
        # For now, just mock some data with small changes
        self._mock_fundamental_data["yield_differential"] = self.yield_differential_threshold + np.random.normal(0, 0.1)
        self._mock_fundamental_data["real_yield"] = -0.2 if self.symbol in ["XAUUSD", "EURUSD"] else 0.7
        self._last_fundamental_update = datetime.now()

        self.log_strategy_event(
            level="INFO",
            message=f"Updated fundamental data for {self.symbol}",
            action="_update_mock_fundamental_data",
            status="updated",
            details=self._mock_fundamental_data
        )

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
        min_bars = max(self.ma_period, self.atr_period) + 10
        if not self.validate_bars(timeframe, min_bars):
            return None

        # Get the latest bar
        current_bar = bars[-1]

        # Get indicators
        ma = self.get_indicator(timeframe, 'ma')
        atr = self.get_indicator(timeframe, 'atr')
        momentum = self.get_indicator(timeframe, 'momentum')

        # Ensure we have all necessary indicators
        if ma is None or atr is None or momentum is None:
            return None

        # Calculate technical conditions
        price_above_ma = current_bar.close > ma[-1]
        price_below_ma = current_bar.close < ma[-1]

        positive_momentum = momentum[-1] > 0
        negative_momentum = momentum[-1] < 0

        # Calculate fundamental conditions
        yield_diff_bullish = self._mock_fundamental_data["yield_differential"] >= self.yield_differential_threshold
        real_yield_bullish = (not self.real_yield_filter) or (
                self._mock_fundamental_data["real_yield"] <= self.max_real_yield_long
        )
        real_yield_bearish = (not self.real_yield_filter) or (
                self._mock_fundamental_data["real_yield"] >= self.min_real_yield_short
        )

        # Currency-specific conditions
        has_boj_dovish_bias = self.boj_dovish_bias == (self._mock_fundamental_data["boj_stance"] == "dovish")
        has_us_yield_rising = self.us_bond_yield_rising == (
                    self._mock_fundamental_data["us_bond_yield_trend"] == "rising")

        # Combine technical and fundamental conditions
        bullish_tech = price_above_ma and positive_momentum
        bearish_tech = price_below_ma and negative_momentum

        bullish_fund = yield_diff_bullish and real_yield_bullish
        bearish_fund = not yield_diff_bullish and real_yield_bearish

        # USDJPY specific carry trade conditions
        if self.symbol == "USDJPY":
            bullish_carry = has_boj_dovish_bias and has_us_yield_rising
            bearish_carry = not has_boj_dovish_bias and not has_us_yield_rising
        else:
            bullish_carry = True  # Not relevant for other pairs
            bearish_carry = True  # Not relevant for other pairs

        # Prepare condition groups for display
        condition_groups = {
            "Bullish Hybrid Conditions": [
                ("Technical: Price above MA", price_above_ma,
                 f"Price: {current_bar.close:.5f}, MA: {ma[-1]:.5f}"),

                ("Technical: Positive Momentum", positive_momentum,
                 f"Momentum: {momentum[-1]:.5f}"),

                ("Fundamental: Yield Differential", yield_diff_bullish,
                 f"Yield Diff: {self._mock_fundamental_data['yield_differential']:.2f} >= {self.yield_differential_threshold}"),

                ("Fundamental: Real Yield Condition", real_yield_bullish,
                 f"Real Yield: {self._mock_fundamental_data['real_yield']:.2f} <= {self.max_real_yield_long}"),

                ("Currency Specific: BOJ Dovish", has_boj_dovish_bias if self.symbol == "USDJPY" else True,
                 f"BOJ Stance: {self._mock_fundamental_data['boj_stance']}" if self.symbol == "USDJPY" else "N/A"),

                ("Currency Specific: US Yield Rising", has_us_yield_rising if self.symbol == "USDJPY" else True,
                 f"US Yield Trend: {self._mock_fundamental_data['us_bond_yield_trend']}" if self.symbol == "USDJPY" else "N/A")
            ],

            "Bearish Hybrid Conditions": [
                ("Technical: Price below MA", price_below_ma,
                 f"Price: {current_bar.close:.5f}, MA: {ma[-1]:.5f}"),

                ("Technical: Negative Momentum", negative_momentum,
                 f"Momentum: {momentum[-1]:.5f}"),

                ("Fundamental: Yield Differential", not yield_diff_bullish,
                 f"Yield Diff: {self._mock_fundamental_data['yield_differential']:.2f} < {self.yield_differential_threshold}"),

                ("Fundamental: Real Yield Condition", real_yield_bearish,
                 f"Real Yield: {self._mock_fundamental_data['real_yield']:.2f} >= {self.min_real_yield_short}"),

                ("Currency Specific: BOJ Hawkish", not has_boj_dovish_bias if self.symbol == "USDJPY" else True,
                 f"BOJ Stance: {self._mock_fundamental_data['boj_stance']}" if self.symbol == "USDJPY" else "N/A"),

                ("Currency Specific: US Yield Falling", not has_us_yield_rising if self.symbol == "USDJPY" else True,
                 f"US Yield Trend: {self._mock_fundamental_data['us_bond_yield_trend']}" if self.symbol == "USDJPY" else "N/A")
            ]
        }
        # Print the conditions
        self.print_strategy_conditions(timeframe, condition_groups)

        # Check for bullish hybrid signal
        if (bullish_tech and bullish_fund and bullish_carry and
                self.last_signal_type[timeframe] != "BUY"):

            # Calculate entry, stop loss, and take profit
            entry_price = current_bar.close
            stop_loss = entry_price - (atr[-1] * self.stop_loss_atr_multiplier)
            take_profit = entry_price + (atr[-1] * self.stop_loss_atr_multiplier * 2)  # 1:2 risk-reward

            # Log the signal
            self.log_strategy_event(
                level="INFO",
                message=f"Bullish hybrid signal detected for {self.symbol} on {timeframe.name}",
                action="calculate_signals",
                status="signal_detected",
                details={
                    "price": current_bar.close,
                    "ma": ma[-1],
                    "momentum": momentum[-1],
                    "yield_differential": self._mock_fundamental_data["yield_differential"],
                    "real_yield": self._mock_fundamental_data["real_yield"]
                }
            )

            # Update last signal type
            self.last_signal_type[timeframe] = "BUY"

            # Create and return signal
            return self.create_signal(
                timeframe=timeframe,
                direction="BUY",
                reason="Hybrid Bullish Signal",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        # Check for bearish hybrid signal
        elif (bearish_tech and bearish_fund and bearish_carry and
              self.last_signal_type[timeframe] != "SELL"):

            # Calculate entry, stop loss, and take profit
            entry_price = current_bar.close
            stop_loss = entry_price + (atr[-1] * self.stop_loss_atr_multiplier)
            take_profit = entry_price - (atr[-1] * self.stop_loss_atr_multiplier * 2)  # 1:2 risk-reward

            # Log the signal
            self.log_strategy_event(
                level="INFO",
                message=f"Bearish hybrid signal detected for {self.symbol} on {timeframe.name}",
                action="calculate_signals",
                status="signal_detected",
                details={
                    "price": current_bar.close,
                    "ma": ma[-1],
                    "momentum": momentum[-1],
                    "yield_differential": self._mock_fundamental_data["yield_differential"],
                    "real_yield": self._mock_fundamental_data["real_yield"]
                }
            )

            # Update last signal type
            self.last_signal_type[timeframe] = "SELL"

            # Create and return signal
            return self.create_signal(
                timeframe=timeframe,
                direction="SELL",
                reason="Hybrid Bearish Signal",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        # Check for exit signals
        elif self.last_signal_type[timeframe] == "BUY":
            # Exit long position when price falls below MA or fundamental condition changes
            if price_below_ma or not bullish_fund:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Hybrid Exit Signal (Price below MA or fundamental change)"
                )

        elif self.last_signal_type[timeframe] == "SELL":
            # Exit short position when price rises above MA or fundamental condition changes
            if price_above_ma or not bearish_fund:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Hybrid Exit Signal (Price above MA or fundamental change)"
                )

        return None