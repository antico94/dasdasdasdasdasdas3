# Strategies/ichimoku_strategy.py
from typing import List, Optional, Set

import numpy as np

from Config.trading_config import TimeFrame
from Database.models import PriceBar
from Events.events import SignalEvent
from Strategies.base_strategy import BaseStrategy
from Strategies.indicator_utils import IndicatorUtils


class IchimokuStrategy(BaseStrategy):
    """
    Ichimoku Cloud trading strategy.

    This strategy uses the Ichimoku Cloud indicator to generate signals based on:
    - Tenkan-sen (Conversion Line) and Kijun-sen (Base Line) crossovers
    - Price position relative to the Kumo (Cloud)
    - Chikou Span (Lagging Span) confirmation
    """

    def __init__(self, name: str, symbol: str, timeframes: Set[TimeFrame],
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_span_b_period: int = 52,
                 displacement: int = 26,
                 require_kumo_breakout: bool = True,
                 require_chikou_confirmation: bool = True,
                 adx_period: int = 14,
                 adx_threshold: float = 25.0,
                 logger=None):
        """
        Initialize the Ichimoku strategy.

        Args:
            name: Strategy name
            symbol: Trading instrument symbol
            timeframes: Set of timeframes this strategy will use
            tenkan_period: Period for Tenkan-sen (Conversion Line)
            kijun_period: Period for Kijun-sen (Base Line)
            senkou_span_b_period: Period for Senkou Span B
            displacement: Displacement period for cloud and Chikou Span
            require_kumo_breakout: Require price to break through the cloud
            require_chikou_confirmation: Require Chikou Span confirmation
            adx_period: Period for ADX calculation (trend strength)
            adx_threshold: ADX threshold for valid signals
            logger: Logger instance
        """
        super().__init__(name, symbol, timeframes, logger)

        # Ichimoku parameters
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.displacement = displacement

        # Signal filters
        self.require_kumo_breakout = require_kumo_breakout
        self.require_chikou_confirmation = require_chikou_confirmation

        # Trend filter
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

        # Track previous signals to avoid duplicates
        self.last_signal_type = {tf: None for tf in timeframes}

    def update_indicators(self, timeframe: TimeFrame):
        """
        Calculate and cache Ichimoku Cloud indicators.

        Args:
            timeframe: The timeframe to update indicators for
        """
        # Get completed bars
        bars = self.get_completed_bars(timeframe)

        # Ensure we have enough bars
        required_bars = max(self.tenkan_period, self.kijun_period,
                            self.senkou_span_b_period, self.displacement) + 10
        if len(bars) < required_bars:
            return

        # Convert to numpy arrays
        arrays = self.get_bars_as_arrays(timeframe)

        # Calculate Ichimoku Cloud components
        ichimoku_data = IndicatorUtils.ichimoku(
            arrays['high'], arrays['low'], arrays['close'],
            tenkan_period=self.tenkan_period,
            kijun_period=self.kijun_period,
            senkou_span_b_period=self.senkou_span_b_period,
            displacement=self.displacement
        )

        # Store all components
        for component, values in ichimoku_data.items():
            self.set_indicator(timeframe, component, values)

        # Calculate ADX for trend confirmation
        adx, plus_di, minus_di = IndicatorUtils.adx(
            arrays['high'], arrays['low'], arrays['close'], self.adx_period
        )
        self.set_indicator(timeframe, 'adx', adx)
        self.set_indicator(timeframe, 'plus_di', plus_di)
        self.set_indicator(timeframe, 'minus_di', minus_di)

        # Calculate bullish and bearish signals
        bullish_signals, bearish_signals = IndicatorUtils.detect_ichimoku_signals(
            ichimoku_data, arrays['close'], self.displacement
        )
        self.set_indicator(timeframe, 'bullish_signals', bullish_signals)
        self.set_indicator(timeframe, 'bearish_signals', bearish_signals)

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
        min_bars = max(self.tenkan_period, self.kijun_period,
                       self.senkou_span_b_period, self.displacement) + 10
        if not self.validate_bars(timeframe, min_bars):
            return None

        # Get the latest bar
        current_bar = bars[-1]

        # Get Ichimoku components
        tenkan_sen = self.get_indicator(timeframe, 'tenkan_sen')
        kijun_sen = self.get_indicator(timeframe, 'kijun_sen')
        senkou_span_a = self.get_indicator(timeframe, 'senkou_span_a')
        senkou_span_b = self.get_indicator(timeframe, 'senkou_span_b')
        chikou_span = self.get_indicator(timeframe, 'chikou_span')

        # Get signal indicators
        bullish_signals = self.get_indicator(timeframe, 'bullish_signals')
        bearish_signals = self.get_indicator(timeframe, 'bearish_signals')

        # Get ADX values
        adx = self.get_indicator(timeframe, 'adx')
        plus_di = self.get_indicator(timeframe, 'plus_di')
        minus_di = self.get_indicator(timeframe, 'minus_di')

        # Ensure we have all necessary indicators
        if (tenkan_sen is None or kijun_sen is None or
                senkou_span_a is None or senkou_span_b is None or
                chikou_span is None or adx is None):
            return None

        # Calculate conditions
        cloud_top = max(senkou_span_a[-1], senkou_span_b[-1])
        cloud_bottom = min(senkou_span_a[-1], senkou_span_b[-1])

        # Tenkan-Kijun cross conditions
        tk_cross_bullish = tenkan_sen[-1] > kijun_sen[-1]
        tk_cross_bearish = tenkan_sen[-1] < kijun_sen[-1]
        fresh_bullish_cross = tenkan_sen[-2] <= kijun_sen[-2] and tk_cross_bullish
        fresh_bearish_cross = tenkan_sen[-2] >= kijun_sen[-2] and tk_cross_bearish

        # Price vs cloud conditions
        price_above_cloud = current_bar.close > cloud_top
        price_below_cloud = current_bar.close < cloud_bottom
        price_in_cloud = current_bar.close <= cloud_top and current_bar.close >= cloud_bottom

        # Cloud color
        green_cloud = senkou_span_a[-1] > senkou_span_b[-1]

        # Chikou span conditions
        chikou_bullish = False
        chikou_bearish = False

        if len(bars) > self.displacement:
            chikou_idx = -self.displacement - 1
            if chikou_idx >= -len(chikou_span):
                chikou_bullish = chikou_span[chikou_idx] > bars[chikou_idx].close
                chikou_bearish = chikou_span[chikou_idx] < bars[chikou_idx].close

        # Prepare condition groups for display
        condition_groups = {
            "Bullish Conditions": [
                ("Tenkan > Kijun", tk_cross_bullish,
                 f"T:{tenkan_sen[-1]:.4f}, K:{kijun_sen[-1]:.4f}"),

                ("Fresh Bullish Cross", fresh_bullish_cross,
                 f"Previous: T:{tenkan_sen[-2]:.4f} vs K:{kijun_sen[-2]:.4f}"),

                ("Price Above Cloud", price_above_cloud,
                 f"Price:{current_bar.close:.4f}, CloudTop:{cloud_top:.4f}"),

                ("Green Cloud", green_cloud,
                 f"SpanA:{senkou_span_a[-1]:.4f}, SpanB:{senkou_span_b[-1]:.4f}"),

                ("Chikou Confirmation", chikou_bullish,
                 "Above price" if chikou_bullish else "Below price"),

                ("Strong Trend (ADX)", adx[-1] > self.adx_threshold,
                 f"ADX:{adx[-1]:.2f}, Threshold:{self.adx_threshold}")
            ],

            "Bearish Conditions": [
                ("Tenkan < Kijun", tk_cross_bearish,
                 f"T:{tenkan_sen[-1]:.4f}, K:{kijun_sen[-1]:.4f}"),

                ("Fresh Bearish Cross", fresh_bearish_cross,
                 f"Previous: T:{tenkan_sen[-2]:.4f} vs K:{kijun_sen[-2]:.4f}"),

                ("Price Below Cloud", price_below_cloud,
                 f"Price:{current_bar.close:.4f}, CloudBottom:{cloud_bottom:.4f}"),

                ("Red Cloud", not green_cloud,
                 f"SpanA:{senkou_span_a[-1]:.4f}, SpanB:{senkou_span_b[-1]:.4f}"),

                ("Chikou Confirmation", chikou_bearish,
                 "Below price" if chikou_bearish else "Above price"),

                ("Strong Trend (ADX)", adx[-1] > self.adx_threshold,
                 f"ADX:{adx[-1]:.2f}, Threshold:{self.adx_threshold}")
            ]
        }

        # Print the conditions
        self.print_strategy_conditions(timeframe, condition_groups)

        # Check for bullish signal
        if bullish_signals is not None and bullish_signals[-1] and self.last_signal_type[timeframe] != "BUY":
            # Additional confirmations
            if self._check_bullish_confirmations(timeframe, bars, adx):
                # Calculate entry, stop loss, and take profit levels
                entry_price = current_bar.close
                stop_loss = kijun_sen[-1]  # Use Kijun-sen as stop loss
                take_profit = entry_price + (entry_price - stop_loss) * 2  # 1:2 risk-reward ratio

                # Log the signal details
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bullish Ichimoku signal detected for {self.symbol} on {timeframe.name}",
                    action="calculate_signals",
                    status="signal_detected",
                    details={
                        "tenkan": tenkan_sen[-1],
                        "kijun": kijun_sen[-1],
                        "price": current_bar.close,
                        "cloud_top": cloud_top,
                        "cloud_bottom": cloud_bottom,
                        "adx": adx[-1] if adx is not None else None
                    }
                )

                # Update last signal type
                self.last_signal_type[timeframe] = "BUY"

                # Create and return signal event
                return self.create_signal(
                    timeframe=timeframe,
                    direction="BUY",
                    reason="Ichimoku Bullish Signal",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

        # Check for bearish signal
        elif bearish_signals is not None and bearish_signals[-1] and self.last_signal_type[timeframe] != "SELL":
            # Additional confirmations
            if self._check_bearish_confirmations(timeframe, bars, adx):
                # Calculate entry, stop loss, and take profit levels
                entry_price = current_bar.close
                stop_loss = kijun_sen[-1]  # Use Kijun-sen as stop loss
                take_profit = entry_price - (stop_loss - entry_price) * 2  # 1:2 risk-reward ratio

                # Log the signal details
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bearish Ichimoku signal detected for {self.symbol} on {timeframe.name}",
                    action="calculate_signals",
                    status="signal_detected",
                    details={
                        "tenkan": tenkan_sen[-1],
                        "kijun": kijun_sen[-1],
                        "price": current_bar.close,
                        "cloud_top": cloud_top,
                        "cloud_bottom": cloud_bottom,
                        "adx": adx[-1] if adx is not None else None
                    }
                )

                # Update last signal type
                self.last_signal_type[timeframe] = "SELL"

                # Create and return signal event
                return self.create_signal(
                    timeframe=timeframe,
                    direction="SELL",
                    reason="Ichimoku Bearish Signal",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

        # Check for exit signals
        elif self.last_signal_type[timeframe] == "BUY" and tenkan_sen[-1] < kijun_sen[-1]:
            # Exit bullish position
            self.last_signal_type[timeframe] = None

            return self.create_signal(
                timeframe=timeframe,
                direction="CLOSE",
                reason="Ichimoku Exit Signal (Tenkan crossed below Kijun)"
            )

        elif self.last_signal_type[timeframe] == "SELL" and tenkan_sen[-1] > kijun_sen[-1]:
            # Exit bearish position
            self.last_signal_type[timeframe] = None

            return self.create_signal(
                timeframe=timeframe,
                direction="CLOSE",
                reason="Ichimoku Exit Signal (Tenkan crossed above Kijun)"
            )

        return None

    def _check_bullish_confirmations(self, timeframe: TimeFrame, bars: List[PriceBar], adx: np.ndarray) -> bool:
        """
        Check additional bullish confirmations.

        Args:
            timeframe: The timeframe to check
            bars: List of price bars
            adx: ADX values

        Returns:
            True if confirmations pass, False otherwise
        """
        current_bar = bars[-1]

        # Get Ichimoku components
        tenkan_sen = self.get_indicator(timeframe, 'tenkan_sen')
        kijun_sen = self.get_indicator(timeframe, 'kijun_sen')
        senkou_span_a = self.get_indicator(timeframe, 'senkou_span_a')
        senkou_span_b = self.get_indicator(timeframe, 'senkou_span_b')
        chikou_span = self.get_indicator(timeframe, 'chikou_span')

        # Check trend strength with ADX
        if adx is not None and adx[-1] < self.adx_threshold:
            self.log_strategy_event(
                level="INFO",
                message=f"Bullish signal rejected due to weak trend for {self.symbol}",
                action="_check_bullish_confirmations",
                status="rejected",
                details={"adx": adx[-1], "threshold": self.adx_threshold}
            )
            return False

        # Check Kumo breakout if required
        if self.require_kumo_breakout:
            # Get cloud top at current bar
            cloud_top = max(senkou_span_a[-1], senkou_span_b[-1])

            # Price must be above the cloud
            if current_bar.close <= cloud_top:
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bullish signal rejected: price not above cloud for {self.symbol}",
                    action="_check_bullish_confirmations",
                    status="rejected",
                    details={
                        "price": current_bar.close,
                        "cloud_top": cloud_top
                    }
                )
                return False

        # Check Chikou Span confirmation if required
        if self.require_chikou_confirmation:
            # Get the right index for comparison (chikou is displaced)
            if len(bars) > self.displacement:
                # Price must be above chikou
                chikou_idx = -self.displacement - 1  # Adjust for zero-based indexing
                if chikou_idx >= -len(chikou_span) and chikou_span[chikou_idx] <= bars[chikou_idx].close:
                    self.log_strategy_event(
                        level="INFO",
                        message=f"Bullish signal rejected: chikou not confirming for {self.symbol}",
                        action="_check_bullish_confirmations",
                        status="rejected",
                        details={
                            "chikou": chikou_span[chikou_idx],
                            "price": bars[chikou_idx].close
                        }
                    )
                    return False

        return True

    def _check_bearish_confirmations(self, timeframe: TimeFrame, bars: List[PriceBar], adx: np.ndarray) -> bool:
        """
        Check additional bearish confirmations.

        Args:
            timeframe: The timeframe to check
            bars: List of price bars
            adx: ADX values

        Returns:
            True if confirmations pass, False otherwise
        """
        current_bar = bars[-1]

        # Get Ichimoku components
        tenkan_sen = self.get_indicator(timeframe, 'tenkan_sen')
        kijun_sen = self.get_indicator(timeframe, 'kijun_sen')
        senkou_span_a = self.get_indicator(timeframe, 'senkou_span_a')
        senkou_span_b = self.get_indicator(timeframe, 'senkou_span_b')
        chikou_span = self.get_indicator(timeframe, 'chikou_span')

        # Check trend strength with ADX
        if adx is not None and adx[-1] < self.adx_threshold:
            self.log_strategy_event(
                level="INFO",
                message=f"Bearish signal rejected due to weak trend for {self.symbol}",
                action="_check_bearish_confirmations",
                status="rejected",
                details={"adx": adx[-1], "threshold": self.adx_threshold}
            )
            return False

        # Check Kumo breakout if required
        if self.require_kumo_breakout:
            # Get cloud bottom at current bar
            cloud_bottom = min(senkou_span_a[-1], senkou_span_b[-1])

            # Price must be below the cloud
            if current_bar.close >= cloud_bottom:
                self.log_strategy_event(
                    level="INFO",
                    message=f"Bearish signal rejected: price not below cloud for {self.symbol}",
                    action="_check_bearish_confirmations",
                    status="rejected",
                    details={
                        "price": current_bar.close,
                        "cloud_bottom": cloud_bottom
                    }
                )
                return False

        # Check Chikou Span confirmation if required
        if self.require_chikou_confirmation:
            # Get the right index for comparison (chikou is displaced)
            if len(bars) > self.displacement:
                # Price must be below chikou
                chikou_idx = -self.displacement - 1  # Adjust for zero-based indexing
                if chikou_idx >= -len(chikou_span) and chikou_span[chikou_idx] >= bars[chikou_idx].close:
                    self.log_strategy_event(
                        level="INFO",
                        message=f"Bearish signal rejected: chikou not confirming for {self.symbol}",
                        action="_check_bearish_confirmations",
                        status="rejected",
                        details={
                            "chikou": chikou_span[chikou_idx],
                            "price": bars[chikou_idx].close
                        }
                    )
                    return False

        return True