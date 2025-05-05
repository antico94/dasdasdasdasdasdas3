# Strategies/mean_reversion_strategy.py
from typing import Optional, Set

from Config.trading_config import TimeFrame
from Database.models import PriceBar
from Events.events import SignalEvent
from Strategies.base_strategy import BaseStrategy
from Strategies.indicator_utils import IndicatorUtils


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using RSI, Stochastic, and Bollinger Bands.

    This strategy looks for oversold/overbought conditions in ranging markets:
    - RSI below/above threshold levels
    - Stochastic below/above threshold levels
    - Price at or outside Bollinger Bands
    - Optional IBS (Internal Bar Strength) filter
    """

    def __init__(self, name: str, symbol: str, timeframes: Set[TimeFrame],
                 rsi_period: int = 14,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 stoch_k_period: int = 14,
                 stoch_d_period: int = 3,
                 stoch_oversold: float = 20.0,
                 stoch_overbought: float = 80.0,
                 bb_period: int = 20,
                 bb_std_dev: float = 2.0,
                 use_ibs: bool = True,
                 ibs_threshold_low: float = 0.2,
                 ibs_threshold_high: float = 0.8,
                 atr_period: int = 14,
                 stop_loss_atr_multiplier: float = 1.5,
                 take_profit_atr_multiplier: float = 2.0,
                 logger=None):
        """
        Initialize the Mean Reversion strategy.

        Args:
            name: Strategy name
            symbol: Trading instrument symbol
            timeframes: Set of timeframes this strategy will use
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI threshold for oversold condition
            rsi_overbought: RSI threshold for overbought condition
            stoch_k_period: Period for Stochastic %K
            stoch_d_period: Period for Stochastic %D
            stoch_oversold: Stochastic threshold for oversold condition
            stoch_overbought: Stochastic threshold for overbought condition
            bb_period: Period for Bollinger Bands
            bb_std_dev: Standard deviation multiplier for Bollinger Bands
            use_ibs: Whether to use Internal Bar Strength filter
            ibs_threshold_low: IBS threshold for oversold condition
            ibs_threshold_high: IBS threshold for overbought condition
            atr_period: Period for ATR calculation
            stop_loss_atr_multiplier: ATR multiplier for stop loss
            take_profit_atr_multiplier: ATR multiplier for take profit
            logger: Logger instance
        """
        super().__init__(name, symbol, timeframes, logger)

        # RSI parameters
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Stochastic parameters
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought

        # Bollinger Bands parameters
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev

        # IBS filter
        self.use_ibs = use_ibs
        self.ibs_threshold_low = ibs_threshold_low
        self.ibs_threshold_high = ibs_threshold_high

        # Risk management
        self.atr_period = atr_period
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier

        # Track previous signals to avoid duplicates
        self.last_signal_type = {tf: None for tf in timeframes}

    def update_indicators(self, timeframe: TimeFrame):
        """
        Calculate and cache indicators for the Mean Reversion strategy.

        Args:
            timeframe: The timeframe to update indicators for
        """
        # Get completed bars
        bars = self.get_completed_bars(timeframe)

        # Ensure we have enough bars
        required_bars = max(self.rsi_period, self.stoch_k_period, self.bb_period, self.atr_period) + 10
        if len(bars) < required_bars:
            return

        # Convert to numpy arrays
        arrays = self.get_bars_as_arrays(timeframe)

        # Calculate RSI
        rsi = IndicatorUtils.rsi(arrays['close'], self.rsi_period)
        self.set_indicator(timeframe, 'rsi', rsi)

        # Calculate Stochastic
        stoch_k, stoch_d = IndicatorUtils.stochastic(
            arrays['high'], arrays['low'], arrays['close'],
            self.stoch_k_period, 1, self.stoch_d_period
        )
        self.set_indicator(timeframe, 'stoch_k', stoch_k)
        self.set_indicator(timeframe, 'stoch_d', stoch_d)

        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = IndicatorUtils.bollinger_bands(
            arrays['close'], self.bb_period, self.bb_std_dev
        )
        self.set_indicator(timeframe, 'bb_upper', bb_upper)
        self.set_indicator(timeframe, 'bb_middle', bb_middle)
        self.set_indicator(timeframe, 'bb_lower', bb_lower)

        # Calculate ATR for stop loss and take profit
        atr = IndicatorUtils.atr(
            arrays['high'], arrays['low'], arrays['close'], self.atr_period
        )
        self.set_indicator(timeframe, 'atr', atr)

        # Calculate IBS if needed
        if self.use_ibs:
            ibs = IndicatorUtils.internal_bar_strength(
                arrays['open'], arrays['high'], arrays['low'], arrays['close']
            )
            self.set_indicator(timeframe, 'ibs', ibs)

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
        min_bars = max(self.rsi_period, self.stoch_k_period, self.bb_period, self.atr_period) + 10
        if not self.validate_bars(timeframe, min_bars):
            return None

        # Get the latest bar
        current_bar = bars[-1]

        # Get indicators
        rsi = self.get_indicator(timeframe, 'rsi')
        stoch_k = self.get_indicator(timeframe, 'stoch_k')
        stoch_d = self.get_indicator(timeframe, 'stoch_d')
        bb_upper = self.get_indicator(timeframe, 'bb_upper')
        bb_middle = self.get_indicator(timeframe, 'bb_middle')
        bb_lower = self.get_indicator(timeframe, 'bb_lower')
        atr = self.get_indicator(timeframe, 'atr')
        ibs = self.get_indicator(timeframe, 'ibs') if self.use_ibs else None

        # Ensure we have all necessary indicators
        if (rsi is None or stoch_k is None or stoch_d is None or
                bb_upper is None or bb_middle is None or bb_lower is None or
                atr is None or (self.use_ibs and ibs is None)):
            return None

        # Calculate oversold/overbought conditions
        is_rsi_oversold = rsi[-1] < self.rsi_oversold
        is_rsi_overbought = rsi[-1] > self.rsi_overbought

        is_stoch_oversold = stoch_k[-1] < self.stoch_oversold and stoch_d[-1] < self.stoch_oversold
        is_stoch_overbought = stoch_k[-1] > self.stoch_overbought and stoch_d[-1] > self.stoch_overbought

        is_price_at_lower_band = current_bar.close <= bb_lower[-1]
        is_price_at_upper_band = current_bar.close >= bb_upper[-1]

        is_ibs_low = self.use_ibs and ibs is not None and ibs[-1] < self.ibs_threshold_low
        is_ibs_high = self.use_ibs and ibs is not None and ibs[-1] > self.ibs_threshold_high

        # Prepare condition groups for display
        condition_groups = {
            "Bullish (Oversold) Conditions": [
                ("RSI Oversold", is_rsi_oversold,
                 f"RSI: {rsi[-1]:.2f} < {self.rsi_oversold:.2f}"),

                ("Stochastic Oversold", is_stoch_oversold,
                 f"K: {stoch_k[-1]:.2f}, D: {stoch_d[-1]:.2f} < {self.stoch_oversold:.2f}"),

                ("Price at/below Lower BB", is_price_at_lower_band,
                 f"Price: {current_bar.close:.5f}, BB Lower: {bb_lower[-1]:.5f}"),
            ],

            "Bearish (Overbought) Conditions": [
                ("RSI Overbought", is_rsi_overbought,
                 f"RSI: {rsi[-1]:.2f} > {self.rsi_overbought:.2f}"),

                ("Stochastic Overbought", is_stoch_overbought,
                 f"K: {stoch_k[-1]:.2f}, D: {stoch_d[-1]:.2f} > {self.stoch_overbought:.2f}"),

                ("Price at/above Upper BB", is_price_at_upper_band,
                 f"Price: {current_bar.close:.5f}, BB Upper: {bb_upper[-1]:.5f}"),
            ]
        }

        # Add IBS conditions if used
        if self.use_ibs:
            condition_groups["Bullish (Oversold) Conditions"].append(
                ("IBS Low", is_ibs_low,
                 f"IBS: {ibs[-1]:.2f} < {self.ibs_threshold_low:.2f}" if ibs is not None else "N/A")
            )
            condition_groups["Bearish (Overbought) Conditions"].append(
                ("IBS High", is_ibs_high,
                 f"IBS: {ibs[-1]:.2f} > {self.ibs_threshold_high:.2f}" if ibs is not None else "N/A")
            )

        # Print the conditions
        self.print_strategy_conditions(timeframe, condition_groups)

        # Check for oversold condition (long signal)
        if self._is_oversold(timeframe, current_bar) and self.last_signal_type[timeframe] != "BUY":
            # Calculate entry, stop loss, and take profit
            entry_price = current_bar.close
            stop_loss = entry_price - (atr[-1] * self.stop_loss_atr_multiplier)
            take_profit = entry_price + (atr[-1] * self.take_profit_atr_multiplier)

            # Log the signal
            self.log_strategy_event(
                level="INFO",
                message=f"Oversold condition detected for {self.symbol} on {timeframe.name}",
                action="calculate_signals",
                status="signal_detected",
                details={
                    "rsi": rsi[-1],
                    "stoch_k": stoch_k[-1],
                    "stoch_d": stoch_d[-1],
                    "price": current_bar.close,
                    "bb_lower": bb_lower[-1],
                    "ibs": ibs[-1] if self.use_ibs else None
                }
            )

            # Update last signal type
            self.last_signal_type[timeframe] = "BUY"

            # Create and return signal
            return self.create_signal(
                timeframe=timeframe,
                direction="BUY",
                reason="Mean Reversion Oversold Signal",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        # Check for overbought condition (short signal)
        elif self._is_overbought(timeframe, current_bar) and self.last_signal_type[timeframe] != "SELL":
            # Calculate entry, stop loss, and take profit
            entry_price = current_bar.close
            stop_loss = entry_price + (atr[-1] * self.stop_loss_atr_multiplier)
            take_profit = entry_price - (atr[-1] * self.take_profit_atr_multiplier)

            # Log the signal
            self.log_strategy_event(
                level="INFO",
                message=f"Overbought condition detected for {self.symbol} on {timeframe.name}",
                action="calculate_signals",
                status="signal_detected",
                details={
                    "rsi": rsi[-1],
                    "stoch_k": stoch_k[-1],
                    "stoch_d": stoch_d[-1],
                    "price": current_bar.close,
                    "bb_upper": bb_upper[-1],
                    "ibs": ibs[-1] if self.use_ibs else None
                }
            )

            # Update last signal type
            self.last_signal_type[timeframe] = "SELL"

            # Create and return signal
            return self.create_signal(
                timeframe=timeframe,
                direction="SELL",
                reason="Mean Reversion Overbought Signal",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        # Check for exit signals
        elif self.last_signal_type[timeframe] == "BUY":
            # Exit long position when price reaches middle band
            bb_middle_val = bb_middle[-1]
            if current_bar.close >= bb_middle_val:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Mean Reversion Exit Signal (Price reached mean)"
                )

        elif self.last_signal_type[timeframe] == "SELL":
            # Exit short position when price reaches middle band
            bb_middle_val = bb_middle[-1]
            if current_bar.close <= bb_middle_val:
                # Reset signal type
                self.last_signal_type[timeframe] = None

                return self.create_signal(
                    timeframe=timeframe,
                    direction="CLOSE",
                    reason="Mean Reversion Exit Signal (Price reached mean)"
                )

        return None

    def _is_oversold(self, timeframe: TimeFrame, current_bar: PriceBar) -> bool:
        """
        Check if the market is in an oversold condition.

        Args:
            timeframe: The timeframe to check
            current_bar: The current price bar

        Returns:
            True if oversold, False otherwise
        """
        # Get indicators
        rsi = self.get_indicator(timeframe, 'rsi')
        stoch_k = self.get_indicator(timeframe, 'stoch_k')
        stoch_d = self.get_indicator(timeframe, 'stoch_d')
        bb_lower = self.get_indicator(timeframe, 'bb_lower')
        ibs = self.get_indicator(timeframe, 'ibs') if self.use_ibs else None

        # Count how many indicators confirm oversold condition
        oversold_count = 0
        # total_indicators = 3 + (1 if self.use_ibs else 0)

        # Check RSI
        if rsi is not None and rsi[-1] < self.rsi_oversold:
            oversold_count += 1

        # Check Stochastic
        if stoch_k is not None and stoch_d is not None:
            if stoch_k[-1] < self.stoch_oversold and stoch_d[-1] < self.stoch_oversold:
                oversold_count += 1

        # Check Bollinger Bands
        if bb_lower is not None and current_bar.close <= bb_lower[-1]:
            oversold_count += 1

        # Check IBS
        if self.use_ibs and ibs is not None and ibs[-1] < self.ibs_threshold_low:
            oversold_count += 1

        # Require at least 2 indicators to confirm
        return oversold_count >= 2

    def _is_overbought(self, timeframe: TimeFrame, current_bar: PriceBar) -> bool:
        """
        Check if the market is in an overbought condition.

        Args:
            timeframe: The timeframe to check
            current_bar: The current price bar

        Returns:
            True if overbought, False otherwise
        """
        # Get indicators
        rsi = self.get_indicator(timeframe, 'rsi')
        stoch_k = self.get_indicator(timeframe, 'stoch_k')
        stoch_d = self.get_indicator(timeframe, 'stoch_d')
        bb_upper = self.get_indicator(timeframe, 'bb_upper')
        ibs = self.get_indicator(timeframe, 'ibs') if self.use_ibs else None

        # Count how many indicators confirm overbought condition
        overbought_count = 0
        # total_indicators = 3 + (1 if self.use_ibs else 0)

        # Check RSI
        if rsi is not None and rsi[-1] > self.rsi_overbought:
            overbought_count += 1

        # Check Stochastic
        if stoch_k is not None and stoch_d is not None:
            if stoch_k[-1] > self.stoch_overbought and stoch_d[-1] > self.stoch_overbought:
                overbought_count += 1

        # Check Bollinger Bands
        if bb_upper is not None and current_bar.close >= bb_upper[-1]:
            overbought_count += 1

        # Check IBS
        if self.use_ibs and ibs is not None and ibs[-1] > self.ibs_threshold_high:
            overbought_count += 1

        # Require at least 2 indicators to confirm
        return overbought_count >= 2