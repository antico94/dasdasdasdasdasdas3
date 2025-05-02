# Strategies/indicator_utils.py
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
from datetime import datetime, time


class MAType(Enum):
    """Types of Moving Averages"""
    SMA = "Simple Moving Average"
    EMA = "Exponential Moving Average"
    WMA = "Weighted Moving Average"
    HULL = "Hull Moving Average"
    TEMA = "Triple Exponential Moving Average"


class IndicatorUtils:
    """
    Utility class for calculating technical indicators.

    This class provides static methods for calculating common technical indicators
    used across various trading strategies.
    """

    @staticmethod
    def moving_average(prices: np.ndarray, period: int, ma_type: MAType = MAType.SMA) -> np.ndarray:
        """
        Calculate moving average.

        Args:
            prices: Array of price values
            period: MA period
            ma_type: Type of moving average

        Returns:
            Array with moving average values (with leading zeros for invalid values)
        """
        if len(prices) < period:
            return np.zeros(len(prices))

        result = np.zeros(len(prices))

        if ma_type == MAType.SMA:
            # Simple Moving Average
            for i in range(period - 1, len(prices)):
                result[i] = np.mean(prices[i - period + 1:i + 1])

        elif ma_type == MAType.EMA:
            # Exponential Moving Average
            alpha = 2 / (period + 1)
            result[period - 1] = np.mean(prices[:period])  # First value is SMA

            for i in range(period, len(prices)):
                result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]

        elif ma_type == MAType.WMA:
            # Weighted Moving Average
            weights = np.arange(1, period + 1)
            weight_sum = np.sum(weights)

            for i in range(period - 1, len(prices)):
                window = prices[i - period + 1:i + 1]
                result[i] = np.sum(window * weights) / weight_sum

        elif ma_type == MAType.HULL:
            # Hull Moving Average
            half_period = period // 2

            # Calculate WMAs
            wma1 = IndicatorUtils.moving_average(prices, half_period, MAType.WMA)
            wma2 = IndicatorUtils.moving_average(prices, period, MAType.WMA)

            # Calculate 2*WMA(n/2) - WMA(n)
            raw_hma = 2 * wma1 - wma2

            # Calculate WMA of raw_hma with sqrt(n) period
            sqrt_period = int(np.sqrt(period))
            result = IndicatorUtils.moving_average(raw_hma, sqrt_period, MAType.WMA)

        elif ma_type == MAType.TEMA:
            # Triple Exponential Moving Average
            ema1 = IndicatorUtils.moving_average(prices, period, MAType.EMA)
            ema2 = IndicatorUtils.moving_average(ema1, period, MAType.EMA)
            ema3 = IndicatorUtils.moving_average(ema2, period, MAType.EMA)

            result = 3 * ema1 - 3 * ema2 + ema3

        return result

    @staticmethod
    def macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26,
             signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Array of price values
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate EMAs
        fast_ema = IndicatorUtils.moving_average(prices, fast_period, MAType.EMA)
        slow_ema = IndicatorUtils.moving_average(prices, slow_period, MAType.EMA)

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line
        signal_line = IndicatorUtils.moving_average(macd_line, signal_period, MAType.EMA)

        # Calculate histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            prices: Array of price values
            period: RSI period

        Returns:
            Array with RSI values (with leading zeros for invalid values)
        """
        if len(prices) <= period:
            return np.zeros(len(prices))

        # Calculate price changes
        delta = np.zeros(len(prices))
        delta[1:] = prices[1:] - prices[:-1]

        # Separate gains and losses
        gain = np.zeros(len(delta))
        loss = np.zeros(len(delta))

        gain[delta > 0] = delta[delta > 0]
        loss[delta < 0] = -delta[delta < 0]

        # Calculate average gain and loss
        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))

        # First values
        avg_gain[period] = np.mean(gain[1:period + 1])
        avg_loss[period] = np.mean(loss[1:period + 1])

        # Calculate subsequent values
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

        # Calculate RS and RSI
        rs = np.zeros(len(prices))
        rsi = np.zeros(len(prices))

        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))

        return rsi

    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20,
                        deviation: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Array of price values
            period: Period for moving average
            deviation: Number of standard deviations

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(prices) < period:
            return np.zeros(len(prices)), np.zeros(len(prices)), np.zeros(len(prices))

        # Calculate middle band (SMA)
        middle_band = IndicatorUtils.moving_average(prices, period, MAType.SMA)

        # Calculate standard deviation
        std = np.zeros(len(prices))

        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1], ddof=1)

        # Calculate upper and lower bands
        upper_band = middle_band + (std * deviation)
        lower_band = middle_band - (std * deviation)

        return upper_band, middle_band, lower_band

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> np.ndarray:
        """
        Calculate ATR (Average True Range).

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: ATR period

        Returns:
            Array with ATR values
        """
        if len(high) < period + 1:
            return np.zeros(len(high))

        # Calculate True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]  # First TR is simply high - low

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate ATR
        atr = np.zeros(len(high))
        atr[period - 1] = np.mean(tr[:period])

        for i in range(period, len(high)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   k_period: int = 14, k_slowing: int = 3,
                   d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            k_period: %K period
            k_slowing: %K slowing period
            d_period: %D period

        Returns:
            Tuple of (%K, %D)
        """
        if len(close) < k_period:
            return np.zeros(len(close)), np.zeros(len(close))

        # Calculate %K
        k = np.zeros(len(close))

        for i in range(k_period - 1, len(close)):
            lowest_low = np.min(low[i - k_period + 1:i + 1])
            highest_high = np.max(high[i - k_period + 1:i + 1])

            if highest_high - lowest_low > 0:
                k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k[i] = 50  # Default to midpoint if range is zero

        # Apply slowing to %K
        if k_slowing > 1:
            k_slowed = np.zeros(len(close))

            for i in range(k_period + k_slowing - 2, len(close)):
                k_slowed[i] = np.mean(k[i - k_slowing + 1:i + 1])

            k = k_slowed

        # Calculate %D (SMA of %K)
        d = IndicatorUtils.moving_average(k, d_period, MAType.SMA)

        return k, d

    @staticmethod
    def donchian_channel(high: np.ndarray, low: np.ndarray,
                         period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Donchian Channel.

        Args:
            high: Array of high prices
            low: Array of low prices
            period: Channel period

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(high) < period:
            return np.zeros(len(high)), np.zeros(len(high)), np.zeros(len(high))

        upper_band = np.zeros(len(high))
        lower_band = np.zeros(len(high))

        for i in range(period - 1, len(high)):
            upper_band[i] = np.max(high[i - period + 1:i + 1])
            lower_band[i] = np.min(low[i - period + 1:i + 1])

        # Calculate middle band
        middle_band = (upper_band + lower_band) / 2

        return upper_band, middle_band, lower_band

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ADX (Average Directional Index).

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: ADX period

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        if len(high) < 2 * period:
            return np.zeros(len(high)), np.zeros(len(high)), np.zeros(len(high))

        # Calculate True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        # Calculate Directional Movement
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))

        for i in range(1, len(high)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0

            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0

        # Calculate smoothed values
        smoothed_plus_dm = np.zeros(len(high))
        smoothed_minus_dm = np.zeros(len(high))
        smoothed_tr = np.zeros(len(high))

        # First values
        smoothed_plus_dm[period] = np.sum(plus_dm[1:period + 1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1:period + 1])
        smoothed_tr[period] = np.sum(tr[1:period + 1])

        # Calculate subsequent values
        for i in range(period + 1, len(high)):
            smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / period) + minus_dm[i]
            smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / period) + tr[i]

        # Calculate Directional Indicators
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))

        for i in range(period, len(high)):
            if smoothed_tr[i] > 0:
                plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]

        # Calculate DX
        dx = np.zeros(len(high))

        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

        # Calculate ADX
        adx = np.zeros(len(high))

        # First ADX value is average of DX for period
        adx[2 * period - 1] = np.mean(dx[period:2 * period])

        # Calculate subsequent values
        for i in range(2 * period, len(high)):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx, plus_di, minus_di

    @staticmethod
    def ichimoku(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_span_b_period: int = 52, displacement: int = 26) -> Dict[str, np.ndarray]:
        """
        Calculate Ichimoku Cloud components.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_span_b_period: Senkou Span B (Leading Span B) period
            displacement: Displacement period for Senkou Span

        Returns:
            Dictionary with all Ichimoku components
        """
        max_period = max(tenkan_period, kijun_period, senkou_span_b_period)
        if len(high) < max_period:
            return {
                'tenkan_sen': np.zeros(len(high)),
                'kijun_sen': np.zeros(len(high)),
                'senkou_span_a': np.zeros(len(high)),
                'senkou_span_b': np.zeros(len(high)),
                'chikou_span': np.zeros(len(high))
            }

        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for tenkan_period
        tenkan_sen = np.zeros(len(high))
        for i in range(tenkan_period - 1, len(high)):
            tenkan_sen[i] = (np.max(high[i - tenkan_period + 1:i + 1]) + np.min(low[i - tenkan_period + 1:i + 1])) / 2

        # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for kijun_period
        kijun_sen = np.zeros(len(high))
        for i in range(kijun_period - 1, len(high)):
            kijun_sen[i] = (np.max(high[i - kijun_period + 1:i + 1]) + np.min(low[i - kijun_period + 1:i + 1])) / 2

        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, displaced forward
        senkou_span_a = np.zeros(len(high) + displacement)
        for i in range(max(tenkan_period, kijun_period) - 1, len(high)):
            senkou_span_a[i + displacement] = (tenkan_sen[i] + kijun_sen[i]) / 2

        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for senkou_span_b_period, displaced forward
        senkou_span_b = np.zeros(len(high) + displacement)
        for i in range(senkou_span_b_period - 1, len(high)):
            senkou_span_b[i + displacement] = (np.max(high[i - senkou_span_b_period + 1:i + 1]) + np.min(
                low[i - senkou_span_b_period + 1:i + 1])) / 2

        # Calculate Chikou Span (Lagging Span): Current closing price, displaced backward
        chikou_span = np.zeros(len(close) + displacement)
        chikou_span[:len(close)] = close

        # Return only the valid portion of the arrays
        return {
            'tenkan_sen': tenkan_sen[:len(high)],
            'kijun_sen': kijun_sen[:len(high)],
            'senkou_span_a': senkou_span_a[:len(high)],
            'senkou_span_b': senkou_span_b[:len(high)],
            'chikou_span': chikou_span[:len(high)]
        }

    @staticmethod
    def internal_bar_strength(open_prices: np.ndarray, high: np.ndarray,
                              low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate Internal Bar Strength (IBS).
        IBS = (Close - Low) / (High - Low)

        Args:
            open_prices: Array of open prices
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices

        Returns:
            Array with IBS values
        """
        ibs = np.zeros(len(close))

        for i in range(len(close)):
            if high[i] - low[i] > 0:
                ibs[i] = (close[i] - low[i]) / (high[i] - low[i])
            else:
                ibs[i] = 0.5  # Default to midpoint if high = low

        return ibs

    @staticmethod
    def is_valid_session(timestamp: datetime, session: str) -> bool:
        """
        Check if the given timestamp is within a valid trading session.

        Args:
            timestamp: The datetime to check
            session: Session name ('london', 'new_york', 'tokyo', etc.)

        Returns:
            True if within the specified session, False otherwise
        """
        hour = timestamp.hour

        if session == "london":
            # London session: 7:00-16:00 UTC
            return 7 <= hour < 16
        elif session == "new_york":
            # New York session: 12:00-21:00 UTC
            return 12 <= hour < 21
        elif session == "london_new_york":
            # London + New York overlap: 12:00-16:00 UTC
            return 12 <= hour < 16
        elif session == "tokyo":
            # Tokyo session: 0:00-9:00 UTC
            return 0 <= hour < 9
        elif session == "tokyo_ny_overlap":
            # Tokyo + New York overlap: 8:00-12:00 UTC
            return 8 <= hour < 12
        elif session == "asian":
            # Asian session (broader): 22:00-8:00 UTC
            return hour >= 22 or hour < 8

        # Default to True if session not recognized
        return True

    @staticmethod
    def calculate_pivot_points(high: float, low: float, close: float,
                               pivot_type: str = "standard") -> Dict[str, float]:
        """
        Calculate pivot points for the next period.

        Args:
            high: Previous period's high
            low: Previous period's low
            close: Previous period's close
            pivot_type: Type of pivot calculation ('standard', 'fibonacci', 'camarilla', etc.)

        Returns:
            Dictionary with pivot levels
        """
        result = {}

        if pivot_type == "standard":
            # Standard pivot points
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)

            result = {
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                's1': s1,
                's2': s2,
                's3': s3
            }

        elif pivot_type == "fibonacci":
            # Fibonacci pivot points
            pivot = (high + low + close) / 3
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + (high - low)
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - (high - low)

            result = {
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                's1': s1,
                's2': s2,
                's3': s3
            }

        elif pivot_type == "camarilla":
            # Camarilla pivot points
            r1 = close + 1.1 / 12.0 * (high - low)
            r2 = close + 1.1 / 6.0 * (high - low)
            r3 = close + 1.1 / 4.0 * (high - low)
            r4 = close + 1.1 / 2.0 * (high - low)
            s1 = close - 1.1 / 12.0 * (high - low)
            s2 = close - 1.1 / 6.0 * (high - low)
            s3 = close - 1.1 / 4.0 * (high - low)
            s4 = close - 1.1 / 2.0 * (high - low)

            result = {
                'r1': r1,
                'r2': r2,
                'r3': r3,
                'r4': r4,
                's1': s1,
                's2': s2,
                's3': s3,
                's4': s4
            }

        return result

    @staticmethod
    def calculate_risk_reward(entry: float, stop_loss: float, take_profit: float) -> float:
        """
        Calculate risk-reward ratio.

        Args:
            entry: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Risk-reward ratio (reward / risk)
        """
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)

        if risk == 0:
            return 0

        return reward / risk