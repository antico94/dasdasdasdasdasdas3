# Strategies/indicator_utils.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
from enum import Enum
from datetime import datetime, time, timedelta, timezone


class TradingSession(Enum):
    """Trading session definitions with UTC time ranges"""
    SYDNEY = {"name": "Sydney", "start": time(20, 0), "end": time(5, 0), "next_day_end": True}
    TOKYO = {"name": "Tokyo", "start": time(0, 0), "end": time(9, 0), "next_day_end": False}
    LONDON = {"name": "London", "start": time(7, 0), "end": time(16, 0), "next_day_end": False}
    NEWYORK = {"name": "New York", "start": time(12, 0), "end": time(21, 0), "next_day_end": False}
    LONDON_NEWYORK_OVERLAP = {"name": "London/NY Overlap", "start": time(12, 0), "end": time(16, 0),
                              "next_day_end": False}
    TOKYO_LONDON_OVERLAP = {"name": "Tokyo/London Overlap", "start": time(7, 0), "end": time(9, 0),
                            "next_day_end": False}
    ASIAN = {"name": "Asian Session", "start": time(22, 0), "end": time(8, 0), "next_day_end": True}
    EUROPEAN = {"name": "European Session", "start": time(7, 0), "end": time(16, 0), "next_day_end": False}
    AMERICAN = {"name": "American Session", "start": time(12, 0), "end": time(21, 0), "next_day_end": False}


class MAType(Enum):
    """Types of Moving Averages"""
    SMA = "Simple Moving Average"
    EMA = "Exponential Moving Average"
    WMA = "Weighted Moving Average"
    HULL = "Hull Moving Average"
    TEMA = "Triple Exponential Moving Average"
    T3 = "T3 Moving Average"


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

        elif ma_type == MAType.T3:
            # T3 Moving Average (Tillson T3)
            # Default volume factor
            vfactor = 0.7

            # Calculate multiple EMAs
            e1 = IndicatorUtils.moving_average(prices, period, MAType.EMA)
            e2 = IndicatorUtils.moving_average(e1, period, MAType.EMA)
            e3 = IndicatorUtils.moving_average(e2, period, MAType.EMA)
            e4 = IndicatorUtils.moving_average(e3, period, MAType.EMA)
            e5 = IndicatorUtils.moving_average(e4, period, MAType.EMA)
            e6 = IndicatorUtils.moving_average(e5, period, MAType.EMA)

            # Calculate T3 using Tim Tillson's formula
            c1 = -vfactor ** 3
            c2 = 3 * vfactor ** 2 + 3 * vfactor ** 3
            c3 = -6 * vfactor ** 2 - 3 * vfactor - 3 * vfactor ** 3
            c4 = 1 + 3 * vfactor + vfactor ** 3 + 3 * vfactor ** 2

            result = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

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
    def keltner_channel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        period: int = 20, atr_period: int = 10,
                        atr_multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Keltner Channels.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: Period for the middle line (EMA)
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(close) < max(period, atr_period):
            return np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close))

        # Calculate middle line (EMA of close)
        middle_band = IndicatorUtils.moving_average(close, period, MAType.EMA)

        # Calculate ATR
        atr = IndicatorUtils.atr(high, low, close, atr_period)

        # Calculate bands
        upper_band = middle_band + (atr * atr_multiplier)
        lower_band = middle_band - (atr * atr_multiplier)

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
    def is_in_session(timestamp: datetime, session: Union[str, TradingSession]) -> bool:
        """
        Check if the given timestamp is within a valid trading session.

        Args:
            timestamp: The datetime to check (must be in UTC)
            session: Session name or TradingSession enum

        Returns:
            True if within the specified session, False otherwise
        """
        # Convert timestamp to UTC if it has timezone info but is not UTC
        if timestamp.tzinfo is not None and timestamp.tzinfo != timezone.utc:
            timestamp = timestamp.astimezone(timezone.utc)

        # If timestamp has no timezone info, assume it's UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Get the hour and minute
        current_time = timestamp.time()

        # Convert string session name to enum if needed
        if isinstance(session, str):
            try:
                session_enum = TradingSession[session.upper()]
            except KeyError:
                raise ValueError(f"Unknown session name: {session}")
        else:
            session_enum = session

        session_data = session_enum.value
        session_start = session_data["start"]
        session_end = session_data["end"]
        next_day_end = session_data["next_day_end"]

        # Check if we're in the session
        if next_day_end:
            # Session crosses midnight
            if current_time >= session_start or current_time <= session_end:
                return True
        else:
            # Session within the same day
            if session_start <= current_time <= session_end:
                return True

        return False

    @staticmethod
    def is_valid_session(timestamp: datetime, session_name: str) -> bool:
        """
        Check if the given timestamp is within a valid trading session.
        Backward compatibility method.

        Args:
            timestamp: The datetime to check
            session_name: Session name ('london', 'new_york', 'tokyo', etc.)

        Returns:
            True if within the specified session, False otherwise
        """
        # Convert session_name to standard format
        session_name = session_name.upper().replace('-', '_').replace(' ', '_')

        # Map legacy session names to TradingSession enum
        session_mapping = {
            "LONDON": TradingSession.LONDON,
            "NEW_YORK": TradingSession.NEWYORK,
            "NEWYORK": TradingSession.NEWYORK,
            "TOKYO": TradingSession.TOKYO,
            "LONDON_NEW_YORK": TradingSession.LONDON_NEWYORK_OVERLAP,
            "LONDON_NEWYORK": TradingSession.LONDON_NEWYORK_OVERLAP,
            "LONDON_NY": TradingSession.LONDON_NEWYORK_OVERLAP,
            "LONDONNY": TradingSession.LONDON_NEWYORK_OVERLAP,
            "TOKYO_LONDON": TradingSession.TOKYO_LONDON_OVERLAP,
            "TOKYO_NY_OVERLAP": TradingSession.TOKYO_LONDON_OVERLAP,
            "ASIAN": TradingSession.ASIAN,
            "EUROPEAN": TradingSession.EUROPEAN,
            "AMERICAN": TradingSession.AMERICAN
        }

        try:
            session = session_mapping.get(session_name)
            if session is None:
                # Try direct mapping to enum
                session = TradingSession[session_name]

            return IndicatorUtils.is_in_session(timestamp, session)
        except (KeyError, ValueError):
            # Default to True if session not recognized for backward compatibility
            return True

    @staticmethod
    def get_overnight_range(high: np.ndarray, low: np.ndarray, timestamps: List[datetime],
                            session_start: TradingSession, lookback_days: int = 1) -> Tuple[float, float]:
        """
        Calculate the overnight range before a trading session.

        Args:
            high: Array of high prices
            low: Array of low prices
            timestamps: List of timestamp for each bar
            session_start: Trading session start definition
            lookback_days: Number of days to look back

        Returns:
            Tuple of (range_high, range_low)
        """
        if len(high) != len(timestamps) or len(low) != len(timestamps):
            raise ValueError("Length of high, low, and timestamps arrays must match")

        # Convert timestamps to UTC if needed
        utc_timestamps = []
        for ts in timestamps:
            if ts.tzinfo is None:
                utc_ts = ts.replace(tzinfo=timezone.utc)
            elif ts.tzinfo != timezone.utc:
                utc_ts = ts.astimezone(timezone.utc)
            else:
                utc_ts = ts
            utc_timestamps.append(utc_ts)

        # Get the most recent timestamp
        latest_ts = utc_timestamps[-1]

        # Find the start of the session for the latest day
        session_data = session_start.value
        session_start_time = session_data["start"]

        # Create a session start datetime for today
        session_start_dt = datetime(
            latest_ts.year,
            latest_ts.month,
            latest_ts.day,
            session_start_time.hour,
            session_start_time.minute,
            0,
            tzinfo=timezone.utc
        )

        # If the latest timestamp is before today's session start, use today's session
        # Otherwise, the overnight range is for tomorrow's session, so we need to find bars
        # between yesterday's session end and today's session start
        if latest_ts < session_start_dt:
            # Today's overnight range (before session start)
            range_start_dt = session_start_dt - timedelta(days=1)
        else:
            # Tomorrow's overnight range (after today's session)
            session_start_dt = session_start_dt + timedelta(days=1)
            range_start_dt = session_start_dt - timedelta(days=1)

        # Find the end time of the previous session
        session_end_time = session_data["end"]
        if session_data["next_day_end"]:
            # If the session ends on the next day, add a day to the end time
            range_start_dt = datetime(
                range_start_dt.year,
                range_start_dt.month,
                range_start_dt.day,
                session_end_time.hour,
                session_end_time.minute,
                0,
                tzinfo=timezone.utc
            ) + timedelta(days=1)
        else:
            range_start_dt = datetime(
                range_start_dt.year,
                range_start_dt.month,
                range_start_dt.day,
                session_end_time.hour,
                session_end_time.minute,
                0,
                tzinfo=timezone.utc
            )

        # Find bars in the overnight range
        overnight_high = float('-inf')
        overnight_low = float('inf')
        bars_found = False

        for i in range(len(utc_timestamps)):
            # Check if the bar is in the overnight range
            if range_start_dt <= utc_timestamps[i] < session_start_dt:
                bars_found = True
                overnight_high = max(overnight_high, high[i])
                overnight_low = min(overnight_low, low[i])

        # If no bars were found in the overnight range or if the range is invalid
        if not bars_found or overnight_high == float('-inf') or overnight_low == float('inf'):
            # Look back additional days if specified
            if lookback_days > 1:
                additional_days = 1
                while additional_days < lookback_days and (
                        not bars_found or overnight_high == float('-inf') or overnight_low == float('inf')):
                    # Adjust range to look back another day
                    range_start_dt = range_start_dt - timedelta(days=1)
                    session_start_dt = session_start_dt - timedelta(days=1)

                    # Search again with expanded range
                    for i in range(len(utc_timestamps)):
                        if range_start_dt <= utc_timestamps[i] < session_start_dt:
                            bars_found = True
                            overnight_high = max(overnight_high, high[i])
                            overnight_low = min(overnight_low, low[i])

                    additional_days += 1

            # If still no valid range found
            if overnight_high == float('-inf') or overnight_low == float('inf'):
                # Return the most recent high and low as a fallback
                if len(high) > 0 and len(low) > 0:
                    return high[-1], low[-1]
                else:
                    raise ValueError("Could not determine overnight range - insufficient data")

        return overnight_high, overnight_low

    @staticmethod
    def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Calculate the momentum indicator.

        Args:
            prices: Array of price values
            period: Momentum period

        Returns:
            Array with momentum values
        """
        if len(prices) <= period:
            return np.zeros(len(prices))

        momentum = np.zeros(len(prices))

        for i in range(period, len(prices)):
            momentum[i] = prices[i] - prices[i - period]

        return momentum

    @staticmethod
    def price_crosses_above(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        """
        Detect when array1 crosses above array2.

        Args:
            array1: First array of values
            array2: Second array of values

        Returns:
            Array of booleans, True when crossover occurs
        """
        if len(array1) != len(array2):
            raise ValueError("Arrays must be of equal length")

        crosses = np.zeros(len(array1), dtype=bool)

        # Cannot cross with only one value
        if len(array1) <= 1:
            return crosses

        for i in range(1, len(array1)):
            if array1[i - 1] <= array2[i - 1] and array1[i] > array2[i]:
                crosses[i] = True

        return crosses

    @staticmethod
    def price_crosses_below(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        """
        Detect when array1 crosses below array2.

        Args:
            array1: First array of values
            array2: Second array of values

        Returns:
            Array of booleans, True when crossunder occurs
        """
        if len(array1) != len(array2):
            raise ValueError("Arrays must be of equal length")

        crosses = np.zeros(len(array1), dtype=bool)

        # Cannot cross with only one value
        if len(array1) <= 1:
            return crosses

        for i in range(1, len(array1)):
            if array1[i - 1] >= array2[i - 1] and array1[i] < array2[i]:
                crosses[i] = True

        return crosses

    @staticmethod
    def is_bullish_engulfing(open_prices: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Detect bullish engulfing candlestick patterns.

        Args:
            open_prices: Array of open prices
            close: Array of close prices

        Returns:
            Array of booleans, True for bullish engulfing pattern
        """
        if len(open_prices) != len(close):
            raise ValueError("Arrays must be of equal length")

        pattern = np.zeros(len(close), dtype=bool)

        # Need at least two candles for engulfing pattern
        if len(close) <= 1:
            return pattern

        for i in range(1, len(close)):
            # Previous candle is bearish (close < open)
            # Current candle is bullish (close > open)
            # Current candle engulfs previous (open < prev close AND close > prev open)
            if (close[i - 1] < open_prices[i - 1] and  # Previous candle is bearish
                    close[i] > open_prices[i] and  # Current candle is bullish
                    open_prices[i] < close[i - 1] and  # Current open is lower than previous close
                    close[i] > open_prices[i - 1]):  # Current close is higher than previous open
                pattern[i] = True

        return pattern

    @staticmethod
    def is_bearish_engulfing(open_prices: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Detect bearish engulfing candlestick patterns.

        Args:
            open_prices: Array of open prices
            close: Array of close prices

        Returns:
            Array of booleans, True for bearish engulfing pattern
        """
        if len(open_prices) != len(close):
            raise ValueError("Arrays must be of equal length")

        pattern = np.zeros(len(close), dtype=bool)

        # Need at least two candles for engulfing pattern
        if len(close) <= 1:
            return pattern

        for i in range(1, len(close)):
            # Previous candle is bullish (close > open)
            # Current candle is bearish (close < open)
            # Current candle engulfs previous (open > prev close AND close < prev open)
            if (close[i - 1] > open_prices[i - 1] and  # Previous candle is bullish
                    close[i] < open_prices[i] and  # Current candle is bearish
                    open_prices[i] > close[i - 1] and  # Current open is higher than previous close
                    close[i] < open_prices[i - 1]):  # Current close is lower than previous open
                pattern[i] = True

        return pattern

    @staticmethod
    def average_volume(volume: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Calculate the average volume over a period.

        Args:
            volume: Array of volume values
            period: Lookback period for average

        Returns:
            Array with average volume values
        """
        return IndicatorUtils.moving_average(volume, period, MAType.SMA)

    @staticmethod
    def volume_spike(volume: np.ndarray, period: int = 20, threshold: float = 1.5) -> np.ndarray:
        """
        Detect volume spikes where current volume exceeds average by threshold factor.

        Args:
            volume: Array of volume values
            period: Lookback period for average volume
            threshold: Multiplier for average to consider as spike

        Returns:
            Array of booleans, True when volume spike occurs
        """
        if len(volume) <= period:
            return np.zeros(len(volume), dtype=bool)

        avg_volume = IndicatorUtils.average_volume(volume, period)

        # Check for volume spikes
        spikes = np.zeros(len(volume), dtype=bool)

        for i in range(period, len(volume)):
            if volume[i] > avg_volume[i] * threshold:
                spikes[i] = True

        return spikes

    @staticmethod
    def round_numbers(min_price: float, max_price: float,
                      pip_value: float = 0.0001, levels: int = 5) -> List[float]:
        """
        Generate significant round number price levels.

        Args:
            min_price: Minimum price for range
            max_price: Maximum price for range
            pip_value: Value of one pip
            levels: Number of significant levels to identify

        Returns:
            List of price levels
        """
        if min_price >= max_price:
            raise ValueError("min_price must be less than max_price")

        if pip_value <= 0:
            raise ValueError("pip_value must be positive")

        # Determine decimal places based on pip value
        if pip_value >= 0.01:  # JPY pairs
            decimal_places = 2
            round_factor = 1.0  # Look for whole numbers
        elif pip_value >= 0.0001:  # Most FX pairs
            decimal_places = 4
            round_factor = 0.01  # Look for 100 pip intervals (0.01)
        else:
            decimal_places = 5
            round_factor = 0.001  # Look for 100 pip intervals (0.001)

        # Find potential round numbers
        potential_levels = []

        # Start at a round number below min_price
        start_level = np.floor(min_price / round_factor) * round_factor

        # Generate potential levels
        current_level = start_level
        while current_level <= max_price:
            if current_level >= min_price:
                potential_levels.append(current_level)
            current_level = np.round(current_level + round_factor, decimal_places)

        # If we have more potential levels than requested, select most significant ones
        if len(potential_levels) > levels:
            # For simplicity, evenly distribute the levels
            step = len(potential_levels) // levels
            selected_levels = potential_levels[::step][:levels]
            return selected_levels
        else:
            return potential_levels

    @staticmethod
    def is_price_near_level(price: float, level: float, pips_distance: int = 10,
                            pip_value: float = 0.0001) -> bool:
        """
        Check if price is within N pips of a given level.

        Args:
            price: Current price
            level: Price level to check against
            pips_distance: Number of pips to consider as "near"
            pip_value: Value of one pip

        Returns:
            True if price is within pips_distance of level
        """
        distance_in_pips = abs(price - level) / pip_value
        return distance_in_pips <= pips_distance

    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float, is_uptrend: bool = True) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            high: High price point
            low: Low price point
            is_uptrend: True for uptrend (retracements from low to high), False for downtrend

        Returns:
            Dictionary with retracement levels
        """
        if high <= low:
            raise ValueError("high must be greater than low")

        range_size = high - low

        # Standard Fibonacci retracement levels
        levels = {
            "0.0": low if is_uptrend else high,
            "0.236": low + 0.236 * range_size if is_uptrend else high - 0.236 * range_size,
            "0.382": low + 0.382 * range_size if is_uptrend else high - 0.382 * range_size,
            "0.5": low + 0.5 * range_size if is_uptrend else high - 0.5 * range_size,
            "0.618": low + 0.618 * range_size if is_uptrend else high - 0.618 * range_size,
            "0.786": low + 0.786 * range_size if is_uptrend else high - 0.786 * range_size,
            "1.0": high if is_uptrend else low
        }

        # Add extension levels
        levels.update({
            "1.272": high + 0.272 * range_size if is_uptrend else low - 0.272 * range_size,
            "1.618": high + 0.618 * range_size if is_uptrend else low - 0.618 * range_size,
            "2.0": high + range_size if is_uptrend else low - range_size,
            "2.618": high + 1.618 * range_size if is_uptrend else low - 1.618 * range_size
        })

        return levels

    @staticmethod
    def bollinger_band_width(prices: np.ndarray, period: int = 20,
                             deviation: float = 2.0) -> np.ndarray:
        """
        Calculate Bollinger Band width as an indicator of volatility.

        Args:
            prices: Array of price values
            period: Period for moving average
            deviation: Number of standard deviations

        Returns:
            Array with Bollinger Band width values
        """
        upper, middle, lower = IndicatorUtils.bollinger_bands(prices, period, deviation)

        # Calculate width as percentage of middle band
        width = np.zeros(len(prices))

        for i in range(period - 1, len(prices)):
            if middle[i] > 0:  # Avoid division by zero
                width[i] = (upper[i] - lower[i]) / middle[i]

        return width

    @staticmethod
    def detect_bollinger_squeeze(prices: np.ndarray, period: int = 20,
                                 deviation: float = 2.0, width_period: int = 50,
                                 squeeze_percentage: float = 0.5) -> np.ndarray:
        """
        Detect when Bollinger Bands are in a squeeze (narrowing) compared to recent history.

        Args:
            prices: Array of price values
            period: Period for Bollinger Bands
            deviation: Number of standard deviations
            width_period: Period to compare current width against
            squeeze_percentage: Percentage of normal width to consider as squeeze

        Returns:
            Array of booleans, True when bands are in a squeeze
        """
        if len(prices) < period + width_period:
            return np.zeros(len(prices), dtype=bool)

        # Calculate Bollinger Band width
        bb_width = IndicatorUtils.bollinger_band_width(prices, period, deviation)

        # Detect squeeze
        squeeze = np.zeros(len(prices), dtype=bool)

        for i in range(period + width_period - 1, len(prices)):
            # Calculate the average width over the width_period
            avg_width = np.mean(bb_width[i - width_period + 1:i + 1])

            # Check if current width is below squeeze threshold
            if bb_width[i] < avg_width * squeeze_percentage:
                squeeze[i] = True

        return squeeze

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

    @staticmethod
    def detect_ma_cross(prices: np.ndarray, short_period: int, long_period: int,
                        ma_type: MAType = MAType.EMA) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect moving average crossovers.

        Args:
            prices: Array of price values
            short_period: Period for short-term MA
            long_period: Period for long-term MA
            ma_type: Type of moving average

        Returns:
            Tuple of (crosses_above, crosses_below)
        """
        # Calculate moving averages
        short_ma = IndicatorUtils.moving_average(prices, short_period, ma_type)
        long_ma = IndicatorUtils.moving_average(prices, long_period, ma_type)

        # Detect crossovers
        crosses_above = IndicatorUtils.price_crosses_above(short_ma, long_ma)
        crosses_below = IndicatorUtils.price_crosses_below(short_ma, long_ma)

        return crosses_above, crosses_below

    @staticmethod
    def detect_triple_ma_setup(prices: np.ndarray, short_period: int,
                               medium_period: int, long_period: int,
                               ma_type: MAType = MAType.EMA) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect triple moving average setups (short > medium > long for bullish, reverse for bearish).

        Args:
            prices: Array of price values
            short_period: Period for short-term MA
            medium_period: Period for medium-term MA
            long_period: Period for long-term MA
            ma_type: Type of moving average

        Returns:
            Tuple of (bullish_setups, bearish_setups)
        """
        # Calculate moving averages
        short_ma = IndicatorUtils.moving_average(prices, short_period, ma_type)
        medium_ma = IndicatorUtils.moving_average(prices, medium_period, ma_type)
        long_ma = IndicatorUtils.moving_average(prices, long_period, ma_type)

        # Detect setups
        bullish_setups = np.zeros(len(prices), dtype=bool)
        bearish_setups = np.zeros(len(prices), dtype=bool)

        min_period = max(short_period, medium_period, long_period)

        for i in range(min_period, len(prices)):
            # Bullish: short > medium > long
            if short_ma[i] > medium_ma[i] and medium_ma[i] > long_ma[i]:
                bullish_setups[i] = True

            # Bearish: short < medium < long
            if short_ma[i] < medium_ma[i] and medium_ma[i] < long_ma[i]:
                bearish_setups[i] = True

        return bullish_setups, bearish_setups

    @staticmethod
    def detect_ichimoku_signals(ichimoku_data: Dict[str, np.ndarray],
                                close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect Ichimoku bullish and bearish signals.

        Args:
            ichimoku_data: Dictionary with Ichimoku components from ichimoku() function
            close: Array of close prices

        Returns:
            Tuple of (bullish_signals, bearish_signals)
        """
        if len(close) < 1:
            return np.zeros(len(close), dtype=bool), np.zeros(len(close), dtype=bool)

        tenkan = ichimoku_data['tenkan_sen']
        kijun = ichimoku_data['kijun_sen']
        senkou_a = ichimoku_data['senkou_span_a']
        senkou_b = ichimoku_data['senkou_span_b']

        # Detect TK cross
        tk_bull_cross = IndicatorUtils.price_crosses_above(tenkan, kijun)
        tk_bear_cross = IndicatorUtils.price_crosses_below(tenkan, kijun)

        # Check price position relative to the cloud
        bullish_signals = np.zeros(len(close), dtype=bool)
        bearish_signals = np.zeros(len(close), dtype=bool)

        for i in range(26, len(close)):  # Start after cloud displacement
            # Define the cloud top and bottom
            cloud_top = max(senkou_a[i], senkou_b[i])
            cloud_bottom = min(senkou_a[i], senkou_b[i])

            # Bullish signal: TK Cross + Price above cloud + Cloud is green (A > B)
            if (tk_bull_cross[i] and
                    close[i] > cloud_top and
                    senkou_a[i] > senkou_b[i]):
                bullish_signals[i] = True

            # Bearish signal: TK Cross + Price below cloud + Cloud is red (B > A)
            if (tk_bear_cross[i] and
                    close[i] < cloud_bottom and
                    senkou_b[i] > senkou_a[i]):
                bearish_signals[i] = True

        return bullish_signals, bearish_signals