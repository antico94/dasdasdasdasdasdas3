# Strategies/test_indicator_utils.py
import unittest
import numpy as np
from datetime import datetime, time, timezone
from Strategies.indicator_utils import IndicatorUtils, MAType, TradingSession


class TestIndicatorUtils(unittest.TestCase):

    def test_moving_average_sma(self):
        # Test Simple Moving Average
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = IndicatorUtils.moving_average(prices, 3, MAType.SMA)

        # Expected: zeros for first 2 points, then 3-point averages
        expected = np.array([0.0, 0.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(result, expected)

    def test_moving_average_ema(self):
        # Test Exponential Moving Average
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = IndicatorUtils.moving_average(prices, 3, MAType.EMA)

        # For EMA with alpha = 2/(3+1) = 0.5
        # First value at index 2 is SMA = 2.0
        # EMA[3] = 0.5 * 4.0 + 0.5 * 2.0 = 3.0
        # EMA[4] = 0.5 * 5.0 + 0.5 * 3.0 = 4.0
        expected = np.array([0.0, 0.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(result, expected)

    def test_macd(self):
        # Simple test case for MACD
        prices = np.ones(30) * 10.0  # Constant price, MACD should be 0
        macd, signal, hist = IndicatorUtils.macd(prices, 12, 26, 9)

        # All values should be close to 0
        self.assertTrue(np.all(np.abs(macd[-10:]) < 1e-10))
        self.assertTrue(np.all(np.abs(signal[-5:]) < 1e-10))
        self.assertTrue(np.all(np.abs(hist[-5:]) < 1e-10))

    def test_is_in_session(self):
        # Test London session
        london_time = datetime(2023, 5, 1, 10, 0, tzinfo=timezone.utc)  # 10:00 UTC, should be in London session
        result = IndicatorUtils.is_in_session(london_time, TradingSession.LONDON)
        self.assertTrue(result)

        non_london_time = datetime(2023, 5, 1, 6, 0, tzinfo=timezone.utc)  # 6:00 UTC, should not be in London session
        result = IndicatorUtils.is_in_session(non_london_time, TradingSession.LONDON)
        self.assertFalse(result)

        # Test session that crosses midnight
        sydney_time = datetime(2023, 5, 1, 22, 0, tzinfo=timezone.utc)  # 22:00 UTC, should be in Sydney session
        result = IndicatorUtils.is_in_session(sydney_time, TradingSession.SYDNEY)
        self.assertTrue(result)

        sydney_time_2 = datetime(2023, 5, 1, 4, 0, tzinfo=timezone.utc)  # 4:00 UTC, should be in Sydney session
        result = IndicatorUtils.is_in_session(sydney_time_2, TradingSession.SYDNEY)
        self.assertTrue(result)

        non_sydney_time = datetime(2023, 5, 1, 10, 0, tzinfo=timezone.utc)  # 10:00 UTC, should not be in Sydney session
        result = IndicatorUtils.is_in_session(non_sydney_time, TradingSession.SYDNEY)
        self.assertFalse(result)

    def test_price_crosses_above(self):
        array1 = np.array([1.0, 2.0, 3.0, 2.0, 4.0])
        array2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

        result = IndicatorUtils.price_crosses_above(array1, array2)
        expected = np.array([False, False, True, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_price_crosses_below(self):
        array1 = np.array([3.0, 2.0, 1.0, 2.0, 1.0])
        array2 = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

        result = IndicatorUtils.price_crosses_below(array1, array2)
        expected = np.array([False, False, True, False, True])

        np.testing.assert_array_equal(result, expected)

    def test_bollinger_band_width(self):
        # Constant prices should have constant width
        prices = np.ones(30) * 100.0
        width = IndicatorUtils.bollinger_band_width(prices, 20, 2.0)

        # Should be close to zero since std dev is zero
        self.assertTrue(np.all(width[20:] < 1e-10))

        # Increasing volatility should increase width
        prices = np.concatenate([np.ones(20) * 100.0, np.array([100.0, 110.0, 90.0, 120.0, 80.0])])
        width = IndicatorUtils.bollinger_band_width(prices, 5, 2.0)

        # Width should increase with volatility
        self.assertTrue(width[-1] > width[5])

    def test_round_numbers(self):
        # Test with EURUSD-like pair
        levels = IndicatorUtils.round_numbers(1.2000, 1.3000, 0.0001, 5)

        # Should return round numbers at 100 pip intervals
        self.assertEqual(len(levels), 5)
        self.assertTrue(1.20 in levels)
        self.assertTrue(1.21 in levels)

        # Test with USDJPY-like pair
        levels = IndicatorUtils.round_numbers(100.00, 110.00, 0.01, 5)

        # Should return round numbers
        self.assertEqual(len(levels), 5)
        self.assertTrue(100.0 in levels)
        self.assertTrue(102.0 in levels or 105.0 in levels)

    def test_fibonacci_levels(self):
        # Test uptrend
        fib_up = IndicatorUtils.calculate_fibonacci_levels(1.5000, 1.4000, True)

        # 0% should be at low point
        self.assertAlmostEqual(fib_up["0.0"], 1.4000)
        # 100% should be at high point
        self.assertAlmostEqual(fib_up["1.0"], 1.5000)
        # 50% should be midpoint
        self.assertAlmostEqual(fib_up["0.5"], 1.4500)

        # Test downtrend
        fib_down = IndicatorUtils.calculate_fibonacci_levels(1.5000, 1.4000, False)

        # 0% should be at high point
        self.assertAlmostEqual(fib_down["0.0"], 1.5000)
        # 100% should be at low point
        self.assertAlmostEqual(fib_down["1.0"], 1.4000)
        # 50% should be midpoint
        self.assertAlmostEqual(fib_down["0.5"], 1.4500)


if __name__ == "__main__":
    unittest.main()