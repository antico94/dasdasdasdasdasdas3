# tests/test_base_strategy.py

import unittest
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

from Config.trading_config import TimeFrame
from Database.models import PriceBar
from Events.events import SignalEvent
from Strategies.base_strategy import BaseStrategy


class TestStrategy(BaseStrategy):
    """Test strategy implementation for unit testing"""

    def __init__(self, name, symbol, timeframes, logger=None):
        super().__init__(name, symbol, timeframes, logger)
        self.signals_called = 0
        self.mock_signal = None

    def calculate_signals(self, timeframe: TimeFrame) -> Optional[SignalEvent]:
        self.signals_called += 1
        return self.mock_signal


class TestBaseStrategy(unittest.TestCase):
    """Test the BaseStrategy class enhancements"""

    def setUp(self):
        """Set up the test"""
        self.logger = MagicMock()
        self.timeframe = TimeFrame.M1
        self.timeframes = {self.timeframe}
        self.strategy = TestStrategy("test_strategy", "EURUSD", self.timeframes, self.logger)
        self.strategy.set_instrument_info(1, "EURUSD")
        self.strategy.set_timeframe_ids({self.timeframe: 1})
        self.strategy.mock_signal = None

    def create_bars(self, count, interval_minutes=1, start_time=None):
        """Create test bars"""
        if start_time is None:
            start_time = datetime.now() - timedelta(minutes=count * interval_minutes)

        bars = []
        for i in range(count):
            timestamp = start_time + timedelta(minutes=i * interval_minutes)
            bar = PriceBar(
                instrument_id=1,
                timeframe_id=1,
                timestamp=timestamp,
                open=1.0 + i * 0.01,
                high=1.0 + i * 0.01 + 0.005,
                low=1.0 + i * 0.01 - 0.005,
                close=1.0 + i * 0.01 + 0.002,
                volume=100.0 + i
            )
            bars.append(bar)
        return bars

    def test_on_bar_insufficient_bars(self):
        """Test on_bar with insufficient bars"""
        # Only one bar (should be rejected)
        bars = self.create_bars(1)
        result = self.strategy.on_bar(self.timeframe, bars)
        self.assertIsNone(result)
        self.assertEqual(self.strategy.signals_called, 0)

    def test_on_bar_with_completed_bars(self):
        """Test on_bar with completed bars"""
        # Two bars (one completed, one forming)
        bars = self.create_bars(2)
        result = self.strategy.on_bar(self.timeframe, bars)
        self.assertIsNone(result)  # No signal set
        self.assertEqual(self.strategy.signals_called, 1)

        # Verify the last processed timestamp is set
        self.assertEqual(self.strategy.last_processed_timestamps[self.timeframe], bars[-2].timestamp)

        # Verify completed bars are stored (excluding forming bar)
        self.assertEqual(len(self.strategy.completed_bars[self.timeframe]), 1)
        self.assertEqual(self.strategy.completed_bars[self.timeframe][0].timestamp, bars[0].timestamp)

    def test_signal_throttling(self):
        """Test signal throttling mechanism"""
        # Set up mock signal
        mock_signal = MagicMock(spec=SignalEvent)
        self.strategy.mock_signal = mock_signal

        # Two bars (one completed, one forming)
        bars = self.create_bars(2)

        # First call should generate a signal
        result1 = self.strategy.on_bar(self.timeframe, bars)
        self.assertEqual(result1, mock_signal)
        self.assertEqual(self.strategy.signals_called, 1)

        # Second call with same bars should be throttled
        self.strategy.signals_called = 0
        result2 = self.strategy.on_bar(self.timeframe, bars)
        self.assertIsNone(result2)
        self.assertEqual(self.strategy.signals_called, 0)  # calculate_signals not called due to throttling

        # New bar should not be throttled
        new_bars = self.create_bars(3)  # Add one more bar
        self.strategy.signals_called = 0
        result3 = self.strategy.on_bar(self.timeframe, new_bars)
        self.assertEqual(result3, mock_signal)
        self.assertEqual(self.strategy.signals_called, 1)

    def test_validate_bars(self):
        """Test bars validation"""
        # Not enough bars
        self.strategy.completed_bars[self.timeframe] = self.create_bars(2)
        self.assertFalse(self.strategy.validate_bars(self.timeframe, 3))

        # Enough bars
        self.strategy.completed_bars[self.timeframe] = self.create_bars(5)
        self.assertTrue(self.strategy.validate_bars(self.timeframe, 3))

        # Future bars
        future_bars = self.create_bars(3, start_time=datetime.now() + timedelta(minutes=10))
        self.strategy.completed_bars[self.timeframe] = future_bars
        self.assertFalse(self.strategy.validate_bars(self.timeframe, 3))


if __name__ == '__main__':
    unittest.main()