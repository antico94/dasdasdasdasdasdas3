# Tests/test_timeframe_manager.py
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from Config.trading_config import TimeFrame
from Events.event_bus import EventBus
from Logger.logger import DBLogger
from Strategies.timeframe_manager import TimeframeManager


class TestTimeframeManager(unittest.TestCase):
    """Test cases for the TimeframeManager class"""

    @patch('Logger.logger.DBLogger')
    @patch('Events.event_bus.EventBus')
    def setUp(self, mock_event_bus, mock_logger):
        # Create mocks
        self.mock_logger = mock_logger.return_value
        self.mock_event_bus = mock_event_bus.return_value

        # Create the timeframe manager
        self.timeframe_manager = TimeframeManager(
            logger=self.mock_logger,
            event_bus=self.mock_event_bus
        )

        # Reset singleton for clean testing
        TimeframeManager._instance = None

        # Define test data
        self.strategy_name = "test_strategy"
        self.M1 = TimeFrame.M1
        self.M5 = TimeFrame.M5
        self.M15 = TimeFrame.M15
        self.H1 = TimeFrame.H1

        # Current time for testing
        self.current_time = datetime.now()

    # Tests/test_timeframe_manager.py (continued)
    def test_register_strategy_timeframes(self):
        """Test registering timeframe dependencies"""
        # Register a strategy with M1 as primary and M5, M15 as dependencies
        dependent_timeframes = {self.M5, self.M15}
        self.timeframe_manager.register_strategy_timeframes(
            self.strategy_name,
            self.M1,
            dependent_timeframes
        )

        # Check if dependencies were registered correctly
        self.assertIn(self.strategy_name, self.timeframe_manager.timeframe_dependencies)
        self.assertIn(self.M1, self.timeframe_manager.timeframe_dependencies[self.strategy_name])

        # Check that all dependencies are in the set
        registered_deps = self.timeframe_manager.timeframe_dependencies[self.strategy_name][self.M1]
        self.assertEqual(len(registered_deps), 3)  # M1, M5, M15
        self.assertIn(self.M1, registered_deps)
        self.assertIn(self.M5, registered_deps)
        self.assertIn(self.M15, registered_deps)

    def test_mark_timeframe_updated(self):
        """Test marking a timeframe as updated"""
        # Register a strategy first
        self.timeframe_manager.register_strategy_timeframes(
            self.strategy_name,
            self.M1,
            {self.M5}
        )

        # Mark the timeframe as updated
        test_time = self.current_time
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M1,
            test_time
        )

        # Check if the update was recorded
        key = (self.strategy_name, self.M1)
        self.assertIn(key, self.timeframe_manager.last_update_time)
        self.assertEqual(self.timeframe_manager.last_update_time[key], test_time)

    def test_check_timeframe_ready_no_dependencies(self):
        """Test checking if a timeframe is ready when it has no dependencies"""
        # Register a strategy with just one timeframe
        self.timeframe_manager.register_strategy_timeframes(
            self.strategy_name,
            self.M1,
            None  # No dependencies
        )

        # Mark the timeframe as updated
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M1,
            self.current_time
        )

        # Check if it's ready
        self.assertTrue(
            self.timeframe_manager.check_timeframe_ready(self.strategy_name, self.M1)
        )

    def test_check_timeframe_ready_with_dependencies(self):
        """Test checking if a timeframe is ready when it has dependencies"""
        # Register a strategy with dependencies
        self.timeframe_manager.register_strategy_timeframes(
            self.strategy_name,
            self.M15,
            {self.M1, self.M5}
        )

        # Initially, no updates have happened
        self.assertFalse(
            self.timeframe_manager.check_timeframe_ready(self.strategy_name, self.M15)
        )

        # Update M15 only
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M15,
            self.current_time
        )

        # Should still not be ready (missing M1 and M5)
        self.assertFalse(
            self.timeframe_manager.check_timeframe_ready(self.strategy_name, self.M15)
        )

        # Update M1
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M1,
            self.current_time
        )

        # Still not ready (missing M5)
        self.assertFalse(
            self.timeframe_manager.check_timeframe_ready(self.strategy_name, self.M15)
        )

        # Update M5
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M5,
            self.current_time
        )

        # Now it should be ready
        self.assertTrue(
            self.timeframe_manager.check_timeframe_ready(self.strategy_name, self.M15)
        )

    def test_check_timeframe_ready_older_dependency(self):
        """Test that a dependency must be as new or newer than the primary timeframe"""
        # Register a strategy with dependencies
        self.timeframe_manager.register_strategy_timeframes(
            self.strategy_name,
            self.H1,
            {self.M5}
        )

        # Update H1 with current time
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.H1,
            self.current_time
        )

        # Update M5 with older time
        older_time = self.current_time - timedelta(minutes=10)
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M5,
            older_time
        )

        # Should not be ready because M5 is older than H1
        self.assertFalse(
            self.timeframe_manager.check_timeframe_ready(self.strategy_name, self.H1)
        )

        # Update M5 with newer time
        newer_time = self.current_time + timedelta(minutes=10)
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M5,
            newer_time
        )

        # Now it should be ready
        self.assertTrue(
            self.timeframe_manager.check_timeframe_ready(self.strategy_name, self.H1)
        )

    def test_get_update_status(self):
        """Test getting the update status for a strategy"""
        # Register a strategy with dependencies
        self.timeframe_manager.register_strategy_timeframes(
            self.strategy_name,
            self.H1,
            {self.M15, self.M5, self.M1}
        )

        # Update some timeframes
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.H1,
            self.current_time
        )

        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M15,
            self.current_time
        )

        # Get status
        status = self.timeframe_manager.get_update_status(self.strategy_name)

        # Check status format and content
        self.assertIn(self.H1.name, status)
        self.assertIn('last_update', status[self.H1.name])
        self.assertIn('dependencies', status[self.H1.name])
        self.assertIn('ready', status[self.H1.name])

        # Verify dependency status
        dependencies = status[self.H1.name]['dependencies']
        self.assertIn(self.M1.name, dependencies)
        self.assertIn(self.M5.name, dependencies)
        self.assertIn(self.M15.name, dependencies)

        # H1 should not be ready (missing M1 and M5 updates)
        self.assertFalse(status[self.H1.name]['ready'])

    def test_clear_strategy(self):
        """Test clearing all timeframe info for a strategy"""
        # Register a strategy with dependencies
        self.timeframe_manager.register_strategy_timeframes(
            self.strategy_name,
            self.H1,
            {self.M15}
        )

        # Add some updates
        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.H1,
            self.current_time
        )

        self.timeframe_manager.mark_timeframe_updated(
            self.strategy_name,
            self.M15,
            self.current_time
        )

        # Clear the strategy
        self.timeframe_manager.clear_strategy(self.strategy_name)

        # Check that all data was cleared
        self.assertNotIn(self.strategy_name, self.timeframe_manager.timeframe_dependencies)

        # Check that update times were cleared
        h1_key = (self.strategy_name, self.H1)
        m15_key = (self.strategy_name, self.M15)
        self.assertNotIn(h1_key, self.timeframe_manager.last_update_time)
        self.assertNotIn(m15_key, self.timeframe_manager.last_update_time)

    def test_on_new_bar(self):
        """Test handling new bar events"""
        # Register a strategy with dependencies
        self.timeframe_manager.register_strategy_timeframes(
            self.strategy_name,
            self.H1,
            {self.M15}
        )

        # Create a mock NewBarEvent
        mock_event = MagicMock()
        mock_event.symbol = "EURUSD"
        mock_event.timeframe_name = "M15"
        mock_event.timestamp = self.current_time

        # Process the event
        self.timeframe_manager._on_new_bar(mock_event)

        # Check if the timeframe was marked as updated
        key = (self.strategy_name, self.M15)
        self.assertIn(key, self.timeframe_manager.last_update_time)
        self.assertEqual(self.timeframe_manager.last_update_time[key], self.current_time)