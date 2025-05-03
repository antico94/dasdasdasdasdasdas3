from datetime import datetime
from typing import Dict, Set, Any

from Config.trading_config import TimeFrame
from Logger.logger import DBLogger


class TimeframeManager:
    """Manages multi-timeframe dependencies and synchronization"""

    def __init__(self, logger: DBLogger = None):
        """
        Initialize the timeframe manager.

        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.timeframe_dependencies = {}  # {strategy_name: {timeframe: set(dependency_timeframes)}}
        self.last_update_time = {}  # {(strategy_name, timeframe): timestamp}

    def register_strategy_timeframes(self, strategy_name: str,
                                     primary_timeframe: TimeFrame,
                                     dependent_timeframes: Set[TimeFrame] = None):
        """
        Register a strategy's timeframe dependencies.

        Args:
            strategy_name: Name of the strategy
            primary_timeframe: The primary timeframe the strategy operates on
            dependent_timeframes: Set of timeframes this strategy depends on
        """
        if strategy_name not in self.timeframe_dependencies:
            self.timeframe_dependencies[strategy_name] = {}

        # Create the dependency entry
        self.timeframe_dependencies[strategy_name][primary_timeframe] = \
            dependent_timeframes or set()

        if self.logger:
            self.logger.log_event(
                level="INFO",
                message=f"Registered timeframe dependencies for {strategy_name}",
                event_type="TIMEFRAME_MANAGER",
                component="timeframe_manager",
                action="register_strategy_timeframes",
                status="success",
                details={
                    "strategy_name": strategy_name,
                    "primary_timeframe": primary_timeframe.name,
                    "dependent_timeframes": [tf.name for tf in (dependent_timeframes or set())]
                }
            )

    def check_timeframe_ready(self, strategy_name: str, timeframe: TimeFrame) -> bool:
        """
        Check if all dependent timeframes for a strategy have been updated.

        Args:
            strategy_name: Name of the strategy
            timeframe: The timeframe to check

        Returns:
            True if all dependencies are updated, False otherwise
        """
        # If strategy or timeframe not registered, assume it's ready
        if strategy_name not in self.timeframe_dependencies or \
                timeframe not in self.timeframe_dependencies[strategy_name]:
            return True

        # Get the dependencies
        dependencies = self.timeframe_dependencies[strategy_name][timeframe]

        # If no dependencies, it's ready
        if not dependencies:
            return True

        # Check if all dependencies have been updated
        for dep_timeframe in dependencies:
            # Get the last update times
            primary_update_time = self.last_update_time.get((strategy_name, timeframe))
            dep_update_time = self.last_update_time.get((strategy_name, dep_timeframe))

            # If either is missing, not ready
            if primary_update_time is None or dep_update_time is None:
                return False

            # If dependent timeframe is older than primary, not ready
            if dep_update_time < primary_update_time:
                return False

        # All dependencies are up to date
        return True

    def mark_timeframe_updated(self, strategy_name: str, timeframe: TimeFrame,
                               timestamp: datetime = None):
        """
        Mark a timeframe as updated for a strategy.

        Args:
            strategy_name: Name of the strategy
            timeframe: The timeframe that was updated
            timestamp: The timestamp of the update (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.last_update_time[(strategy_name, timeframe)] = timestamp

        if self.logger:
            self.logger.log_event(
                level="DEBUG",
                message=f"Marked timeframe {timeframe.name} as updated for {strategy_name}",
                event_type="TIMEFRAME_MANAGER",
                component="timeframe_manager",
                action="mark_timeframe_updated",
                status="success",
                details={
                    "strategy_name": strategy_name,
                    "timeframe": timeframe.name,
                    "timestamp": str(timestamp)
                }
            )

    def get_update_status(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get the update status for all timeframes of a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with timeframe update status
        """
        if strategy_name not in self.timeframe_dependencies:
            return {}

        status = {}

        for timeframe in self.timeframe_dependencies[strategy_name]:
            last_update = self.last_update_time.get((strategy_name, timeframe))
            dependencies = self.timeframe_dependencies[strategy_name][timeframe]

            dep_status = {}
            for dep in dependencies:
                dep_update = self.last_update_time.get((strategy_name, dep))
                dep_status[dep.name] = str(dep_update) if dep_update else "Never"

            status[timeframe.name] = {
                "last_update": str(last_update) if last_update else "Never",
                "dependencies": dep_status,
                "ready": self.check_timeframe_ready(strategy_name, timeframe)
            }

        return status