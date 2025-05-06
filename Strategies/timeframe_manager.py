# Strategies/timeframe_manager.py
import threading
from datetime import datetime
from typing import Dict, Set, Any, Optional

from Config.trading_config import TimeFrame
from Events.event_bus import EventBus
from Logger.logger import DBLogger


class TimeframeManager:
    """
    Manages multi-timeframe dependencies and synchronization.

    This class ensures that strategies only execute when all required
    timeframes have been updated with the latest data. It prevents
    strategies from running with incomplete or outdated data.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TimeframeManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, logger: Optional[DBLogger] = None, event_bus: Optional[EventBus] = None):
        """
        Initialize the timeframe manager.

        Args:
            logger: Logger instance for recording events
            event_bus: Event bus for publishing/subscribing to events
        """
        with self._lock:
            if self._initialized:
                return

            # Create connection string from credentials
            from Config.credentials import (
                SQL_SERVER,
                SQL_DATABASE,
                SQL_DRIVER,
                USE_WINDOWS_AUTH,
                SQL_USERNAME,
                SQL_PASSWORD
            )

            # Create connection string
            if USE_WINDOWS_AUTH:
                conn_string = f"DRIVER={{{SQL_DRIVER}}};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};Trusted_Connection=yes;"
            else:
                conn_string = f"DRIVER={{{SQL_DRIVER}}};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};UID={SQL_USERNAME};PWD={SQL_PASSWORD}"

            self.logger = logger or DBLogger(
                conn_string=conn_string,
                enabled_levels={'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
                console_output=True
            )

            self.event_bus = event_bus or EventBus()

            # Strategy-specific timeframe dependencies
            # {strategy_name: {timeframe: set(dependency_timeframes)}}
            self.timeframe_dependencies = {}

            # Timestamp of the last update for each (strategy, timeframe)
            # {(strategy_name, timeframe): timestamp}
            self.last_update_time = {}

            # Register listeners for timeframe updates
            self._register_event_listeners()

            self._initialized = True

            self.logger.log_event(
                level="INFO",
                message="TimeframeManager initialized",
                event_type="TIMEFRAME_MANAGER",
                component="timeframe_manager",
                action="__init__",
                status="success"
            )

    def _register_event_listeners(self):
        """Register event listeners for timeframe updates"""
        from Events.events import NewBarEvent

        # Subscribe to new bar events to track timeframe updates
        self.event_bus.subscribe(NewBarEvent, self._on_new_bar)

        self.logger.log_event(
            level="INFO",
            message="Registered event listeners for timeframe updates",
            event_type="TIMEFRAME_MANAGER",
            component="timeframe_manager",
            action="_register_event_listeners",
            status="success"
        )

    def _on_new_bar(self, event):
        """
        Handle new bar events to track timeframe updates.

        Args:
            event: The new bar event containing timeframe information
        """
        try:
            # Extract the relevant information from the event
            symbol = event.symbol
            timeframe_name = event.timeframe_name
            timestamp = event.timestamp

            # Find the corresponding TimeFrame enum
            timeframe = None
            for tf in TimeFrame:
                if tf.name == timeframe_name:
                    timeframe = tf
                    break

            if not timeframe:
                self.logger.log_event(
                    level="WARNING",
                    message=f"Unknown timeframe name in event: {timeframe_name}",
                    event_type="TIMEFRAME_MANAGER",
                    component="timeframe_manager",
                    action="_on_new_bar",
                    status="warning",
                    details={"symbol": symbol, "timeframe_name": timeframe_name}
                )
                return

            # Update all strategies that use this symbol and timeframe
            for strategy_name, tf_dependencies in self.timeframe_dependencies.items():
                if timeframe in tf_dependencies:
                    # Update the last update time for this strategy and timeframe
                    self.mark_timeframe_updated(strategy_name, timeframe, timestamp)

                    self.logger.log_event(
                        level="DEBUG",
                        message=f"Updated timeframe {timeframe.name} for strategy {strategy_name}",
                        event_type="TIMEFRAME_MANAGER",
                        component="timeframe_manager",
                        action="_on_new_bar",
                        status="success",
                        details={
                            "strategy_name": strategy_name,
                            "timeframe": timeframe.name,
                            "symbol": symbol,
                            "timestamp": str(timestamp)
                        }
                    )

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error handling new bar event: {str(e)}",
                exception_type=type(e).__name__,
                function="_on_new_bar",
                traceback=str(e),
                context={
                    "event_type": type(event).__name__,
                    "symbol": getattr(event, "symbol", None),
                    "timeframe_name": getattr(event, "timeframe_name", None)
                }
            )

    def register_strategy_timeframes(self, strategy_name: str,
                                     primary_timeframe: TimeFrame,
                                     dependent_timeframes: Optional[Set[TimeFrame]] = None):
        """
        Register a strategy's timeframe dependencies.

        Args:
            strategy_name: Name of the strategy
            primary_timeframe: The primary timeframe the strategy operates on
            dependent_timeframes: Set of timeframes this strategy depends on
        """
        if not strategy_name:
            raise ValueError("Strategy name cannot be empty")

        if not primary_timeframe:
            raise ValueError("Primary timeframe cannot be None")

        if strategy_name not in self.timeframe_dependencies:
            self.timeframe_dependencies[strategy_name] = {}

        # Ensure dependent_timeframes is a set (or empty set if None)
        if dependent_timeframes is None:
            dependent_timeframes = set()
        elif not isinstance(dependent_timeframes, set):
            raise ValueError("dependent_timeframes must be a set or None")

        # Add the primary timeframe as a dependency for itself
        all_dependencies = dependent_timeframes.copy()
        all_dependencies.add(primary_timeframe)

        # Create the dependency entry
        self.timeframe_dependencies[strategy_name][primary_timeframe] = all_dependencies

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
                "dependent_timeframes": [tf.name for tf in dependent_timeframes]
            }
        )

    def check_timeframe_ready(self, strategy_name: str, timeframe: TimeFrame) -> bool:
        """
        Check if all dependent timeframes for a strategy have been updated.
        Enhanced with additional logging.

        Args:
            strategy_name: Name of the strategy
            timeframe: The timeframe to check

        Returns:
            True if all dependencies are updated, False otherwise
        """
        try:
            # If strategy or timeframe not registered, assume it's not ready
            if strategy_name not in self.timeframe_dependencies:
                self.logger.log_event(
                    level="ERROR",  # Changed from WARNING to ERROR
                    message=f"Strategy {strategy_name} not registered in TimeframeManager",
                    event_type="TIMEFRAME_MANAGER",
                    component="timeframe_manager",
                    action="check_timeframe_ready",
                    status="not_registered",
                    details={"registered_strategies": list(self.timeframe_dependencies.keys())}
                )
                return False

            if timeframe not in self.timeframe_dependencies[strategy_name]:
                self.logger.log_event(
                    level="ERROR",  # Changed from WARNING to ERROR
                    message=f"Timeframe {timeframe.name} not registered for strategy {strategy_name}",
                    event_type="TIMEFRAME_MANAGER",
                    component="timeframe_manager",
                    action="check_timeframe_ready",
                    status="not_registered",
                    details={
                        "strategy_name": strategy_name,
                        "timeframe": timeframe.name,
                        "registered_timeframes": [tf.name for tf in self.timeframe_dependencies[strategy_name].keys()]
                    }
                )
                return False

            # Get the dependencies
            dependencies = self.timeframe_dependencies[strategy_name][timeframe]

            # If no dependencies, it's ready
            if not dependencies:
                return True

            # Check if all dependencies have been updated
            missing_deps = []
            for dep_timeframe in dependencies:
                # Get the last update times
                primary_update_time = self.last_update_time.get((strategy_name, timeframe))
                dep_update_time = self.last_update_time.get((strategy_name, dep_timeframe))

                # If either is missing, not ready
                if primary_update_time is None:
                    self.logger.log_event(
                        level="INFO",  # Changed from DEBUG to INFO
                        message=f"Primary timeframe {timeframe.name} not yet updated for {strategy_name}",
                        event_type="TIMEFRAME_MANAGER",
                        component="timeframe_manager",
                        action="check_timeframe_ready",
                        status="not_ready",
                        details={
                            "strategy_name": strategy_name,
                            "timeframe": timeframe.name,
                            "dependency_timeframes": [tf.name for tf in dependencies],
                            "update_times": {k[1].name: str(v) for k, v in self.last_update_time.items() if
                                             k[0] == strategy_name}
                        }
                    )
                    return False

                if dep_update_time is None:
                    missing_deps.append(dep_timeframe.name)
                    self.logger.log_event(
                        level="INFO",  # Changed from DEBUG to INFO
                        message=f"Dependency {dep_timeframe.name} not yet updated for {strategy_name}",
                        event_type="TIMEFRAME_MANAGER",
                        component="timeframe_manager",
                        action="check_timeframe_ready",
                        status="not_ready",
                        details={
                            "strategy_name": strategy_name,
                            "timeframe": timeframe.name,
                            "dependency": dep_timeframe.name,
                            "primary_update_time": str(primary_update_time)
                        }
                    )
                    continue  # Continue checking other dependencies

                # Dependent timeframe must be newer or equal to primary
                if dep_update_time < primary_update_time:
                    missing_deps.append(dep_timeframe.name)
                    self.logger.log_event(
                        level="INFO",  # Changed from DEBUG to INFO
                        message=f"Dependency {dep_timeframe.name} is older than primary {timeframe.name} for {strategy_name}",
                        event_type="TIMEFRAME_MANAGER",
                        component="timeframe_manager",
                        action="check_timeframe_ready",
                        status="not_ready",
                        details={
                            "strategy_name": strategy_name,
                            "timeframe": timeframe.name,
                            "dependency": dep_timeframe.name,
                            "primary_time": str(primary_update_time),
                            "dependency_time": str(dep_update_time)
                        }
                    )

            # If any dependencies are missing or outdated, return false
            if missing_deps:
                return False

            # All dependencies are up to date
            self.logger.log_event(
                level="INFO",  # Changed from DEBUG to INFO
                message=f"All dependencies are up to date for {strategy_name} on {timeframe.name}",
                event_type="TIMEFRAME_MANAGER",
                component="timeframe_manager",
                action="check_timeframe_ready",
                status="ready",
                details={
                    "strategy_name": strategy_name,
                    "timeframe": timeframe.name,
                    "dependencies": [tf.name for tf in dependencies],
                    "update_time": str(self.last_update_time.get((strategy_name, timeframe)))
                }
            )
            return True

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error checking if timeframe is ready: {str(e)}",
                exception_type=type(e).__name__,
                function="check_timeframe_ready",
                traceback=str(e),
                context={"strategy_name": strategy_name, "timeframe": timeframe.name if timeframe else None}
            )
            # When in doubt, assume not ready to avoid trading with incomplete data
            return False

    def mark_timeframe_updated(self, strategy_name: str, timeframe: TimeFrame,
                               timestamp: Optional[datetime] = None):
        """
        Mark a timeframe as updated for a strategy.

        Args:
            strategy_name: Name of the strategy
            timeframe: The timeframe that was updated
            timestamp: The timestamp of the update (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Validate inputs
        if not strategy_name:
            raise ValueError("Strategy name cannot be empty")

        if not timeframe:
            raise ValueError("Timeframe cannot be None")

        self.last_update_time[(strategy_name, timeframe)] = timestamp

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
        if not strategy_name:
            raise ValueError("Strategy name cannot be empty")

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

    def clear_strategy(self, strategy_name: str):
        """
        Clear all timeframe information for a strategy.

        Args:
            strategy_name: Name of the strategy to clear
        """
        if not strategy_name:
            raise ValueError("Strategy name cannot be empty")

        # Remove from dependencies
        if strategy_name in self.timeframe_dependencies:
            del self.timeframe_dependencies[strategy_name]

        # Remove from last update times
        keys_to_remove = []
        for key in self.last_update_time:
            if key[0] == strategy_name:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.last_update_time[key]

        self.logger.log_event(
            level="INFO",
            message=f"Cleared timeframe information for strategy {strategy_name}",
            event_type="TIMEFRAME_MANAGER",
            component="timeframe_manager",
            action="clear_strategy",
            status="success",
            details={"strategy_name": strategy_name}
        )