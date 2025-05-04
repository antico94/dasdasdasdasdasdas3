# Strategies/strategy_manager.py
import threading
from typing import List, Set, Type, Optional, Dict, Any

from Config.trading_config import TimeFrame
from Database.db_manager import DatabaseManager
from Events.event_bus import EventBus
from Events.events import NewBarEvent
from Logger.logger import DBLogger
from Strategies.base_strategy import BaseStrategy
from Strategies.timeframe_manager import TimeframeManager


class StrategyManager:
    """
    Manages trading strategies and routes events to them.

    This class:
    1. Registers and manages strategy instances
    2. Subscribes to relevant events
    3. Routes events to appropriate strategies
    4. Processes signals from strategies
    """

    def __init__(self, event_bus: EventBus, db_manager: DatabaseManager, logger: DBLogger):
        """
        Initialize the strategy manager.

        Args:
            event_bus: Event bus instance
            db_manager: Database manager instance
            logger: Logger instance
        """
        self.event_bus = event_bus
        self.db_manager = db_manager
        self.logger = logger

        # Strategy instances by symbol and name
        self.strategies = {}  # {symbol: {strategy_name: strategy_instance}}

        # Initialize the timeframe manager
        self.timeframe_manager = TimeframeManager(logger=logger, event_bus=event_bus)

        # Timeframe mappings
        self.timeframe_ids = {}  # {TimeFrame: id}
        self.timeframe_by_id = {}  # {id: TimeFrame}

        # Instrument mappings
        self.instrument_ids = {}  # {symbol: id}

        # Lock for thread safety
        self.lock = threading.RLock()

        # Initialize mappings
        self._initialize_mappings()

        # Subscribe to events
        self._subscribe_to_events()

    # First, let's modify the _initialize_mappings method in Strategies/strategy_manager.py

    def _initialize_mappings(self):
        """Initialize timeframe and instrument mappings from database"""
        try:
            # Always use a fresh session to ensure objects are attached
            with self.db_manager._db_session.session_scope() as session:
                # Import models directly to avoid circular imports
                from Database.models import Timeframe, Instrument

                # Get all timeframes in this session
                timeframes = session.query(Timeframe).all()

                # Build timeframe mappings within the active session
                for tf in timeframes:
                    for enum_tf in TimeFrame:
                        if enum_tf.name == tf.name:
                            self.timeframe_ids[enum_tf] = tf.id
                            self.timeframe_by_id[tf.id] = enum_tf
                            break

                # Get all instruments in this same session
                instruments = session.query(Instrument).all()

                # Build instrument mappings within the active session
                for instrument in instruments:
                    self.instrument_ids[instrument.symbol] = instrument.id

            self.logger.log_event(
                level="INFO",
                message=f"Initialized mappings: {len(self.timeframe_ids)} timeframes, {len(self.instrument_ids)} instruments",
                event_type="STRATEGY_MANAGER",
                component="strategy_manager",
                action="initialize_mappings",
                status="success"
            )

        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize mappings: {str(e)}",
                exception_type=type(e).__name__,
                function="_initialize_mappings",
                traceback=str(e),
                context={}
            )
            raise

    def _subscribe_to_events(self):
        """Subscribe to relevant events"""
        try:
            # Subscribe to new bar events
            self.event_bus.subscribe(NewBarEvent, self._on_new_bar)

            self.logger.log_event(
                level="INFO",
                message="Subscribed to events",
                event_type="STRATEGY_MANAGER",
                component="strategy_manager",
                action="subscribe_to_events",
                status="success"
            )

        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to subscribe to events: {str(e)}",
                exception_type=type(e).__name__,
                function="_subscribe_to_events",
                traceback=str(e),
                context={}
            )
            raise

    def register_strategy(self, strategy_class: Type[BaseStrategy],
                          name: str, symbol: str, timeframes: Set[TimeFrame],
                          **kwargs) -> BaseStrategy:
        """
        Register a new strategy.

        Args:
            strategy_class: Strategy class to instantiate
            name: Strategy name
            symbol: Symbol to trade
            timeframes: Set of timeframes to use
            **kwargs: Additional parameters for the strategy

        Returns:
            The created strategy instance
        """
        with self.lock:
            # Create strategy instance
            strategy = strategy_class(name=name, symbol=symbol, timeframes=timeframes,
                                      logger=self.logger, **kwargs)

            # Set instrument and timeframe info
            strategy.set_instrument_info(
                instrument_id=self.instrument_ids.get(symbol),
                symbol=symbol
            )

            # Create timeframe ID mapping for the strategy
            timeframe_ids = {}
            for tf in timeframes:
                if tf in self.timeframe_ids:
                    timeframe_ids[tf] = self.timeframe_ids[tf]

            strategy.set_timeframe_ids(timeframe_ids)

            # Store the strategy
            if symbol not in self.strategies:
                self.strategies[symbol] = {}

            self.strategies[symbol][name] = strategy

            # Register timeframe dependencies with the TimeframeManager
            # Each timeframe depends on all other timeframes in the strategy
            for primary_tf in timeframes:
                # A timeframe depends on all other timeframes
                # We'll exclude the primary timeframe from its own dependencies
                dependent_tfs = set(tf for tf in timeframes if tf != primary_tf)

                # Register with TimeframeManager
                self.timeframe_manager.register_strategy_timeframes(
                    strategy_name=name,
                    primary_timeframe=primary_tf,
                    dependent_timeframes=dependent_tfs
                )

            self.logger.log_event(
                level="INFO",
                message=f"Registered strategy {name} for {symbol}",
                event_type="STRATEGY_MANAGER",
                component="strategy_manager",
                action="register_strategy",
                status="success",
                details={
                    "strategy_name": name,
                    "symbol": symbol,
                    "timeframes": [tf.name for tf in timeframes]
                }
            )

            return strategy

    def unregister_strategy(self, symbol: str, name: str) -> bool:
        """
        Unregister a strategy.

        Args:
            symbol: Symbol the strategy is trading
            name: Strategy name

        Returns:
            True if successful, False if not found
        """
        with self.lock:
            if symbol in self.strategies and name in self.strategies[symbol]:
                # Clean up TimeframeManager entries
                self.timeframe_manager.clear_strategy(name)

                # Remove the strategy from our registry
                del self.strategies[symbol][name]

                if not self.strategies[symbol]:
                    del self.strategies[symbol]

                self.logger.log_event(
                    level="INFO",
                    message=f"Unregistered strategy {name} for {symbol}",
                    event_type="STRATEGY_MANAGER",
                    component="strategy_manager",
                    action="unregister_strategy",
                    status="success"
                )

                return True

            return False

    def get_strategy(self, symbol: str, name: str) -> Optional[BaseStrategy]:
        """
        Get a strategy by symbol and name.

        Args:
            symbol: Symbol the strategy is trading
            name: Strategy name

        Returns:
            Strategy instance or None if not found
        """
        return self.strategies.get(symbol, {}).get(name)

    def get_strategies_for_symbol(self, symbol: str) -> List[BaseStrategy]:
        """
        Get all strategies for a symbol.

        Args:
            symbol: Symbol to get strategies for

        Returns:
            List of strategy instances
        """
        return list(self.strategies.get(symbol, {}).values())

    def get_all_strategies(self) -> List[BaseStrategy]:
        """
        Get all registered strategies.

        Returns:
            List of all strategy instances
        """
        all_strategies = []
        for symbol_strategies in self.strategies.values():
            all_strategies.extend(symbol_strategies.values())
        return all_strategies

    def _on_new_bar(self, event: NewBarEvent):
        """
        Handle new bar events.

        Args:
            event: New bar event
        """
        try:
            # Get the symbol and timeframe
            symbol = event.symbol
            timeframe_id = event.timeframe_id

            # Skip if we don't have the timeframe in our mapping
            if timeframe_id not in self.timeframe_by_id:
                return

            timeframe = self.timeframe_by_id[timeframe_id]

            # Skip if we don't have strategies for this symbol
            if symbol not in self.strategies:
                return

            # Get all strategies for this symbol
            symbol_strategies = self.strategies[symbol]

            # Get all bars for this symbol and timeframe
            instrument_id = self.instrument_ids.get(symbol)
            bars = self.db_manager.get_latest_bars(instrument_id, timeframe_id)

            if not bars or len(bars) < 2:  # Need at least 1 completed + 1 forming
                return

            # Process each strategy
            for strategy_name, strategy in symbol_strategies.items():
                # Skip strategies that don't use this timeframe
                if timeframe not in strategy.timeframes:
                    continue

                try:
                    # Check if timeframes are ready with the TimeframeManager
                    if not self.timeframe_manager.check_timeframe_ready(strategy_name, timeframe):
                        self.logger.log_event(
                            level="DEBUG",
                            message=f"Timeframe {timeframe.name} not ready for strategy {strategy_name}",
                            event_type="STRATEGY_EXECUTION",
                            component="strategy_manager",
                            action="_on_new_bar",
                            status="not_ready",
                            details={
                                "strategy_name": strategy_name,
                                "symbol": symbol,
                                "timeframe": timeframe.name
                            }
                        )
                        continue

                    # Call the strategy's on_bar method
                    signal = strategy.on_bar(timeframe, bars)

                    # If a signal was generated, publish it
                    if signal:
                        self.event_bus.publish(signal)

                        self.logger.log_event(
                            level="INFO",
                            message=f"Signal generated by {strategy_name} for {symbol} on {timeframe.name}",
                            event_type="STRATEGY_SIGNAL",
                            component="strategy_manager",
                            action="process_signal",
                            status="success",
                            details={
                                "strategy_name": strategy_name,
                                "symbol": symbol,
                                "timeframe": timeframe.name,
                                "direction": signal.direction,
                                "reason": signal.reason
                            }
                        )

                except Exception as e:
                    self.logger.log_error(
                        level="ERROR",
                        message=f"Error processing strategy {strategy_name} for {symbol} on {timeframe.name}: {str(e)}",
                        exception_type=type(e).__name__,
                        function="_on_new_bar",
                        traceback=str(e),
                        context={
                            "strategy_name": strategy_name,
                            "symbol": symbol,
                            "timeframe": timeframe.name
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
                    "event_type": "NewBarEvent",
                    "symbol": getattr(event, "symbol", None),
                    "timeframe_id": getattr(event, "timeframe_id", None)
                }
            )

    def get_timeframe_update_status(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get the timeframe update status for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with timeframe update status
        """
        try:
            if not strategy_name:
                raise ValueError("Strategy name cannot be empty")

            # Get the status from the TimeframeManager
            return self.timeframe_manager.get_update_status(strategy_name)
        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error getting timeframe update status: {str(e)}",
                exception_type=type(e).__name__,
                function="get_timeframe_update_status",
                traceback=str(e),
                context={"strategy_name": strategy_name}
            )
            return {}