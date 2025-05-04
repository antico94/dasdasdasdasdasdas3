# startup.py
"""Trading Bot Startup Module"""

import sys
import traceback
from typing import Dict, Any, List, Optional, Set

from Config.trading_config import ConfigManager, TimeFrame
from Database.models import Instrument, Timeframe
from Strategies.config import trading_strategies_config, StrategyType, TimeFrameType
from Database.db_manager import DatabaseManager
from Events.event_bus import EventBus
from Logger.logger import DBLogger
from Strategies.timeframe_manager import TimeframeManager
from Strategies.breakout_strategy import BreakoutStrategy
from Strategies.strategy_manager import StrategyManager
from execution.order_manager import OrderManager
from MT5.mt5_manager import MT5Manager
from Data.data_fetcher import DataFetcher
from Data.scheduled_updates import TimeframeUpdateScheduler


class TradingBotStartup:
    """Handles initialization and startup of the trading bot"""

    def __init__(self, logger: DBLogger):
        self.logger = logger
        self.config_manager = ConfigManager()
        self.components = {}

    def initialize_components(self) -> bool:
        """Initialize all trading bot components"""
        try:
            self.logger.log_event(
                level="INFO",
                message="Initializing trading bot components",
                event_type="SYSTEM_INIT",
                component="startup",
                action="initialize_components",
                status="starting"
            )

            # Initialize components with proper error handling
            self._init_database()
            self._init_mt5()
            self._init_event_bus()  # Event bus should be initialized early
            self._init_timeframe_manager()  # Initialize TimeframeManager
            self._init_data_fetcher()
            self._init_scheduled_updates()
            self._init_order_manager()
            self._init_strategy_manager()
            self._register_strategies()

            self.logger.log_event(
                level="INFO",
                message="All trading bot components initialized successfully",
                event_type="SYSTEM_INIT",
                component="startup",
                action="initialize_components",
                status="success"
            )
            return True

        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize trading bot components: {str(e)}",
                exception_type=type(e).__name__,
                function="initialize_components",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def _init_database(self) -> None:
        """Initialize database manager"""
        try:
            db_manager = DatabaseManager()
            db_manager.initialize_database()
            self.components['db_manager'] = db_manager

            self.logger.log_event(
                level="INFO",
                message="Database manager initialized",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_database",
                status="success"
            )
        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize database: {str(e)}",
                exception_type=type(e).__name__,
                function="_init_database",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _init_mt5(self) -> None:
        """Initialize MT5 connection"""
        try:
            mt5_manager = MT5Manager()

            # Initialize with strict validation - fail immediately if MT5 not connected
            if not mt5_manager.initialize():
                raise ConnectionError("Failed to initialize MT5 connection")

            self.components['mt5_manager'] = mt5_manager

            self.logger.log_event(
                level="INFO",
                message="MT5 manager initialized",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_mt5",
                status="success"
            )
        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize MT5: {str(e)}",
                exception_type=type(e).__name__,
                function="_init_mt5",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _init_event_bus(self) -> None:
        """Initialize event bus"""
        try:
            event_bus = EventBus()
            self.components['event_bus'] = event_bus

            self.logger.log_event(
                level="INFO",
                message="Event bus initialized",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_event_bus",
                status="success"
            )
        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize event bus: {str(e)}",
                exception_type=type(e).__name__,
                function="_init_event_bus",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _init_timeframe_manager(self) -> None:
        """Initialize timeframe manager"""
        try:
            # TimeframeManager is a singleton, so this will get the same instance every time
            timeframe_manager = TimeframeManager(
                logger=self.logger,
                event_bus=self.components['event_bus']
            )

            self.components['timeframe_manager'] = timeframe_manager

            self.logger.log_event(
                level="INFO",
                message="Timeframe manager initialized",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_timeframe_manager",
                status="success"
            )
        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize timeframe manager: {str(e)}",
                exception_type=type(e).__name__,
                function="_init_timeframe_manager",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _init_data_fetcher(self) -> None:
        """Initialize data fetcher"""
        try:
            data_fetcher = DataFetcher()

            # Initialize with strict validation
            if not data_fetcher.initialize():
                raise ConnectionError("Failed to initialize data fetcher")

            self.components['data_fetcher'] = data_fetcher

            self.logger.log_event(
                level="INFO",
                message="Data fetcher initialized",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_data_fetcher",
                status="success"
            )
        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize data fetcher: {str(e)}",
                exception_type=type(e).__name__,
                function="_init_data_fetcher",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _init_scheduled_updates(self) -> None:
        """Initialize timeframe update scheduler"""
        try:
            update_scheduler = TimeframeUpdateScheduler(
                data_fetcher=self.components['data_fetcher'],
                mt5_manager=self.components['mt5_manager'],
                logger=self.logger
            )

            self.components['update_scheduler'] = update_scheduler

            self.logger.log_event(
                level="INFO",
                message="Timeframe update scheduler initialized",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_scheduled_updates",
                status="success"
            )
        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize update scheduler: {str(e)}",
                exception_type=type(e).__name__,
                function="_init_scheduled_updates",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _init_order_manager(self) -> None:
        """Initialize order manager"""
        try:
            # Get global risk settings from trading strategies config
            global_risk = trading_strategies_config.global_risk_management

            order_manager = OrderManager(
                mt5_manager=self.components['mt5_manager'],
                logger=self.logger,
                event_bus=self.components['event_bus']
            )

            # Configure risk settings from global config
            order_manager.base_risk_percent = global_risk.max_risk_per_trade_percent
            order_manager.max_daily_risk_percent = global_risk.max_risk_per_trade_percent * 3
            order_manager.max_positions_per_symbol = global_risk.max_positions_per_instrument

            self.components['order_manager'] = order_manager

            self.logger.log_event(
                level="INFO",
                message="Order manager initialized",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_order_manager",
                status="success",
                details={
                    "base_risk_percent": order_manager.base_risk_percent,
                    "max_daily_risk_percent": order_manager.max_daily_risk_percent,
                    "max_positions_per_symbol": order_manager.max_positions_per_symbol
                }
            )
        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize order manager: {str(e)}",
                exception_type=type(e).__name__,
                function="_init_order_manager",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _init_strategy_manager(self) -> None:
        """Initialize strategy manager"""
        try:
            # Import the database models
            from Database.models import Timeframe as DbTimeframe, Instrument

            # Create raw data mappings

            timeframe_data = []
            instrument_data = []

            # Get database manager - it's already initialized
            db_manager = self.components['db_manager']

            # Get all timeframes using the database manager's public methods
            timeframes = db_manager.get_all_timeframes()
            instruments = db_manager.get_all_instruments()

            # Extract the data we need without keeping references to SQLAlchemy objects
            timeframe_data = [(tf.id, tf.name) for tf in timeframes]
            instrument_data = [(instr.id, instr.symbol) for instr in instruments]

            # Create strategy manager
            strategy_manager = StrategyManager(
                event_bus=self.components['event_bus'],
                db_manager=db_manager,
                logger=self.logger
            )

            # Initialize the mappings manually without relying on the manager's initialization
            strategy_manager.timeframe_ids = {}
            strategy_manager.timeframe_by_id = {}
            strategy_manager.instrument_ids = {}

            # Build mappings from raw data
            for tf_id, tf_name in timeframe_data:
                for enum_tf in TimeFrame:
                    if enum_tf.name == tf_name:
                        strategy_manager.timeframe_ids[enum_tf] = tf_id
                        strategy_manager.timeframe_by_id[tf_id] = enum_tf
                        break

            for instr_id, symbol in instrument_data:
                strategy_manager.instrument_ids[symbol] = instr_id

            self.components['strategy_manager'] = strategy_manager

            self.logger.log_event(
                level="INFO",
                message="Strategy manager initialized with manually created mappings",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_strategy_manager",
                status="success",
                details={
                    "timeframes_count": len(timeframe_data),
                    "instruments_count": len(instrument_data)
                }
            )
        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize strategy manager: {str(e)}",
                exception_type=type(e).__name__,
                function="_init_strategy_manager",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _register_strategies(self) -> None:
        """Register trading strategies using the existing Strategies/config.py"""
        try:
            # Get all timeframes
            timeframes = {tf.name: tf for tf in TimeFrame}

            # Register each strategy per instrument
            for symbol, instrument_config in trading_strategies_config.instruments.items():
                for strategy_name, strategy_config in instrument_config.strategies.items():
                    # Skip disabled strategies
                    if not strategy_config.enabled:
                        self.logger.log_event(
                            level="INFO",
                            message=f"Skipping disabled strategy {strategy_name} for {symbol}",
                            event_type="STRATEGY_REGISTRATION",
                            component="startup",
                            action="_register_strategies",
                            status="skipped",
                            details={"strategy_name": strategy_name, "symbol": symbol}
                        )
                        continue

                    # Convert timeframe types to TimeFrame enums
                    tf_set = set()
                    for tf_type in strategy_config.timeframes:
                        tf_name = tf_type.name  # This is a TimeFrameType enum
                        if tf_name in timeframes:
                            tf_set.add(timeframes[tf_name])
                        else:
                            self.logger.log_event(
                                level="WARNING",
                                message=f"Unknown timeframe {tf_name} for {strategy_name}",
                                event_type="STRATEGY_REGISTRATION",
                                component="startup",
                                action="_register_strategies",
                                status="warning",
                                details={"strategy_name": strategy_name, "timeframe": tf_name}
                            )

                    # Skip if no valid timeframes
                    if not tf_set:
                        self.logger.log_error(
                            level="ERROR",
                            message=f"No valid timeframes for strategy {strategy_name} on {symbol}",
                            exception_type="ConfigError",
                            function="_register_strategies",
                            traceback="",
                            context={"strategy_name": strategy_name, "symbol": symbol}
                        )
                        continue

                    # Create specific strategy instance based on strategy type
                    if strategy_config.strategy_type == StrategyType.BREAKOUT:
                        # Extract breakout-specific settings
                        indicators = strategy_config.indicators
                        risk = strategy_config.risk_management
                        custom_params = strategy_config.custom_parameters

                        donchian_config = indicators.donchian_channels
                        bb_config = indicators.bollinger_bands
                        atr_config = indicators.atr
                        adx_config = indicators.adx

                        # Register the strategy
                        self.components['strategy_manager'].register_strategy(
                            strategy_class=BreakoutStrategy,
                            name=strategy_name,
                            symbol=symbol,
                            timeframes=tf_set,
                            # Extract parameters from nested configs
                            donchian_period=donchian_config.period if donchian_config else 20,
                            bollinger_period=bb_config.period if bb_config else 20,
                            bollinger_deviation=bb_config.deviation if bb_config else 2.0,
                            atr_period=atr_config.period if atr_config else 14,
                            min_volatility_trigger=custom_params.get('min_volatility_trigger', 1.2),
                            session_filter=custom_params.get('session_filter'),
                            stop_loss_atr_multiplier=risk.stop_loss_atr_multiplier,
                            take_profit_atr_multiplier=risk.take_profit_atr_multiplier,
                            adx_period=adx_config.period if adx_config else 14,
                            adx_threshold=adx_config.threshold if adx_config else 25.0
                        )

                        self.logger.log_event(
                            level="INFO",
                            message=f"Registered {strategy_name} for {symbol}",
                            event_type="STRATEGY_REGISTRATION",
                            component="startup",
                            action="_register_strategies",
                            status="success",
                            details={
                                "strategy_name": strategy_name,
                                "symbol": symbol,
                                "timeframes": [tf.name for tf in tf_set],
                                "strategy_type": strategy_config.strategy_type.value
                            }
                        )
                    else:
                        # Log unsupported strategy type
                        self.logger.log_error(
                            level="ERROR",
                            message=f"Unsupported strategy type {strategy_config.strategy_type.value} for {strategy_name}",
                            exception_type="ConfigError",
                            function="_register_strategies",
                            traceback="",
                            context={"strategy_name": strategy_name,
                                     "strategy_type": strategy_config.strategy_type.value}
                        )

            # Log summary
            self.logger.log_event(
                level="INFO",
                message="Strategy registration complete",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_register_strategies",
                status="success"
            )

        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to register strategies: {str(e)}",
                exception_type=type(e).__name__,
                function="_register_strategies",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def start_components(self) -> bool:
        """Start all trading bot components"""
        try:
            self.logger.log_event(
                level="INFO",
                message="Starting trading bot components",
                event_type="SYSTEM_START",
                component="startup",
                action="start_components",
                status="starting"
            )

            # Start update scheduler
            update_scheduler = self.components.get('update_scheduler')
            if update_scheduler and not update_scheduler.start():
                self.logger.log_error(
                    level="CRITICAL",
                    message="Failed to start timeframe update scheduler",
                    exception_type="StartupError",
                    function="start_components",
                    traceback="",
                    context={}
                )
                return False

            # Start data fetcher
            data_fetcher = self.components.get('data_fetcher')
            if data_fetcher and not data_fetcher.start():
                self.logger.log_error(
                    level="CRITICAL",
                    message="Failed to start data fetcher",
                    exception_type="StartupError",
                    function="start_components",
                    traceback="",
                    context={}
                )
                return False

            self.logger.log_event(
                level="INFO",
                message="All components started successfully",
                event_type="SYSTEM_START",
                component="startup",
                action="start_components",
                status="success"
            )
            return True

        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"Failed to start trading bot components: {str(e)}",
                exception_type=type(e).__name__,
                function="start_components",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def verify_system_readiness(self) -> bool:
        """Perform final checks before trading begins"""
        try:
            self.logger.log_event(
                level="INFO",
                message="Verifying system readiness for trading",
                event_type="SYSTEM_START",
                component="startup",
                action="verify_system_readiness",
                status="starting"
            )

            # Check MT5 connection
            mt5_manager = self.components.get('mt5_manager')
            if not mt5_manager or not mt5_manager.ensure_connection():
                self.logger.log_error(
                    level="CRITICAL",
                    message="MT5 connection not available",
                    exception_type="ConnectionError",
                    function="verify_system_readiness",
                    traceback="",
                    context={}
                )
                return False

            # Verify account information
            account_info = mt5_manager.get_account_info()
            if not account_info:
                self.logger.log_error(
                    level="CRITICAL",
                    message="Failed to get MT5 account information",
                    exception_type="ConnectionError",
                    function="verify_system_readiness",
                    traceback="",
                    context={}
                )
                return False

            # Verify symbols available
            config = self.config_manager.config
            for instrument in config.instruments:
                if not mt5_manager.check_symbol_available(instrument.symbol):
                    self.logger.log_error(
                        level="CRITICAL",
                        message=f"Symbol {instrument.symbol} not available in MT5",
                        exception_type="ConfigError",
                        function="verify_system_readiness",
                        traceback="",
                        context={"symbol": instrument.symbol}
                    )
                    return False

            # Log account information
            self.logger.log_event(
                level="INFO",
                message="MT5 account verified",
                event_type="SYSTEM_VERIFY",
                component="startup",
                action="verify_system_readiness",
                status="success",
                details={
                    "login": account_info.get('login'),
                    "balance": account_info.get('balance'),
                    "equity": account_info.get('equity'),
                    "margin_level": account_info.get('margin_level'),
                    "leverage": account_info.get('leverage')
                }
            )

            # Check that data fetcher is running
            data_fetcher = self.components.get('data_fetcher')
            if not data_fetcher or not getattr(data_fetcher, '_running', False):
                self.logger.log_error(
                    level="CRITICAL",
                    message="Data fetcher not running",
                    exception_type="ComponentError",
                    function="verify_system_readiness",
                    traceback="",
                    context={}
                )
                return False

            # Check that update scheduler is running
            update_scheduler = self.components.get('update_scheduler')
            if not update_scheduler or not getattr(update_scheduler, '_running', False):
                self.logger.log_error(
                    level="CRITICAL",
                    message="Update scheduler not running",
                    exception_type="ComponentError",
                    function="verify_system_readiness",
                    traceback="",
                    context={}
                )
                return False

            # Verify strategies registered
            strategy_manager = self.components.get('strategy_manager')
            strategies = strategy_manager.get_all_strategies() if strategy_manager else []
            if not strategies:
                self.logger.log_error(
                    level="CRITICAL",
                    message="No strategies registered",
                    exception_type="ConfigError",
                    function="verify_system_readiness",
                    traceback="",
                    context={}
                )
                return False

            # Log strategy information
            for strategy in strategies:
                self.logger.log_event(
                    level="INFO",
                    message=f"Strategy verified: {strategy.name} for {strategy.symbol}",
                    event_type="SYSTEM_VERIFY",
                    component="startup",
                    action="verify_system_readiness",
                    status="success",
                    details={
                        "strategy_name": strategy.name,
                        "symbol": strategy.symbol,
                        "timeframes": [tf.name for tf in strategy.timeframes]
                    }
                )

            self.logger.log_event(
                level="INFO",
                message="System ready for trading",
                event_type="SYSTEM_START",
                component="startup",
                action="verify_system_readiness",
                status="success"
            )
            return True

        except Exception as e:
            self.logger.log_error(
                level="CRITICAL",
                message=f"System readiness verification failed: {str(e)}",
                exception_type=type(e).__name__,
                function="verify_system_readiness",
                traceback=traceback.format_exc(),
                context={}
            )
            return False