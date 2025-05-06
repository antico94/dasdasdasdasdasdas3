# startup.py
"""Trading Bot Startup Module"""

import traceback

from Config.trading_config import ConfigManager, TimeFrame
from Data.data_fetcher import DataFetcher
from Data.scheduled_updates import TimeframeUpdateScheduler
from Database.db_manager import DatabaseManager
from Events.event_bus import EventBus
from Logger.logger import DBLogger
from MT5.mt5_manager import MT5Manager
from Strategies.breakout_strategy import BreakoutStrategy
from Strategies.config import trading_strategies_config, StrategyType
from Strategies.hybrid_strategy import HybridStrategy
from Strategies.ichimoku_strategy import IchimokuStrategy
from Strategies.mean_reversion_strategy import MeanReversionStrategy
from Strategies.momentum_strategy import MomentumStrategy
from Strategies.scalping_strategy import ScalpingStrategy
from Strategies.strategy_manager import StrategyManager
from Strategies.timeframe_manager import TimeframeManager
from Strategies.triple_ma_strategy import TripleMAStrategy
from execution.order_manager import OrderManager


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
            # Get database manager - it's already initialized
            db_manager = self.components['db_manager']

            # Create strategy manager instance
            strategy_manager = StrategyManager(
                event_bus=self.components['event_bus'],
                db_manager=db_manager,
                logger=self.logger
            )

            # Store the manager in components dictionary first
            self.components['strategy_manager'] = strategy_manager

            # Now use a single session scope to initialize all mappings
            with db_manager._db_session.session_scope() as session:
                # Import the models
                from Database.models import Timeframe, Instrument

                # Get all timeframes and instruments in a single query
                timeframes = session.query(Timeframe).all()
                instruments = session.query(Instrument).all()

                # Initialize mappings directly while session is still active
                strategy_manager.timeframe_ids = {}
                strategy_manager.timeframe_by_id = {}
                strategy_manager.instrument_ids = {}

                # Build timeframe mappings
                for tf in timeframes:
                    for enum_tf in TimeFrame:
                        if enum_tf.name == tf.name:
                            strategy_manager.timeframe_ids[enum_tf] = tf.id
                            strategy_manager.timeframe_by_id[tf.id] = enum_tf
                            break

                # Build instrument mappings
                for instr in instruments:
                    strategy_manager.instrument_ids[instr.symbol] = instr.id

            self.logger.log_event(
                level="INFO",
                message="Strategy manager initialized with manually created mappings",
                event_type="COMPONENT_INIT",
                component="startup",
                action="_init_strategy_manager",
                status="success",
                details={
                    "timeframes_count": len(strategy_manager.timeframe_ids),
                    "instruments_count": len(strategy_manager.instrument_ids)
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

                    # Get common parameters
                    indicators = strategy_config.indicators
                    risk = strategy_config.risk_management
                    custom_params = strategy_config.custom_parameters

                    # Create specific strategy instance based on strategy type
                    if strategy_config.strategy_type == StrategyType.BREAKOUT:
                        # Extract breakout-specific settings
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

                    elif strategy_config.strategy_type == StrategyType.ICHIMOKU:
                        # Extract Ichimoku-specific settings
                        ichimoku_config = indicators.ichimoku
                        adx_config = indicators.adx

                        # Register the strategy
                        self.components['strategy_manager'].register_strategy(
                            strategy_class=IchimokuStrategy,
                            name=strategy_name,
                            symbol=symbol,
                            timeframes=tf_set,
                            # Extract parameters from nested configs
                            tenkan_period=ichimoku_config.tenkan_period if ichimoku_config else 9,
                            kijun_period=ichimoku_config.kijun_period if ichimoku_config else 26,
                            senkou_span_b_period=ichimoku_config.senkou_span_b_period if ichimoku_config else 52,
                            displacement=ichimoku_config.displacement if ichimoku_config else 26,
                            require_kumo_breakout=custom_params.get('require_kumo_breakout', True),
                            require_chikou_confirmation=custom_params.get('require_chikou_confirmation', True),
                            adx_period=adx_config.period if adx_config else 14,
                            adx_threshold=adx_config.threshold if adx_config else 25.0
                        )

                    elif strategy_config.strategy_type == StrategyType.TRIPLE_MA:
                        # Extract Triple MA-specific settings
                        triple_ma_config = indicators.triple_ma
                        adx_config = indicators.adx
                        macd_config = indicators.macd

                        # Register the strategy
                        self.components['strategy_manager'].register_strategy(
                            strategy_class=TripleMAStrategy,
                            name=strategy_name,
                            symbol=symbol,
                            timeframes=tf_set,
                            # Extract parameters from nested configs
                            short_period=triple_ma_config.short_period if triple_ma_config else 8,
                            medium_period=triple_ma_config.medium_period if triple_ma_config else 21,
                            long_period=triple_ma_config.long_period if triple_ma_config else 55,
                            ma_type=triple_ma_config.ma_type if triple_ma_config else "EMA",
                            adx_period=adx_config.period if adx_config else 14,
                            adx_threshold=adx_config.threshold if adx_config else 25.0,
                            require_macd_confirmation=custom_params.get('require_macd_confirmation', False),
                            stop_loss_atr_multiplier=risk.stop_loss_atr_multiplier,
                            take_profit_atr_multiplier=risk.take_profit_atr_multiplier
                        )

                    elif strategy_config.strategy_type == StrategyType.MEAN_REVERSION:
                        # Extract Mean Reversion-specific settings
                        rsi_config = indicators.rsi
                        stoch_config = indicators.stochastic
                        bb_config = indicators.bollinger_bands

                        # Register the strategy
                        self.components['strategy_manager'].register_strategy(
                            strategy_class=MeanReversionStrategy,
                            name=strategy_name,
                            symbol=symbol,
                            timeframes=tf_set,
                            # Extract parameters from nested configs
                            rsi_period=rsi_config.period if rsi_config else 14,
                            rsi_oversold=rsi_config.oversold_level if rsi_config else 30.0,
                            rsi_overbought=rsi_config.overbought_level if rsi_config else 70.0,
                            stoch_k_period=stoch_config.k_period if stoch_config else 14,
                            stoch_d_period=stoch_config.d_period if stoch_config else 3,
                            stoch_oversold=stoch_config.oversold_level if stoch_config else 20.0,
                            stoch_overbought=stoch_config.overbought_level if stoch_config else 80.0,
                            bb_period=bb_config.period if bb_config else 20,
                            bb_std_dev=bb_config.deviation if bb_config else 2.0,
                            use_ibs=custom_params.get('internal_bar_strength', True),
                            ibs_threshold_low=custom_params.get('ibs_threshold_low', 0.2),
                            ibs_threshold_high=custom_params.get('ibs_threshold_high', 0.8),
                            stop_loss_atr_multiplier=risk.stop_loss_atr_multiplier,
                            take_profit_atr_multiplier=risk.take_profit_atr_multiplier
                        )

                    elif strategy_config.strategy_type == StrategyType.MOMENTUM:
                        # Extract Momentum-specific settings
                        macd_config = indicators.macd
                        rsi_config = indicators.rsi
                        ma_config = indicators.ma

                        # Register the strategy
                        self.components['strategy_manager'].register_strategy(
                            strategy_class=MomentumStrategy,
                            name=strategy_name,
                            symbol=symbol,
                            timeframes=tf_set,
                            # Extract parameters from nested configs
                            macd_fast_period=macd_config.fast_period if macd_config else 12,
                            macd_slow_period=macd_config.slow_period if macd_config else 26,
                            macd_signal_period=macd_config.signal_period if macd_config else 9,
                            rsi_period=rsi_config.period if rsi_config else 14,
                            rsi_threshold_low=custom_params.get('rsi_threshold_low', 40.0),
                            rsi_threshold_high=custom_params.get('rsi_threshold_high', 60.0),
                            ma_period=ma_config.period if ma_config else 50,
                            ma_type=ma_config.ma_type if ma_config else "EMA",
                            require_volume_confirmation=custom_params.get('require_volume_confirmation', True),
                            volume_threshold=custom_params.get('volume_threshold', 1.5),
                            stop_loss_atr_multiplier=risk.stop_loss_atr_multiplier,
                            take_profit_atr_multiplier=risk.take_profit_atr_multiplier
                        )

                    # Add handler for SCALPING strategy type
                    elif strategy_config.strategy_type == StrategyType.SCALPING:
                        # Extract Scalping-specific settings
                        macd_config = indicators.macd
                        rsi_config = indicators.rsi
                        ma_config = indicators.ma

                        # Register the strategy
                        self.components['strategy_manager'].register_strategy(
                            strategy_class=ScalpingStrategy,
                            name=strategy_name,
                            symbol=symbol,
                            timeframes=tf_set,
                            macd_fast_period=macd_config.fast_period if macd_config else 3,
                            macd_slow_period=macd_config.slow_period if macd_config else 10,
                            macd_signal_period=macd_config.signal_period if macd_config else 16,
                            rsi_period=rsi_config.period if rsi_config else 5,
                            ma_period=ma_config.period if ma_config else 20,
                            ma_type=ma_config.ma_type if ma_config else "EMA",
                            require_volume_confirmation=custom_params.get('require_volume_confirmation', False),
                            volume_threshold=custom_params.get('volume_threshold', 1.5),
                            ema_values=custom_params.get('ema_values', [5, 20, 55]),
                            rapid_exit=custom_params.get('rapid_exit', False),
                            max_spread_pips=custom_params.get('max_spread_pips', 1.0),
                            alternative_macd_settings=custom_params.get('alternative_macd_settings', None),
                            stop_loss_atr_multiplier=risk.stop_loss_atr_multiplier,
                            take_profit_atr_multiplier=risk.take_profit_atr_multiplier
                        )

                    # Add handler for HYBRID strategy type
                    elif strategy_config.strategy_type == StrategyType.HYBRID:
                        # Extract Moving Average config if present
                        ma_config = indicators.ma

                        # Register the strategy
                        self.components['strategy_manager'].register_strategy(
                            strategy_class=HybridStrategy,
                            name=strategy_name,
                            symbol=symbol,
                            timeframes=tf_set,
                            # Extract parameters from nested configs
                            ma_period=ma_config.period if ma_config else 100,
                            ma_type=ma_config.ma_type if ma_config else "EMA",
                            yield_differential_threshold=custom_params.get('yield_differential_threshold', 0.5),
                            boj_dovish_bias=custom_params.get('boj_dovish_bias', True),
                            us_bond_yield_rising=custom_params.get('us_bond_yield_rising', True),
                            real_yield_filter=custom_params.get('real_yield_filter', False),
                            max_real_yield_long=custom_params.get('max_real_yield_long', 0.0),
                            min_real_yield_short=custom_params.get('min_real_yield_short', 0.5),
                            stop_loss_atr_multiplier=risk.stop_loss_atr_multiplier
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
                        continue

                    # If strategy was registered successfully, log it
                    if strategy_config.strategy_type in [StrategyType.BREAKOUT, StrategyType.ICHIMOKU,
                                                         StrategyType.TRIPLE_MA, StrategyType.MEAN_REVERSION,
                                                         StrategyType.MOMENTUM, StrategyType.SCALPING,
                                                         StrategyType.HYBRID]:
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