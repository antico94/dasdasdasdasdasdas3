# main.py
"""
Trading Bot Main Module
This is the entry point for the trading bot application
"""

import os
import signal
import sys
import time
import threading
import traceback
from datetime import datetime

from Config.trading_config import ConfigManager
from Config.logging_config import LoggingConfig
from Logger.logger import DBLogger
from startup import TradingBotStartup
from system_monitor import SystemMonitor


class TradingBot:
    """Main trading bot application"""

    def __init__(self):
        # Load configuration first
        self.config_manager = ConfigManager()
        config_path = "Config/trading_config.json"

        if os.path.exists(config_path):
            print(f"Loading configuration from {config_path}")
            self.config_manager.load_from_file(config_path)
            print("Configuration loaded successfully")
        else:
            print(f"Config file not found: {config_path}")
            print("Creating default configuration...")
            self.config_manager.save_default_config(config_path)
            self.config_manager.load_from_file(config_path)
            print(f"Default configuration created and loaded at {config_path}")

        # Initialize logger with proper configuration
        self._logger = self._initialize_logger()

        self._startup = TradingBotStartup(self._logger)
        self._system_monitor = None
        self._running = False
        self._stop_event = threading.Event()
        self._shutdown_complete_event = threading.Event()
        self._shutdown_in_progress = False
        self._shutdown_lock = threading.Lock()

    def _initialize_logger(self) -> DBLogger:
        """Initialize the logger with the configuration"""
        # Create connection string from credentials
        from Config.credentials import (
            SQL_SERVER,
            SQL_DATABASE,
            SQL_DRIVER,
            USE_WINDOWS_AUTH,
            SQL_USERNAME,
            SQL_PASSWORD
        )

        # Create SQLAlchemy connection string
        if USE_WINDOWS_AUTH:
            conn_string = f"mssql+pyodbc://{SQL_SERVER}/{SQL_DATABASE}?driver={SQL_DRIVER.replace(' ', '+')}&trusted_connection=yes"
        else:
            conn_string = f"mssql+pyodbc://{SQL_USERNAME}:{SQL_PASSWORD}@{SQL_SERVER}/{SQL_DATABASE}?driver={SQL_DRIVER.replace(' ', '+')}"

        # Create logging config
        logging_config = LoggingConfig(
            conn_string=conn_string,
            max_records=10000,
            console_output=True,
            enabled_levels={'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            component_configs={
                'data_fetcher': {
                    'enabled_levels': {'WARNING', 'ERROR', 'CRITICAL'},
                    'console_output': False
                },
                'mt5_manager': {
                    'enabled_levels': {'WARNING', 'ERROR', 'CRITICAL'},
                    'console_output': True
                },
                'order_manager': {
                    'enabled_levels': {'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
                    'console_output': True
                },
                'strategy_manager': {
                    'enabled_levels': {'WARNING', 'ERROR', 'CRITICAL'},
                    'console_output': True
                },
                'event_bus': {
                    'enabled_levels': {'WARNING', 'ERROR', 'CRITICAL'},
                    'console_output': False
                },
                'timeframe_manager': {
                    'enabled_levels': {'WARNING', 'ERROR', 'CRITICAL'},
                    'console_output': False
                },
                'timeframe_scheduler': {
                    'enabled_levels': {'WARNING', 'ERROR', 'CRITICAL'},
                    'console_output': False
                },
                'system_monitor': {
                    'enabled_levels': {'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
                    'console_output': True
                }
            }
        )

        # Initialize logger
        logger = DBLogger(conn_string=conn_string)

        # Load the configuration
        logger.load_config(logging_config)

        return logger

    def initialize(self) -> bool:
        """Initialize the trading bot"""
        try:
            self._logger.log_event(
                level="INFO",
                message="Initializing trading bot",
                event_type="SYSTEM_INIT",
                component="trading_bot",
                action="initialize",
                status="starting"
            )

            # Pass the config manager to startup
            self._startup.config_manager = self.config_manager

            # Initialize all components
            if not self._startup.initialize_components():
                self._logger.log_error(
                    level="CRITICAL",
                    message="Failed to initialize components",
                    exception_type="InitializationError",
                    function="initialize",
                    traceback="",
                    context={}
                )
                return False

            # Get required references for monitoring
            components = self._startup.components
            if 'mt5_manager' in components and 'order_manager' in components:
                self._system_monitor = SystemMonitor(
                    logger=self._logger,
                    mt5_manager=components['mt5_manager'],
                    order_manager=components['order_manager']
                )

            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self._logger.log_event(
                level="INFO",
                message="Trading bot initialized successfully",
                event_type="SYSTEM_INIT",
                component="trading_bot",
                action="initialize",
                status="success"
            )

            return True

        except Exception as e:
            self._logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize trading bot: {str(e)}",
                exception_type=type(e).__name__,
                function="initialize",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def start(self) -> bool:
        """Start the trading bot"""
        if self._running:
            self._logger.log_event(
                level="WARNING",
                message="Trading bot already running",
                event_type="SYSTEM_START",
                component="trading_bot",
                action="start",
                status="ignored"
            )
            return True

        try:
            self._stop_event.clear()
            self._shutdown_complete_event.clear()
            self._shutdown_in_progress = False

            # Start components
            if not self._startup.start_components():
                self._logger.log_error(
                    level="CRITICAL",
                    message="Failed to start components",
                    exception_type="StartupError",
                    function="start",
                    traceback="",
                    context={}
                )
                return False

            # Verify system readiness
            if not self._startup.verify_system_readiness():
                self._logger.log_error(
                    level="CRITICAL",
                    message="System readiness verification failed",
                    exception_type="StartupError",
                    function="start",
                    traceback="",
                    context={}
                )
                return False

            # Start system monitor
            if self._system_monitor:
                self._system_monitor.start()

            self._running = True

            self._logger.log_event(
                level="INFO",
                message="Trading bot started successfully",
                event_type="SYSTEM_START",
                component="trading_bot",
                action="start",
                status="success"
            )

            return True

        except Exception as e:
            self._running = False
            self._logger.log_error(
                level="CRITICAL",
                message=f"Failed to start trading bot: {str(e)}",
                exception_type=type(e).__name__,
                function="start",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def stop(self) -> None:
        """Stop the trading bot with graceful shutdown of all components"""
        # Use a lock to prevent multiple shutdown attempts
        with self._shutdown_lock:
            if not self._running or self._shutdown_in_progress:
                return

            self._shutdown_in_progress = True

        try:
            self._logger.log_event(
                level="INFO",
                message="Initiating graceful shutdown of trading bot",
                event_type="SYSTEM_STOP",
                component="trading_bot",
                action="stop",
                status="starting"
            )

            # Signal all threads to stop
            self._stop_event.set()

            # Stop system monitor first
            if self._system_monitor:
                self._logger.log_event(
                    level="INFO",
                    message="Stopping system monitor",
                    event_type="SYSTEM_STOP",
                    component="trading_bot",
                    action="stop_component",
                    status="progress",
                    details={"component": "system_monitor"}
                )
                self._system_monitor.stop()

            # Get components for shutdown
            components = self._startup.components

            # Stop data fetcher
            if 'data_fetcher' in components:
                self._logger.log_event(
                    level="INFO",
                    message="Stopping data fetcher",
                    event_type="SYSTEM_STOP",
                    component="trading_bot",
                    action="stop_component",
                    status="progress",
                    details={"component": "data_fetcher"}
                )
                components['data_fetcher'].stop()

            # Stop event bus
            if 'event_bus' in components:
                self._logger.log_event(
                    level="INFO",
                    message="Stopping event bus",
                    event_type="SYSTEM_STOP",
                    component="trading_bot",
                    action="stop_component",
                    status="progress",
                    details={"component": "event_bus"}
                )
                components['event_bus'].stop()

            # Shut down MT5 connection
            if 'mt5_manager' in components:
                self._logger.log_event(
                    level="INFO",
                    message="Shutting down MT5 connection",
                    event_type="SYSTEM_STOP",
                    component="trading_bot",
                    action="stop_component",
                    status="progress",
                    details={"component": "mt5_manager"}
                )
                components['mt5_manager'].shutdown()

            self._running = False
            self._shutdown_in_progress = False
            self._shutdown_complete_event.set()

            self._logger.log_event(
                level="INFO",
                message="Trading bot stopped successfully",
                event_type="SYSTEM_STOP",
                component="trading_bot",
                action="stop",
                status="success"
            )

        except Exception as e:
            self._shutdown_in_progress = False
            self._logger.log_error(
                level="ERROR",
                message=f"Error during trading bot shutdown: {str(e)}",
                exception_type=type(e).__name__,
                function="stop",
                traceback=traceback.format_exc(),
                context={}
            )
            # Even if there's an error, signal that shutdown is complete
            self._shutdown_complete_event.set()

    def run(self) -> None:
        """Run the trading bot until stopped"""
        if not self.initialize():
            sys.exit(1)

        if not self.start():
            sys.exit(1)

        try:
            # Print banner
            print("\n" + "=" * 80)
            print(" Trading Bot Running - Press Ctrl+C to stop gracefully".center(80))
            print(" Started at: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")).center(80))
            print("=" * 80 + "\n")

            # Keep the main thread alive
            while not self._stop_event.is_set():
                time.sleep(0.5)

        except KeyboardInterrupt:
            # This should not be needed as signal handler will catch it,
            # but added as a fallback
            print("\nKeyboard interrupt received. Shutting down...")
            self.stop()
        finally:
            # Wait for shutdown to complete with timeout
            if not self._shutdown_complete_event.wait(timeout=30):
                self._logger.log_error(
                    level="CRITICAL",
                    message="Shutdown timed out after 30 seconds",
                    exception_type="ShutdownError",
                    function="run",
                    traceback="",
                    context={}
                )
                print("\nWARNING: Shutdown timed out after 30 seconds. Forcing exit.")

    def _signal_handler(self, sig, frame) -> None:
        """Handle termination signals with graceful shutdown"""
        # Prevent the default KeyboardInterrupt traceback
        signal_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"

        print(f"\n{signal_name} received. Initiating graceful shutdown (this may take a moment)...")

        self._logger.log_event(
            level="INFO",
            message=f"Received signal {signal_name}, shutting down gracefully",
            event_type="SYSTEM_SIGNAL",
            component="trading_bot",
            action="signal_handler",
            status="process"
        )

        # Initiate graceful shutdown
        self.stop()

        # Wait for shutdown to complete with timeout
        if not self._shutdown_complete_event.wait(timeout=20):
            self._logger.log_error(
                level="WARNING",
                message=f"Graceful shutdown taking longer than expected after {signal_name}",
                exception_type="ShutdownDelayError",
                function="_signal_handler",
                traceback="",
                context={"signal": signal_name}
            )
            print("\nShutdown is taking longer than expected. Please wait...")

            # Give it a bit more time
            if not self._shutdown_complete_event.wait(timeout=10):
                self._logger.log_error(
                    level="CRITICAL",
                    message="Shutdown timed out after 30 seconds",
                    exception_type="ShutdownError",
                    function="_signal_handler",
                    traceback="",
                    context={"signal": signal_name}
                )
                print("\nWARNING: Shutdown timed out after 30 seconds. Forcing exit.")
                # Exit non-zero to indicate error
                os._exit(1)


if __name__ == "__main__":
    try:
        # Initialize and run trading bot
        bot = TradingBot()
        bot.run()
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        traceback.print_exc()
        sys.exit(1)