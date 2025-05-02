# trading_bot.py
import os
import sys
import time
import signal
import threading
import json
from typing import Dict, Any, Optional

from Config.trading_config import ConfigManager
from Data.data_fetcher import DataFetcher
from Events.event_bus import EventBus
from Logger.logger import DBLogger
from container import Container
from dependency_injector.wiring import Provide, inject


class TradingBot:
    """Main trading bot application"""

    def __init__(self):
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

        self._logger = DBLogger(
            conn_string=conn_string,
            enabled_levels={'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            console_output=True
        )
        self._data_fetcher = DataFetcher()
        self._event_bus = EventBus()
        self._running = False
        self._stop_event = threading.Event()

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

            # Initialize data fetcher
            if not self._data_fetcher.initialize():
                self._logger.log_error(
                    level="CRITICAL",
                    message="Failed to initialize data fetcher",
                    exception_type="InitializationError",
                    function="initialize",
                    traceback="",
                    context={}
                )
                return False

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
                traceback=str(e),
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

            # Start data fetcher
            if not self._data_fetcher.start():
                self._logger.log_error(
                    level="CRITICAL",
                    message="Failed to start data fetcher",
                    exception_type="StartupError",
                    function="start",
                    traceback="",
                    context={}
                )
                return False

            self._running = True

            self._logger.log_event(
                level="INFO",
                message="Trading bot started",
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
                traceback=str(e),
                context={}
            )
            return False

    def stop(self) -> None:
        """Stop the trading bot"""
        if not self._running:
            return

        try:
            self._stop_event.set()

            # Stop data fetcher
            self._data_fetcher.stop()

            # Stop event bus
            self._event_bus.stop()

            self._running = False

            self._logger.log_event(
                level="INFO",
                message="Trading bot stopped",
                event_type="SYSTEM_STOP",
                component="trading_bot",
                action="stop",
                status="success"
            )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error stopping trading bot: {str(e)}",
                exception_type=type(e).__name__,
                function="stop",
                traceback=str(e),
                context={}
            )

    def run(self) -> None:
        """Run the trading bot until stopped"""
        if not self.initialize():
            sys.exit(1)

        if not self.start():
            sys.exit(1)

        try:
            # Keep the main thread alive
            while not self._stop_event.is_set():
                time.sleep(1)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _signal_handler(self, sig, frame) -> None:
        """Handle termination signals"""
        self._logger.log_event(
            level="INFO",
            message=f"Received signal {sig}, shutting down",
            event_type="SYSTEM_SIGNAL",
            component="trading_bot",
            action="signal_handler",
            status="process"
        )
        self.stop()


def configure_container() -> Container:
    """Configure and initialize container"""
    container = Container()

    # Load configuration from file
    config_path = os.environ.get('CONFIG_PATH', 'trading_config.json')
    config_manager = ConfigManager()

    if not os.path.exists(config_path):
        # Create default config if not exists
        config_manager.save_default_config(config_path)
        print(f"Created default configuration at: {config_path}")

    # Load configuration
    config_manager.load_from_file(config_path)

    # Configure container using credentials directly
    from Config.credentials import (
        SQL_SERVER,
        SQL_DATABASE,
        SQL_DRIVER,
        USE_WINDOWS_AUTH
    )

    container.config.from_dict({
        'db': {
            'server': SQL_SERVER,
            'database': SQL_DATABASE,
            'driver': SQL_DRIVER,
            'trusted_connection': USE_WINDOWS_AUTH
        },
        'logging': {
            'enabled_levels': {'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            'console_output': True,
            'max_records': 10000,
            'color_scheme': {
                'DEBUG': '\033[37m',  # White
                'INFO': '\033[36m',  # Cyan
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',  # Red
                'CRITICAL': '\033[41m',  # Red background
            }
        }
    })

    return container


@inject
def main(logger=Provide[Container.db_logger]) -> None:
    """Main entry point for the trading bot"""

    logger.log_event(
        level="INFO",
        message="Starting trading bot application",
        event_type="SYSTEM_STARTUP",
        component="main",
        action="initialize",
        status="starting",
        details={"version": "1.0.0"}
    )

    # Create and run trading bot
    bot = TradingBot()
    bot.run()

    logger.log_event(
        level="INFO",
        message="Trading bot application exiting",
        event_type="SYSTEM_SHUTDOWN",
        component="main",
        action="exit",
        status="success"
    )


if __name__ == "__main__":
    container = configure_container()
    container.wire(modules=[__name__])

    # Run main application
    main()