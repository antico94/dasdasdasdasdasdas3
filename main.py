# Initialize ConfigManager first
import os
from Config.trading_config import ConfigManager
import signal
import sys
import threading
import time

from Strategies.timeframe_manager import TimeframeManager

# Create and load configuration
config_manager = ConfigManager()
config_path = "Config/trading_config.json"

if os.path.exists(config_path):
    print(f"Loading configuration from {config_path}")
    config_manager.load_from_file(config_path)
    print("Configuration loaded successfully")
else:
    print(f"Config file not found: {config_path}")
    print("Creating default configuration...")
    config_manager.save_default_config(config_path)
    config_manager.load_from_file(config_path)
    print(f"Default configuration created and loaded at {config_path}")



from dependency_injector.wiring import inject, Provide
from container import Container

from Data.data_fetcher import DataFetcher
from Events.event_bus import EventBus
from Logger.logger import DBLogger
from MT5.mt5_manager import MT5Manager
from Data.scheduled_updates import TimeframeUpdateScheduler


class TradingBot:
    """Main trading bot application"""

    @inject
    def __init__(
            self,
            logger: DBLogger = Provide[Container.db_logger],
            data_fetcher: DataFetcher = Provide[Container.data_fetcher],
            event_bus: EventBus = Provide[Container.event_bus],
            mt5_manager: MT5Manager = Provide[Container.mt5_manager],
            timeframe_manager: TimeframeManager = Provide[Container.timeframe_manager]
    ):
        self._timeframe_manager = timeframe_manager
        self._logger = logger
        self._data_fetcher = data_fetcher
        self._event_bus = event_bus
        self._mt5_manager = mt5_manager
        self._update_scheduler = TimeframeUpdateScheduler(self._data_fetcher, self._mt5_manager, self._logger)
        self._running = False
        self._stop_event = threading.Event()
        self._shutdown_complete_event = threading.Event()
        self._shutdown_in_progress = False
        self._shutdown_lock = threading.Lock()

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

            # Initialize TimeframeManager (it's already a singleton, so this just ensures initialization)
            from Strategies.timeframe_manager import TimeframeManager
            self._timeframe_manager = TimeframeManager(logger=self._logger, event_bus=self._event_bus)

            self._logger.log_event(
                level="INFO",
                message="TimeframeManager initialized",
                event_type="SYSTEM_INIT",
                component="trading_bot",
                action="initialize",
                status="progress"
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
            self._shutdown_complete_event.clear()
            self._shutdown_in_progress = False

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

            # Start the update scheduler
            if not self._update_scheduler.start():
                self._logger.log_error(
                    level="CRITICAL",
                    message="Failed to start update scheduler",
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

            # Stop update scheduler
            self._logger.log_event(
                level="INFO",
                message="Stopping update scheduler",
                event_type="SYSTEM_STOP",
                component="trading_bot",
                action="stop_component",
                status="progress",
                details={"component": "update_scheduler"}
            )
            self._update_scheduler.stop()

            # Stop data fetcher
            self._logger.log_event(
                level="INFO",
                message="Stopping data fetcher",
                event_type="SYSTEM_STOP",
                component="trading_bot",
                action="stop_component",
                status="progress",
                details={"component": "data_fetcher"}
            )
            self._data_fetcher.stop()

            # Stop event bus
            self._logger.log_event(
                level="INFO",
                message="Stopping event bus",
                event_type="SYSTEM_STOP",
                component="trading_bot",
                action="stop_component",
                status="progress",
                details={"component": "event_bus"}
            )
            self._event_bus.stop()

            # Shut down MT5 connection
            self._logger.log_event(
                level="INFO",
                message="Shutting down MT5 connection",
                event_type="SYSTEM_STOP",
                component="trading_bot",
                action="stop_component",
                status="progress",
                details={"component": "mt5_manager"}
            )
            self._mt5_manager.shutdown()

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
                traceback=str(e),
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


def init_container():
    # Create and configure the container
    container = Container()

    # Load configuration
    container.config.from_dict({
        "db": {
            "server": "DESKTOP-B1AT4R0\\SQLEXPRESS",
            "database": "GeneralTradingBot",
            "driver": "ODBC Driver 17 for SQL Server",
            "trusted_connection": True
        },
        "logging": {
            "enabled_levels": ['INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            "console_output": True,
            "max_records": 10000,
            "color_scheme": {
                'DEBUG': '\033[37m',  # White
                'INFO': '\033[36m',  # Cyan
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',  # Red
                'CRITICAL': '\033[41m',  # Red background
            }
        }
    })

    # Initialize the components
    container.init_resources()

    # Wire the container with the current module
    container.wire(modules=[__name__])

    return container


if __name__ == "__main__":
    # Initialize the container
    container = init_container()

    # Setup complete - create and run the trading bot (dependencies injected automatically)
    bot = TradingBot()
    bot.run()