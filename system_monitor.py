# system_monitor.py
"""
Trading Bot System Monitor
Monitors system resources and trading bot components
"""

import threading
import time
from typing import Dict, Any, Optional, List

from Logger.logger import DBLogger


class SystemMonitor:
    """
    Monitors system resources and trading bot components.

    This class provides:
    1. System resource monitoring (CPU, memory)
    2. MT5 connection health checks
    3. Trading activity monitoring
    4. Periodic garbage collection
    """

    def __init__(self, logger: DBLogger, mt5_manager,
                 order_manager=None,
                 check_interval: int = 30):
        """
        Initialize the system monitor.

        Args:
            logger: Logger instance
            mt5_manager: MT5 manager instance
            order_manager: Optional order manager instance
            check_interval: Check interval in seconds (default: 30)
        """
        self.logger = logger
        self.mt5_manager = mt5_manager
        self.order_manager = order_manager
        self.check_interval = check_interval

        self._running = False
        self._thread = None
        self._stop_event = threading.Event()

    def start(self) -> bool:
        """Start the system monitor"""
        if self._running:
            return True

        try:
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.daemon = True
            self._thread.start()

            self.logger.log_event(
                level="INFO",
                message="System monitor started",
                event_type="SYSTEM_MONITOR",
                component="system_monitor",
                action="start",
                status="success"
            )
            return True

        except Exception as e:
            self._running = False
            self.logger.log_error(
                level="ERROR",
                message=f"Failed to start system monitor: {str(e)}",
                exception_type=type(e).__name__,
                function="start",
                traceback=str(e),
                context={}
            )
            return False

    def stop(self) -> None:
        """Stop the system monitor"""
        if not self._running:
            return

        try:
            self._stop_event.set()

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=10)

            self._running = False

            self.logger.log_event(
                level="INFO",
                message="System monitor stopped",
                event_type="SYSTEM_MONITOR",
                component="system_monitor",
                action="stop",
                status="success"
            )

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error stopping system monitor: {str(e)}",
                exception_type=type(e).__name__,
                function="stop",
                traceback=str(e),
                context={}
            )
            # Mark as not running even if there was an error
            self._running = False

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        last_gc = time.time()
        gc_interval = 300  # Run garbage collection every 5 minutes

        while not self._stop_event.is_set():
            try:
                # Check MT5 connection
                self._check_mt5_connection()

                # Check system resources if psutil is available
                try:
                    import psutil
                    self._check_system_resources()
                except ImportError:
                    # psutil not available, skip resource check
                    pass

                # Check trading activity
                if self.order_manager:
                    self._check_trading_activity()

                # Run periodic garbage collection
                current_time = time.time()
                if current_time - last_gc >= gc_interval:
                    import gc
                    gc.collect()
                    last_gc = current_time

                    self.logger.log_event(
                        level="INFO",
                        message="Performed garbage collection",
                        event_type="SYSTEM_MONITOR",
                        component="system_monitor",
                        action="_monitor_loop",
                        status="gc_completed"
                    )

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                self.logger.log_error(
                    level="ERROR",
                    message=f"Error in system monitor loop: {str(e)}",
                    exception_type=type(e).__name__,
                    function="_monitor_loop",
                    traceback=str(e),
                    context={}
                )
                # Sleep before retrying to avoid excessive logging
                time.sleep(5)

    def _check_mt5_connection(self) -> None:
        """Check MT5 connection status"""
        try:
            if not self.mt5_manager.ensure_connection():
                self.logger.log_error(
                    level="WARNING",
                    message="MT5 connection lost, attempting to reconnect",
                    exception_type="MT5ConnectionError",
                    function="_check_mt5_connection",
                    traceback="",
                    context={}
                )

                # Connection will be automatically restored by ensure_connection
            else:
                # Get server time to verify active connection
                server_time = self.mt5_manager.get_server_time()
                if server_time:
                    self.logger.log_event(
                        level="DEBUG",
                        message="MT5 connection verified",
                        event_type="SYSTEM_MONITOR",
                        component="system_monitor",
                        action="_check_mt5_connection",
                        status="success",
                        details={"server_time": str(server_time)}
                    )
                else:
                    self.logger.log_error(
                        level="WARNING",
                        message="MT5 connected but cannot get server time",
                        exception_type="MT5DataError",
                        function="_check_mt5_connection",
                        traceback="",
                        context={}
                    )

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error checking MT5 connection: {str(e)}",
                exception_type=type(e).__name__,
                function="_check_mt5_connection",
                traceback=str(e),
                context={}
            )

    def _check_system_resources(self) -> None:
        """Check system resource usage"""
        try:
            import psutil

            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Get process information
            process = psutil.Process()
            process_cpu = process.cpu_percent(interval=1)
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Log if resource usage is high
            if cpu_percent > 80 or memory_percent > 80:
                self.logger.log_event(
                    level="WARNING",
                    message="High system resource usage detected",
                    event_type="SYSTEM_MONITOR",
                    component="system_monitor",
                    action="_check_system_resources",
                    status="high_usage",
                    details={
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "process_cpu": process_cpu,
                        "process_memory_mb": process_memory
                    }
                )
            else:
                self.logger.log_event(
                    level="DEBUG",
                    message="System resources normal",
                    event_type="SYSTEM_MONITOR",
                    component="system_monitor",
                    action="_check_system_resources",
                    status="normal",
                    details={
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "process_cpu": process_cpu,
                        "process_memory_mb": process_memory
                    }
                )

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error checking system resources: {str(e)}",
                exception_type=type(e).__name__,
                function="_check_system_resources",
                traceback=str(e),
                context={}
            )

    def _check_trading_activity(self) -> None:
        """Check trading activity and account status"""
        try:
            # Get open positions
            positions = self.mt5_manager.get_positions()

            # Get account info
            account_info = self.mt5_manager.get_account_info()

            if not account_info:
                self.logger.log_error(
                    level="WARNING",
                    message="Failed to get account information",
                    exception_type="MT5DataError",
                    function="_check_trading_activity",
                    traceback="",
                    context={}
                )
                return

            # Check margin level
            margin_level = account_info.get('margin_level', 0)

            # Warning if margin level is low
            if margin_level < 200 and margin_level > 0:
                self.logger.log_event(
                    level="WARNING",
                    message=f"Low margin level: {margin_level}%",
                    event_type="SYSTEM_MONITOR",
                    component="system_monitor",
                    action="_check_trading_activity",
                    status="low_margin",
                    details={
                        "margin_level": margin_level,
                        "balance": account_info.get('balance'),
                        "equity": account_info.get('equity')
                    }
                )

            # Log trading activity
            self.logger.log_event(
                level="INFO",
                message=f"Trading activity: {len(positions)} open positions",
                event_type="SYSTEM_MONITOR",
                component="system_monitor",
                action="_check_trading_activity",
                status="success",
                details={
                    "open_positions": len(positions),
                    "balance": account_info.get('balance'),
                    "equity": account_info.get('equity'),
                    "margin_level": margin_level
                }
            )

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error checking trading activity: {str(e)}",
                exception_type=type(e).__name__,
                function="_check_trading_activity",
                traceback=str(e),
                context={}
            )