# Logger/logger.py
import logging
import sys
from typing import Optional, Dict, Any, Set
import threading
import json
import os

from Database.db_session import DatabaseSession
from .formatters import ColorFormatter
from .handlers import SQLAlchemyHandler


class DBLogger:
    """SQL-based logger with optional console output using SQLAlchemy"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DBLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, conn_string: str, enabled_levels: Set[str] = None, console_output: bool = True,
                 color_scheme: Dict[str, str] = None):
        with self._lock:
            if self._initialized:
                return

            self.conn_string = conn_string
            self.enabled_levels = enabled_levels or {'ERROR', 'CRITICAL'}
            self.console_output = console_output
            self.color_scheme = color_scheme
            self.logger = logging.getLogger('trading_bot')
            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False

            # Clear existing handlers to avoid duplicates on re-initialization
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

            # SQL Handler (always enabled)
            self.sql_handler = SQLAlchemyHandler(conn_string, max_records=10000)
            self.sql_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(self.sql_handler)

            # Console Handler (conditional)
            if console_output:
                self.console_handler = logging.StreamHandler(sys.stdout)
                self.console_handler.setFormatter(
                    ColorFormatter('%(asctime)s - %(levelname)s - %(message)s', self.color_scheme))

                # Set level based on enabled_levels
                min_level = self._get_min_level()
                self.console_handler.setLevel(min_level)
                self.logger.addHandler(self.console_handler)

            self._initialized = True

    def _get_min_level(self) -> int:
        """Get minimum logging level based on enabled_levels"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if not self.enabled_levels or 'NONE' in self.enabled_levels:
            return logging.CRITICAL + 1  # Higher than any level

        return min(level_map.get(level, logging.CRITICAL) for level in self.enabled_levels)

    def debug(self, message: str, **kwargs):
        """Log a debug message"""
        self.logger.debug(message, extra={'extra': kwargs})

    def info(self, message: str, **kwargs):
        """Log an info message"""
        self.logger.info(message, extra={'extra': kwargs})

    def warning(self, message: str, **kwargs):
        """Log a warning message"""
        self.logger.warning(message, extra={'extra': kwargs})

    def error(self, message: str, **kwargs):
        """Log an error message"""
        self.logger.error(message, extra={'extra': kwargs})

    def critical(self, message: str, **kwargs):
        """Log a critical message"""
        self.logger.critical(message, extra={'extra': kwargs})

    def log_event(self, level: str, message: str, event_type: str, component: str,
                  action: str = None, status: str = "success", details: Dict[str, Any] = None):
        """
        Log an event with enhanced context

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Main log message
            event_type: Type of event (SYSTEM_INIT, CONFIG_CHANGE, TRADE_SIGNAL, etc.)
            component: Component that generated the event
            action: Specific action that was taken
            status: Outcome status (success, failure, pending, etc.)
            details: Additional structured details as a dictionary
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, extra={
            'extra': {
                'entry_type': 'event',
                'event_type': event_type,
                'component': component,
                'action': action,
                'status': status,
                'details': details
            }
        })

    def log_error(self, level: str, message: str, exception_type: str, function: str,
                  traceback: str, context: Dict[str, Any] = None):
        """Log an error with detailed information"""
        log_method = getattr(self.logger, level.lower(), self.logger.error)
        log_method(message, extra={
            'extra': {
                'entry_type': 'error',
                'exception_type': exception_type,
                'function': function,
                'traceback': traceback,
                'context': context
            }
        })

    def log_trade(self, level: str, message: str, symbol: str, operation: str,
                  price: float, volume: float, order_id: Optional[int] = None,
                  strategy: Optional[str] = None):
        """Log a trade operation with details"""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, extra={
            'extra': {
                'entry_type': 'trade',
                'symbol': symbol,
                'operation': operation,
                'price': price,
                'volume': volume,
                'order_id': order_id,
                'strategy': strategy
            }
        })

    def log_with_print(self, level: str, message: str, **kwargs):
        """Force console output regardless of enabled levels"""
        original_console_output = self.console_output
        original_enabled_levels = self.enabled_levels

        try:
            if not self.console_output:
                self.console_output = True
                if not hasattr(self, 'console_handler'):
                    self.console_handler = logging.StreamHandler(sys.stdout)
                    self.console_handler.setFormatter(
                        ColorFormatter('%(asctime)s - %(levelname)s - %(message)s', self.color_scheme))
                    self.logger.addHandler(self.console_handler)

            # Temporarily enable this level
            self.enabled_levels = self.enabled_levels.union({level.upper()})
            self.console_handler.setLevel(getattr(logging, level.upper()))

            # Log with the appropriate method
            log_method = getattr(self, level.lower(), self.info)
            log_method(message, **kwargs)

        finally:
            # Restore original settings
            self.console_output = original_console_output
            self.enabled_levels = original_enabled_levels
            if hasattr(self, 'console_handler'):
                self.console_handler.setLevel(self._get_min_level())