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

    def __init__(self, conn_string: str = None, enabled_levels: Set[str] = None, console_output: bool = True,
                 color_scheme: Dict[str, str] = None, component_name: str = None):
        with self._lock:
            # For a component-specific logger, we should already have a base instance initialized
            if component_name and self._initialized:
                return

            # If this is a new base instance, we need a connection string
            if not self._initialized and conn_string is None:
                raise ValueError("Connection string is required for initial logger setup")

            # Initialize base logger if not done yet
            if not self._initialized:
                self._initialize_base_logger(conn_string, enabled_levels, console_output, color_scheme)

    def _initialize_base_logger(self, conn_string, enabled_levels, console_output, color_scheme):
        """Initialize the base logger with global settings"""
        self.conn_string = conn_string
        self.enabled_levels = enabled_levels or {'WARNING', 'ERROR', 'CRITICAL'}
        self.console_output = console_output
        self.color_scheme = color_scheme
        self.config = None  # Will hold the LoggingConfig when loaded
        self._component_loggers = {}  # Store component-specific loggers

        # Create the main logger
        self.logger = logging.getLogger('trading_bot')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # SQL Handler (always enabled)
        self.sql_handler = SQLAlchemyHandler(conn_string, max_records=10000)
        self.sql_handler.setLevel(logging.DEBUG)  # Log everything to database
        self.logger.addHandler(self.sql_handler)

        # Console Handler (conditional)
        self.console_handler = None
        if console_output:
            self._setup_console_handler(self.enabled_levels, self.color_scheme)

        self._initialized = True

    def _setup_console_handler(self, enabled_levels, color_scheme):
        """Set up console handler with appropriate levels"""
        if self.console_handler:
            self.logger.removeHandler(self.console_handler)

        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(
            ColorFormatter('%(asctime)s - %(levelname)s - %(message)s', color_scheme))

        # Set level based on enabled_levels
        min_level = self._get_min_level(enabled_levels)
        self.console_handler.setLevel(min_level)
        self.logger.addHandler(self.console_handler)

    def _get_min_level(self, enabled_levels):
        """Get minimum logging level based on enabled_levels"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if not enabled_levels or 'NONE' in enabled_levels:
            return logging.CRITICAL + 1  # Higher than any level

        return min(level_map.get(level, logging.CRITICAL) for level in enabled_levels)

    # Add factory method for component loggers
    @classmethod
    def get_component_logger(cls, component_name: str):
        """Get a logger for a specific component"""
        if cls._instance is None or not cls._instance._initialized:
            raise RuntimeError("Base logger must be initialized before creating component loggers")

        # Return a new logger instance with the component name
        logger = cls(component_name=component_name)
        return logger

    def load_config(self, config):
        """Load configuration from LoggingConfig object"""
        self.config = config
        self.conn_string = config.conn_string
        self.enabled_levels = config.enabled_levels
        self.console_output = config.console_output
        self.color_scheme = config.color_scheme

        # Update base logger with new settings
        if self.console_output:
            self._setup_console_handler(self.enabled_levels, self.color_scheme)
        elif self.console_handler:
            self.logger.removeHandler(self.console_handler)
            self.console_handler = None

        # Update SQL handler with max_records
        if hasattr(self.sql_handler, 'max_records'):
            self.sql_handler.max_records = config.max_records

        # Initialize component loggers based on config
        self._init_component_loggers()

    def _init_component_loggers(self):
        """Initialize loggers for all components in config"""
        if not self.config or not hasattr(self.config, 'component_configs'):
            return

        for component_name, comp_config in self.config.component_configs.items():
            self._init_component_logger(component_name, comp_config)

    def _init_component_logger(self, component_name, comp_config=None):
        """Initialize a component-specific logger"""
        if comp_config is None and self.config and hasattr(self.config, 'component_configs'):
            comp_config = self.config.component_configs.get(component_name, {})
        elif comp_config is None:
            comp_config = {}

        # Get component settings (fallback to base logger settings)
        comp_console_output = comp_config.get('console_output', self.console_output)
        comp_enabled_levels = comp_config.get('enabled_levels', self.enabled_levels)

        # Create component logger (child of base logger)
        component_logger = logging.getLogger(f'trading_bot.{component_name}')
        component_logger.setLevel(logging.DEBUG)
        component_logger.propagate = False

        # Clear existing handlers
        for handler in component_logger.handlers[:]:
            component_logger.removeHandler(handler)

        # Always add SQL handler
        component_logger.addHandler(self.sql_handler)

        # Add console handler if enabled for this component
        if comp_console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                ColorFormatter('%(asctime)s - %(levelname)s - %(message)s', self.color_scheme))
            min_level = self._get_min_level(comp_enabled_levels)
            console_handler.setLevel(min_level)
            component_logger.addHandler(console_handler)

        # Store the component logger
        self._component_loggers[component_name] = component_logger

    def get_logger(self, component_name=None):
        """Get a logger for the specified component or the default logger"""
        if component_name is None:
            return self.logger

        # Initialize component logger if not done yet
        if component_name not in self._component_loggers:
            self._init_component_logger(component_name)

        return self._component_loggers.get(component_name, self.logger)

    # Core logging methods using the appropriate logger
    def debug(self, message: str, component: str = None, **kwargs):
        """Log a debug message"""
        logger = self.get_logger(component) if component else self.logger
        logger.debug(message, extra={'extra': kwargs})

    def info(self, message: str, component: str = None, **kwargs):
        """Log an info message"""
        logger = self.get_logger(component) if component else self.logger
        logger.info(message, extra={'extra': kwargs})

    def warning(self, message: str, component: str = None, **kwargs):
        """Log a warning message"""
        logger = self.get_logger(component) if component else self.logger
        logger.warning(message, extra={'extra': kwargs})

    def error(self, message: str, component: str = None, **kwargs):
        """Log an error message"""
        logger = self.get_logger(component) if component else self.logger
        logger.error(message, extra={'extra': kwargs})

    def critical(self, message: str, component: str = None, **kwargs):
        """Log a critical message"""
        logger = self.get_logger(component) if component else self.logger
        logger.critical(message, extra={'extra': kwargs})

    def log_event(self, level: str, message: str, event_type: str, component: str,
                  action: str = None, status: str = "success", details: Dict[str, Any] = None):
        """Log an event with enhanced context, with truncation protection"""
        # Get the appropriate logger for this component
        logger = self.get_logger(component)

        # Ensure message fits within database field limits (500 chars)
        if len(message) > 450:
            message = message[:447] + "..."

        # Ensure event_type, component, action, status fields stay within limits
        if event_type and len(event_type) > 45:
            event_type = event_type[:42] + "..."

        if component and len(component) > 45:
            component = component[:42] + "..."

        if action and len(action) > 95:
            action = action[:92] + "..."

        if status and len(status) > 15:
            status = status[:12] + "..."

        # Convert details to a simple short string to avoid truncation issues
        safe_details = None
        if details:
            try:
                # First try to serialize with minimal information
                simple_details = {}
                for k, v in details.items():
                    if isinstance(v, (int, float, bool, str)):
                        # For strings, limit length
                        if isinstance(v, str) and len(v) > 20:
                            simple_details[k] = v[:17] + "..."
                        else:
                            simple_details[k] = v
                    else:
                        # For complex types, just use type name
                        simple_details[k] = f"{type(v).__name__}"

                # Serialize and check length
                json_details = json.dumps(simple_details)
                if len(json_details) > 450:  # Be conservative with JSON size
                    # If still too long, create a minimal version
                    safe_details = json.dumps({"truncated": True, "original_keys": list(details.keys())[:5]})
                else:
                    safe_details = json_details
            except Exception:
                # If serialization fails, provide a fallback
                safe_details = json.dumps({"error": "Could not serialize details"})

        # Call the logger method with safe parameters
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message, extra={
            'extra': {
                'entry_type': 'event',
                'event_type': event_type,
                'component': component,
                'action': action,
                'status': status,
                'details': safe_details
            }
        })

    def log_error(self, level: str, message: str, exception_type: str, function: str,
                  traceback: str, context: Dict[str, Any] = None):
        """Log an error with detailed information"""
        try:
            # Truncate message, traceback and context if too long to prevent SQL errors
            max_message_length = 450  # Below SQL field size limit
            max_traceback_length = 1000  # Adjust based on your database field size

            # Truncate message if needed
            if len(message) > max_message_length:
                message = message[:max_message_length] + "..."

            # Truncate traceback if needed
            if traceback and len(traceback) > max_traceback_length:
                traceback = traceback[:max_traceback_length] + "..."

            # Truncate context values if needed
            if context:
                for key, value in context.items():
                    if isinstance(value, str) and len(value) > max_message_length:
                        context[key] = value[:max_message_length] + "..."

            # Convert context to JSON string or None
            context_json = json.dumps(context) if context else None

            # Check if the resulting JSON string is too long and truncate if needed
            if context_json and len(context_json) > 4000:  # Adjust based on your database field size
                context_json = context_json[:4000] + "..."

            # Determine the appropriate log method
            log_method = getattr(self.logger, level.lower(), self.logger.error)

            # Log the error with truncated values
            log_method(message, extra={
                'extra': {
                    'entry_type': 'error',
                    'exception_type': exception_type,
                    'function': function,
                    'traceback': traceback,
                    'context': context_json
                }
            })

        except Exception as e:
            # Fallback to simple stderr logging if there's an error in the logging itself
            sys.stderr.write(f"Error logging error: {str(e)}. Original error: {message}\n")

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