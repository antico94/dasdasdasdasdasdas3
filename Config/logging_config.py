from dataclasses import dataclass, field
from typing import Set, Dict, Any


@dataclass
class LoggingConfig:
    # Database connection configuration
    conn_string: str
    max_records: int = 10000

    # Console output configuration
    console_output: bool = True
    enabled_levels: Set[str] = field(default_factory=lambda: {'WARNING', 'ERROR', 'CRITICAL'})
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        'DEBUG': '\033[37m',  # White
        'INFO': '\033[36m',  # Cyan
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41m',  # Red background
    })

    # Component-specific logging configuration
    # Can be extended for each strategy, pair, etc.
    component_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'data_fetcher': {
            'enabled_levels': {'WARNING', 'ERROR', 'CRITICAL'},
            'console_output': True
        },
        'trade_executor': {
            'enabled_levels': {'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            'console_output': True
        }
        # Add more component-specific configs as needed
    })