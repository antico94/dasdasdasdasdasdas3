# Config/trading_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum
import json
import os
from Config.credentials import (
    SQL_SERVER,
    SQL_DATABASE,
    SQL_DRIVER,
    USE_WINDOWS_AUTH,
    SQL_USERNAME,
    SQL_PASSWORD,
    MT5_SERVER,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_TIMEOUT
)


class TimeFrame(Enum):
    M1  = {"name": "M1", "description": "1 Minute", "minutes": 1, "mt5_timeframe": 1, "history_size": 300}
    M5  = {"name": "M5", "description": "5 Minutes", "minutes": 5, "mt5_timeframe": 5, "history_size": 300}
    M15 = {"name": "M15", "description": "15 Minutes", "minutes": 15, "mt5_timeframe": 15, "history_size": 300}
    M30 = {"name": "M30", "description": "30 Minutes", "minutes": 30, "mt5_timeframe": 30, "history_size": 300}
    H1  = {"name": "H1", "description": "1 Hour", "minutes": 60, "mt5_timeframe": 16385, "history_size": 300}
    H4  = {"name": "H4", "description": "4 Hours", "minutes": 240, "mt5_timeframe": 16388, "history_size": 300}
    D1  = {"name": "D1", "description": "1 Day", "minutes": 1440, "mt5_timeframe": 16408, "history_size": 300}
    W1  = {"name": "W1", "description": "1 Week", "minutes": 10080, "mt5_timeframe": 32769, "history_size": 200}
    MN1 = {"name": "MN1", "description": "1 Month", "minutes": 43200, "mt5_timeframe": 49153, "history_size": 150}

    @property
    def minutes(self):
        return self.value["minutes"]

    @property
    def mt5_timeframe(self):
        return self.value["mt5_timeframe"]

    @property
    def description(self):
        return self.value["description"]

    @property
    def history_size(self):
        return self.value["history_size"]


@dataclass
class InstrumentConfig:
    symbol: str
    description: str
    pip_value: float
    timeframes: List[TimeFrame]
    history_size: Dict[TimeFrame, int] = field(default_factory=dict)

    def __post_init__(self):
        # Set default history size if not specified
        for tf in self.timeframes:
            if tf not in self.history_size:
                self.history_size[tf] = 300


@dataclass
class MT5Config:
    server: str
    login: int
    password: str
    timeout: int = 60000


@dataclass
class DatabaseConfig:
    server: str
    database: str
    driver: str
    trusted_connection: bool
    username: Optional[str] = None
    password: Optional[str] = None

    @property
    def connection_string(self) -> str:
        if self.trusted_connection:
            return f"DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;"
        else:
            return f"DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"


@dataclass
class TradingBotConfig:
    instruments: List[InstrumentConfig]
    mt5: MT5Config
    database: DatabaseConfig
    sync_interval_seconds: int = 10
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5


class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = None
        return cls._instance

    def load_from_file(self, file_path: str) -> None:
        """Load configuration from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r') as f:
            config_data = json.load(f)

        # Create database configuration from credentials
        db_config = DatabaseConfig(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            driver=SQL_DRIVER,
            trusted_connection=USE_WINDOWS_AUTH,
            username=SQL_USERNAME,
            password=SQL_PASSWORD
        )

        # Create MT5 configuration from credentials
        mt5_config = MT5Config(
            server=MT5_SERVER,
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            timeout=MT5_TIMEOUT
        )

        # Parse instruments
        instruments = []
        for instr_data in config_data["instruments"]:
            timeframes = [TimeFrame[tf] for tf in instr_data["timeframes"]]

            # Parse history sizes if provided
            history_size = {}
            if "history_size" in instr_data:
                for tf_str, size in instr_data["history_size"].items():
                    history_size[TimeFrame[tf_str]] = size

            instrument = InstrumentConfig(
                symbol=instr_data["symbol"],
                description=instr_data["description"],
                pip_value=instr_data["pip_value"],
                timeframes=timeframes,
                history_size=history_size
            )
            instruments.append(instrument)

        # Create config object
        self._config = TradingBotConfig(
            instruments=instruments,
            mt5=mt5_config,
            database=db_config,
            sync_interval_seconds=config_data.get("sync_interval_seconds", 10),
            max_retry_attempts=config_data.get("max_retry_attempts", 3),
            retry_delay_seconds=config_data.get("retry_delay_seconds", 5)
        )

    def load_from_dict(self, config_dict: Dict) -> None:
        """Load configuration from dictionary"""
        # Create database configuration from credentials
        db_config = DatabaseConfig(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            driver=SQL_DRIVER,
            trusted_connection=USE_WINDOWS_AUTH,
            username=SQL_USERNAME,
            password=SQL_PASSWORD
        )

        # Create MT5 configuration from credentials
        mt5_config = MT5Config(
            server=MT5_SERVER,
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            timeout=MT5_TIMEOUT
        )

        # Parse instruments
        instruments = []
        for instr_data in config_dict["instruments"]:
            timeframes = [TimeFrame[tf] for tf in instr_data["timeframes"]]

            # Parse history sizes if provided
            history_size = {}
            if "history_size" in instr_data:
                for tf_str, size in instr_data["history_size"].items():
                    history_size[TimeFrame[tf_str]] = size

            instrument = InstrumentConfig(
                symbol=instr_data["symbol"],
                description=instr_data["description"],
                pip_value=instr_data["pip_value"],
                timeframes=timeframes,
                history_size=history_size
            )
            instruments.append(instrument)

        # Create config object
        self._config = TradingBotConfig(
            instruments=instruments,
            mt5=mt5_config,
            database=db_config,
            sync_interval_seconds=config_dict.get("sync_interval_seconds", 10),
            max_retry_attempts=config_dict.get("max_retry_attempts", 3),
            retry_delay_seconds=config_dict.get("retry_delay_seconds", 5)
        )

    def generate_default_config(self) -> Dict:
        """Generate a default configuration dictionary"""
        return {
            "instruments": [
                {
                    "symbol": "EURUSD",
                    "description": "Euro vs US Dollar",
                    "pip_value": 0.0001,
                    "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
                    "history_size": {
                        "M1": 300,
                        "M5": 300,
                        "M15": 300,
                        "H1": 300,
                        "H4": 300,
                        "D1": 300
                    }
                },
                {
                    "symbol": "GBPUSD",
                    "description": "Great Britain Pound vs US Dollar",
                    "pip_value": 0.0001,
                    "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
                    "history_size": {
                        "M1": 300,
                        "M5": 300,
                        "M15": 300,
                        "H1": 300,
                        "H4": 300,
                        "D1": 300
                    }
                },
                {
                    "symbol": "USDJPY",
                    "description": "US Dollar vs Japanese Yen",
                    "pip_value": 0.01,
                    "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
                    "history_size": {
                        "M1": 300,
                        "M5": 300,
                        "M15": 300,
                        "H1": 300,
                        "H4": 300,
                        "D1": 300
                    }
                },
                {
                    "symbol": "XAUUSD",
                    "description": "Gold vs US Dollar",
                    "pip_value": 0.01,
                    "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
                    "history_size": {
                        "M1": 300,
                        "M5": 300,
                        "M15": 300,
                        "H1": 300,
                        "H4": 300,
                        "D1": 300
                    }
                }
            ],
            "sync_interval_seconds": 10,
            "max_retry_attempts": 3,
            "retry_delay_seconds": 5
        }

    def save_default_config(self, file_path: str) -> None:
        """Save default configuration to a file"""
        config = self.generate_default_config()
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)

    @property
    def config(self) -> TradingBotConfig:
        if self._config is None:
            raise ValueError("Configuration not loaded")
        return self._config