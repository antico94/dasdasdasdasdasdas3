# Config/db_config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """SQL Server database configuration with Windows authentication"""
    server: str
    database: str
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = True  # Windows authentication

    # Optional parameters for SQL authentication (not used with Windows auth)
    username: Optional[str] = None
    password: Optional[str] = None

    @property
    def connection_string(self) -> str:
        """Generate connection string based on configuration"""
        if self.trusted_connection:
            # Windows authentication
            return f"DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;"
        else:
            # SQL authentication
            return f"DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"