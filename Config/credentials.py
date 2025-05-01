# Config/credentials.py
"""
Trading bot credentials configuration
This file should never be committed to version control
"""

# SQL Server configuration
SQL_SERVER = r"DESKTOP-B1AT4R0\SQLEXPRESS"
SQL_DATABASE = "GeneralTradingBot"
SQL_DRIVER = "ODBC Driver 17 for SQL Server"
USE_WINDOWS_AUTH = True

# MetaTrader5 credentials (for future use)
MT5_SERVER = ""
MT5_LOGIN = 0
MT5_PASSWORD = ""
MT5_TIMEOUT = 60000