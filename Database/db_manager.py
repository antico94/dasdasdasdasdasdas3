# Database/db_manager.py
import sys
import threading
from datetime import datetime
from typing import List, Optional

import pyodbc

from Config.trading_config import ConfigManager
from Database.models import Instrument, Timeframe, PriceBar
from Logger.logger import DBLogger


class DatabaseManager:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Create connection string from credentials
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

        self._config = ConfigManager().config
        self._conn_string = conn_string
        self._logger = DBLogger(
            conn_string=conn_string,
            enabled_levels={'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            console_output=True
        )
        self._initialized = True

    def initialize_database(self) -> None:
        """Initialize database schema and tables"""
        try:
            self._logger.log_event(
                level="INFO",
                message="Initializing database schema",
                event_type="DATABASE_INIT",
                component="database_manager",
                action="initialize_database",
                status="starting"
            )

            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()

                # Create Trading schema if it doesn't exist
                cursor.execute(
                    "IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'Trading') EXEC('CREATE SCHEMA Trading')")

                # Create Instruments table
                cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Instruments' AND schema_id = SCHEMA_ID('Trading'))
                BEGIN
                    CREATE TABLE Trading.Instruments (
                        Id INT IDENTITY(1,1) PRIMARY KEY,
                        Symbol NVARCHAR(20) NOT NULL UNIQUE,
                        Description NVARCHAR(100) NOT NULL,
                        PipValue FLOAT NOT NULL
                    )
                END
                """)

                # Create Timeframes table
                cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Timeframes' AND schema_id = SCHEMA_ID('Trading'))
                BEGIN
                    CREATE TABLE Trading.Timeframes (
                        Id INT IDENTITY(1,1) PRIMARY KEY,
                        Name NVARCHAR(10) NOT NULL UNIQUE,
                        Description NVARCHAR(50) NOT NULL,
                        Minutes INT NOT NULL
                    )
                END
                """)

                # Create PriceData table
                cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'PriceData' AND schema_id = SCHEMA_ID('Trading'))
                BEGIN
                    CREATE TABLE Trading.PriceData (
                        Id INT IDENTITY(1,1) PRIMARY KEY,
                        InstrumentId INT NOT NULL,
                        TimeframeId INT NOT NULL,
                        Timestamp DATETIME2 NOT NULL,
                        [Open] FLOAT NOT NULL,
                        [High] FLOAT NOT NULL,
                        [Low] FLOAT NOT NULL,
                        [Close] FLOAT NOT NULL,
                        [Volume] FLOAT NOT NULL,
                        [Spread] FLOAT NULL,
                        CONSTRAINT FK_PriceData_Instrument FOREIGN KEY (InstrumentId)
                            REFERENCES Trading.Instruments(Id),
                        CONSTRAINT FK_PriceData_Timeframe FOREIGN KEY (TimeframeId)
                            REFERENCES Trading.Timeframes(Id),
                        CONSTRAINT UQ_PriceData_InstrumentTimeframeTimestamp UNIQUE
                            (InstrumentId, TimeframeId, Timestamp)
                    )
                END
                """)

                # Create index on Timestamp for efficient querying
                cursor.execute("""
                IF NOT EXISTS (
                    SELECT * FROM sys.indexes
                    WHERE name = 'IX_PriceData_Timestamp'
                    AND object_id = OBJECT_ID('Trading.PriceData')
                )
                BEGIN
                    CREATE INDEX IX_PriceData_Timestamp
                    ON Trading.PriceData (InstrumentId, TimeframeId, Timestamp DESC)
                END
                """)

                conn.commit()

            # Initialize timeframes from configuration
            self._initialize_timeframes()

            # Initialize instruments from configuration
            self._initialize_instruments()

            self._logger.log_event(
                level="INFO",
                message="Database schema initialized successfully",
                event_type="DATABASE_INIT",
                component="database_manager",
                action="initialize_database",
                status="success"
            )

        except Exception as e:
            self._logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize database: {str(e)}",
                exception_type=type(e).__name__,
                function="initialize_database",
                traceback=str(e),
                context={"conn_string": self._conn_string}
            )
            sys.exit(1)  # Critical failure, terminate application

    def _initialize_timeframes(self) -> None:
        """Initialize timeframes from configuration"""
        try:
            # Get unique timeframes from all instruments
            unique_timeframes = set()
            for instrument in self._config.instruments:
                for tf in instrument.timeframes:
                    unique_timeframes.add(tf)

            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()

                # Insert or update timeframes
                for tf in unique_timeframes:
                    cursor.execute("""
                    IF NOT EXISTS (SELECT 1 FROM Trading.Timeframes WHERE Name = ?)
                    BEGIN
                        INSERT INTO Trading.Timeframes (Name, Description, Minutes)
                        VALUES (?, ?, ?)
                    END
                    ELSE
                    BEGIN
                        UPDATE Trading.Timeframes
                        SET Description = ?, Minutes = ?
                        WHERE Name = ?
                    END
                    """,
                                   tf.name,
                                   tf.name, tf.description, tf.minutes,
                                   tf.description, tf.minutes, tf.name)

                conn.commit()

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to initialize timeframes: {str(e)}",
                exception_type=type(e).__name__,
                function="_initialize_timeframes",
                traceback=str(e),
                context={}
            )
            raise

    def _initialize_instruments(self) -> None:
        """Initialize instruments from configuration"""
        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()

                # Insert or update instruments
                for instrument in self._config.instruments:
                    cursor.execute("""
                    IF NOT EXISTS (SELECT 1 FROM Trading.Instruments WHERE Symbol = ?)
                    BEGIN
                        INSERT INTO Trading.Instruments (Symbol, Description, PipValue)
                        VALUES (?, ?, ?)
                    END
                    ELSE
                    BEGIN
                        UPDATE Trading.Instruments
                        SET Description = ?, PipValue = ?
                        WHERE Symbol = ?
                    END
                    """,
                                   instrument.symbol,
                                   instrument.symbol, instrument.description, instrument.pip_value,
                                   instrument.description, instrument.pip_value, instrument.symbol)

                conn.commit()

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to initialize instruments: {str(e)}",
                exception_type=type(e).__name__,
                function="_initialize_instruments",
                traceback=str(e),
                context={}
            )
            raise

    def get_instrument_id(self, symbol: str) -> int:
        """Get instrument ID by symbol"""
        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT Id FROM Trading.Instruments WHERE Symbol = ?", symbol)
                result = cursor.fetchone()

                if result:
                    return result[0]
                else:
                    raise ValueError(f"Instrument not found: {symbol}")

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to get instrument ID: {str(e)}",
                exception_type=type(e).__name__,
                function="get_instrument_id",
                traceback=str(e),
                context={"symbol": symbol}
            )
            raise

    def get_timeframe_id(self, timeframe_name: str) -> int:
        """Get timeframe ID by name"""
        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT Id FROM Trading.Timeframes WHERE Name = ?", timeframe_name)
                result = cursor.fetchone()

                if result:
                    return result[0]
                else:
                    raise ValueError(f"Timeframe not found: {timeframe_name}")

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to get timeframe ID: {str(e)}",
                exception_type=type(e).__name__,
                function="get_timeframe_id",
                traceback=str(e),
                context={"timeframe_name": timeframe_name}
            )
            raise

    def get_all_instruments(self) -> List[Instrument]:
        """Get all instruments from database"""
        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT Id, Symbol, Description, PipValue FROM Trading.Instruments")

                instruments = []
                for row in cursor.fetchall():
                    instruments.append(Instrument.from_db_row(row))

                return instruments

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to get all instruments: {str(e)}",
                exception_type=type(e).__name__,
                function="get_all_instruments",
                traceback=str(e),
                context={}
            )
            raise

    def get_all_timeframes(self) -> List[Timeframe]:
        """Get all timeframes from database"""
        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT Id, Name, Description, Minutes FROM Trading.Timeframes")

                timeframes = []
                for row in cursor.fetchall():
                    timeframes.append(Timeframe.from_db_row(row))

                return timeframes

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to get all timeframes: {str(e)}",
                exception_type=type(e).__name__,
                function="get_all_timeframes",
                traceback=str(e),
                context={}
            )
            raise

    def insert_price_bars(self, bars: List[PriceBar]) -> None:
        """Insert multiple price bars into database"""
        if not bars:
            return

        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()

                # Use transaction for better performance
                cursor.execute("BEGIN TRANSACTION")

                for bar in bars:
                    # *** FIX: Removed leading space inside brackets for [Open] and [Close] ***
                    cursor.execute("""
                                   INSERT INTO Trading.PriceData
                                   (InstrumentId, TimeframeId, Timestamp, [Open], [High], [Low], [Close], [Volume], [Spread])
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                   """,
                                   bar.instrument_id, bar.timeframe_id, bar.timestamp,
                                   bar.open, bar.high, bar.low, bar.close,
                                   bar.volume, bar.spread)

                cursor.execute("COMMIT")

            self._logger.log_event(
                level="INFO",
                message=f"Inserted {len(bars)} price bars",
                event_type="DATA_INSERT",
                component="database_manager",
                action="insert_price_bars",
                status="success",
                details={"count": len(bars), "instrument_id": bars[0].instrument_id,
                         "timeframe_id": bars[0].timeframe_id}
            )

        except pyodbc.Error as e: # Catch pyodbc specific errors for more details
            sqlstate = e.args[0]
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to insert price bars (SQLSTATE: {sqlstate}): {str(e)}",
                exception_type=type(e).__name__,
                function="insert_price_bars",
                traceback=str(e),
                context={"count": len(bars), "sqlstate": sqlstate}
            )
            # Rollback transaction on error
            try:
                conn.rollback()
            except Exception as rb_ex:
                 self._logger.log_error(level="ERROR", message=f"Failed to rollback transaction: {rb_ex}", function="insert_price_bars")
            raise # Re-raise the original exception
        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to insert price bars: {str(e)}",
                exception_type=type(e).__name__,
                function="insert_price_bars",
                traceback=str(e),
                context={"count": len(bars)}
            )
             # Rollback transaction on error
            try:
                conn.rollback()
            except Exception as rb_ex:
                 self._logger.log_error(level="ERROR", message=f"Failed to rollback transaction: {rb_ex}", function="insert_price_bars")
            raise # Re-raise the original exception

    def get_latest_bars(self, instrument_id: int, timeframe_id: int, limit: int = 300) -> List[PriceBar]:
        """Get latest price bars for an instrument and timeframe"""
        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                               SELECT TOP(?)
                                          Id,
                                      InstrumentId,
                                      TimeframeId, Timestamp, [Open], [High], [Low], [Close], [Volume], [Spread]
                               FROM Trading.PriceData
                               WHERE InstrumentId = ? AND TimeframeId = ?
                               ORDER BY Timestamp DESC
                               """, limit, instrument_id, timeframe_id)

                bars = []
                for row in cursor.fetchall():
                    bars.append(PriceBar.from_db_row(row))

                # Return in ascending order (oldest first)
                return list(reversed(bars))

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to get latest bars: {str(e)}",
                exception_type=type(e).__name__,
                function="get_latest_bars",
                traceback=str(e),
                context={"instrument_id": instrument_id, "timeframe_id": timeframe_id, "limit": limit}
            )
            raise

    def get_latest_timestamp(self, instrument_id: int, timeframe_id: int) -> Optional[datetime]:
        """Get timestamp of most recent bar for an instrument and timeframe"""
        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                               SELECT TOP 1 Timestamp
                               FROM Trading.PriceData
                               WHERE InstrumentId = ? AND TimeframeId = ?
                               ORDER BY Timestamp DESC
                               """, instrument_id, timeframe_id)

                result = cursor.fetchone()
                return result[0] if result else None

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to get latest timestamp: {str(e)}",
                exception_type=type(e).__name__,
                function="get_latest_timestamp",
                traceback=str(e),
                context={"instrument_id": instrument_id, "timeframe_id": timeframe_id}
            )
            raise

    def maintain_bar_limit(self, instrument_id: int, timeframe_id: int, max_bars: int) -> None:
        """Delete oldest bars to maintain maximum number of bars"""
        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()

                # Get current count
                cursor.execute("""
                               SELECT COUNT(*)
                               FROM Trading.PriceData
                               WHERE InstrumentId = ?
                                 AND TimeframeId = ?
                               """, instrument_id, timeframe_id)

                count = cursor.fetchone()[0]

                # If over limit, delete oldest bars
                if count > max_bars:
                    delete_count = count - max_bars

                    cursor.execute("""
                                   WITH RowsToDelete AS (
                                       SELECT TOP(?) Id
                                       FROM Trading.PriceData
                                       WHERE InstrumentId = ?
                                         AND TimeframeId = ?
                                       ORDER BY Timestamp ASC
                                   )
                                   DELETE FROM RowsToDelete
                                   """, delete_count, instrument_id, timeframe_id)

                    conn.commit()

                    self._logger.log_event(
                        level="INFO",
                        message=f"Deleted {delete_count} oldest bars to maintain limit",
                        event_type="DATA_MAINTENANCE",
                        component="database_manager",
                        action="maintain_bar_limit",
                        status="success",
                        details={"instrument_id": instrument_id, "timeframe_id": timeframe_id, "max_bars": max_bars,
                                 "deleted": delete_count}
                    )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to maintain bar limit: {str(e)}",
                exception_type=type(e).__name__,
                function="maintain_bar_limit",
                traceback=str(e),
                context={"instrument_id": instrument_id, "timeframe_id": timeframe_id, "max_bars": max_bars}
            )
            raise

    def upsert_price_bars(self, bars: List[PriceBar]) -> None:
        """Insert or update price bars in database"""
        if not bars:
            return

        try:
            with pyodbc.connect(self._conn_string) as conn:
                cursor = conn.cursor()

                # Use transaction for better performance
                # Note: Using MERGE might be more efficient if available and syntax is suitable
                cursor.execute("BEGIN TRANSACTION")

                for bar in bars:
                    # Check if the record exists
                    cursor.execute("""
                        SELECT COUNT(1) FROM Trading.PriceData
                        WHERE InstrumentId = ? AND TimeframeId = ? AND Timestamp = ?
                    """, bar.instrument_id, bar.timeframe_id, bar.timestamp)
                    exists = cursor.fetchone()[0] > 0

                    if exists:
                        # Update existing record
                        cursor.execute("""
                            UPDATE Trading.PriceData
                            SET [Open] = ?, [High] = ?, [Low] = ?, [Close] = ?, [Volume] = ?, [Spread] = ?
                            WHERE InstrumentId = ? AND TimeframeId = ? AND Timestamp = ?
                        """,
                                       bar.open, bar.high, bar.low, bar.close, bar.volume, bar.spread,
                                       bar.instrument_id, bar.timeframe_id, bar.timestamp)
                    else:
                        # Insert new record
                        cursor.execute("""
                            INSERT INTO Trading.PriceData
                            (InstrumentId, TimeframeId, Timestamp, [Open], [High], [Low], [Close], [Volume], [Spread])
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                                       bar.instrument_id, bar.timeframe_id, bar.timestamp,
                                       bar.open, bar.high, bar.low, bar.close, bar.volume, bar.spread)

                cursor.execute("COMMIT")

            self._logger.log_event(
                level="INFO",
                message=f"Upserted {len(bars)} price bars",
                event_type="DATA_UPSERT",
                component="database_manager",
                action="upsert_price_bars",
                status="success",
                details={"count": len(bars), "instrument_id": bars[0].instrument_id,
                         "timeframe_id": bars[0].timeframe_id}
            )

        except pyodbc.Error as e: # Catch pyodbc specific errors
            sqlstate = e.args[0]
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to upsert price bars (SQLSTATE: {sqlstate}): {str(e)}",
                exception_type=type(e).__name__,
                function="upsert_price_bars",
                traceback=str(e),
                context={"count": len(bars), "sqlstate": sqlstate}
            )
            # Rollback transaction on error
            try:
                conn.rollback()
            except Exception as rb_ex:
                self._logger.log_error(level="ERROR", message=f"Failed to rollback transaction: {rb_ex}", function="upsert_price_bars")
            raise # Re-raise the original exception
        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to upsert price bars: {str(e)}",
                exception_type=type(e).__name__,
                function="upsert_price_bars",
                traceback=str(e),
                context={"count": len(bars)}
            )
            # Rollback transaction on error
            try:
                conn.rollback()
            except Exception as rb_ex:
                self._logger.log_error(level="ERROR", message=f"Failed to rollback transaction: {rb_ex}", function="upsert_price_bars")
            raise # Re-raise the original exception