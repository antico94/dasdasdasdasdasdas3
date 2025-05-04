# Database/db_manager.py
import sys
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

from sqlalchemy import desc, func, and_
from sqlalchemy.exc import SQLAlchemyError

from Config.trading_config import ConfigManager
from Database.models import Instrument, Timeframe, PriceBar, Base
from Database.db_session import DatabaseSession
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

        # Create SQLAlchemy connection string from credentials
        from Config.credentials import (
            SQL_SERVER,
            SQL_DATABASE,
            SQL_DRIVER,
            USE_WINDOWS_AUTH,
            SQL_USERNAME,
            SQL_PASSWORD
        )

        # Create SQLAlchemy connection string
        if USE_WINDOWS_AUTH:
            conn_string = f"mssql+pyodbc://{SQL_SERVER}/{SQL_DATABASE}?driver={SQL_DRIVER.replace(' ', '+')}&trusted_connection=yes"
        else:
            conn_string = f"mssql+pyodbc://{SQL_USERNAME}:{SQL_PASSWORD}@{SQL_SERVER}/{SQL_DATABASE}?driver={SQL_DRIVER.replace(' ', '+')}"

        self._config = ConfigManager().config
        self._db_session = DatabaseSession(conn_string)
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

            # Create all tables from SQLAlchemy models
            self._db_session.create_all_tables()

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

        except SQLAlchemyError as e:
            self._logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize database: {str(e)}",
                exception_type=type(e).__name__,
                function="initialize_database",
                traceback=str(e),
                context={"conn_string": self._db_session.get_connection_string()}
            )
            sys.exit(1)  # Critical failure, terminate application
        except Exception as e:
            self._logger.log_error(
                level="CRITICAL",
                message=f"Unexpected error initializing database: {str(e)}",
                exception_type=type(e).__name__,
                function="initialize_database",
                traceback=str(e),
                context={}
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

            with self._db_session.session_scope() as session:
                # Insert or update timeframes
                for tf in unique_timeframes:
                    existing = session.query(Timeframe).filter(Timeframe.name == tf.name).first()

                    if not existing:
                        new_timeframe = Timeframe(
                            name=tf.name,
                            description=tf.description,
                            minutes=tf.minutes
                        )
                        session.add(new_timeframe)
                    else:
                        existing.description = tf.description
                        existing.minutes = tf.minutes

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                # Insert or update instruments
                for instrument in self._config.instruments:
                    existing = session.query(Instrument).filter(Instrument.symbol == instrument.symbol).first()

                    if not existing:
                        new_instrument = Instrument(
                            symbol=instrument.symbol,
                            description=instrument.description,
                            pip_value=instrument.pip_value
                        )
                        session.add(new_instrument)
                    else:
                        existing.description = instrument.description
                        existing.pip_value = instrument.pip_value

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                instrument = session.query(Instrument).filter(Instrument.symbol == symbol).first()

                if instrument:
                    return instrument.id
                else:
                    raise ValueError(f"Instrument not found: {symbol}")

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                timeframe = session.query(Timeframe).filter(Timeframe.name == timeframe_name).first()

                if timeframe:
                    return timeframe.id
                else:
                    raise ValueError(f"Timeframe not found: {timeframe_name}")

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                return session.query(Instrument).all()

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                return session.query(Timeframe).all()

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                # Add all bars to session
                for bar in bars:
                    session.add(bar)

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

        except SQLAlchemyError as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to insert price bars: {str(e)}",
                exception_type=type(e).__name__,
                function="insert_price_bars",
                traceback=str(e),
                context={"count": len(bars)}
            )
            raise

    def get_latest_bars(self, instrument_id: int, timeframe_id: int, limit: int = 300) -> List[PriceBar]:
        """Get latest price bars for an instrument and timeframe"""
        try:
            with self._db_session.session_scope() as session:
                # Use subquery to get the latest N timestamps
                subquery = session.query(PriceBar.timestamp) \
                    .filter(and_(PriceBar.instrument_id == instrument_id,
                                 PriceBar.timeframe_id == timeframe_id)) \
                    .order_by(desc(PriceBar.timestamp)) \
                    .limit(limit) \
                    .subquery()

                # Get complete bars for these timestamps
                bars = session.query(PriceBar) \
                    .filter(and_(PriceBar.instrument_id == instrument_id,
                                 PriceBar.timeframe_id == timeframe_id,
                                 PriceBar.timestamp.in_(subquery))) \
                    .order_by(desc(PriceBar.timestamp)) \
                    .all()

                # Return in ascending order (oldest first)
                return list(reversed(bars))

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                result = session.query(func.max(PriceBar.timestamp)) \
                    .filter(and_(PriceBar.instrument_id == instrument_id,
                                 PriceBar.timeframe_id == timeframe_id)) \
                    .scalar()

                return result  # Will be None if no bars exist

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                # Get current count
                count = session.query(func.count(PriceBar.id)) \
                    .filter(and_(PriceBar.instrument_id == instrument_id,
                                 PriceBar.timeframe_id == timeframe_id)) \
                    .scalar()

                # If over limit, delete oldest bars
                if count > max_bars:
                    delete_count = count - max_bars

                    # Get cutoff timestamp
                    cutoff_timestamp = session.query(PriceBar.timestamp) \
                        .filter(and_(PriceBar.instrument_id == instrument_id,
                                     PriceBar.timeframe_id == timeframe_id)) \
                        .order_by(PriceBar.timestamp) \
                        .offset(delete_count - 1) \
                        .limit(1) \
                        .scalar()

                    # Delete bars older than cutoff
                    deleted = session.query(PriceBar) \
                        .filter(and_(PriceBar.instrument_id == instrument_id,
                                     PriceBar.timeframe_id == timeframe_id,
                                     PriceBar.timestamp <= cutoff_timestamp)) \
                        .delete(synchronize_session=False)

                    self._logger.log_event(
                        level="INFO",
                        message=f"Deleted {deleted} oldest bars to maintain limit",
                        event_type="DATA_MAINTENANCE",
                        component="database_manager",
                        action="maintain_bar_limit",
                        status="success",
                        details={"instrument_id": instrument_id, "timeframe_id": timeframe_id,
                                 "max_bars": max_bars, "deleted": deleted}
                    )

        except SQLAlchemyError as e:
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
            with self._db_session.session_scope() as session:
                for bar in bars:
                    # Check if record exists
                    existing = session.query(PriceBar) \
                        .filter(and_(PriceBar.instrument_id == bar.instrument_id,
                                     PriceBar.timeframe_id == bar.timeframe_id,
                                     PriceBar.timestamp == bar.timestamp)) \
                        .first()

                    if existing:
                        # Update existing record
                        existing.open = bar.open
                        existing.high = bar.high
                        existing.low = bar.low
                        existing.close = bar.close
                        existing.volume = bar.volume
                        existing.spread = bar.spread
                    else:
                        # Add new record
                        session.add(bar)

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

        except SQLAlchemyError as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to upsert price bars: {str(e)}",
                exception_type=type(e).__name__,
                function="upsert_price_bars",
                traceback=str(e),
                context={"count": len(bars)}
            )
            raise

    # First, modify the DatabaseManager methods to return detached data

    # In Database/db_manager.py
    def get_all_timeframes_detached(self) -> List[dict]:
        """Get all timeframes as detached dictionaries"""
        try:
            with self._db_session.session_scope() as session:
                timeframes = session.query(Timeframe).all()
                # Convert to detached dictionaries
                return [{"id": tf.id, "name": tf.name, "description": tf.description, "minutes": tf.minutes} for tf in
                        timeframes]
        except SQLAlchemyError as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to get detached timeframes: {str(e)}",
                exception_type=type(e).__name__,
                function="get_all_timeframes_detached",
                traceback=str(e),
                context={}
            )
            raise

    def get_all_instruments_detached(self) -> List[dict]:
        """Get all instruments as detached dictionaries"""
        try:
            with self._db_session.session_scope() as session:
                instruments = session.query(Instrument).all()
                # Convert to detached dictionaries
                return [{"id": instr.id, "symbol": instr.symbol, "description": instr.description} for instr in
                        instruments]
        except SQLAlchemyError as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to get detached instruments: {str(e)}",
                exception_type=type(e).__name__,
                function="get_all_instruments_detached",
                traceback=str(e),
                context={}
            )
            raise