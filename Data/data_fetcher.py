# Data/data_fetcher.py
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
import time
import sys
import traceback

from sqlalchemy import and_, desc, func
from sqlalchemy.exc import SQLAlchemyError

from Config.trading_config import ConfigManager, TimeFrame, InstrumentConfig
from Database.db_session import DatabaseSession
from Database.models import PriceBar, Instrument, Timeframe as DbTimeframe
from MT5.mt5_manager import MT5Manager
from Logger.logger import DBLogger
from Events.event_bus import EventBus
from Events.events import NewBarEvent


class DataFetcher:
    """Fetches data from MT5 and stores it in the database"""

    def __init__(self):
        # Create connection string from credentials
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
        self._mt5_manager = MT5Manager()
        self._logger = DBLogger(
            conn_string=conn_string,
            enabled_levels={'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            console_output=True
        )
        self._event_bus = EventBus()
        self._instrument_timeframe_map = {}
        self._running = False
        self._fetch_thread = None
        self._stop_event = threading.Event()

    def initialize(self) -> bool:
        """Initialize data fetcher"""
        try:
            self._logger.log_event(
                level="INFO",
                message="Initializing data fetcher",
                event_type="DATA_FETCHER_INIT",
                component="data_fetcher",
                action="initialize",
                status="starting"
            )

            # Initialize database connection
            if not self._db_session.initialize():
                self._logger.log_error(
                    level="CRITICAL",
                    message="Failed to initialize database session",
                    exception_type="DBConnectionError",
                    function="initialize",
                    traceback="",
                    context={}
                )
                return False

            # Initialize MT5 connection
            if not self._mt5_manager.initialize():
                self._logger.log_error(
                    level="CRITICAL",
                    message="Failed to initialize MT5 connection",
                    exception_type="MT5ConnectionError",
                    function="initialize",
                    traceback="",
                    context={}
                )
                return False

            # Build instrument-timeframe mapping
            self._build_instrument_timeframe_map()

            # Verify symbols availability
            if not self._verify_symbols():
                return False

            # Check for initial data and load if needed
            if not self._load_initial_data():
                return False

            self._logger.log_event(
                level="INFO",
                message="Data fetcher initialized successfully",
                event_type="DATA_FETCHER_INIT",
                component="data_fetcher",
                action="initialize",
                status="success"
            )

            return True

        except Exception as e:
            self._logger.log_error(
                level="CRITICAL",
                message=f"Failed to initialize data fetcher: {str(e)}",
                exception_type=type(e).__name__,
                function="initialize",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def _build_instrument_timeframe_map(self) -> None:
        """Build mapping of instrument ID and timeframe ID"""
        try:
            instrument_map = {}
            timeframe_map = {}

            # Get all instruments and timeframes in a single session
            with self._db_session.session_scope() as session:
                # Get all instruments
                db_instruments = session.query(Instrument).all()
                for instr in db_instruments:
                    # Store only the ID and symbol to avoid session binding issues
                    instrument_map[instr.symbol] = instr.id

                # Get all timeframes
                db_timeframes = session.query(DbTimeframe).all()
                for tf in db_timeframes:
                    # Store only the ID and name to avoid session binding issues
                    timeframe_map[tf.name] = tf.id

            # Now build the mapping using the extracted IDs
            for instrument in self._config.instruments:
                if instrument.symbol not in instrument_map:
                    self._logger.log_error(
                        level="ERROR",
                        message=f"Instrument not found in database: {instrument.symbol}",
                        exception_type="ConfigError",
                        function="_build_instrument_timeframe_map",
                        traceback="",
                        context={"symbol": instrument.symbol}
                    )
                    continue

                instrument_id = instrument_map[instrument.symbol]

                for tf in instrument.timeframes:
                    if tf.name not in timeframe_map:
                        self._logger.log_error(
                            level="ERROR",
                            message=f"Timeframe not found in database: {tf.name}",
                            exception_type="ConfigError",
                            function="_build_instrument_timeframe_map",
                            traceback="",
                            context={"timeframe": tf.name}
                        )
                        continue

                    timeframe_id = timeframe_map[tf.name]

                    # Get history size
                    history_size = instrument.history_size.get(tf, 300)

                    # Store in map
                    key = (instrument.symbol, tf)
                    self._instrument_timeframe_map[key] = {
                        "instrument_id": instrument_id,
                        "timeframe_id": timeframe_id,
                        "history_size": history_size
                    }

            self._logger.log_event(
                level="INFO",
                message=f"Built instrument-timeframe map with {len(self._instrument_timeframe_map)} entries",
                event_type="DATA_FETCHER_CONFIG",
                component="data_fetcher",
                action="_build_instrument_timeframe_map",
                status="success",
                details={"map_size": len(self._instrument_timeframe_map)}
            )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to build instrument-timeframe map: {str(e)}",
                exception_type=type(e).__name__,
                function="_build_instrument_timeframe_map",
                traceback=traceback.format_exc(),
                context={}
            )
            raise

    def _verify_symbols(self) -> bool:
        """Verify that all configured symbols are available in MT5"""
        try:
            mt5_symbols = self._mt5_manager.get_symbols()

            if mt5_symbols is None:
                self._logger.log_error(
                    level="CRITICAL",
                    message="Failed to get symbols from MT5",
                    exception_type="MT5DataError",
                    function="_verify_symbols",
                    traceback="",
                    context={}
                )
                return False

            # Check each configured symbol
            missing_symbols = []
            for instrument in self._config.instruments:
                if instrument.symbol not in mt5_symbols:
                    missing_symbols.append(instrument.symbol)

            if missing_symbols:
                missing_str = ", ".join(missing_symbols)
                self._logger.log_error(
                    level="CRITICAL",
                    message=f"The following symbols are not available in MT5: {missing_str}",
                    exception_type="ConfigError",
                    function="_verify_symbols",
                    traceback="",
                    context={"missing_symbols": missing_symbols}
                )
                return False

            return True

        except Exception as e:
            self._logger.log_error(
                level="CRITICAL",
                message=f"Failed to verify symbols: {str(e)}",
                exception_type=type(e).__name__,
                function="_verify_symbols",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def _load_initial_data(self) -> bool:
        """Load initial historical data if needed"""
        try:
            for (symbol, timeframe), mapping in self._instrument_timeframe_map.items():
                instrument_id = mapping["instrument_id"]
                timeframe_id = mapping["timeframe_id"]
                history_size = mapping["history_size"]

                # Check if data exists for this instrument/timeframe
                latest_timestamp = None
                with self._db_session.session_scope() as session:
                    latest_timestamp = session.query(func.max(PriceBar.timestamp)) \
                        .filter(and_(PriceBar.instrument_id == instrument_id,
                                     PriceBar.timeframe_id == timeframe_id)) \
                        .scalar()

                if latest_timestamp is None:
                    # No data exists, load initial data
                    self._logger.log_event(
                        level="INFO",
                        message=f"Loading initial data for {symbol}/{timeframe.name}",
                        event_type="DATA_LOAD",
                        component="data_fetcher",
                        action="_load_initial_data",
                        status="starting",
                        details={"symbol": symbol, "timeframe": timeframe.name, "count": history_size}
                    )

                    # Get historical data from MT5
                    mt5_data = self._mt5_manager.copy_rates_from_pos(symbol, timeframe, 0, history_size)

                    if mt5_data is None or len(mt5_data) == 0:
                        self._logger.log_error(
                            level="ERROR",
                            message=f"Failed to load initial data for {symbol}/{timeframe.name}",
                            exception_type="MT5DataError",
                            function="_load_initial_data",
                            traceback="",
                            context={"symbol": symbol, "timeframe": timeframe.name}
                        )
                        continue

                    # Convert to price bars
                    bars = self._convert_to_price_bars(mt5_data, instrument_id, timeframe_id)

                    # Insert into database using SQLAlchemy
                    with self._db_session.session_scope() as session:
                        for bar in bars:
                            session.add(bar)

                    self._logger.log_event(
                        level="INFO",
                        message=f"Loaded {len(bars)} initial bars for {symbol}/{timeframe.name}",
                        event_type="DATA_LOAD",
                        component="data_fetcher",
                        action="_load_initial_data",
                        status="success",
                        details={"symbol": symbol, "timeframe": timeframe.name, "count": len(bars)}
                    )

                    # Notify listeners of new data
                    for bar in bars:
                        event = NewBarEvent(instrument_id, timeframe_id, bar, symbol, timeframe.name)
                        self._event_bus.publish(event)
                else:
                    # Verify we have enough historical data
                    with self._db_session.session_scope() as session:
                        bars_count = session.query(func.count(PriceBar.id)) \
                            .filter(and_(PriceBar.instrument_id == instrument_id,
                                         PriceBar.timeframe_id == timeframe_id)) \
                            .scalar()

                    if bars_count < history_size:
                        missing_count = history_size - bars_count

                        self._logger.log_event(
                            level="INFO",
                            message=f"Missing {missing_count} historical bars for {symbol}/{timeframe.name}, loading now",
                            event_type="DATA_LOAD",
                            component="data_fetcher",
                            action="_load_initial_data",
                            status="starting",
                            details={"symbol": symbol, "timeframe": timeframe.name, "missing": missing_count}
                        )

                        # Get oldest bar timestamp
                        oldest_timestamp = None
                        with self._db_session.session_scope() as session:
                            oldest_timestamp = session.query(func.min(PriceBar.timestamp)) \
                                .filter(and_(PriceBar.instrument_id == instrument_id,
                                             PriceBar.timeframe_id == timeframe_id)) \
                                .scalar()

                            # Get all bars to have complete data available
                            bars = session.query(PriceBar) \
                                .filter(and_(PriceBar.instrument_id == instrument_id,
                                             PriceBar.timeframe_id == timeframe_id)) \
                                .order_by(PriceBar.timestamp.asc()) \
                                .all()

                        if oldest_timestamp:
                            # Calculate start date based on timeframe
                            minutes = timeframe.minutes
                            from_date = oldest_timestamp - timedelta(minutes=minutes * missing_count)

                            # Get more historical data
                            mt5_data = self._mt5_manager.copy_rates_range(
                                symbol,
                                timeframe,
                                from_date,
                                oldest_timestamp - timedelta(minutes=minutes)
                            )

                            if mt5_data and len(mt5_data) > 0:
                                # Convert to price bars
                                additional_bars = self._convert_to_price_bars(mt5_data, instrument_id, timeframe_id)

                                # Insert into database
                                with self._db_session.session_scope() as session:
                                    for bar in additional_bars:
                                        session.add(bar)

                                self._logger.log_event(
                                    level="INFO",
                                    message=f"Loaded {len(additional_bars)} additional historical bars for {symbol}/{timeframe.name}",
                                    event_type="DATA_LOAD",
                                    component="data_fetcher",
                                    action="_load_initial_data",
                                    status="success",
                                    details={"symbol": symbol, "timeframe": timeframe.name,
                                             "count": len(additional_bars)}
                                )

                                # Notify listeners of new data
                                for bar in additional_bars:
                                    event = NewBarEvent(instrument_id, timeframe_id, bar, symbol, timeframe.name)
                                    self._event_bus.publish(event)

            return True

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to load initial data: {str(e)}",
                exception_type=type(e).__name__,
                function="_load_initial_data",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def _convert_to_price_bars(self, mt5_data: List[Dict[str, Any]], instrument_id: int, timeframe_id: int) -> List[
        PriceBar]:
        """Convert MT5 data to price bars"""
        bars = []
        for rate in mt5_data:
            # Check if rate is a numpy structured array or a dict
            if hasattr(rate, 'dtype') and hasattr(rate, '__getitem__'):
                # Convert from numpy array
                bar = PriceBar(
                    instrument_id=int(instrument_id),
                    timeframe_id=int(timeframe_id),
                    timestamp=datetime.fromtimestamp(float(rate[0])),  # time
                    open=float(rate[1]),  # open
                    high=float(rate[2]),  # high
                    low=float(rate[3]),  # low
                    close=float(rate[4]),  # close
                    volume=float(rate[5]),  # tick_volume
                    spread=float(rate[6]) if len(rate) > 6 else None  # spread
                )
            else:
                # Convert from dict
                bar = PriceBar(
                    instrument_id=int(instrument_id),
                    timeframe_id=int(timeframe_id),
                    timestamp=rate["time"],
                    open=float(rate["open"]),
                    high=float(rate["high"]),
                    low=float(rate["low"]),
                    close=float(rate["close"]),
                    volume=float(rate["tick_volume"]),
                    spread=float(rate["spread"]) if "spread" in rate else None
                )
            bars.append(bar)
        return bars

    def start(self) -> bool:
        """Start data fetcher"""
        if self._running:
            self._logger.log_event(
                level="WARNING",
                message="Data fetcher already running",
                event_type="DATA_FETCHER_START",
                component="data_fetcher",
                action="start",
                status="ignored"
            )
            return True

        try:
            self._stop_event.clear()
            self._running = True
            self._fetch_thread = threading.Thread(target=self._fetch_loop)
            self._fetch_thread.daemon = True
            self._fetch_thread.start()

            self._logger.log_event(
                level="INFO",
                message="Data fetcher started",
                event_type="DATA_FETCHER_START",
                component="data_fetcher",
                action="start",
                status="success"
            )

            return True

        except Exception as e:
            self._running = False
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to start data fetcher: {str(e)}",
                exception_type=type(e).__name__,
                function="start",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def stop(self) -> None:
        """Stop data fetcher"""
        if not self._running:
            return

        try:
            self._stop_event.set()
            if self._fetch_thread:
                self._fetch_thread.join(timeout=10)
            self._running = False

            self._logger.log_event(
                level="INFO",
                message="Data fetcher stopped",
                event_type="DATA_FETCHER_STOP",
                component="data_fetcher",
                action="stop",
                status="success"
            )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error stopping data fetcher: {str(e)}",
                exception_type=type(e).__name__,
                function="stop",
                traceback=traceback.format_exc(),
                context={}
            )

    def _fetch_loop(self) -> None:
        """Main data fetching loop"""
        while not self._stop_event.is_set():
            try:
                start_time = time.time()

                # Fetch new data for all instrument/timeframe pairs
                self._fetch_all_new_data()

                # Calculate sleep time to maintain sync interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self._config.sync_interval_seconds - elapsed)

                if sleep_time > 0:
                    # Sleep until next sync, but check stop event periodically
                    for _ in range(int(sleep_time)):
                        if self._stop_event.is_set():
                            break
                        time.sleep(1)

                    # Handle remainder
                    remainder = sleep_time - int(sleep_time)
                    if remainder > 0 and not self._stop_event.is_set():
                        time.sleep(remainder)

            except Exception as e:
                self._logger.log_error(
                    level="ERROR",
                    message=f"Error in data fetcher loop: {str(e)}",
                    exception_type=type(e).__name__,
                    function="_fetch_loop",
                    traceback=traceback.format_exc(),
                    context={}
                )

                # Sleep a bit before retrying
                time.sleep(5)

    def _fetch_all_new_data(self) -> None:
        """Fetch new data for all configured instrument/timeframe pairs"""
        for (symbol, timeframe), mapping in self._instrument_timeframe_map.items():
            try:
                instrument_id = mapping["instrument_id"]
                timeframe_id = mapping["timeframe_id"]
                history_size = mapping["history_size"]

                # Get latest timestamp from database
                latest_timestamp = None
                with self._db_session.session_scope() as session:
                    latest_timestamp = session.query(func.max(PriceBar.timestamp)) \
                        .filter(and_(PriceBar.instrument_id == instrument_id,
                                     PriceBar.timeframe_id == timeframe_id)) \
                        .scalar()

                if latest_timestamp is None:
                    # No data yet, skip (should have been loaded in initialize)
                    continue

                # Calculate from_date for getting new data
                minutes = timeframe.minutes
                from_date = latest_timestamp + timedelta(minutes=minutes)
                now = datetime.now()

                # Skip if we're already up to date
                if from_date > now:
                    continue

                # Fetch new data from MT5
                mt5_data = self._mt5_manager.copy_rates_from(symbol, timeframe, from_date, 100)

                if mt5_data and len(mt5_data) > 0:
                    # Convert to price bars
                    bars = self._convert_to_price_bars(mt5_data, instrument_id, timeframe_id)

                    if bars:
                        # Insert into database using SQLAlchemy
                        with self._db_session.session_scope() as session:
                            for bar in bars:
                                # Check if bar already exists
                                existing = session.query(PriceBar).filter(
                                    and_(
                                        PriceBar.instrument_id == bar.instrument_id,
                                        PriceBar.timeframe_id == bar.timeframe_id,
                                        PriceBar.timestamp == bar.timestamp
                                    )
                                ).first()

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

                        # Maintain history limit
                        self._maintain_history_limit(instrument_id, timeframe_id, history_size)

                        self._logger.log_event(
                            level="INFO",
                            message=f"Fetched {len(bars)} new bars for {symbol}/{timeframe.name}",
                            event_type="DATA_FETCH",
                            component="data_fetcher",
                            action="_fetch_all_new_data",
                            status="success",
                            details={"symbol": symbol, "timeframe": timeframe.name, "count": len(bars)}
                        )

                        # Notify listeners of new data
                        for bar in bars:
                            event = NewBarEvent(instrument_id, timeframe_id, bar, symbol, timeframe.name)
                            self._event_bus.publish(event)

            except Exception as e:
                self._logger.log_error(
                    level="ERROR",
                    message=f"Error fetching data for {symbol}/{timeframe.name}: {str(e)}",
                    exception_type=type(e).__name__,
                    function="_fetch_all_new_data",
                    traceback=traceback.format_exc(),
                    context={"symbol": symbol, "timeframe": str(timeframe.name)}
                )

    def _maintain_history_limit(self, instrument_id: int, timeframe_id: int, max_bars: int) -> None:
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

                    # Find cutoff timestamp
                    cutoff_timestamp = session.query(PriceBar.timestamp) \
                        .filter(and_(PriceBar.instrument_id == instrument_id,
                                     PriceBar.timeframe_id == timeframe_id)) \
                        .order_by(PriceBar.timestamp) \
                        .offset(delete_count - 1) \
                        .limit(1) \
                        .scalar()

                    # Delete all bars older than cutoff
                    deleted = session.query(PriceBar) \
                        .filter(and_(
                        PriceBar.instrument_id == instrument_id,
                        PriceBar.timeframe_id == timeframe_id,
                        PriceBar.timestamp <= cutoff_timestamp
                    )) \
                        .delete(synchronize_session=False)

                    self._logger.log_event(
                        level="INFO",
                        message=f"Deleted {deleted} oldest bars to maintain limit",
                        event_type="DATA_MAINTENANCE",
                        component="data_fetcher",
                        action="maintain_bar_limit",
                        status="success",
                        details={"instrument_id": instrument_id, "timeframe_id": timeframe_id,
                                 "max_bars": max_bars, "deleted": deleted}
                    )
        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to maintain bar limit: {str(e)}",
                exception_type=type(e).__name__,
                function="_maintain_history_limit",
                traceback=traceback.format_exc(),
                context={"instrument_id": instrument_id, "timeframe_id": timeframe_id, "max_bars": max_bars}
            )

    def get_latest_bars(self, symbol: str, timeframe: TimeFrame, count: int = 100) -> Optional[List[PriceBar]]:
        """Get latest bars for a specific instrument and timeframe"""
        try:
            key = (symbol, timeframe)
            if key not in self._instrument_timeframe_map:
                self._logger.log_error(
                    level="ERROR",
                    message=f"Invalid instrument/timeframe combination: {symbol}/{timeframe.name}",
                    exception_type="ConfigError",
                    function="get_latest_bars",
                    traceback="",
                    context={"symbol": symbol, "timeframe": timeframe.name}
                )
                return None

            mapping = self._instrument_timeframe_map[key]
            instrument_id = mapping["instrument_id"]
            timeframe_id = mapping["timeframe_id"]

            # Get bars from database using SQLAlchemy
            with self._db_session.session_scope() as session:
                bars = session.query(PriceBar) \
                    .filter(and_(
                    PriceBar.instrument_id == instrument_id,
                    PriceBar.timeframe_id == timeframe_id
                )) \
                    .order_by(desc(PriceBar.timestamp)) \
                    .limit(count) \
                    .all()

                # Convert to list in ascending order (oldest first)
                return list(reversed(bars))

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error getting latest bars for {symbol}/{timeframe.name}: {str(e)}",
                exception_type=type(e).__name__,
                function="get_latest_bars",
                traceback=traceback.format_exc(),
                context={"symbol": symbol, "timeframe": timeframe.name, "count": count}
            )
            return None

    def force_sync(self, symbol: Optional[str] = None, timeframe: Optional[TimeFrame] = None) -> bool:
        """Force immediate synchronization for specific or all instruments/timeframes"""
        try:
            if symbol and timeframe:
                # Sync specific instrument/timeframe
                key = (symbol, timeframe)
                if key not in self._instrument_timeframe_map:
                    self._logger.log_error(
                        level="ERROR",
                        message=f"Invalid instrument/timeframe combination: {symbol}/{timeframe.name}",
                        exception_type="ConfigError",
                        function="force_sync",
                        traceback="",
                        context={"symbol": symbol, "timeframe": timeframe.name}
                    )
                    return False

                self._logger.log_event(
                    level="INFO",
                    message=f"Forcing sync for {symbol}/{timeframe.name}",
                    event_type="DATA_SYNC",
                    component="data_fetcher",
                    action="force_sync",
                    status="starting",
                    details={"symbol": symbol, "timeframe": timeframe.name}
                )

                # Get latest timestamp
                mapping = self._instrument_timeframe_map[key]
                instrument_id = mapping["instrument_id"]
                timeframe_id = mapping["timeframe_id"]
                history_size = mapping["history_size"]

                # Get latest timestamp from database using SQLAlchemy
                latest_timestamp = None
                with self._db_session.session_scope() as session:
                    latest_timestamp = session.query(func.max(PriceBar.timestamp)) \
                        .filter(and_(PriceBar.instrument_id == instrument_id,
                                     PriceBar.timeframe_id == timeframe_id)) \
                        .scalar()

                if latest_timestamp is None:
                    # No data yet, load initial data
                    mt5_data = self._mt5_manager.copy_rates_from_pos(symbol, timeframe, 0, history_size)
                else:
                    # Fetch new data since latest timestamp
                    minutes = timeframe.minutes
                    from_date = latest_timestamp + timedelta(minutes=minutes)
                    mt5_data = self._mt5_manager.copy_rates_from(symbol, timeframe, from_date, 100)

                if mt5_data and len(mt5_data) > 0:
                    # Convert to price bars
                    bars = self._convert_to_price_bars(mt5_data, instrument_id, timeframe_id)

                    if bars:
                        # Insert into database using SQLAlchemy
                        with self._db_session.session_scope() as session:
                            for bar in bars:
                                # Check if bar already exists
                                existing = session.query(PriceBar).filter(
                                    and_(
                                        PriceBar.instrument_id == bar.instrument_id,
                                        PriceBar.timeframe_id == bar.timeframe_id,
                                        PriceBar.timestamp == bar.timestamp
                                    )
                                ).first()

                                if existing:
                                    # Update existing bar
                                    existing.open = bar.open
                                    existing.high = bar.high
                                    existing.low = bar.low
                                    existing.close = bar.close
                                    existing.volume = bar.volume
                                    existing.spread = bar.spread
                                else:
                                    # Add new bar
                                    session.add(bar)

                        # Maintain history limit
                        self._maintain_history_limit(instrument_id, timeframe_id, history_size)

                        self._logger.log_event(
                            level="INFO",
                            message=f"Forced sync completed for {symbol}/{timeframe.name}: {len(bars)} bars",
                            event_type="DATA_SYNC",
                            component="data_fetcher",
                            action="force_sync",
                            status="success",
                            details={"symbol": symbol, "timeframe": timeframe.name, "count": len(bars)}
                        )

                        # Notify listeners of new data
                        for bar in bars:
                            event = NewBarEvent(instrument_id, timeframe_id, bar, symbol, timeframe.name)
                            self._event_bus.publish(event)
            else:
                # Sync all instruments/timeframes
                self._logger.log_event(
                    level="INFO",
                    message="Forcing sync for all instruments/timeframes",
                    event_type="DATA_SYNC",
                    component="data_fetcher",
                    action="force_sync",
                    status="starting"
                )

                self._fetch_all_new_data()

                self._logger.log_event(
                    level="INFO",
                    message="Forced sync completed for all instruments/timeframes",
                    event_type="DATA_SYNC",
                    component="data_fetcher",
                    action="force_sync",
                    status="success"
                )

            return True

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error forcing sync: {str(e)}",
                exception_type=type(e).__name__,
                function="force_sync",
                traceback=traceback.format_exc(),
                context={"symbol": symbol, "timeframe": timeframe.name if timeframe else None}
            )
            return False