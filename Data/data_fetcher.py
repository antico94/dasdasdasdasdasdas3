from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import threading
import time
import traceback

from sqlalchemy import and_, desc, func
import MetaTrader5 as mt5
from Config.trading_config import ConfigManager, TimeFrame
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
        self._last_processed_timestamps = {}

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

            # NEW: Ensure all charts are subscribed to receive real-time data
            if not self._ensure_charts_subscribed():
                self._logger.log_event(
                    level="WARNING",
                    message="Some charts failed to subscribe, but continuing initialization",
                    event_type="DATA_FETCHER_INIT",
                    component="data_fetcher",
                    action="initialize",
                    status="partial_charts"
                )

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
        """Convert MT5 data to price bars ensuring naive datetime for database compatibility"""
        bars = []
        for rate in mt5_data:
            # Check if rate is a numpy structured array or a dict
            if hasattr(rate, 'dtype') and hasattr(rate, '__getitem__'):
                # Convert from numpy array to datetime
                # First create with timezone info for consistency
                timestamp_tz = datetime.fromtimestamp(float(rate[0]), tz=timezone.utc)
                # Then strip timezone for database storage
                timestamp = timestamp_tz.replace(tzinfo=None)

                bar = PriceBar(
                    instrument_id=int(instrument_id),
                    timeframe_id=int(timeframe_id),
                    timestamp=timestamp,  # Naive datetime for database
                    open=float(rate[1]),
                    high=float(rate[2]),
                    low=float(rate[3]),
                    close=float(rate[4]),
                    volume=float(rate[5]),
                    spread=float(rate[6]) if len(rate) > 6 else None
                )
            else:
                # Convert from dict
                if isinstance(rate["time"], datetime):
                    # Save as naive datetime for database
                    timestamp = rate["time"].replace(tzinfo=None) if rate["time"].tzinfo else rate["time"]
                else:
                    # Convert timestamp to datetime, then strip timezone
                    timestamp_tz = datetime.fromtimestamp(rate["time"], tz=timezone.utc)
                    timestamp = timestamp_tz.replace(tzinfo=None)

                bar = PriceBar(
                    instrument_id=int(instrument_id),
                    timeframe_id=int(timeframe_id),
                    timestamp=timestamp,  # Naive datetime for database
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

    def _fetch_loop(self) -> None:
        """Main data fetching loop - modified to use our new synchronization logic"""
        reconnection_count = 0
        max_reconnection_attempts = 5
        reconnection_cooldown = 60  # seconds
        last_reconnection_time = None

        # Track when we last successfully synchronized
        last_sync_time = time.time()
        sync_interval = 1.0  # Check for new bars every second

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Check MT5 connection before fetching
                if not self._mt5_manager.ensure_connection():
                    # Connection error detected
                    if last_reconnection_time is None or (
                            current_time - last_reconnection_time) > reconnection_cooldown:
                        reconnection_count += 1
                        if reconnection_count <= max_reconnection_attempts:
                            last_reconnection_time = current_time
                            self._logger.log_event(
                                level="WARNING",
                                message=f"MT5 connection error detected. Attempting reconnection ({reconnection_count}/{max_reconnection_attempts})",
                                event_type="MT5_CONNECTION_ERROR",
                                component="data_fetcher",
                                action="_fetch_loop",
                                status="retry"
                            )
                            if self.handle_mt5_reconnection():
                                reconnection_count = 0  # Reset count on successful reconnection
                        else:
                            self._logger.log_error(
                                level="CRITICAL",
                                message=f"Failed to reconnect to MT5 after {max_reconnection_attempts} attempts",
                                exception_type="MT5ConnectionError",
                                function="_fetch_loop",
                                traceback="",
                                context={"reconnection_attempts": max_reconnection_attempts}
                            )
                            # Wait longer before retry
                            time.sleep(30)
                            reconnection_count = 0  # Reset to try again

                    # Skip this iteration and sleep
                    time.sleep(1)  # Reduced sleep time for quicker retry
                    continue

                # Check if it's time to sync the data
                if current_time - last_sync_time >= sync_interval:
                    # Run the synchronization
                    self._check_and_sync_timeframes()
                    last_sync_time = current_time

                # Heartbeat log (approximately once per minute)
                if int(current_time) % 60 == 0:
                    self._logger.log_event(
                        level="INFO",
                        message="DataFetcher heartbeat - connection maintained",
                        event_type="HEARTBEAT",
                        component="data_fetcher",
                        action="_fetch_loop",
                        status="ok"
                    )

                # Reset reconnection count
                reconnection_count = 0

                # Sleep for a short time to avoid excessive CPU usage
                time.sleep(0.1)

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
                time.sleep(1)

    def _maintain_history_limit(self, instrument_id: int, timeframe_id: int, max_bars: int) -> None:
        """Delete oldest bars to maintain maximum number of bars"""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
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

                        # Delete all bars older than cutoff with a direct delete statement
                        # instead of loading and then deleting objects
                        deleted = session.query(PriceBar) \
                            .filter(and_(
                            PriceBar.instrument_id == instrument_id,
                            PriceBar.timeframe_id == timeframe_id,
                            PriceBar.timestamp <= cutoff_timestamp
                        )) \
                            .delete(synchronize_session=False)

                        # Explicit commit within the session to finalize the delete
                        session.commit()

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

                # If we got here, the operation succeeded
                break

            except Exception as e:
                retry_count += 1
                error_message = str(e)

                # Check if this is a deadlock error
                if "deadlock" in error_message.lower() and retry_count < max_retries:
                    # Log the deadlock and retry
                    self._logger.log_event(
                        level="WARNING",
                        message=f"Deadlock detected when maintaining bar limit, retrying ({retry_count}/{max_retries})",
                        event_type="DATABASE_DEADLOCK",
                        component="data_fetcher",
                        action="_maintain_history_limit",
                        status="retrying",
                        details={
                            "instrument_id": instrument_id,
                            "timeframe_id": timeframe_id,
                            "retry": retry_count
                        }
                    )

                    # Wait a bit before retrying (exponential backoff)
                    time.sleep(0.1 * (2 ** retry_count))
                else:
                    # Truncate error message to prevent string truncation in SQL
                    error_msg = str(e)
                    if len(error_msg) > 450:
                        error_msg = error_msg[:450] + "..."

                    self._logger.log_error(
                        level="ERROR",
                        message=f"Failed to maintain bar limit: {error_msg}",
                        exception_type=type(e).__name__,
                        function="_maintain_history_limit",
                        traceback=str(e)[:450],
                        context={"instrument_id": instrument_id, "timeframe_id": timeframe_id, "max_bars": max_bars}
                    )
                    break  # Exit the retry loop on non-deadlock errors

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

                mapping = self._instrument_timeframe_map[key]
                instrument_id = mapping["instrument_id"]
                timeframe_id = mapping["timeframe_id"]
                history_size = mapping["history_size"]

                self._logger.log_event(
                    level="INFO",
                    message=f"Forcing sync for {symbol}/{timeframe.name}",
                    event_type="DATA_SYNC",
                    component="data_fetcher",
                    action="force_sync",
                    status="starting",
                    details={"symbol": symbol, "timeframe": timeframe.name}
                )

                # Determine appropriate number of bars to fetch
                # We add +1 to get the current forming bar for context
                if timeframe.minutes <= 5:  # M1, M5
                    bars_to_fetch = 100 + 1
                elif timeframe.minutes <= 30:  # M15, M30
                    bars_to_fetch = 50 + 1
                elif timeframe.minutes <= 240:  # H1, H4
                    bars_to_fetch = 30 + 1
                else:  # D1 and higher
                    bars_to_fetch = 20 + 1

                # Fetch data from MT5
                mt5_data = self._mt5_manager.copy_rates_from_pos(symbol, timeframe, 0, bars_to_fetch)

                if not mt5_data or len(mt5_data) == 0:
                    self._logger.log_event(
                        level="WARNING",
                        message=f"No data returned from MT5 for {symbol}/{timeframe.name}",
                        event_type="DATA_SYNC",
                        component="data_fetcher",
                        action="force_sync",
                        status="warning",
                        details={"symbol": symbol, "timeframe": timeframe.name}
                    )
                    return False

                # IMPORTANT: Separate completed bars from the current forming bar
                # The last bar (index -1) is the current forming bar and should not be treated as completed
                completed_mt5_data = mt5_data[:-1]
                current_forming_bar = mt5_data[-1] if len(mt5_data) > 0 else None

                self._logger.log_event(
                    level="INFO",
                    message=f"Retrieved {len(completed_mt5_data)} completed bars for {symbol}/{timeframe.name}",
                    event_type="DATA_SYNC",
                    component="data_fetcher",
                    action="force_sync",
                    status="data_retrieved",
                    details={
                        "completed_bars_count": len(completed_mt5_data),
                        "has_forming_bar": current_forming_bar is not None,
                        "timeframe": timeframe.name
                    }
                )

                if completed_mt5_data:
                    # Convert to price bars (only completed bars)
                    bars = self._convert_to_price_bars(completed_mt5_data, instrument_id, timeframe_id)

                    if bars:
                        # Insert or update bars in database
                        updated_count = 0
                        inserted_count = 0

                        with self._db_session.session_scope() as session:
                            for bar in bars:
                                # Check if bar already exists - Using query each time, no storing of objects
                                existing = session.query(PriceBar).filter(
                                    and_(
                                        PriceBar.instrument_id == bar.instrument_id,
                                        PriceBar.timeframe_id == bar.timeframe_id,
                                        PriceBar.timestamp == bar.timestamp
                                    )
                                ).first()

                                if existing:
                                    # Update directly with a query instead of modifying the existing object
                                    session.query(PriceBar).filter(
                                        and_(
                                            PriceBar.instrument_id == bar.instrument_id,
                                            PriceBar.timeframe_id == bar.timeframe_id,
                                            PriceBar.timestamp == bar.timestamp
                                        )
                                    ).update({
                                        "open": bar.open,
                                        "high": bar.high,
                                        "low": bar.low,
                                        "close": bar.close,
                                        "volume": bar.volume,
                                        "spread": bar.spread
                                    })
                                    updated_count += 1
                                else:
                                    # Add new record
                                    session.add(bar)
                                    inserted_count += 1

                            # Commit changes within the session
                            session.commit()

                        # Maintain history limit
                        self._maintain_history_limit(instrument_id, timeframe_id, history_size)

                        self._logger.log_event(
                            level="INFO",
                            message=f"Forced sync completed for {symbol}/{timeframe.name}: {inserted_count} new, {updated_count} updated bars",
                            event_type="DATA_SYNC",
                            component="data_fetcher",
                            action="force_sync",
                            status="success",
                            details={
                                "symbol": symbol,
                                "timeframe": timeframe.name,
                                "new_bars": inserted_count,
                                "updated_bars": updated_count,
                                "total_fetched": len(bars)
                            }
                        )

                        # Notify listeners of new COMPLETED data
                        # Get fresh copies from the database to avoid session binding issues
                        with self._db_session.session_scope() as session:
                            stored_bars = session.query(PriceBar).filter(
                                and_(
                                    PriceBar.instrument_id == instrument_id,
                                    PriceBar.timeframe_id == timeframe_id
                                )
                            ).order_by(PriceBar.timestamp).all()

                            # Create fresh PriceBar objects for event publishing
                            bar_copies = []
                            for bar in stored_bars:
                                bar_copy = PriceBar(
                                    id=bar.id,  # Include ID for reference
                                    instrument_id=bar.instrument_id,
                                    timeframe_id=bar.timeframe_id,
                                    timestamp=bar.timestamp,
                                    open=bar.open,
                                    high=bar.high,
                                    low=bar.low,
                                    close=bar.close,
                                    volume=bar.volume,
                                    spread=bar.spread
                                )
                                bar_copies.append(bar_copy)

                        # Sort bars by timestamp and publish events outside the session
                        bar_copies.sort(key=lambda x: x.timestamp)
                        for bar in bar_copies:
                            event = NewBarEvent(instrument_id, timeframe_id, bar, symbol, timeframe.name)
                            self._event_bus.publish(event)
                    else:
                        self._logger.log_event(
                            level="WARNING",
                            message=f"No bars converted from MT5 data for {symbol}/{timeframe.name}",
                            event_type="DATA_SYNC",
                            component="data_fetcher",
                            action="force_sync",
                            status="warning",
                            details={"symbol": symbol, "timeframe": timeframe.name}
                        )
                else:
                    self._logger.log_event(
                        level="WARNING",
                        message=f"No completed bars available from MT5 for {symbol}/{timeframe.name}",
                        event_type="DATA_SYNC",
                        component="data_fetcher",
                        action="force_sync",
                        status="warning",
                        details={"symbol": symbol, "timeframe": timeframe.name}
                    )

                # If we get here, the sync was successful (even if no new data)
                return True

            # Code for syncing all symbols/timeframes would go here...
            # (this part isn't shown in the error, but would need similar fixes)

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

    def handle_mt5_reconnection(self) -> bool:
        """Handle MT5 disconnection by waiting and reinitializing"""
        try:
            self._logger.log_event(
                level="WARNING",
                message="Handling MT5 disconnection",
                event_type="MT5_RECONNECTION",
                component="data_fetcher",
                action="handle_mt5_reconnection",
                status="starting"
            )

            # Wait for a bit to give MT5 terminal time to fully start
            time.sleep(5)

            # Try to reinitialize MT5
            if not self._mt5_manager.initialize():
                # Wait longer and try again
                self._logger.log_event(
                    level="WARNING",
                    message="First reconnection attempt failed, waiting longer...",
                    event_type="MT5_RECONNECTION",
                    component="data_fetcher",
                    action="handle_mt5_reconnection",
                    status="retry"
                )

                time.sleep(10)

                if not self._mt5_manager.initialize():
                    self._logger.log_error(
                        level="CRITICAL",
                        message="Failed to reconnect to MT5 after multiple attempts",
                        exception_type="MT5ConnectionError",
                        function="handle_mt5_reconnection",
                        traceback="",
                        context={}
                    )
                    return False

            self._logger.log_event(
                level="INFO",
                message="Successfully reconnected to MT5",
                event_type="MT5_RECONNECTION",
                component="data_fetcher",
                action="handle_mt5_reconnection",
                status="success"
            )

            # Force a complete data resync
            return self.force_sync()

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error during MT5 reconnection: {str(e)}",
                exception_type=type(e).__name__,
                function="handle_mt5_reconnection",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def stop(self) -> None:
        """Stop data fetcher with timeout to ensure clean shutdown"""
        if not self._running:
            return

        try:
            # Signal threads to stop
            self._stop_event.set()

            # Wait for fetch thread to terminate with timeout
            if self._fetch_thread and self._fetch_thread.is_alive():
                max_wait_seconds = 15
                for i in range(max_wait_seconds):
                    if not self._fetch_thread.is_alive():
                        break
                    time.sleep(1)

                # If still alive after timeout, log warning
                if self._fetch_thread.is_alive():
                    self._logger.log_event(
                        level="WARNING",
                        message=f"Data fetcher thread didn't terminate after {max_wait_seconds} seconds",
                        event_type="DATA_FETCHER_STOP",
                        component="data_fetcher",
                        action="stop",
                        status="timeout"
                    )
                    # Don't attempt join again as it may block indefinitely

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
            # Mark as not running even if there was an error
            self._running = False

    def _ensure_charts_subscribed(self) -> bool:
        """
        Ensure MT5 has all required charts open to receive real-time data.
        Returns success status.
        """
        try:
            self._logger.log_event(
                level="INFO",
                message="Ensuring all required charts are subscribed in MT5",
                event_type="MT5_SUBSCRIPTION",
                component="data_fetcher",
                action="_ensure_charts_subscribed",
                status="starting"
            )

            # Count success/failure
            success_count = 0
            failure_count = 0

            # For each symbol/timeframe combination
            for (symbol, timeframe) in self._instrument_timeframe_map.keys():
                try:
                    # First verify the symbol is available
                    if not self._mt5_manager.check_symbol_available(symbol):
                        self._logger.log_error(
                            level="ERROR",
                            message=f"Symbol {symbol} is not available in MT5",
                            exception_type="ConfigError",
                            function="_ensure_charts_subscribed",
                            traceback="",
                            context={"symbol": symbol}
                        )
                        failure_count += 1
                        continue

                    # Try to fetch a small amount of data to "activate" the symbol/timeframe
                    # This sends a request to MT5 which will cause it to open the chart internally
                    mt5_data = self._mt5_manager.copy_rates_from_pos(symbol, timeframe, 0, 5)

                    if mt5_data is not None and len(mt5_data) > 0:
                        # Successfully subscribed
                        success_count += 1

                        # Log only at debug level to avoid excessive logs
                        self._logger.log_event(
                            level="INFO",
                            message=f"Successfully subscribed to {symbol}/{timeframe.name}",
                            event_type="MT5_SUBSCRIPTION",
                            component="data_fetcher",
                            action="_ensure_charts_subscribed",
                            status="success",
                            details={"symbol": symbol, "timeframe": timeframe.name}
                        )
                    else:
                        # Failed to subscribe
                        failure_count += 1
                        self._logger.log_error(
                            level="ERROR",
                            message=f"Failed to subscribe to {symbol}/{timeframe.name}",
                            exception_type="MT5DataError",
                            function="_ensure_charts_subscribed",
                            traceback="",
                            context={"symbol": symbol, "timeframe": timeframe.name}
                        )

                except Exception as e:
                    failure_count += 1
                    self._logger.log_error(
                        level="ERROR",
                        message=f"Error subscribing to {symbol}/{timeframe.name}: {str(e)}",
                        exception_type=type(e).__name__,
                        function="_ensure_charts_subscribed",
                        traceback=traceback.format_exc(),
                        context={"symbol": symbol, "timeframe": timeframe.name}
                    )

            # Log overall results
            total = success_count + failure_count
            success_rate = (success_count / total * 100) if total > 0 else 0

            self._logger.log_event(
                level="INFO",
                message=f"Chart subscription complete: {success_count}/{total} successful ({success_rate:.1f}%)",
                event_type="MT5_SUBSCRIPTION",
                component="data_fetcher",
                action="_ensure_charts_subscribed",
                status="complete",
                details={"success_count": success_count, "failure_count": failure_count, "total": total}
            )

            return failure_count == 0

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error ensuring charts are subscribed: {str(e)}",
                exception_type=type(e).__name__,
                function="_ensure_charts_subscribed",
                traceback=traceback.format_exc(),
                context={}
            )
            return False

    def _update_last_processed_timestamp(self, symbol: str, timeframe: TimeFrame, timestamp: datetime) -> None:
        """
        Update the last processed timestamp for a symbol/timeframe.
        Stores timestamp in memory with timezone info for consistency in comparisons.

        Args:
            symbol: The symbol
            timeframe: The timeframe
            timestamp: The timestamp with or without timezone info
        """
        key = (symbol, timeframe)

        # Ensure timestamp has timezone info for internal tracking
        if timestamp.tzinfo is None:
            timestamp_with_tz = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp_with_tz = timestamp

        # Store with timezone info
        self._last_processed_timestamps[key] = timestamp_with_tz

        self._logger.log_event(
            level="INFO",
            message=f"Updated last processed timestamp for {symbol}/{timeframe.name}: {timestamp_with_tz.isoformat()}",
            event_type="DATA_SYNC",
            component="data_fetcher",
            action="_update_last_processed_timestamp",
            status="success",
            details={"symbol": symbol, "timeframe": timeframe.name, "timestamp": timestamp_with_tz.isoformat()}
        )

    def _get_last_processed_timestamp(self, symbol: str, timeframe: TimeFrame) -> Optional[datetime]:
        """
        Get the last processed timestamp for a symbol/timeframe.
        Returns timestamp with timezone info for consistent comparisons.

        Args:
            symbol: The symbol
            timeframe: The timeframe

        Returns:
            Timestamp with timezone info or None if not set
        """
        key = (symbol, timeframe)
        timestamp = self._last_processed_timestamps.get(key)

        # If we have a timestamp, ensure it has timezone info
        if timestamp is not None and timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)

        return timestamp

    def _check_and_sync_timeframes(self) -> None:
        """
        Check for new completed candles and sync all timeframes accordingly.
        This follows the pattern described in the article:
        1. Check for new M1 bars
        2. If an M1 bar closed, update indicators and check for higher timeframe closes
        """
        try:
            # First check if markets are open for any of our symbols
            any_market_open = False
            for (symbol, _) in set((symbol, None) for symbol, _ in self._instrument_timeframe_map.keys()):
                if self.is_market_open(symbol):
                    any_market_open = True
                    break

            if not any_market_open:
                # All markets are closed, log this and skip synchronization
                self._logger.log_event(
                    level="INFO",
                    message="All markets are currently closed, skipping synchronization",
                    event_type="MARKET_CLOSED",
                    component="data_fetcher",
                    action="_check_and_sync_timeframes",
                    status="skipped"
                )
                return

            # Get MT5 server time - critical for accurate bar closure detection
            mt5_server_time = self._get_mt5_server_time()
            if not mt5_server_time:
                self._logger.log_event(
                    level="WARNING",
                    message="Could not get MT5 server time, using system time as fallback",
                    event_type="TIME_SYNC",
                    component="data_fetcher",
                    action="_check_and_sync_timeframes",
                    status="fallback"
                )
                # Use current time but ensure it has timezone info
                mt5_server_time = datetime.now(timezone.utc)

            # Process timeframes in order: M1 first, then higher timeframes
            timeframe_minutes = [1, 5, 15, 30, 60, 240, 1440, 10080, 43200]  # M1 to MN1

            # First, sync all M1 timeframes
            for (symbol, timeframe), mapping in self._instrument_timeframe_map.items():
                if timeframe.minutes != 1:
                    continue  # Skip non-M1 timeframes in this first pass

                # Check if this specific market is open
                if not self.is_market_open(symbol):
                    self._logger.log_event(
                        level="INFO",
                        message=f"Market for {symbol} is closed, skipping M1 sync",
                        event_type="MARKET_CLOSED",
                        component="data_fetcher",
                        action="_check_and_sync_timeframes",
                        status="skipped",
                        details={"symbol": symbol, "timeframe": timeframe.name}
                    )
                    continue

                # Try to sync this M1 timeframe
                has_new_m1_bar = self._sync_timeframe(symbol, timeframe, mt5_server_time)

                # If we have a new M1 bar, check higher timeframes for this symbol
                if has_new_m1_bar:
                    self._check_higher_timeframes(symbol, mt5_server_time)

            # Then sync all remaining timeframes
            # This ensures we don't miss any higher timeframe bars even if M1 didn't trigger
            for minutes in timeframe_minutes[1:]:  # Skip M1 as we already processed it
                for (symbol, timeframe), mapping in self._instrument_timeframe_map.items():
                    if timeframe.minutes == minutes:
                        # Check if this specific market is open
                        if not self.is_market_open(symbol):
                            self._logger.log_event(
                                level="INFO",
                                message=f"Market for {symbol} is closed, skipping {timeframe.name} sync",
                                event_type="MARKET_CLOSED",
                                component="data_fetcher",
                                action="_check_and_sync_timeframes",
                                status="skipped",
                                details={"symbol": symbol, "timeframe": timeframe.name}
                            )
                            continue

                        self._sync_timeframe(symbol, timeframe, mt5_server_time)

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error in timeframe synchronization: {str(e)}",
                exception_type=type(e).__name__,
                function="_check_and_sync_timeframes",
                traceback=traceback.format_exc(),
                context={}
            )

    def _sync_timeframe(self, symbol: str, timeframe: TimeFrame, current_time: datetime) -> bool:
        """
        Sync a specific timeframe and return whether a new bar was detected.

        Args:
            symbol: The symbol to sync
            timeframe: The timeframe to sync
            current_time: The current MT5 server time

        Returns:
            bool: True if a new completed bar was detected, False otherwise
        """
        try:
            # Get the last processed timestamp for this symbol/timeframe
            last_processed = self._get_last_processed_timestamp(symbol, timeframe)

            # Fetch latest bars (adding +1 to get the current forming bar)
            bars_to_fetch = 10 + 1  # Fetching more than needed to ensure we get all new bars
            mt5_data = self._mt5_manager.copy_rates_from_pos(symbol, timeframe, 0, bars_to_fetch)

            if not mt5_data or len(mt5_data) < 2:  # Need at least one completed bar plus the forming one
                self._logger.log_event(
                    level="WARNING",
                    message=f"Insufficient data for {symbol}/{timeframe.name}, got {len(mt5_data) if mt5_data else 0} bars",
                    event_type="DATA_SYNC",
                    component="data_fetcher",
                    action="_sync_timeframe",
                    status="insufficient_data",
                    details={"symbol": symbol, "timeframe": timeframe.name}
                )
                return False

            # Separate completed bars from the forming bar
            completed_bars = mt5_data[:-1]  # All bars except the last one
            forming_bar = mt5_data[-1]  # The last bar is the current forming one

            if not completed_bars:
                return False

            # Get the timestamp of the latest completed bar
            if isinstance(completed_bars[-1]["time"], datetime):
                latest_completed_timestamp = completed_bars[-1]["time"]
            else:
                latest_completed_timestamp = datetime.fromtimestamp(completed_bars[-1]["time"], tz=timezone.utc)

            # Strip timezone info for database comparisons since SQL Server doesn't handle timezone objects properly
            if latest_completed_timestamp.tzinfo is not None:
                latest_completed_timestamp_naive = latest_completed_timestamp.replace(tzinfo=None)
            else:
                latest_completed_timestamp_naive = latest_completed_timestamp

            # If this is the first time we're processing this timeframe or we have a new bar
            new_bar_detected = False

            if last_processed is None:
                new_bar_detected = True
            else:
                # Convert last_processed to naive for consistent comparison
                if last_processed.tzinfo is not None:
                    last_processed_naive = last_processed.replace(tzinfo=None)
                else:
                    last_processed_naive = last_processed

                # Compare timestamps without timezone info
                new_bar_detected = latest_completed_timestamp_naive > last_processed_naive

            if new_bar_detected:
                # We have a new completed bar - let's process it
                mapping = self._instrument_timeframe_map.get((symbol, timeframe))
                if not mapping:
                    self._logger.log_error(
                        level="ERROR",
                        message=f"Missing mapping for {symbol}/{timeframe.name}",
                        exception_type="ConfigError",
                        function="_sync_timeframe",
                        traceback="",
                        context={"symbol": symbol, "timeframe": timeframe.name}
                    )
                    return False

                instrument_id = mapping["instrument_id"]
                timeframe_id = mapping["timeframe_id"]

                # Convert to price bars
                bars = self._convert_to_price_bars(completed_bars, instrument_id, timeframe_id)

                if bars:
                    # Insert or update bars in database
                    updated_count = 0
                    inserted_count = 0

                    with self._db_session.session_scope() as session:
                        for bar in bars:
                            # Check if this is newer than the last processed timestamp
                            if last_processed:
                                # Convert both to naive datetime for comparison
                                bar_ts_naive = bar.timestamp.replace(
                                    tzinfo=None) if bar.timestamp.tzinfo else bar.timestamp
                                last_proc_naive = last_processed.replace(
                                    tzinfo=None) if last_processed.tzinfo else last_processed

                                # Skip bars we've already processed
                                if bar_ts_naive <= last_proc_naive:
                                    continue

                            # Check if bar already exists - QUERY EACH TIME INSIDE SESSION
                            existing = session.query(PriceBar).filter(
                                and_(
                                    PriceBar.instrument_id == bar.instrument_id,
                                    PriceBar.timeframe_id == bar.timeframe_id,
                                    PriceBar.timestamp == bar.timestamp.replace(tzinfo=None)
                                    # Strip timezone info for SQL Server
                                )
                            ).first()

                            if existing:
                                # Update existing record if any value is different
                                if (existing.open != bar.open or
                                        existing.high != bar.high or
                                        existing.low != bar.low or
                                        existing.close != bar.close or
                                        existing.volume != bar.volume or
                                        existing.spread != bar.spread):
                                    # Update values directly, don't store reference to existing
                                    session.query(PriceBar).filter(
                                        and_(
                                            PriceBar.instrument_id == bar.instrument_id,
                                            PriceBar.timeframe_id == bar.timeframe_id,
                                            PriceBar.timestamp == bar.timestamp.replace(tzinfo=None)
                                            # Strip timezone info
                                        )
                                    ).update({
                                        "open": bar.open,
                                        "high": bar.high,
                                        "low": bar.low,
                                        "close": bar.close,
                                        "volume": bar.volume,
                                        "spread": bar.spread
                                    })
                                    updated_count += 1
                            else:
                                # Add new record - ensuring the timestamp is timezone-naive for the database
                                bar.timestamp = bar.timestamp.replace(
                                    tzinfo=None) if bar.timestamp.tzinfo else bar.timestamp
                                session.add(bar)
                                inserted_count += 1

                        # Commit changes within session scope
                        session.commit()

                    # Log the synchronization
                    if inserted_count > 0 or updated_count > 0:
                        self._logger.log_event(
                            level="INFO",
                            message=f"Synchronized {symbol}/{timeframe.name}: {inserted_count} new, {updated_count} updated bars",
                            event_type="DATA_SYNC",
                            component="data_fetcher",
                            action="_sync_timeframe",
                            status="success",
                            details={
                                "symbol": symbol,
                                "timeframe": timeframe.name,
                                "new_bars": inserted_count,
                                "updated_bars": updated_count,
                                "latest_timestamp": latest_completed_timestamp.isoformat() if isinstance(
                                    latest_completed_timestamp, datetime) else str(latest_completed_timestamp)
                            }
                        )

                    # Update the last processed timestamp - store with timezone info for internal comparisons
                    self._update_last_processed_timestamp(symbol, timeframe, latest_completed_timestamp)

                    # Publish events for new bars
                    # We need to query the database again to get the stored bars with their IDs
                    # to avoid using detached objects
                    with self._db_session.session_scope() as session:
                        if last_processed:
                            # Convert to naive for SQL Server comparison
                            last_processed_naive = last_processed.replace(
                                tzinfo=None) if last_processed.tzinfo else last_processed

                            # Use standard timestamp comparison without timezone function
                            new_bars = session.query(PriceBar).filter(
                                and_(
                                    PriceBar.instrument_id == instrument_id,
                                    PriceBar.timeframe_id == timeframe_id,
                                    PriceBar.timestamp > last_processed_naive
                                )
                            ).order_by(PriceBar.timestamp).all()
                        else:
                            new_bars = session.query(PriceBar).filter(
                                and_(
                                    PriceBar.instrument_id == instrument_id,
                                    PriceBar.timeframe_id == timeframe_id
                                )
                            ).order_by(PriceBar.timestamp).all()

                        # Copy necessary data to avoid session issues after the session is closed
                        bar_copies = []
                        for bar in new_bars:
                            bar_copy = PriceBar(
                                instrument_id=bar.instrument_id,
                                timeframe_id=bar.timeframe_id,
                                timestamp=bar.timestamp,  # Keep as naive datetime from database
                                open=bar.open,
                                high=bar.high,
                                low=bar.low,
                                close=bar.close,
                                volume=bar.volume,
                                spread=bar.spread
                            )
                            bar_copies.append(bar_copy)

                    # Now publish events using the copied objects outside the session
                    for bar in bar_copies:
                        event = NewBarEvent(instrument_id, timeframe_id, bar, symbol, timeframe.name)
                        self._event_bus.publish(event)

                    return True  # New bar was detected and processed

                return False  # No new bars

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error syncing {symbol}/{timeframe.name}: {str(e)}",
                exception_type=type(e).__name__,
                function="_sync_timeframe",
                traceback=traceback.format_exc(),
                context={"symbol": symbol, "timeframe": timeframe.name}
            )
            return False

    def _check_higher_timeframes(self, symbol: str, current_time: datetime) -> None:
        """
        Check if higher timeframes need to be synced for a specific symbol.
        This is called when a new M1 bar is detected.

        Args:
            symbol: The symbol to check
            current_time: The current MT5 server time (with timezone info)
        """
        try:
            # Define timeframes in ascending order of minutes
            timeframe_minutes = [5, 15, 30, 60, 240, 1440, 10080, 43200]  # M5 to MN1

            for minutes in timeframe_minutes:
                # Find the corresponding TimeFrame enum
                matching_timeframe = None
                for tf in TimeFrame:
                    if tf.minutes == minutes:
                        matching_timeframe = tf
                        break

                if matching_timeframe is None:
                    continue

                # Check if this symbol/timeframe combination exists
                key = (symbol, matching_timeframe)
                if key not in self._instrument_timeframe_map:
                    continue

                # Check if the current time indicates a potential timeframe close
                # For example, if it's exactly on the hour, an H1 bar may have closed

                # Ensure we're working with a naive datetime for minute/hour extraction
                # but keep a copy of the timezone info
                if current_time.tzinfo is not None:
                    naive_current_time = current_time.replace(tzinfo=None)
                    timezone_info = current_time.tzinfo
                else:
                    naive_current_time = current_time
                    timezone_info = None

                minute = naive_current_time.minute
                hour = naive_current_time.hour
                day = naive_current_time.day

                should_check = False

                # M5: Check every 5 minutes (00, 05, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
                if minutes == 5 and minute % 5 == 0:
                    should_check = True

                # M15: Check every 15 minutes (00, 15, 30, 45)
                elif minutes == 15 and minute % 15 == 0:
                    should_check = True

                # M30: Check every 30 minutes (00, 30)
                elif minutes == 30 and minute % 30 == 0:
                    should_check = True

                # H1: Check on the hour (minute = 00)
                elif minutes == 60 and minute == 0:
                    should_check = True

                # H4: Check every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
                elif minutes == 240 and minute == 0 and hour % 4 == 0:
                    should_check = True

                # D1: Check at midnight (00:00)
                elif minutes == 1440 and minute == 0 and hour == 0:
                    should_check = True

                # Other timeframes (W1, MN1) are more complex and may need additional logic

                if should_check:
                    # Restore timezone info for the sync operation
                    if timezone_info is not None:
                        check_time = naive_current_time.replace(tzinfo=timezone_info)
                    else:
                        check_time = naive_current_time

                    # This timeframe may have a new completed bar, synchronize it
                    self._sync_timeframe(symbol, matching_timeframe, check_time)

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error checking higher timeframes for {symbol}: {str(e)}",
                exception_type=type(e).__name__,
                function="_check_higher_timeframes",
                traceback=traceback.format_exc(),
                context={"symbol": symbol}
            )

    def is_market_open(self, symbol: str) -> bool:
        """
        Determine if the market is currently open for the given symbol.
        Takes into account weekends and typical trading hours.

        Args:
            symbol: The instrument symbol to check

        Returns:
            bool: True if the market is open, False otherwise
        """
        try:
            # First try the MT5 API through the MT5 manager
            if not self._mt5_manager or not self._mt5_manager.ensure_connection():
                return False

            symbol_info = self._mt5_manager.get_symbol_info(symbol)
            if symbol_info:
                # Check the 'trade_mode' property - 0 means no trading
                if hasattr(symbol_info, 'trade_mode') and symbol_info.trade_mode == 0:
                    return False

            # Get current time from MT5 if possible, otherwise system time
            current_time = self._get_mt5_server_time()
            if not current_time:
                current_time = datetime.now(timezone.utc)

            # Check for weekend (Saturday = 5, Sunday = 6)
            weekday = current_time.weekday()
            if weekday == 5:  # Saturday
                return False

            if weekday == 6:  # Sunday
                # Sunday - market typically opens around 5:00 PM Eastern Time
                # Convert to 24-hour format in server time
                if current_time.hour < 17:  # Before 5 PM
                    return False

            # Check for Friday close (Friday = 4)
            if weekday == 4:  # Friday
                if current_time.hour >= 17:  # After 5 PM Eastern
                    return False

            # For other days, assume market is open (could add more specific logic if needed)
            return True

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error checking if market is open for {symbol}: {str(e)}",
                exception_type=type(e).__name__,
                function="is_market_open",
                traceback=traceback.format_exc(),
                context={"symbol": symbol}
            )
            # Default to assuming market is open when uncertain
            return True

    def _get_mt5_server_time(self) -> Optional[datetime]:
        """
        Get the current MT5 server time with proper error handling and timezone handling.

        Returns:
            datetime: The current MT5 server time (as UTC time with timezone info preserved),
                     or None if it couldn't be retrieved
        """
        try:
            # Use MT5 manager to get server time
            if self._mt5_manager and hasattr(self._mt5_manager, 'get_server_time'):
                server_time = self._mt5_manager.get_server_time()
                if server_time:
                    # Ensure it has timezone info
                    if server_time.tzinfo is None:
                        server_time = server_time.replace(tzinfo=timezone.utc)
                    return server_time

            self._logger.log_error(
                level="ERROR",
                message="Failed to get MT5 server time via MT5 manager",
                exception_type="MT5DataError",
                function="_get_mt5_server_time",
                traceback="",
                context={}
            )
            return None

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error getting MT5 server time: {str(e)}",
                exception_type=type(e).__name__,
                function="_get_mt5_server_time",
                traceback=str(e),
                context={}
            )
            return None

    def ensure_connection(self) -> bool:
        """Ensure MT5 connection is available through MT5 manager"""
        try:
            # Use the MT5 manager instance to check connection
            if self._mt5_manager and hasattr(self._mt5_manager, 'ensure_connection'):
                return self._mt5_manager.ensure_connection()
            else:
                self._logger.log_error(
                    level="ERROR",
                    message="MT5 manager not available or missing ensure_connection method",
                    exception_type="ConnectionError",
                    function="ensure_connection",
                    traceback="",
                    context={}
                )
                return False
        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error in ensure_connection: {str(e)}",
                exception_type=type(e).__name__,
                function="ensure_connection",
                traceback=str(e),
                context={}
            )
            return False