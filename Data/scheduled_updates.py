# Data/scheduled_updates.py

import threading
import time
import schedule
from datetime import datetime, timedelta, timezone
from typing import Dict, Set, Optional, List, Any, Tuple
import traceback

from Config.trading_config import TimeFrame
from Logger.logger import DBLogger
from MT5.mt5_manager import MT5Manager
from Data.data_fetcher import DataFetcher
from Events.events import MarketStateEvent, MarketStateType


class TimeframeUpdateScheduler:
    """
    Scheduler for timeframe-specific updates based on candle formation times.
    This ensures updates are performed right after a candle is expected to form.
    """

    def __init__(self, data_fetcher: DataFetcher, mt5_manager: MT5Manager, logger: DBLogger):
        self._data_fetcher = data_fetcher
        self._mt5_manager = mt5_manager
        self._logger = logger
        self._scheduler = schedule.Scheduler()
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
        self._registered_updates = set()
        self._market_state_cache = {}  # Symbol -> MarketStateType

    def start(self) -> bool:
        """Start the scheduler thread"""
        if self._running:
            self._logger.log_event(
                level="WARNING",
                message="Update scheduler already running",
                event_type="SCHEDULER_START",
                component="timeframe_scheduler",
                action="start",
                status="ignored"
            )
            return True

        try:
            self._stop_event.clear()
            self._scheduler.clear()  # Clear any existing jobs
            self._setup_schedules()

            self._running = True
            self._thread = threading.Thread(target=self._scheduler_loop)
            self._thread.daemon = True
            self._thread.start()

            self._logger.log_event(
                level="INFO",
                message="Timeframe update scheduler started",
                event_type="SCHEDULER_START",
                component="timeframe_scheduler",
                action="start",
                status="success"
            )
            return True

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to start update scheduler: {str(e)}",
                exception_type=type(e).__name__,
                function="start",
                traceback=traceback.format_exc(),
                context={}
            )
            self._running = False
            return False

    def stop(self) -> None:
        """Stop the scheduler thread"""
        if not self._running:
            return

        try:
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=10)

            self._scheduler.clear()
            self._running = False

            self._logger.log_event(
                level="INFO",
                message="Timeframe update scheduler stopped",
                event_type="SCHEDULER_STOP",
                component="timeframe_scheduler",
                action="stop",
                status="success"
            )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error stopping update scheduler: {str(e)}",
                exception_type=type(e).__name__,
                function="stop",
                traceback=traceback.format_exc(),
                context={}
            )
            # Mark as not running even if there was an error
            self._running = False

    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        last_market_check = None
        check_interval = 300  # 5 minutes in seconds

        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()

                # Check market state only on startup and then every 5 minutes
                if last_market_check is None or (current_time - last_market_check).total_seconds() >= check_interval:
                    self._update_market_states()
                    last_market_check = current_time

                # Run pending scheduled jobs
                self._scheduler.run_pending()

                # Sleep for a bit to avoid high CPU usage
                time.sleep(1)

            except Exception as e:
                self._logger.log_error(
                    level="ERROR",
                    message=f"Error in scheduler loop: {str(e)}",
                    exception_type=type(e).__name__,
                    function="_scheduler_loop",
                    traceback=traceback.format_exc(),
                    context={}
                )
                # Sleep a bit longer after an error
                time.sleep(5)

    def _setup_schedules(self) -> None:
        """Set up update schedules for different timeframes"""
        # Get all instrument/timeframe combinations from data fetcher
        map_copy = self._data_fetcher._instrument_timeframe_map.copy()

        for (symbol, timeframe), mapping in map_copy.items():
            minutes = timeframe.minutes

            # Schedule based on timeframe
            if minutes == 1:  # M1
                # Every minute, on the 5th second
                self._scheduler.every().minute.at(":05").do(
                    self._run_update_job, symbol=symbol, timeframe=timeframe)
                self._registered_updates.add((symbol, timeframe.name, "minute:05"))

            elif minutes == 5:  # M5
                # Every 5 minutes (00:05, 00:10, 00:15, etc.)
                for minute in range(0, 60, 5):
                    job = self._scheduler.every().hour.at(f"{minute:02d}:05")
                    job.do(self._run_update_job, symbol=symbol, timeframe=timeframe)
                    self._registered_updates.add((symbol, timeframe.name, f"{minute:02d}:05"))

            elif minutes == 15:  # M15
                # Every 15 minutes (00:15, 00:30, 00:45, 01:00)
                for minute in range(0, 60, 15):
                    job = self._scheduler.every().hour.at(f"{minute:02d}:05")
                    job.do(self._run_update_job, symbol=symbol, timeframe=timeframe)
                    self._registered_updates.add((symbol, timeframe.name, f"{minute:02d}:05"))

            elif minutes == 30:  # M30
                # Every 30 minutes (00:30, 01:00)
                for minute in range(0, 60, 30):
                    job = self._scheduler.every().hour.at(f"{minute:02d}:05")
                    job.do(self._run_update_job, symbol=symbol, timeframe=timeframe)
                    self._registered_updates.add((symbol, timeframe.name, f"{minute:02d}:05"))

            elif minutes == 60:  # H1
                # Every hour, at 5 seconds past the hour
                self._scheduler.every().hour.at(":05").do(
                    self._run_update_job, symbol=symbol, timeframe=timeframe)
                self._registered_updates.add((symbol, timeframe.name, "hour:05"))

            elif minutes == 240:  # H4
                # Every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
                # Since schedule doesn't have direct support for "every 4 hours",
                # we need to schedule specific times
                for hour in range(0, 24, 4):
                    job = self._scheduler.every().day.at(f"{hour:02d}:00:05")
                    job.do(self._run_update_job, symbol=symbol, timeframe=timeframe)
                    self._registered_updates.add((symbol, timeframe.name, f"{hour:02d}:00:05"))

            elif minutes == 1440:  # D1
                # Every day at 00:00:05
                self._scheduler.every().day.at("00:00:05").do(
                    self._run_update_job, symbol=symbol, timeframe=timeframe)
                self._registered_updates.add((symbol, timeframe.name, "00:00:05"))

        # Additionally, schedule frequent updates for M1
        # This ensures we get frequent updates for the most active timeframe
        self._scheduler.every(20).seconds.do(self._update_m1_timeframes)

        # Schedule market state checks
        self._scheduler.every(5).minutes.do(self._update_market_states)

        # Log the setup
        self._logger.log_event(
            level="INFO",
            message=f"Scheduled updates set up for {len(self._registered_updates)} timeframe instances",
            event_type="SCHEDULER_SETUP",
            component="timeframe_scheduler",
            action="_setup_schedules",
            status="success",
            details={"update_count": len(self._registered_updates)}
        )

    def _run_update_job(self, symbol: str, timeframe: TimeFrame) -> None:
        """Run update job for a specific symbol and timeframe"""
        try:
            # Check if market is open for this symbol
            market_state = self._market_state_cache.get(symbol)
            if market_state != MarketStateType.OPEN:
                # Skip updates for closed markets
                self._logger.log_event(
                    level="INFO",
                    message=f"Skipping update for {symbol}/{timeframe.name} - market is {market_state}",
                    event_type="SCHEDULER_UPDATE",
                    component="timeframe_scheduler",
                    action="_run_update_job",
                    status="skipped",
                    details={"symbol": symbol, "timeframe": timeframe.name, "market_state": str(market_state)}
                )
                return

            self._logger.log_event(
                level="INFO",
                message=f"Running scheduled update for {symbol}/{timeframe.name}",
                event_type="SCHEDULER_UPDATE",
                component="timeframe_scheduler",
                action="_run_update_job",
                status="starting",
                details={"symbol": symbol, "timeframe": timeframe.name}
            )

            # Force sync for this specific symbol/timeframe
            success = self._data_fetcher.force_sync(symbol, timeframe)

            if success:
                self._logger.log_event(
                    level="INFO",
                    message=f"Scheduled update completed for {symbol}/{timeframe.name}",
                    event_type="SCHEDULER_UPDATE",
                    component="timeframe_scheduler",
                    action="_run_update_job",
                    status="success",
                    details={"symbol": symbol, "timeframe": timeframe.name}
                )
            else:
                self._logger.log_event(
                    level="WARNING",
                    message=f"Scheduled update failed for {symbol}/{timeframe.name}",
                    event_type="SCHEDULER_UPDATE",
                    component="timeframe_scheduler",
                    action="_run_update_job",
                    status="failed",
                    details={"symbol": symbol, "timeframe": timeframe.name}
                )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error running update job for {symbol}/{timeframe.name}: {str(e)}",
                exception_type=type(e).__name__,
                function="_run_update_job",
                traceback=traceback.format_exc(),
                context={"symbol": symbol, "timeframe": str(timeframe.name)}
            )

    def _update_m1_timeframes(self) -> None:
        """Update all M1 timeframes frequently"""
        try:
            # Get all M1 timeframes
            m1_pairs = []
            map_copy = self._data_fetcher._instrument_timeframe_map.copy()

            for (symbol, timeframe), _ in map_copy.items():
                if timeframe.minutes == 1:
                    # Check if market is open for this symbol
                    market_state = self._market_state_cache.get(symbol)
                    if market_state == MarketStateType.OPEN:
                        m1_pairs.append((symbol, timeframe))

            if not m1_pairs:
                return

            self._logger.log_event(
                level="INFO",
                message=f"Updating {len(m1_pairs)} M1 timeframes",
                event_type="SCHEDULER_UPDATE",
                component="timeframe_scheduler",
                action="_update_m1_timeframes",
                status="starting",
                details={"count": len(m1_pairs)}
            )

            # Update each M1 pair
            for symbol, timeframe in m1_pairs:
                self._data_fetcher.force_sync(symbol, timeframe)
                time.sleep(0.2)  # Short pause between updates

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error updating M1 timeframes: {str(e)}",
                exception_type=type(e).__name__,
                function="_update_m1_timeframes",
                traceback=traceback.format_exc(),
                context={}
            )

    def _update_market_states(self) -> None:
        """Update market state for all symbols"""
        try:
            # Get all unique symbols
            symbols = set()
            map_copy = self._data_fetcher._instrument_timeframe_map.copy()

            for (symbol, _), _ in map_copy.items():
                symbols.add(symbol)

            # Only log on first run
            first_run = not hasattr(self, '_market_state_initialized')
            if first_run:
                self._market_state_initialized = True
                self._logger.log_event(
                    level="INFO",
                    message=f"Initializing market state for {len(symbols)} symbols",
                    event_type="MARKET_STATE",
                    component="timeframe_scheduler",
                    action="_update_market_states",
                    status="starting",
                    details={"count": len(symbols)}
                )

            # Check each symbol
            for symbol in symbols:
                if not self._mt5_manager.ensure_connection():
                    self._logger.log_event(
                        level="WARNING",
                        message="Cannot check market state - MT5 connection error",
                        event_type="MARKET_STATE",
                        component="timeframe_scheduler",
                        action="_update_market_states",
                        status="connection_error"
                    )
                    return

                symbol_info = self._mt5_manager.get_symbol_info(symbol)
                if not symbol_info:
                    self._logger.log_event(
                        level="WARNING",
                        message=f"Cannot get symbol info for {symbol}",
                        event_type="MARKET_STATE",
                        component="timeframe_scheduler",
                        action="_update_market_states",
                        status="symbol_error",
                        details={"symbol": symbol}
                    )
                    continue

                # Determine market state
                if hasattr(symbol_info, 'trade_mode'):
                    trade_mode = symbol_info.trade_mode

                    # Always treat trade_mode 4 as OPEN since we can place orders manually
                    if trade_mode == 0:  # No trading
                        state = MarketStateType.CLOSED
                    elif trade_mode in [1, 2, 3, 4]:  # All trading modes including "close only"
                        state = MarketStateType.OPEN
                    else:
                        state = MarketStateType.CLOSED
                else:
                    # No trade_mode attribute, use session info
                    if datetime.now().weekday() >= 5:  # Weekend
                        state = MarketStateType.CLOSED
                    else:
                        state = MarketStateType.OPEN

                # Update cache and ONLY log if state changed
                previous_state = self._market_state_cache.get(symbol)
                if previous_state != state:
                    self._market_state_cache[symbol] = state
                    self._logger.log_event(
                        level="INFO",
                        message=f"Market state for {symbol}: {state}",
                        event_type="MARKET_STATE",
                        component="timeframe_scheduler",
                        action="_update_market_states",
                        status="state_change",
                        details={"symbol": symbol, "state": str(state), "previous": str(previous_state)}
                    )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error updating market states: {str(e)}",
                exception_type=type(e).__name__,
                function="_update_market_states",
                traceback=traceback.format_exc(),
                context={}
            )