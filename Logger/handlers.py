# Logger/handlers.py

import logging
import json
import os
import threading
import sys
from datetime import datetime
import traceback
from typing import Dict, Any, Optional

from sqlalchemy import create_engine, and_, func, desc
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError

from Database.db_session import DatabaseSession
from Database.models import EventLog, ErrorLog, TradeLog


class SQLAlchemyHandler(logging.Handler):
    def __init__(self, conn_string, max_records=10000):
        super().__init__()
        self.conn_string = conn_string
        self.max_records = max_records
        self.lock = threading.RLock()
        self.db_session = DatabaseSession(conn_string)

        # Create tables if they don't exist
        self._ensure_tables()

        # Maintenance interval (check record limits every 50 records)
        self.maintenance_counter = 0
        self.maintenance_interval = 50

    def _ensure_tables(self):
        """Create log tables if they don't exist"""
        try:
            # Create schemas and tables using SQLAlchemy
            self.db_session.create_all_tables()
        except SQLAlchemyError as e:
            # Log to stderr as a fallback
            sys.stderr.write(f"Error creating log tables: {str(e)}\n")

    def emit(self, record):
        """Process and store the log record"""
        with self.lock:
            try:
                log_entry = self.format(record)
                extra = getattr(record, 'extra', {})
                entry_type = extra.get('entry_type', 'event')

                if entry_type == 'error':
                    self._log_error(record, log_entry, extra)
                elif entry_type == 'trade':
                    self._log_trade(record, log_entry, extra)
                else:
                    self._log_event(record, log_entry, extra)

                # Periodic maintenance
                self.maintenance_counter += 1
                if self.maintenance_counter >= self.maintenance_interval:
                    self.maintenance_counter = 0
                    self._enforce_record_limits()
            except Exception:
                self.handleError(record)

    def _log_event(self, record, message, extra):
        """Store event log entry with enhanced information"""
        import os
        import threading

        event_type = extra.get('event_type', 'general')
        component = extra.get('component', 'system')
        details = extra.get('details', {})

        # Get additional context
        source = extra.get('source', record.pathname if hasattr(record, 'pathname') else None)
        username = extra.get('username', os.environ.get('USERNAME', 'unknown'))
        action = extra.get('action', None)
        status = extra.get('status', 'success')
        process_id = os.getpid()
        thread_id = threading.get_ident()

        try:
            with self.db_session.session_scope() as session:
                # Create new event log entry
                event_entry = EventLog(
                    timestamp=datetime.now(),
                    level=record.levelname,
                    event_type=event_type,
                    component=component,
                    message=message,
                    source=source,
                    user_name=username,
                    action=action,
                    status=status,
                    process_id=process_id,
                    thread_id=thread_id,
                    details=json.dumps(details) if details else None
                )

                session.add(event_entry)

        except SQLAlchemyError as e:
            # Log to stderr as a fallback
            sys.stderr.write(f"Error logging event: {str(e)}\n")

    def _log_error(self, record, message, extra):
        """Store error log entry"""
        exception_type = extra.get('exception_type',
                                   type(record.exc_info[1]).__name__ if record.exc_info else 'Unknown')
        function = extra.get('function', record.funcName)
        tb = extra.get('traceback', ''.join(traceback.format_exception(*record.exc_info)) if record.exc_info else '')
        context = extra.get('context', {})

        try:
            with self.db_session.session_scope() as session:
                # Create new error log entry
                error_entry = ErrorLog(
                    timestamp=datetime.now(),
                    level=record.levelname,
                    exception_type=exception_type,
                    function_name=function,
                    message=message,
                    traceback=tb,
                    context=json.dumps(context) if context else None
                )

                session.add(error_entry)

        except SQLAlchemyError as e:
            # Log to stderr as a fallback
            sys.stderr.write(f"Error logging error entry: {str(e)}\n")

    def _log_trade(self, record, message, extra):
        """Store trade log entry"""
        try:
            with self.db_session.session_scope() as session:
                # Create new trade log entry
                trade_entry = TradeLog(
                    timestamp=datetime.now(),
                    level=record.levelname,
                    symbol=extra.get('symbol', 'UNKNOWN'),
                    operation=extra.get('operation', 'UNKNOWN'),
                    price=extra.get('price', 0.0),
                    volume=extra.get('volume', 0.0),
                    order_id=extra.get('order_id'),
                    strategy=extra.get('strategy'),
                    message=message
                )

                session.add(trade_entry)

        except SQLAlchemyError as e:
            # Log to stderr as a fallback
            sys.stderr.write(f"Error logging trade entry: {str(e)}\n")

    def _enforce_record_limits(self):
        """Delete oldest records when limit is reached for all log tables"""
        try:
            tables = ['Events', 'Errors', 'Trades']

            with self.db_session.session_scope() as session:
                for table_name in tables:
                    self._enforce_table_limit(session, table_name)

        except SQLAlchemyError as e:
            # Log to stderr as a fallback
            sys.stderr.write(f"Error enforcing record limits: {str(e)}\n")

    def _enforce_table_limit(self, session, table_name):
        """Delete oldest records for a specific table when limit is reached"""
        try:
            # Select the appropriate model class
            if table_name == 'Events':
                model_class = EventLog
            elif table_name == 'Errors':
                model_class = ErrorLog
            elif table_name == 'Trades':
                model_class = TradeLog
            else:
                return

            # Get current count
            count = session.query(func.count(model_class.id)).scalar()

            # If over limit, delete oldest records
            if count > self.max_records:
                delete_count = count - self.max_records

                # Get IDs of oldest records to delete
                oldest_ids = session.query(model_class.id) \
                    .order_by(model_class.timestamp) \
                    .limit(delete_count) \
                    .all()

                # Extract just the IDs
                ids_to_delete = [record_id for (record_id,) in oldest_ids]

                # Delete the records
                session.query(model_class) \
                    .filter(model_class.id.in_(ids_to_delete)) \
                    .delete(synchronize_session=False)

        except SQLAlchemyError as e:
            # Log to stderr as a fallback
            sys.stderr.write(f"Error enforcing limit for {table_name}: {str(e)}\n")