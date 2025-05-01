# Logger/handlers.py

import logging
import pyodbc
import json
from datetime import datetime
import traceback
import threading


class SQLServerHandler(logging.Handler):
    def __init__(self, conn_string, max_records=10000):
        super().__init__()
        self.conn_string = conn_string
        self.max_records = max_records
        self.lock = threading.RLock()
        self._ensure_tables()

    def _get_connection(self):
        return pyodbc.connect(self.conn_string)

    def _ensure_tables(self):
        """Create log tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create Logs schema if it doesn't exist
            cursor.execute("IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'Logs') EXEC('CREATE SCHEMA Logs')")

            # Events table - Fixed: changed User to UserName to avoid reserved keyword
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Events' AND schema_id = SCHEMA_ID('Logs'))
            BEGIN
                CREATE TABLE Logs.Events (
                    Id INT IDENTITY(1,1) PRIMARY KEY,
                    Timestamp DATETIME2 NOT NULL,
                    Level VARCHAR(10) NOT NULL,
                    EventType VARCHAR(50) NOT NULL,
                    Component VARCHAR(50) NOT NULL,
                    Message NVARCHAR(500) NOT NULL,
                    Source VARCHAR(100) NULL,
                    UserName VARCHAR(100) NULL,
                    Action VARCHAR(100) NULL,
                    Status VARCHAR(20) NULL,
                    ProcessId INT NULL,
                    ThreadId INT NULL,
                    Details NVARCHAR(MAX) NULL
                )
            END
            """)

            # Errors table
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Errors' AND schema_id = SCHEMA_ID('Logs'))
            BEGIN
                CREATE TABLE Logs.Errors (
                    Id INT IDENTITY(1,1) PRIMARY KEY,
                    Timestamp DATETIME2 NOT NULL,
                    Level VARCHAR(10) NOT NULL,
                    ExceptionType VARCHAR(100) NOT NULL,
                    FunctionName VARCHAR(100) NOT NULL,
                    Message NVARCHAR(500) NOT NULL,
                    Traceback NVARCHAR(MAX) NOT NULL,
                    Context NVARCHAR(MAX) NULL
                )
            END
            """)

            # Trades table
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Trades' AND schema_id = SCHEMA_ID('Logs'))
            BEGIN
                CREATE TABLE Logs.Trades (
                    Id INT IDENTITY(1,1) PRIMARY KEY,
                    Timestamp DATETIME2 NOT NULL,
                    Level VARCHAR(10) NOT NULL,
                    Symbol VARCHAR(20) NOT NULL,
                    Operation VARCHAR(20) NOT NULL,
                    Price FLOAT NOT NULL,
                    Volume FLOAT NOT NULL,
                    OrderId INT NULL,
                    Strategy VARCHAR(50) NULL,
                    Message NVARCHAR(500) NOT NULL
                )
            END
            """)

            conn.commit()

    def _enforce_record_limit(self, table):
        """Delete oldest records when limit is reached"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # First check the count
            cursor.execute(f"SELECT COUNT(*) FROM Logs.{table}")
            count = cursor.fetchval()

            # If over the limit, delete oldest records
            if count > self.max_records:
                delete_count = count - self.max_records

                # Use a subquery approach to identify records to delete
                cursor.execute(f"""
                DELETE FROM Logs.{table}
                WHERE Id IN (
                    SELECT TOP({delete_count}) Id 
                    FROM Logs.{table} 
                    ORDER BY Timestamp ASC
                )
                """)

                conn.commit()

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

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO Logs.Events (Timestamp, Level, EventType, Component, Message, Source, UserName,
                                                    Action, Status, ProcessId, ThreadId, Details)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """, (
                               datetime.now(),
                               record.levelname,
                               event_type,
                               component,
                               message,
                               source,
                               username,
                               action,
                               status,
                               process_id,
                               thread_id,
                               json.dumps(details) if details else None
                           ))
            conn.commit()

        self._enforce_record_limit('Events')

    def _log_error(self, record, message, extra):
        """Store error log entry"""
        exception_type = extra.get('exception_type',
                                   type(record.exc_info[1]).__name__ if record.exc_info else 'Unknown')
        function = extra.get('function', record.funcName)
        tb = extra.get('traceback', ''.join(traceback.format_exception(*record.exc_info)) if record.exc_info else '')
        context = extra.get('context', {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO Logs.Errors (Timestamp, Level, ExceptionType, FunctionName, Message, Traceback,
                                                    Context)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           """, (
                               datetime.now(),
                               record.levelname,
                               exception_type,
                               function,
                               message,
                               tb,
                               json.dumps(context) if context else None
                           ))
            conn.commit()

        self._enforce_record_limit('Errors')

    def _log_trade(self, record, message, extra):
        """Store trade log entry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                           INSERT INTO Logs.Trades (Timestamp, Level, Symbol, Operation, Price, Volume, OrderId,
                                                    Strategy, Message)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """, (
                               datetime.now(),
                               record.levelname,
                               extra.get('symbol', 'UNKNOWN'),
                               extra.get('operation', 'UNKNOWN'),
                               extra.get('price', 0.0),
                               extra.get('volume', 0.0),
                               extra.get('order_id'),
                               extra.get('strategy'),
                               message
                           ))
            conn.commit()

        self._enforce_record_limit('Trades')