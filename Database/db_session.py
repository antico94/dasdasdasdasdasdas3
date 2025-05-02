# Database/db_session.py
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import threading
import logging
import sys

from Database.models import Base


class DatabaseSession:
    """Database session manager using SQLAlchemy"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseSession, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, conn_string=None):
        if self._initialized and conn_string is None:
            return

        with self._lock:
            if not self._initialized or conn_string is not None:
                self._conn_string = conn_string
                self._engine = None
                self._session_factory = None
                self._scoped_session = None

                if conn_string:
                    self.initialize(conn_string)

                self._initialized = True

    def initialize(self, conn_string=None):
        """Initialize database connection"""
        if conn_string:
            self._conn_string = conn_string

        if not self._conn_string:
            raise ValueError("Database connection string not provided")

        try:
            # Create SQLAlchemy engine with connection pooling
            self._engine = create_engine(
                self._conn_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_pre_ping=True  # Check connection validity before using
            )

            # Add engine event listeners for monitoring
            event.listen(self._engine, 'connect', self._on_connect)
            event.listen(self._engine, 'checkout', self._on_checkout)
            event.listen(self._engine, 'checkin', self._on_checkin)

            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine)

            # Create thread-local scoped session
            self._scoped_session = scoped_session(self._session_factory)

            return True

        except SQLAlchemyError as e:
            logging.error(f"Failed to initialize database: {str(e)}")
            return False

    def create_all_tables(self):
        """Create all tables defined in Base"""
        if not self._engine:
            raise RuntimeError("Database engine not initialized")

        try:
            # Create schemas directly with raw SQL commands
            # This approach is more reliable for SQL Server
            with self._engine.begin() as connection:
                # Create the Trading schema if it doesn't exist
                connection.execute(text(
                    "IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'Trading') BEGIN EXEC('CREATE SCHEMA Trading') END"))
                # Create the Logs schema if it doesn't exist
                connection.execute(text(
                    "IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'Logs') BEGIN EXEC('CREATE SCHEMA Logs') END"))

            # Now create all tables
            Base.metadata.create_all(self._engine)

            logging.info("Successfully created database schemas and tables")

        except SQLAlchemyError as e:
            sys.stderr.write(f"Error creating database schemas and tables: {str(e)}\n")
            raise

    def get_session(self):
        """Get current thread-local session"""
        if not self._scoped_session:
            raise RuntimeError("Database session not initialized")

        return self._scoped_session()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self):
        """Dispose engine and all connections"""
        if self._scoped_session:
            self._scoped_session.remove()

        if self._engine:
            self._engine.dispose()

    def _on_connect(self, dbapi_connection, connection_record):
        """Event listener for new connections"""
        logging.debug("Database connection created")

    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Event listener for connection checkout from pool"""
        logging.debug("Database connection checked out from pool")

    def _on_checkin(self, dbapi_connection, connection_record):
        """Event listener for connection checkin to pool"""
        logging.debug("Database connection returned to pool")

    def get_engine(self):
        """Get SQLAlchemy engine"""
        return self._engine

    def get_connection_string(self):
        """Get connection string formatted for SQLAlchemy"""
        if not self._conn_string:
            raise ValueError("Database connection string not initialized")
        return self._conn_string