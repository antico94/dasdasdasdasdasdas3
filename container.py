# container.py
from dependency_injector import containers, providers
from Logger.logger import DBLogger
from Config.logging_config import LoggingConfig
from Config.db_config import DatabaseConfig
from Database.db_session import DatabaseSession
from Data.data_fetcher import DataFetcher
from Events.event_bus import EventBus
from MT5.mt5_manager import MT5Manager
from Data.scheduled_updates import TimeframeUpdateScheduler


class Container(containers.DeclarativeContainer):
    """Application container"""

    config = providers.Configuration()

    # Database configuration
    db_config = providers.Factory(
        DatabaseConfig,
        server=config.db.server,
        database=config.db.database,
        driver=config.db.driver,
        trusted_connection=config.db.trusted_connection
    )

    # Create connection string for SQLAlchemy
    db_connection_string = providers.Callable(
        lambda
            config: f"mssql+pyodbc://{config.server}/{config.database}?driver={config.driver.replace(' ', '+')}&trusted_connection={'yes' if config.trusted_connection else 'no'}",
        config=db_config
    )

    # Database session manager
    db_session = providers.Singleton(
        DatabaseSession,
        conn_string=db_connection_string
    )

    # Load logging configuration
    logging_config = providers.Factory(
        LoggingConfig,
        conn_string=db_connection_string,
        enabled_levels=config.logging.enabled_levels,
        console_output=config.logging.console_output,
        max_records=config.logging.max_records,
        color_scheme=config.logging.color_scheme
    )

    # Create logger
    db_logger = providers.Singleton(
        DBLogger,
        conn_string=db_connection_string,
        enabled_levels=config.logging.enabled_levels,
        console_output=config.logging.console_output,
        color_scheme=config.logging.color_scheme
    )

    # MT5 Manager
    mt5_manager = providers.Singleton(
        MT5Manager
    )

    # Event Bus
    event_bus = providers.Singleton(
        EventBus
    )

    # Data Fetcher
    data_fetcher = providers.Singleton(
        DataFetcher
    )

    # Timeframe Update Scheduler
    timeframe_scheduler = providers.Singleton(
        TimeframeUpdateScheduler,
        data_fetcher=data_fetcher,
        mt5_manager=mt5_manager,
        logger=db_logger
    )