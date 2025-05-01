# container.py
from dependency_injector import containers, providers
from Logger.logger import DBLogger
from Config.logging_config import LoggingConfig
from Config.db_config import DatabaseConfig


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

    # Load logging configuration
    logging_config = providers.Factory(
        LoggingConfig,
        conn_string=db_config.provided.connection_string,
        enabled_levels=config.logging.enabled_levels,
        console_output=config.logging.console_output,
        max_records=config.logging.max_records,
        color_scheme=config.logging.color_scheme
    )

    # Create logger
    db_logger = providers.Singleton(
        DBLogger,
        conn_string=db_config.provided.connection_string,
        enabled_levels=config.logging.enabled_levels,
        console_output=config.logging.console_output,
        color_scheme=config.logging.color_scheme
    )