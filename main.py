# main.py
from container import Container
from dependency_injector.wiring import Provide, inject
from Config.credentials import (
    SQL_SERVER,
    SQL_DATABASE,
    SQL_DRIVER,
    USE_WINDOWS_AUTH
)


def configure_container():
    """Configure and initialize container"""
    container = Container()

    container.config.from_dict({
        'db': {
            'server': SQL_SERVER,
            'database': SQL_DATABASE,
            'driver': SQL_DRIVER,
            'trusted_connection': USE_WINDOWS_AUTH
        },
        'logging': {
            'enabled_levels': {'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            'console_output': True,
            'max_records': 10000,
            'color_scheme': {
                'DEBUG': '\033[37m',  # White
                'INFO': '\033[36m',  # Cyan
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',  # Red
                'CRITICAL': '\033[41m',  # Red background
            }
        }
    })

    return container


@inject
def run_application(logger=Provide[Container.db_logger]):
    """Main application entry point"""
    # Basic application startup logging
    logger.log_event(
        level="INFO",
        message="Application started successfully",
        event_type="SYSTEM_STARTUP",
        component="main",
        action="initialize",
        status="success",
        details={"version": "1.0.0", "environment": "development"}
    )

    # Log configuration loaded
    logger.log_event(
        level="INFO",
        message="Configuration loaded from settings",
        event_type="CONFIG_LOAD",
        component="configuration",
        action="load_settings",
        status="success",
        details={"config_source": "credentials.py", "db_configured": True}
    )

    # Log an example trade event
    logger.log_event(
        level="INFO",
        message="Trade signal generated",
        event_type="TRADE_SIGNAL",
        component="strategy_engine",
        action="generate_signal",
        status="pending",
        details={
            "symbol": "EURUSD",
            "direction": "BUY",
            "strategy": "Moving Average Crossover",
            "reason": "50 SMA crossed above 200 SMA"
        }
    )

    # Try something that fails
    try:
        result = 1 / 0
    except Exception as e:
        import traceback
        logger.log_error(
            level="ERROR",
            message="Division by zero error",
            exception_type=type(e).__name__,
            function="run_application",
            traceback=traceback.format_exc(),
            context={"calculation": "1/0"}
        )
        # Also log as an event
        logger.log_event(
            level="ERROR",
            message="Calculation failed",
            event_type="CALCULATION_ERROR",
            component="math_engine",
            action="divide",
            status="failure",
            details={"numerator": 1, "denominator": 0}
        )


if __name__ == "__main__":
    container = configure_container()
    container.wire(modules=[__name__])

    run_application()