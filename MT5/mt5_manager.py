# MT5/mt5_manager.py
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import MetaTrader5 as mt5

from Config.trading_config import ConfigManager, TimeFrame
from Logger.logger import DBLogger


class MT5Manager:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MT5Manager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Create connection string from credentials
        from Config.credentials import (
            SQL_SERVER,
            SQL_DATABASE,
            SQL_DRIVER,
            USE_WINDOWS_AUTH,
            SQL_USERNAME,
            SQL_PASSWORD
        )

        # Create connection string
        if USE_WINDOWS_AUTH:
            conn_string = f"DRIVER={{{SQL_DRIVER}}};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};Trusted_Connection=yes;"
        else:
            conn_string = f"DRIVER={{{SQL_DRIVER}}};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};UID={SQL_USERNAME};PWD={SQL_PASSWORD}"

        self._config = ConfigManager().config
        self._logger = DBLogger(
            conn_string=conn_string,
            enabled_levels={'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            console_output=True
        )
        self._connected = False
        self._last_check = None
        self._connection_lock = threading.Lock()
        self._initialized = True

    def initialize(self) -> bool:
        """Initialize and connect to MT5"""
        try:
            self._logger.log_event(
                level="INFO",
                message="Initializing MT5 connection",
                event_type="MT5_INIT",
                component="mt5_manager",
                action="initialize",
                status="starting"
            )

            # Check if MT5 is already initialized
            if mt5.initialize():
                self._logger.log_event(
                    level="INFO",
                    message="MT5 was already initialized",
                    event_type="MT5_INIT",
                    component="mt5_manager",
                    action="initialize",
                    status="success"
                )
            else:
                # Try to initialize MT5
                # First, ensure any previous instances are shut down
                try:
                    mt5.shutdown()
                    time.sleep(1)  # Give it time to shut down properly
                except:
                    pass  # Ignore errors during shutdown

                # Now try to initialize again
                if not mt5.initialize():
                    error = mt5.last_error()
                    self._logger.log_error(
                        level="CRITICAL",
                        message=f"Failed to initialize MT5: {error}",
                        exception_type="MT5Error",
                        function="initialize",
                        traceback="",
                        context={"error_code": error[0], "error_message": error[1]}
                    )
                    return False

            # Connect to MT5 account
            if not self._login():
                # If login fails, try to reinitialize MT5 and login again
                mt5.shutdown()
                time.sleep(2)  # Give it more time to shut down properly

                if not mt5.initialize():
                    error = mt5.last_error()
                    self._logger.log_error(
                        level="CRITICAL",
                        message=f"Failed to reinitialize MT5 after failed login: {error}",
                        exception_type="MT5Error",
                        function="initialize",
                        traceback="",
                        context={"error_code": error[0], "error_message": error[1]}
                    )
                    return False

                # Try to login again
                if not self._login():
                    return False

            self._connected = True
            self._last_check = datetime.now()

            self._logger.log_event(
                level="INFO",
                message="MT5 initialized successfully",
                event_type="MT5_INIT",
                component="mt5_manager",
                action="initialize",
                status="success"
            )

            return True

        except Exception as e:
            self._logger.log_error(
                level="CRITICAL",
                message=f"MT5 initialization failed with unexpected error: {str(e)}",
                exception_type=type(e).__name__,
                function="initialize",
                traceback=str(e),
                context={}
            )
            return False

    def _login(self) -> bool:
        """Login to MT5"""
        mt5_config = self._config.mt5

        for attempt in range(1, self._config.max_retry_attempts + 1):
            try:
                # First check if we're already logged in
                account_info = mt5.account_info()
                if account_info is not None:
                    # Already logged in
                    self._logger.log_event(
                        level="INFO",
                        message="MT5 already logged in",
                        event_type="MT5_LOGIN",
                        component="mt5_manager",
                        action="login",
                        status="success"
                    )
                    return True

                # Try to log in
                login_result = mt5.login(
                    login=mt5_config.login,
                    password=mt5_config.password,
                    server=mt5_config.server,
                    timeout=mt5_config.timeout
                )

                if login_result:
                    self._logger.log_event(
                        level="INFO",
                        message="MT5 login successful",
                        event_type="MT5_LOGIN",
                        component="mt5_manager",
                        action="login",
                        status="success"
                    )
                    return True
                else:
                    error = mt5.last_error()
                    self._logger.log_error(
                        level="ERROR",
                        message=f"MT5 login failed (attempt {attempt}/{self._config.max_retry_attempts}): {error}",
                        exception_type="MT5LoginError",
                        function="_login",
                        traceback="",
                        context={"error_code": error[0], "error_message": error[1], "attempt": attempt}
                    )

                    # Wait before retry
                    if attempt < self._config.max_retry_attempts:
                        time.sleep(self._config.retry_delay_seconds)

            except Exception as e:
                self._logger.log_error(
                    level="ERROR",
                    message=f"MT5 login failed with exception (attempt {attempt}/{self._config.max_retry_attempts}): {str(e)}",
                    exception_type=type(e).__name__,
                    function="_login",
                    traceback=str(e),
                    context={"attempt": attempt}
                )

                # Wait before retry
                if attempt < self._config.max_retry_attempts:
                    time.sleep(self._config.retry_delay_seconds)

        # All attempts failed
        self._logger.log_error(
            level="CRITICAL",
            message=f"MT5 login failed after {self._config.max_retry_attempts} attempts",
            exception_type="MT5LoginError",
            function="_login",
            traceback="",
            context={"max_attempts": self._config.max_retry_attempts}
        )
        return False

    def shutdown(self) -> None:
        """Shutdown MT5 connection"""
        try:
            mt5.shutdown()
            self._connected = False
            self._logger.log_event(
                level="INFO",
                message="MT5 connection shutdown",
                event_type="MT5_SHUTDOWN",
                component="mt5_manager",
                action="shutdown",
                status="success"
            )
        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error during MT5 shutdown: {str(e)}",
                exception_type=type(e).__name__,
                function="shutdown",
                traceback=str(e),
                context={}
            )

    def ensure_connection(self) -> bool:
        """Ensure MT5 is connected, reconnect if necessary"""
        with self._connection_lock:
            # Check if we need to verify connection (not too frequent)
            now = datetime.now()
            if self._last_check and (now - self._last_check).total_seconds() < 30:
                return self._connected

            self._last_check = now

            # If not connected, try to initialize
            if not self._connected:
                return self.initialize()

            # Check terminal state to verify connection
            try:
                terminal_info = mt5.terminal_info()
                if terminal_info is None:
                    error = mt5.last_error()
                    self._logger.log_error(
                        level="WARNING",
                        message=f"MT5 connection lost: {error}",
                        exception_type="MT5ConnectionError",
                        function="ensure_connection",
                        traceback="",
                        context={"error_code": error[0], "error_message": error[1]}
                    )

                    # Try shutdown/reinitialize cycle
                    try:
                        mt5.shutdown()
                        time.sleep(2)  # Give it time to shut down properly
                    except:
                        pass  # Ignore shutdown errors

                    # Try to reconnect with delay
                    self._connected = False
                    time.sleep(3)  # Wait before attempting reconnection
                    return self.initialize()

                # Also verify that we're logged in
                account_info = mt5.account_info()
                if account_info is None:
                    self._logger.log_error(
                        level="WARNING",
                        message="MT5 account not logged in",
                        exception_type="MT5LoginError",
                        function="ensure_connection",
                        traceback="",
                        context={}
                    )

                    # Try to login again
                    self._connected = False
                    return self.initialize()

                return True

            except Exception as e:
                self._logger.log_error(
                    level="WARNING",
                    message=f"Error checking MT5 connection: {str(e)}",
                    exception_type=type(e).__name__,
                    function="ensure_connection",
                    traceback=str(e),
                    context={}
                )

                # Try to reconnect
                self._connected = False
                return self.initialize()

    def copy_rates_from_pos(self, symbol: str, timeframe: TimeFrame, start_pos: int, count: int) -> Optional[
        List[Dict[str, Any]]]:
        """Copy rates from position"""
        if not self.ensure_connection():
            return None

        try:
            mt5_timeframe = timeframe.mt5_timeframe

            # Copy rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, start_pos, count)

            if rates is None:
                error = mt5.last_error()
                self._logger.log_error(
                    level="ERROR",
                    message=f"Failed to copy rates for {symbol}/{timeframe.name}: {error}",
                    exception_type="MT5DataError",
                    function="copy_rates_from_pos",
                    traceback="",
                    context={
                        "symbol": symbol,
                        "timeframe": timeframe.name,
                        "start_pos": start_pos,
                        "count": count,
                        "error_code": error[0],
                        "error_message": error[1]
                    }
                )
                return None

            # Convert to list of dictionaries
            result = []
            for rate in rates:
                rate_dict = {
                    "time": datetime.fromtimestamp(rate[0]),
                    "open": rate[1],
                    "high": rate[2],
                    "low": rate[3],
                    "close": rate[4],
                    "tick_volume": rate[5],
                    "spread": rate[6],
                    "real_volume": rate[7]
                }
                result.append(rate_dict)

            return result

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error copying rates for {symbol}/{timeframe.name}: {str(e)}",
                exception_type=type(e).__name__,
                function="copy_rates_from_pos",
                traceback=str(e),
                context={"symbol": symbol, "timeframe": timeframe.name, "start_pos": start_pos, "count": count}
            )
            return None

    def copy_rates_range(self, symbol: str, timeframe: TimeFrame, from_date: datetime, to_date: datetime) -> Optional[
        List[Dict[str, Any]]]:
        """Copy rates within specific date range"""
        if not self.ensure_connection():
            return None

        try:
            mt5_timeframe = timeframe.mt5_timeframe

            # Ensure from_date and to_date are timezone-aware before sending to MT5
            if from_date.tzinfo is None:
                from_date = from_date.replace(tzinfo=timezone.utc)

            if to_date.tzinfo is None:
                to_date = to_date.replace(tzinfo=timezone.utc)

            # Copy rates
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_date, to_date)

            if rates is None:
                error = mt5.last_error()
                self._logger.log_error(
                    level="ERROR",
                    message=f"Failed to copy rates for {symbol}/{timeframe.name} from {from_date} to {to_date}: {error}",
                    exception_type="MT5DataError",
                    function="copy_rates_range",
                    traceback="",
                    context={
                        "symbol": symbol,
                        "timeframe": timeframe.name,
                        "from_date": from_date,
                        "to_date": to_date,
                        "error_code": error[0],
                        "error_message": error[1]
                    }
                )
                return None

            # Convert to list of dictionaries
            result = []
            for rate in rates:
                rate_dict = {
                    "time": datetime.fromtimestamp(rate[0], tz=timezone.utc).replace(tzinfo=None),
                    "open": rate[1],
                    "high": rate[2],
                    "low": rate[3],
                    "close": rate[4],
                    "tick_volume": rate[5],
                    "spread": rate[6],
                    "real_volume": rate[7]
                }
                result.append(rate_dict)

            return result
        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error copying rates for {symbol}/{timeframe.name} from {from_date} to {to_date}: {str(e)}",
                exception_type=type(e).__name__,
                function="copy_rates_range",
                traceback=str(e),
                context={"symbol": symbol, "timeframe": timeframe.name, "from_date": from_date, "to_date": to_date}
            )
            return None

    def get_symbols(self) -> Optional[List[str]]:
        """Get list of available symbols"""
        if not self.ensure_connection():
            return None

        try:
            symbols = mt5.symbols_get()

            if symbols is None:
                error = mt5.last_error()
                self._logger.log_error(
                    level="ERROR",
                    message=f"Failed to get symbols: {error}",
                    exception_type="MT5DataError",
                    function="get_symbols",
                    traceback="",
                    context={"error_code": error[0], "error_message": error[1]}
                )
                return None

            # Extract symbol names
            symbol_names = [symbol.name for symbol in symbols]
            return symbol_names

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error getting symbols: {str(e)}",
                exception_type=type(e).__name__,
                function="get_symbols",
                traceback=str(e),
                context={}
            )
            return None

    def check_symbol_available(self, symbol: str) -> bool:
        """Check if symbol is available"""
        if not self.ensure_connection():
            return False

        try:
            symbol_info = mt5.symbol_info(symbol)
            return symbol_info is not None

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error checking symbol availability for {symbol}: {str(e)}",
                exception_type=type(e).__name__,
                function="check_symbol_available",
                traceback=str(e),
                context={"symbol": symbol}
            )
            return False

    def get_server_time(self) -> Optional[datetime]:
        """Get current MT5 server time with timezone information preserved"""
        if not self.ensure_connection():
            return None

        try:
            # Try to use configured symbols first, as they are guaranteed to be available
            for instrument in self._config.instruments:
                symbol = instrument.symbol
                last_tick = mt5.symbol_info_tick(symbol)

                if last_tick:
                    # Get time from the tick (server time)
                    # Keep timezone info for proper comparisons
                    server_time = datetime.fromtimestamp(last_tick.time, tz=timezone.utc)
                    return server_time

            # If all configured symbols fail, try to get any available symbol
            symbols = mt5.symbols_get()
            if not symbols:
                self._logger.log_error(
                    level="ERROR",
                    message="No symbols available to get server time",
                    exception_type="MT5DataError",
                    function="get_server_time",
                    traceback="",
                    context={}
                )
                return None

            # Try each symbol until we get a tick
            for symbol_info in symbols:
                symbol = symbol_info.name
                last_tick = mt5.symbol_info_tick(symbol)

                if last_tick:
                    server_time = datetime.fromtimestamp(last_tick.time, tz=timezone.utc)
                    return server_time

            self._logger.log_error(
                level="ERROR",
                message="Failed to get tick from any symbol",
                exception_type="MT5DataError",
                function="get_server_time",
                traceback="",
                context={}
            )
            return None

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error getting MT5 server time: {str(e)}",
                exception_type=type(e).__name__,
                function="get_server_time",
                traceback=str(e),
                context={}
            )
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        """Get symbol information from MT5"""
        if not self.ensure_connection():
            return None

        try:
            symbol_info = mt5.symbol_info(symbol)
            return symbol_info
        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error getting symbol info for {symbol}: {str(e)}",
                exception_type=type(e).__name__,
                function="get_symbol_info",
                traceback=str(e),
                context={"symbol": symbol}
            )
            return None

    def place_order(self, symbol: str, order_type: int, volume: float, price: float,
                    sl: float = 0, tp: float = 0, comment: str = "") -> Dict[str, Any]:
        """
        Place a trading order.

        Args:
            symbol: The trading symbol
            order_type: Order type (0=BUY, 1=SELL)
            volume: Order volume in lots
            price: Order price (0 for market orders)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment

        Returns:
            Dict containing order result
        """
        if not self.ensure_connection():
            return {'result': False, 'error': 'Not connected to MT5'}

        try:
            # Import inside method to avoid circular imports
            import MetaTrader5 as mt5

            # Validate inputs
            if volume <= 0:
                return {'result': False, 'error': 'Invalid volume'}

            # Set up order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": mt5.ORDER_TYPE_BUY if order_type == 0 else mt5.ORDER_TYPE_SELL,
                "price": 0.0,  # Market price
                "sl": float(sl) if sl else 0.0,
                "tp": float(tp) if tp else 0.0,
                "deviation": 10,  # Allowed price deviation
                "magic": 123456,  # Expert Advisor ID
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK
            }

            # Send order to MT5
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                return {
                    'result': False,
                    'error': f"Error code: {error[0]}, Error description: {error[1]}"
                }

            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Success
                return {
                    'result': True,
                    'order': result.order,
                    'price': result.price,
                    'volume': result.volume
                }
            else:
                # Failed
                return {
                    'result': False,
                    'error': f"Order failed, retcode={result.retcode}, comment={result.comment}"
                }

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error placing order: {str(e)}",
                exception_type=type(e).__name__,
                function="place_order",
                traceback=str(e),
                context={"symbol": symbol, "type": order_type, "volume": volume}
            )
            return {'result': False, 'error': str(e)}

    def close_position(self, ticket: int) -> Dict[str, Any]:
        """
        Close an open position.

        Args:
            ticket: Position ticket number

        Returns:
            Dict containing close result
        """
        if not self.ensure_connection():
            return {'result': False, 'error': 'Not connected to MT5'}

        try:
            # Import inside method to avoid circular imports
            import MetaTrader5 as mt5

            # Get position info
            position = mt5.positions_get(ticket=ticket)

            if not position:
                return {'result': False, 'error': 'Position not found'}

            position = position[0]

            # Set up close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": 0.0,  # Market price
                "deviation": 10,
                "magic": 123456,
                "comment": f"Close #{ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK
            }

            # Send order to MT5
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                return {
                    'result': False,
                    'error': f"Error code: {error[0]}, Error description: {error[1]}"
                }

            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Success
                return {
                    'result': True,
                    'order': result.order,
                    'close_price': result.price,
                    'close_volume': result.volume,
                    'profit': position.profit
                }
            else:
                # Failed
                return {
                    'result': False,
                    'error': f"Close failed, retcode={result.retcode}, comment={result.comment}"
                }

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error closing position: {str(e)}",
                exception_type=type(e).__name__,
                function="close_position",
                traceback=str(e),
                context={"ticket": ticket}
            )
            return {'result': False, 'error': str(e)}

    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> Dict[str, Any]:
        """
        Modify an open position.

        Args:
            ticket: Position ticket number
            sl: New stop loss price (None to keep existing)
            tp: New take profit price (None to keep existing)

        Returns:
            Dict containing modification result
        """
        if not self.ensure_connection():
            return {'result': False, 'error': 'Not connected to MT5'}

        try:
            # Import inside method to avoid circular imports
            import MetaTrader5 as mt5

            # Get position info
            position = mt5.positions_get(ticket=ticket)

            if not position:
                return {'result': False, 'error': 'Position not found'}

            position = position[0]

            # Use existing values if not specified
            if sl is None:
                sl = position.sl

            if tp is None:
                tp = position.tp

            # Set up modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": float(sl) if sl else 0.0,
                "tp": float(tp) if tp else 0.0
            }

            # Send request to MT5
            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                return {
                    'result': False,
                    'error': f"Error code: {error[0]}, Error description: {error[1]}"
                }

            # Check result
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Success
                return {
                    'result': True,
                    'new_sl': sl,
                    'new_tp': tp
                }
            else:
                # Failed
                return {
                    'result': False,
                    'error': f"Modification failed, retcode={result.retcode}, comment={result.comment}"
                }

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error modifying position: {str(e)}",
                exception_type=type(e).__name__,
                function="modify_position",
                traceback=str(e),
                context={"ticket": ticket, "sl": sl, "tp": tp}
            )
            return {'result': False, 'error': str(e)}

    def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get open positions.

        Args:
            symbol: Filter by symbol (None for all positions)

        Returns:
            List of position dictionaries
        """
        if not self.ensure_connection():
            return []

        try:
            # Import inside method to avoid circular imports
            import MetaTrader5 as mt5

            # Get positions
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None or len(positions) == 0:
                return []

            # Convert to dictionaries
            result = []
            for position in positions:
                position_dict = {
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': position.type,  # 0=BUY, 1=SELL
                    'volume': position.volume,
                    'price_open': position.price_open,
                    'price_current': position.price_current,
                    'sl': position.sl,
                    'tp': position.tp,
                    'profit': position.profit,
                    'comment': position.comment,
                    'time': position.time
                }
                result.append(position_dict)

            return result

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error getting positions: {str(e)}",
                exception_type=type(e).__name__,
                function="get_positions",
                traceback=str(e),
                context={"symbol": symbol}
            )
            return []

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dict containing account info
        """
        if not self.ensure_connection():
            return {}

        try:
            # Import inside method to avoid circular imports
            import MetaTrader5 as mt5

            # Get account info
            account_info = mt5.account_info()

            if account_info is None:
                return {}

            # Convert to dictionary
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'currency': account_info.currency,
                'leverage': account_info.leverage
            }

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error getting account info: {str(e)}",
                exception_type=type(e).__name__,
                function="get_account_info",
                traceback=str(e),
                context={}
            )
            return {}
