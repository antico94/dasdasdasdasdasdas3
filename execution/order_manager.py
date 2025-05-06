# execution/order_manager.py
from datetime import datetime, timezone
import threading
from typing import Dict, Any, List

from MT5.mt5_manager import MT5Manager
from Logger.logger import DBLogger
from Events.events import SignalEvent, OrderEvent
from Events.event_bus import EventBus
from Config.trading_config import ConfigManager


class OrderManager:
    """
    Manages order execution with risk management, partial profit-taking,
    and advanced position management.
    """

    def __init__(self, mt5_manager=None, logger: DBLogger = None, event_bus=None):
        """
        Initialize the order manager.

        Args:
            mt5_manager: MT5 manager for order execution
            logger: Logger instance
            event_bus: Event bus for publishing order events
        """
        # Get connection string from credentials
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

        self.mt5_manager = mt5_manager or MT5Manager()
        self.logger = logger or DBLogger(
            conn_string=conn_string,
            enabled_levels={'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            console_output=True
        )
        self.event_bus = event_bus or EventBus()
        self.config = ConfigManager().config

        # Risk management parameters
        self.base_risk_percent = 1.0  # Base risk per trade (%)
        self.max_daily_risk_percent = 3.0  # Maximum daily risk (%)
        self.max_positions_per_symbol = 2  # Maximum positions per symbol

        # Track open positions
        self.open_positions = {}  # {ticket: position_info}

        # Track recent trade performance
        self.recent_trades = []  # Last 10 trades results
        self.max_recent_trades = 10

        # Thread lock for order operations
        self.lock = threading.RLock()

        # Subscribe to events
        self.event_bus.subscribe(SignalEvent, self._on_signal)

        self.logger.log_event(
            level="INFO",
            message="Order Manager initialized",
            event_type="ORDER_MANAGER",
            component="order_manager",
            action="initialize",
            status="success",
            details={"base_risk": f"{self.base_risk_percent}%", "max_daily_risk": f"{self.max_daily_risk_percent}%"}
        )

    def _on_signal(self, event: SignalEvent) -> None:
        """
        Handle incoming signal events.

        Args:
            event: The signal event
        """
        try:
            # Log the signal receipt
            self.logger.log_event(
                level="INFO",
                message=f"Received {event.direction} signal for {event.symbol} from {event.strategy_name}",
                event_type="SIGNAL_RECEIVED",
                component="order_manager",
                action="_on_signal",
                status="received",
                details={
                    "symbol": event.symbol,
                    "direction": event.direction,
                    "strategy": event.strategy_name,
                    "reason": event.reason,
                    "timestamp": str(event.timestamp)
                }
            )

            # Process the signal
            if event.direction in ["BUY", "SELL"]:
                self._process_entry_signal(event)
            elif event.direction == "CLOSE":
                self._process_close_signal(event)
            else:
                self.logger.log_event(
                    level="WARNING",
                    message=f"Unknown signal direction: {event.direction}",
                    event_type="SIGNAL_ERROR",
                    component="order_manager",
                    action="_on_signal",
                    status="invalid_direction",
                    details={"direction": event.direction}
                )

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error processing signal: {str(e)}",
                exception_type=type(e).__name__,
                function="_on_signal",
                traceback=str(e),
                context={"signal_type": event.direction, "symbol": event.symbol}
            )

    # execution/order_manager.py - _process_entry_signal method
    # execution/order_manager.py - _process_entry_signal method
    def _process_entry_signal(self, signal: SignalEvent) -> None:
        """
        Process an entry (BUY/SELL) signal with enhanced error logging.

        Args:
            signal: The entry signal
        """
        with self.lock:
            try:
                # Log start of signal processing
                self.logger.log_event(
                    level="INFO",
                    message=f"Processing {signal.direction} signal for {signal.symbol}",
                    event_type="ORDER_PROCESSING",
                    component="order_manager",
                    action="_process_entry_signal",
                    status="starting",
                    details={
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "strategy": signal.strategy_name,
                        "timeframe": signal.timeframe_name
                    }
                )

                # Validate signal
                if not self._validate_entry_signal(signal):
                    self.logger.log_event(
                        level="WARNING",
                        message=f"Signal validation failed for {signal.symbol} {signal.direction}",
                        event_type="ORDER_VALIDATION",
                        component="order_manager",
                        action="_process_entry_signal",
                        status="validation_failed",
                        details={
                            "symbol": signal.symbol,
                            "direction": signal.direction,
                            "reason": "Signal validation failed"
                        }
                    )
                    return

                # Check market conditions
                if not self._check_market_conditions(signal.symbol):
                    self.logger.log_event(
                        level="WARNING",
                        message=f"Market conditions not suitable for {signal.symbol} {signal.direction}",
                        event_type="ORDER_MARKET_CHECK",
                        component="order_manager",
                        action="_process_entry_signal",
                        status="market_check_failed",
                        details={
                            "symbol": signal.symbol,
                            "direction": signal.direction
                        }
                    )
                    return

                # Check risk limits
                if not self._check_risk_limits(signal.symbol):
                    self.logger.log_event(
                        level="WARNING",
                        message=f"Risk limits would be exceeded for {signal.symbol} {signal.direction}",
                        event_type="ORDER_RISK_CHECK",
                        component="order_manager",
                        action="_process_entry_signal",
                        status="risk_check_failed",
                        details={
                            "symbol": signal.symbol,
                            "direction": signal.direction
                        }
                    )
                    return

                # Calculate dynamic risk percentage
                risk_percent = self._calculate_dynamic_risk(signal)

                # Calculate position size based on risk
                position_size, stop_loss = self._calculate_position_size(signal, risk_percent)

                if position_size <= 0:
                    self.logger.log_event(
                        level="WARNING",
                        message=f"Invalid position size calculated: {position_size} for {signal.symbol}",
                        event_type="ORDER_CALCULATION",
                        component="order_manager",
                        action="_process_entry_signal",
                        status="invalid_size",
                        details={
                            "symbol": signal.symbol,
                            "risk_percent": risk_percent,
                            "position_size": position_size
                        }
                    )
                    return

                # Calculate take profit levels
                take_profit_levels = self._calculate_take_profits(signal, stop_loss)

                # Execute order strategy with enhanced error reporting
                order_result = self._execute_partial_profit_strategy(signal, position_size, stop_loss,
                                                                     take_profit_levels)

                if not order_result:
                    self.logger.log_event(
                        level="ERROR",
                        message=f"Failed to execute order for {signal.symbol} {signal.direction}",
                        event_type="ORDER_EXECUTION",
                        component="order_manager",
                        action="_process_entry_signal",
                        status="execution_failed",
                        details={
                            "symbol": signal.symbol,
                            "direction": signal.direction,
                            "position_size": position_size
                        }
                    )
                else:
                    self.logger.log_event(
                        level="INFO",
                        message=f"Successfully executed order for {signal.symbol} {signal.direction}",
                        event_type="ORDER_EXECUTION",
                        component="order_manager",
                        action="_process_entry_signal",
                        status="success",
                        details={
                            "symbol": signal.symbol,
                            "direction": signal.direction,
                            "position_size": position_size,
                            "order_result": order_result
                        }
                    )

            except Exception as e:
                self.logger.log_error(
                    level="ERROR",
                    message=f"Error processing entry signal: {str(e)}",
                    exception_type=type(e).__name__,
                    function="_process_entry_signal",
                    traceback=str(e),
                    context={
                        "signal_type": signal.direction,
                        "symbol": signal.symbol,
                        "strategy": signal.strategy_name
                    }
                )

    def _validate_entry_signal(self, signal: SignalEvent) -> bool:
        """
        Validate an entry signal.

        Args:
            signal: The signal to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Check direction
        if signal.direction not in ["BUY", "SELL"]:
            self.logger.log_event(
                level="WARNING",
                message=f"Invalid signal direction: {signal.direction}",
                event_type="SIGNAL_VALIDATION",
                component="order_manager",
                action="_validate_entry_signal",
                status="invalid_direction",
                details={"direction": signal.direction}
            )
            return False

        # Check symbol
        if not signal.symbol:
            self.logger.log_event(
                level="WARNING",
                message="Missing symbol in signal",
                event_type="SIGNAL_VALIDATION",
                component="order_manager",
                action="_validate_entry_signal",
                status="missing_symbol"
            )
            return False

        # Check entry price
        if signal.entry_price is not None and signal.entry_price <= 0:
            self.logger.log_event(
                level="WARNING",
                message=f"Invalid entry price: {signal.entry_price}",
                event_type="SIGNAL_VALIDATION",
                component="order_manager",
                action="_validate_entry_signal",
                status="invalid_price",
                details={"entry_price": signal.entry_price}
            )
            return False

        return True

    def _check_market_conditions(self, symbol: str) -> bool:
        """
        Check if market conditions allow trading.

        Args:
            symbol: The symbol to check

        Returns:
            bool: True if conditions are good, False otherwise
        """
        # Check if market is open
        if not self._is_market_open(symbol):
            self.logger.log_event(
                level="WARNING",
                message=f"Market is closed for {symbol}",
                event_type="MARKET_CHECK",
                component="order_manager",
                action="_check_market_conditions",
                status="market_closed",
                details={"symbol": symbol}
            )
            return False

        # Check spread (can be high during major news or low liquidity)
        spread = self._get_current_spread(symbol)
        max_spread_points = 20  # Maximum acceptable spread in points

        if spread > max_spread_points:
            self.logger.log_event(
                level="WARNING",
                message=f"Spread too high for {symbol}: {spread} points",
                event_type="MARKET_CHECK",
                component="order_manager",
                action="_check_market_conditions",
                status="high_spread",
                details={"symbol": symbol, "spread": spread, "max_spread": max_spread_points}
            )
            return False

        return True

    def _is_market_open(self, symbol: str) -> bool:
        """
        Check if market is open for a symbol.

        Args:
            symbol: The symbol to check

        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            # Get symbol info from MT5
            symbol_info = self.mt5_manager.get_symbol_info(symbol)

            if symbol_info:
                # Check trade mode (0 = disabled)
                if hasattr(symbol_info, 'trade_mode') and symbol_info.trade_mode == 0:
                    return False

            # Get current time
            current_time = datetime.now(timezone.utc).replace(tzinfo=None)

            # Check weekend (Saturday = 5, Sunday = 6)
            weekday = current_time.weekday()
            if weekday == 5:  # Saturday
                return False

            if weekday == 6:  # Sunday
                # Sunday - market opens around 5 PM EST
                if current_time.hour < 17:
                    return False

            # Check Friday close (Friday = 4)
            if weekday == 4 and current_time.hour >= 17:
                return False

            return True

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error checking if market is open: {str(e)}",
                exception_type=type(e).__name__,
                function="_is_market_open",
                traceback=str(e),
                context={"symbol": symbol}
            )
            return False

    def _get_current_spread(self, symbol: str) -> float:
        """
        Get current spread for a symbol.

        Args:
            symbol: The symbol to check

        Returns:
            float: Current spread in points
        """
        try:
            # Get symbol info from MT5
            symbol_info = self.mt5_manager.get_symbol_info(symbol)

            if symbol_info and hasattr(symbol_info, 'spread'):
                return symbol_info.spread

            return 0

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error getting spread: {str(e)}",
                exception_type=type(e).__name__,
                function="_get_current_spread",
                traceback=str(e),
                context={"symbol": symbol}
            )
            return 0

    def _check_risk_limits(self, symbol: str) -> bool:
        """
        Check if adding a position would exceed risk limits.

        Args:
            symbol: The symbol to check

        Returns:
            bool: True if within limits, False otherwise
        """
        try:
            # Check max positions per symbol
            symbol_positions = self._get_positions_for_symbol(symbol)

            if len(symbol_positions) >= self.max_positions_per_symbol:
                self.logger.log_event(
                    level="WARNING",
                    message=f"Maximum positions reached for {symbol}: {len(symbol_positions)}/{self.max_positions_per_symbol}",
                    event_type="RISK_CHECK",
                    component="order_manager",
                    action="_check_risk_limits",
                    status="max_positions_reached",
                    details={"symbol": symbol, "current": len(symbol_positions), "max": self.max_positions_per_symbol}
                )
                return False

            # Check daily risk limit
            current_risk = self._calculate_current_risk()

            if current_risk >= self.max_daily_risk_percent:
                self.logger.log_event(
                    level="WARNING",
                    message=f"Maximum daily risk reached: {current_risk:.2f}%/{self.max_daily_risk_percent}%",
                    event_type="RISK_CHECK",
                    component="order_manager",
                    action="_check_risk_limits",
                    status="max_risk_reached",
                    details={"current_risk": current_risk, "max_risk": self.max_daily_risk_percent}
                )
                return False

            return True

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error checking risk limits: {str(e)}",
                exception_type=type(e).__name__,
                function="_check_risk_limits",
                traceback=str(e),
                context={"symbol": symbol}
            )
            return False

    def _get_positions_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all open positions for a symbol.

        Args:
            symbol: The symbol to get positions for

        Returns:
            List of position dictionaries
        """
        try:
            # Get all positions from MT5
            all_positions = self.mt5_manager.get_positions()

            # Filter by symbol
            symbol_positions = [p for p in all_positions if p.get('symbol') == symbol]

            return symbol_positions

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error getting positions for symbol: {str(e)}",
                exception_type=type(e).__name__,
                function="_get_positions_for_symbol",
                traceback=str(e),
                context={"symbol": symbol}
            )
            return []

    def _calculate_current_risk(self) -> float:
        """
        Calculate current risk exposure across all positions.

        Returns:
            float: Current risk as percentage of account
        """
        try:
            # Get account info from MT5
            account_info = self.mt5_manager.get_account_info()

            if not account_info or 'balance' not in account_info or account_info['balance'] <= 0:
                return 0

            balance = account_info['balance']

            # Get all positions
            positions = self.mt5_manager.get_positions()

            # Calculate total risk amount
            total_risk = 0

            for position in positions:
                entry = position.get('price_open', 0)
                stop = position.get('sl', 0)  # Stop loss price
                volume = position.get('volume', 0)

                # Skip positions without stops
                if stop == 0:
                    continue

                # Calculate pip value (would be more complex in a real system)
                pip_value = self._calculate_pip_value(position.get('symbol', ''), volume)

                # Calculate risk amount
                risk_amount = abs(entry - stop) * pip_value

                # Add to total
                total_risk += risk_amount

            # Convert to percentage
            risk_percent = (total_risk / balance) * 100

            return risk_percent

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error calculating current risk: {str(e)}",
                exception_type=type(e).__name__,
                function="_calculate_current_risk",
                traceback=str(e),
                context={}
            )
            return 0

    def _calculate_pip_value(self, symbol: str, volume: float) -> float:
        """
        Calculate pip value for a symbol and volume.

        Args:
            symbol: The trading symbol
            volume: Position volume in lots

        Returns:
            float: Value of 1 pip in account currency
        """
        # This is a simplified version - in a real system this would be more complex
        if symbol == "XAUUSD":
            return 0.1 * volume * 10  # Gold
        elif symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            return 10 * volume  # Major forex pairs
        else:
            return 10 * volume  # Default for forex

    def _calculate_dynamic_risk(self, signal: SignalEvent) -> float:
        """
        Calculate dynamic risk percentage based on market conditions and recent performance.

        Args:
            signal: The signal to calculate risk for

        Returns:
            float: Risk percentage to use
        """
        try:
            # Start with base risk
            risk_percent = self.base_risk_percent

            # 1. Adjust by current session
            current_hour = datetime.now(timezone.utc).hour

            # London/NY overlap (higher liquidity)
            if 13 <= current_hour < 17:
                risk_percent *= 1.2  # Increase risk by 20%
                self.logger.log_event(
                    level="DEBUG",
                    message="London/NY overlap: Increasing risk by 20%",
                    event_type="RISK_CALCULATION",
                    component="order_manager",
                    action="_calculate_dynamic_risk",
                    status="session_adjustment"
                )
            # Asian session (lower liquidity)
            elif 0 <= current_hour < 6:
                risk_percent *= 0.8  # Decrease risk by 20%
                self.logger.log_event(
                    level="DEBUG",
                    message="Asian session: Decreasing risk by 20%",
                    event_type="RISK_CALCULATION",
                    component="order_manager",
                    action="_calculate_dynamic_risk",
                    status="session_adjustment"
                )

            # 2. Adjust by signal strength
            signal_strength = getattr(signal, 'strength', 0.5)

            # Scale by strength (0.8 to 1.2)
            strength_factor = 0.8 + (signal_strength * 0.4)
            risk_percent *= strength_factor

            # 3. Adjust by recent performance
            if self.recent_trades:
                # Calculate win rate
                wins = sum(1 for result in self.recent_trades if result)
                win_rate = wins / len(self.recent_trades)

                # If winning > 70%, increase risk
                if win_rate > 0.7:
                    risk_percent *= 1.1
                    self.logger.log_event(
                        level="DEBUG",
                        message=f"High win rate ({win_rate:.2f}): Increasing risk by 10%",
                        event_type="RISK_CALCULATION",
                        component="order_manager",
                        action="_calculate_dynamic_risk",
                        status="performance_adjustment"
                    )
                # If winning < 30%, decrease risk
                elif win_rate < 0.3:
                    risk_percent *= 0.9
                    self.logger.log_event(
                        level="DEBUG",
                        message=f"Low win rate ({win_rate:.2f}): Decreasing risk by 10%",
                        event_type="RISK_CALCULATION",
                        component="order_manager",
                        action="_calculate_dynamic_risk",
                        status="performance_adjustment"
                    )

            # 4. Check available risk
            current_risk = self._calculate_current_risk()
            available_risk = self.max_daily_risk_percent - current_risk

            if available_risk < risk_percent:
                old_risk = risk_percent
                risk_percent = max(0.1, available_risk)  # Minimum 0.1%

                self.logger.log_event(
                    level="INFO",
                    message=f"Reducing risk from {old_risk:.2f}% to {risk_percent:.2f}% due to max exposure limit",
                    event_type="RISK_CALCULATION",
                    component="order_manager",
                    action="_calculate_dynamic_risk",
                    status="risk_limit_adjustment",
                    details={"original_risk": old_risk, "adjusted_risk": risk_percent,
                             "max_risk": self.max_daily_risk_percent}
                )

            # Ensure within bounds
            risk_percent = max(0.1, min(risk_percent, 2.0))

            return risk_percent

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error calculating dynamic risk: {str(e)}",
                exception_type=type(e).__name__,
                function="_calculate_dynamic_risk",
                traceback=str(e),
                context={"signal_type": signal.direction, "symbol": signal.symbol}
            )
            return 0.5  # Default to half of base risk

    def _calculate_position_size(self, signal: SignalEvent, risk_percent: float) -> tuple:
        """
        Calculate position size based on risk percentage.

        Args:
            signal: The trading signal
            risk_percent: Risk percentage to use

        Returns:
            tuple: (position_size, stop_loss_price)
        """
        try:
            # Get account info
            account_info = self.mt5_manager.get_account_info()

            if not account_info or 'balance' not in account_info or account_info['balance'] <= 0:
                return 0, 0

            balance = account_info['balance']

            # Get current price if entry price not specified
            current_price = signal.entry_price
            if current_price is None or current_price <= 0:
                # Get latest price from MT5
                symbol_info = self.mt5_manager.get_symbol_info(signal.symbol)
                if symbol_info and hasattr(symbol_info, 'bid') and hasattr(symbol_info, 'ask'):
                    if signal.direction == "BUY":
                        current_price = symbol_info.ask
                    else:
                        current_price = symbol_info.bid
                else:
                    return 0, 0

            # Calculate stop loss if not provided
            stop_loss = signal.stop_loss
            if stop_loss is None or stop_loss <= 0:
                # Use ATR if available in signal (assume it's in the same field that holds the entry price)
                atr_value = getattr(signal, 'atr', None)

                if atr_value:
                    # Use ATR-based stop
                    if signal.direction == "BUY":
                        stop_loss = current_price - (atr_value * 1.5)
                    else:
                        stop_loss = current_price + (atr_value * 1.5)
                else:
                    # Default to 1% stop
                    if signal.direction == "BUY":
                        stop_loss = current_price * 0.99
                    else:
                        stop_loss = current_price * 1.01

            # Calculate risk amount
            risk_amount = balance * (risk_percent / 100)

            # Calculate pip value (would be more complex in a real system)
            # Assume standard lot = 100,000 units
            lot_size = 100000

            # For XAUUSD (Gold), 1 pip is 0.01, for most forex pairs 1 pip is 0.0001
            pip_size = 0.01 if signal.symbol == "XAUUSD" else 0.0001

            # Calculate position size in lots
            if signal.direction == "BUY":
                pips_at_risk = abs(current_price - stop_loss) / pip_size
            else:
                pips_at_risk = abs(stop_loss - current_price) / pip_size

            # Avoid division by zero
            if pips_at_risk <= 0:
                return 0, stop_loss

            # Calculate position size in lots
            position_size = risk_amount / (pips_at_risk * pip_size * lot_size)

            # Get symbol info for min/max lot size
            symbol_info = self.mt5_manager.get_symbol_info(signal.symbol)
            if symbol_info:
                min_lot = getattr(symbol_info, 'volume_min', 0.01)
                max_lot = getattr(symbol_info, 'volume_max', 100.0)
                lot_step = getattr(symbol_info, 'volume_step', 0.01)

                # Round to lot step
                position_size = round(position_size / lot_step) * lot_step

                # Ensure within bounds
                position_size = max(min_lot, min(position_size, max_lot))

            self.logger.log_event(
                level="INFO",
                message=f"Calculated position size: {position_size} lots with stop at {stop_loss}",
                event_type="POSITION_CALCULATION",
                component="order_manager",
                action="_calculate_position_size",
                status="success",
                details={
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "risk_percent": risk_percent,
                    "position_size": position_size,
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "risk_amount": risk_amount
                }
            )

            return position_size, stop_loss

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error calculating position size: {str(e)}",
                exception_type=type(e).__name__,
                function="_calculate_position_size",
                traceback=str(e),
                context={"signal_type": signal.direction, "symbol": signal.symbol}
            )
            return 0, 0

    def _calculate_take_profits(self, signal: SignalEvent, stop_loss: float) -> List[float]:
        """
        Calculate take-profit levels based on risk-reward ratios.

        Args:
            signal: The trading signal
            stop_loss: Stop loss price

        Returns:
            List of take-profit levels
        """
        try:
            # Use take profit if provided in signal
            if signal.take_profit and signal.take_profit > 0:
                return [signal.take_profit]

            # Get current price
            entry_price = signal.entry_price
            if entry_price is None or entry_price <= 0:
                # Get latest price from MT5
                symbol_info = self.mt5_manager.get_symbol_info(signal.symbol)
                if symbol_info and hasattr(symbol_info, 'bid') and hasattr(symbol_info, 'ask'):
                    if signal.direction == "BUY":
                        entry_price = symbol_info.ask
                    else:
                        entry_price = symbol_info.bid
                else:
                    return []

            # Calculate risk (distance to stop)
            risk = abs(entry_price - stop_loss)

            # Default risk-reward ratios
            r1 = 1.0  # First target at 1:1
            r2 = 2.0  # Second target at 2:1

            # Adjust based on strategy if needed
            strategy_name = signal.strategy_name.lower() if signal.strategy_name else ""

            if "breakout" in strategy_name:
                r1 = 1.0
                r2 = 2.0
            elif "momentum" in strategy_name:
                r1 = 1.0
                r2 = 2.0
            elif "trend" in strategy_name:
                r1 = 1.5
                r2 = 3.0
            elif "reversion" in strategy_name or "range" in strategy_name:
                r1 = 1.0
                r2 = 1.5

            # Calculate take profit levels
            if signal.direction == "BUY":
                tp1 = entry_price + (risk * r1)
                tp2 = entry_price + (risk * r2)
            else:
                tp1 = entry_price - (risk * r1)
                tp2 = entry_price - (risk * r2)

            self.logger.log_event(
                level="INFO",
                message=f"Calculated take-profit levels: {[tp1, tp2]}",
                event_type="TP_CALCULATION",
                component="order_manager",
                action="_calculate_take_profits",
                status="success",
                details={
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "strategy": strategy_name,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "risk": risk,
                    "risk_reward_ratios": [r1, r2],
                    "take_profits": [tp1, tp2]
                }
            )

            return [tp1, tp2]

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error calculating take profits: {str(e)}",
                exception_type=type(e).__name__,
                function="_calculate_take_profits",
                traceback=str(e),
                context={"signal_type": signal.direction, "symbol": signal.symbol}
            )
            return []

    def _execute_partial_profit_strategy(self, signal: SignalEvent, position_size: float, stop_loss: float,
                                         take_profit_levels: List[float]) -> Dict:
        """
        Execute a two-part position strategy for better risk management.
        Returns order execution result dictionary.

        Args:
            signal: The trading signal
            position_size: Total position size
            stop_loss: Stop loss price
            take_profit_levels: List of take-profit levels

        Returns:
            Dict containing order execution results
        """
        try:
            # Get minimum lot size
            symbol_info = self.mt5_manager.get_symbol_info(signal.symbol)
            if not symbol_info:
                return {"success": False, "error": "Symbol info not available", "orders": []}

            min_lot = getattr(symbol_info, 'volume_min', 0.01)
            lot_step = getattr(symbol_info, 'volume_step', 0.01)

            # Save order results for return
            order_results = []

            # If position size is too small to split or only one take profit level
            if position_size < 2 * min_lot or len(take_profit_levels) < 2:
                # Use single position with first take profit
                take_profit = take_profit_levels[0] if take_profit_levels else 0

                self.logger.log_event(
                    level="INFO",
                    message=f"Position size too small to split or only one TP level, using single position",
                    event_type="ORDER_EXECUTION",
                    component="order_manager",
                    action="_execute_partial_profit_strategy",
                    status="single_position",
                    details={
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "position_size": position_size,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    }
                )

                # Place single order
                result = self._place_order(signal, position_size, stop_loss, take_profit)
                order_results.append(result)

                return {
                    "success": result.get('result', False) if result else False,
                    "orders": order_results
                }

            # Split position into two parts
            position_size_1 = position_size * 0.5
            position_size_2 = position_size - position_size_1

            # Round to lot step
            position_size_1 = round(position_size_1 / lot_step) * lot_step
            position_size_2 = round(position_size_2 / lot_step) * lot_step

            # Ensure minimum lot sizes
            if position_size_1 < min_lot:
                position_size_1 = min_lot
                position_size_2 = max(min_lot, position_size - min_lot)

            if position_size_2 < min_lot:
                position_size_2 = 0
                position_size_1 = position_size

            # Execute first part with first target
            if position_size_1 >= min_lot:
                self.logger.log_event(
                    level="INFO",
                    message=f"Placing first order (Part 1): {position_size_1} lots with TP at {take_profit_levels[0]}",
                    event_type="ORDER_EXECUTION",
                    component="order_manager",
                    action="_execute_partial_profit_strategy",
                    status="placing_part1",
                    details={
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "position_size": position_size_1,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit_levels[0]
                    }
                )

                # Place first order
                comment = f"{signal.strategy_name}_Part1"
                result1 = self._place_order(signal, position_size_1, stop_loss, take_profit_levels[0], comment)
                order_results.append(result1)

            # Execute second part with second target
            if position_size_2 >= min_lot and len(take_profit_levels) > 1:
                self.logger.log_event(
                    level="INFO",
                    message=f"Placing second order (Part 2): {position_size_2} lots with TP at {take_profit_levels[1]}",
                    event_type="ORDER_EXECUTION",
                    component="order_manager",
                    action="_execute_partial_profit_strategy",
                    status="placing_part2",
                    details={
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "position_size": position_size_2,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit_levels[1]
                    }
                )

                # Place second order
                comment = f"{signal.strategy_name}_Part2"
                result2 = self._place_order(signal, position_size_2, stop_loss, take_profit_levels[1], comment)
                order_results.append(result2)

            # Return complete results dictionary
            success = any(result.get('result', False) for result in order_results if result)
            return {
                "success": success,
                "orders": order_results
            }

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error executing partial profit strategy: {str(e)}",
                exception_type=type(e).__name__,
                function="_execute_partial_profit_strategy",
                traceback=str(e),
                context={
                    "signal_type": signal.direction,
                    "symbol": signal.symbol,
                    "position_size": position_size
                }
            )
            return {"success": False, "error": str(e), "orders": []}

    def _place_order(self, signal: SignalEvent, position_size: float, stop_loss: float, take_profit: float,
                     comment: str = "") -> Dict[str, Any]:
        """
        Place an order with MT5 with improved error handling.

        Args:
            signal: The trading signal
            position_size: Position size in lots
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Optional comment for the order

        Returns:
            Dict containing order result
        """
        try:
            # Validate parameters
            if position_size <= 0:
                error_msg = f"Invalid position size: {position_size}"
                self.logger.log_event(
                    level="ERROR",
                    message=error_msg,
                    event_type="ORDER_VALIDATION",
                    component="order_manager",
                    action="_place_order",
                    status="invalid_size",
                    details={"position_size": position_size}
                )
                return {'result': False, 'error': error_msg}

            # Connect to MT5 before placing order - this is critical
            if not self.mt5_manager.ensure_connection():
                error_msg = "MT5 connection failed"
                self.logger.log_event(
                    level="ERROR",
                    message=error_msg,
                    event_type="MT5_CONNECTION",
                    component="order_manager",
                    action="_place_order",
                    status="connection_failed"
                )
                return {'result': False, 'error': error_msg}

            # Convert signal direction to MT5 order type
            order_type = 0 if signal.direction == "BUY" else 1  # 0=BUY, 1=SELL

            # Add strategy name to comment if not already included
            if not comment and signal.strategy_name:
                comment = signal.strategy_name

            # Get current price if entry price not specified
            entry_price = signal.entry_price
            if entry_price is None or entry_price <= 0:
                # Get latest price from MT5
                symbol_info = self.mt5_manager.get_symbol_info(signal.symbol)
                if symbol_info and hasattr(symbol_info, 'bid') and hasattr(symbol_info, 'ask'):
                    if signal.direction == "BUY":
                        entry_price = symbol_info.ask
                    else:
                        entry_price = symbol_info.bid
                else:
                    error_msg = f"Could not get current price for {signal.symbol}"
                    self.logger.log_event(
                        level="ERROR",
                        message=error_msg,
                        event_type="MT5_DATA",
                        component="order_manager",
                        action="_place_order",
                        status="price_unavailable",
                        details={"symbol": signal.symbol}
                    )
                    return {'result': False, 'error': error_msg}

            # Log order request
            self.logger.log_event(
                level="INFO",
                message=f"Placing {signal.direction} order: {position_size} lots of {signal.symbol} at market",
                event_type="ORDER_REQUEST",
                component="order_manager",
                action="_place_order",
                status="requesting",
                details={
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "position_size": position_size,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "comment": comment
                }
            )

            # Place order via MT5
            result = self.mt5_manager.place_order(
                symbol=signal.symbol,
                order_type=order_type,
                volume=position_size,
                price=0.0,  # Market price
                sl=stop_loss,
                tp=take_profit,
                comment=comment
            )

            # Check if MT5 manager returned anything
            if result is None:
                error_msg = "MT5 place_order returned None"
                self.logger.log_event(
                    level="ERROR",
                    message=error_msg,
                    event_type="MT5_ORDER_ERROR",
                    component="order_manager",
                    action="_place_order",
                    status="null_result"
                )
                return {'result': False, 'error': error_msg}

            if result and 'order' in result and result['order']:
                # Get order ticket
                ticket = result['order']

                # Log successful order
                self.logger.log_event(
                    level="INFO",
                    message=f"Order placed successfully: Ticket #{ticket}",
                    event_type="ORDER_EXECUTION",
                    component="order_manager",
                    action="_place_order",
                    status="success",
                    details={
                        "ticket": ticket,
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "position_size": position_size,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit
                    }
                )

                # Create and publish order event
                order_event = OrderEvent(
                    instrument_id=signal.instrument_id,
                    symbol=signal.symbol,
                    order_type=signal.direction,
                    direction=signal.direction,
                    price=entry_price,
                    volume=position_size,
                    order_id=ticket,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy_name=signal.strategy_name
                )

                self.event_bus.publish(order_event)

                # Update open positions tracker
                self.open_positions[ticket] = {
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'volume': position_size,
                    'open_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'comment': comment,
                    'open_time': datetime.now(),
                    'strategy': signal.strategy_name,
                    'signal_id': getattr(signal, 'id', None)
                }

                return {
                    'result': True,
                    'order': ticket,
                    'price': entry_price,
                    'volume': position_size
                }
            else:
                # Log failed order with detailed error info
                error_info = result.get('error', 'Unknown error') if result else 'No result'

                self.logger.log_event(
                    level="ERROR",
                    message=f"Order placement failed: {error_info}",
                    event_type="ORDER_EXECUTION",
                    component="order_manager",
                    action="_place_order",
                    status="failed",
                    details={
                        "symbol": signal.symbol,
                        "direction": signal.direction,
                        "position_size": position_size,
                        "error": error_info,
                        "raw_result": str(result)
                    }
                )

                return {'result': False, 'error': error_info}

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error placing order: {str(e)}",
                exception_type=type(e).__name__,
                function="_place_order",
                traceback=str(e),
                context={
                    "signal_type": signal.direction,
                    "symbol": signal.symbol,
                    "position_size": position_size
                }
            )
            return {'result': False, 'error': str(e)}

    def _process_close_signal(self, signal: SignalEvent) -> None:
        """
        Process a close signal.

        Args:
            signal: The close signal
        """
        try:
            # Get positions for this symbol
            positions = self._get_positions_for_symbol(signal.symbol)

            if not positions:
                self.logger.log_event(
                    level="WARNING",
                    message=f"No open positions found for {signal.symbol} to close",
                    event_type="CLOSE_SIGNAL",
                    component="order_manager",
                    action="_process_close_signal",
                    status="no_positions",
                    details={"symbol": signal.symbol}
                )
                return

            # Log close signal
            self.logger.log_event(
                level="INFO",
                message=f"Processing close signal for {signal.symbol}",
                event_type="CLOSE_SIGNAL",
                component="order_manager",
                action="_process_close_signal",
                status="processing",
                details={
                    "symbol": signal.symbol,
                    "positions_count": len(positions),
                    "strategy": signal.strategy_name
                }
            )

            # Close each position
            for position in positions:
                # Skip positions from other strategies if strategy specified
                if signal.strategy_name and position.get('comment', '').find(signal.strategy_name) == -1:
                    continue

                # Get position details
                ticket = position.get('ticket')
                volume = position.get('volume')

                # Log close request
                self.logger.log_event(
                    level="INFO",
                    message=f"Closing position #{ticket}: {volume} lots of {signal.symbol}",
                    event_type="CLOSE_REQUEST",
                    component="order_manager",
                    action="_process_close_signal",
                    status="requesting",
                    details={
                        "ticket": ticket,
                        "symbol": signal.symbol,
                        "volume": volume
                    }
                )

                # Close position via MT5
                result = self.mt5_manager.close_position(ticket)

                if result and result.get('result'):
                    # Log successful close
                    self.logger.log_event(
                        level="INFO",
                        message=f"Position #{ticket} closed successfully",
                        event_type="CLOSE_EXECUTION",
                        component="order_manager",
                        action="_process_close_signal",
                        status="success",
                        details={
                            "ticket": ticket,
                            "symbol": signal.symbol,
                            "volume": volume,
                            "profit": result.get('profit', 0)
                        }
                    )

                    # Update recent trades for risk adjustment
                    if ticket in self.open_positions:
                        # Get profit/loss
                        profit = result.get('profit', 0)

                        # Update performance tracking (True for profit, False for loss)
                        self.recent_trades.append(profit > 0)

                        # Keep only the most recent trades
                        if len(self.recent_trades) > self.max_recent_trades:
                            self.recent_trades.pop(0)

                        # Remove from open positions
                        del self.open_positions[ticket]
                else:
                    # Log failed close
                    error_info = result.get('error', 'Unknown error') if result else 'No result'

                    self.logger.log_event(
                        level="ERROR",
                        message=f"Failed to close position #{ticket}: {error_info}",
                        event_type="CLOSE_EXECUTION",
                        component="order_manager",
                        action="_process_close_signal",
                        status="failed",
                        details={
                            "ticket": ticket,
                            "symbol": signal.symbol,
                            "error": error_info
                        }
                    )

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error processing close signal: {str(e)}",
                exception_type=type(e).__name__,
                function="_process_close_signal",
                traceback=str(e),
                context={"symbol": signal.symbol}
            )