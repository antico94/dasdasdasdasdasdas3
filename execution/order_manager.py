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

    def __init__(self, mt5_manager: MT5Manager, logger: DBLogger, event_bus: EventBus):
        """
        Initialize the order manager.

        Args:
            mt5_manager: MT5 manager instance
            logger: Logger instance
            event_bus: Event bus instance
        """
        self.mt5_manager = mt5_manager
        self.logger = logger
        self.event_bus = event_bus

        # Risk management settings
        self.base_risk_percent = 2.0  # Risk 2% of account per trade
        self.max_daily_risk_percent = 6.0  # Maximum 6% risk per day
        self.max_positions_per_symbol = 1  # Maximum positions per symbol

        # Position tracking
        self.open_positions = {}  # {symbol: [position_data]}
        self.daily_risk_used = 0.0  # Risk used today

        # Lock for thread safety
        self.lock = threading.RLock()

        # Subscribe to signal events
        self.event_bus.subscribe(SignalEvent, self._on_signal)

        self.logger.log_event(
            level="INFO",
            message="OrderManager subscribed to signal events",
            event_type="ORDER_MANAGER",
            component="order_manager",
            action="__init__",
            status="subscribed"
        )

        # Initialize position tracking
        self._update_position_tracking()

        self.logger.log_event(
            level="INFO",
            message="Order Manager initialized",
            event_type="ORDER_MANAGER_INIT",
            component="order_manager",
            action="__init__",
            status="success",
            details={
                "base_risk_percent": self.base_risk_percent,
                "max_daily_risk_percent": self.max_daily_risk_percent,
                "max_positions_per_symbol": self.max_positions_per_symbol
            }
        )

    def _on_signal(self, event: SignalEvent):
        """
        Handle signal events from strategies.

        Args:
            event: Signal event with trade information
        """
        try:
            self.logger.log_event(
                level="INFO",
                message=f"Received {event.direction} signal for {event.symbol} on {event.timeframe_name}",
                event_type="ORDER_SIGNAL",
                component="order_manager",
                action="_on_signal",
                status="received",
                details={
                    "symbol": event.symbol,
                    "direction": event.direction,
                    "timeframe": event.timeframe_name,
                    "strategy": event.strategy_name,
                    "reason": event.reason
                }
            )

            # Different actions based on signal direction
            if event.direction == "BUY":
                self._place_buy_order(event)
            elif event.direction == "SELL":
                self._place_sell_order(event)
            elif event.direction == "CLOSE":
                self._close_positions(event)
            else:
                self.logger.log_event(
                    level="ERROR",
                    message=f"Unknown signal direction: {event.direction}",
                    event_type="ORDER_SIGNAL",
                    component="order_manager",
                    action="_on_signal",
                    status="invalid_direction",
                    details={"direction": event.direction}
                )

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error handling signal event: {str(e)}",
                exception_type=type(e).__name__,
                function="_on_signal",
                traceback=str(e),
                context={
                    "symbol": event.symbol,
                    "direction": event.direction,
                    "timeframe": event.timeframe_name,
                    "strategy": event.strategy_name
                }
            )

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

    def _place_buy_order(self, event: SignalEvent):
        """Place a buy order based on signal"""
        # Update position tracking first
        self._update_position_tracking()

        # Check if we can place more orders for this symbol
        symbol_positions = self.open_positions.get(event.symbol, [])
        if len(symbol_positions) >= self.max_positions_per_symbol:
            self.logger.log_event(
                level="WARNING",
                message=f"Maximum positions reached for {event.symbol}, skipping BUY signal",
                event_type="ORDER_LIMIT",
                component="order_manager",
                action="_place_buy_order",
                status="max_positions",
                details={
                    "symbol": event.symbol,
                    "current_positions": len(symbol_positions),
                    "max_positions": self.max_positions_per_symbol
                }
            )
            return

        # Calculate lot size based on risk management
        lot_size = self._calculate_position_size(event.symbol, event.entry_price, event.stop_loss)

        # Place buy order
        result = self.mt5_manager.place_order(
            symbol=event.symbol,
            order_type=0,  # BUY
            volume=lot_size,
            price=0.0,  # Market price (0.0 for market orders)
            sl=event.stop_loss,
            tp=event.take_profit,
            comment=f"{event.strategy_name}_{event.timeframe_name}"
        )

        if result.get('result', False):
            self.logger.log_trade(
                level="INFO",
                message=f"BUY order placed for {event.symbol} at market price",
                symbol=event.symbol,
                operation="BUY",
                price=result.get('price', event.entry_price),
                volume=lot_size,
                order_id=result.get('order'),
                strategy=event.strategy_name
            )

            # Update position tracking after order
            self._update_position_tracking()

            # Publish order event
            order_event = OrderEvent(
                instrument_id=event.instrument_id,
                symbol=event.symbol,
                order_type="MARKET",
                direction="BUY",
                price=result.get('price', event.entry_price),
                volume=lot_size,
                order_id=result.get('order'),
                stop_loss=event.stop_loss,
                take_profit=event.take_profit,
                strategy_name=event.strategy_name
            )
            self.event_bus.publish(order_event)
        else:
            self.logger.log_error(
                level="ERROR",
                message=f"Failed to place BUY order: {result.get('error', 'Unknown error')}",
                exception_type="OrderError",
                function="_place_buy_order",
                traceback="",
                context={
                    "symbol": event.symbol,
                    "price": event.entry_price,
                    "volume": lot_size,
                    "error": result.get('error')
                }
            )

    def _place_sell_order(self, event: SignalEvent):
        """Place a sell order based on signal"""
        # Update position tracking first
        self._update_position_tracking()

        # Check if we can place more orders for this symbol
        symbol_positions = self.open_positions.get(event.symbol, [])
        if len(symbol_positions) >= self.max_positions_per_symbol:
            self.logger.log_event(
                level="WARNING",
                message=f"Maximum positions reached for {event.symbol}, skipping SELL signal",
                event_type="ORDER_LIMIT",
                component="order_manager",
                action="_place_sell_order",
                status="max_positions",
                details={
                    "symbol": event.symbol,
                    "current_positions": len(symbol_positions),
                    "max_positions": self.max_positions_per_symbol
                }
            )
            return

        # Calculate lot size based on risk management
        lot_size = self._calculate_position_size(event.symbol, event.entry_price, event.stop_loss)

        # Place sell order
        result = self.mt5_manager.place_order(
            symbol=event.symbol,
            order_type=1,  # SELL
            volume=lot_size,
            price=0.0,  # Market price (0.0 for market orders)
            sl=event.stop_loss,
            tp=event.take_profit,
            comment=f"{event.strategy_name}_{event.timeframe_name}"
        )

        if result.get('result', False):
            self.logger.log_trade(
                level="INFO",
                message=f"SELL order placed for {event.symbol} at market price",
                symbol=event.symbol,
                operation="SELL",
                price=result.get('price', event.entry_price),
                volume=lot_size,
                order_id=result.get('order'),
                strategy=event.strategy_name
            )

            # Update position tracking after order
            self._update_position_tracking()

            # Publish order event
            order_event = OrderEvent(
                instrument_id=event.instrument_id,
                symbol=event.symbol,
                order_type="MARKET",
                direction="SELL",
                price=result.get('price', event.entry_price),
                volume=lot_size,
                order_id=result.get('order'),
                stop_loss=event.stop_loss,
                take_profit=event.take_profit,
                strategy_name=event.strategy_name
            )
            self.event_bus.publish(order_event)
        else:
            self.logger.log_error(
                level="ERROR",
                message=f"Failed to place SELL order: {result.get('error', 'Unknown error')}",
                exception_type="OrderError",
                function="_place_sell_order",
                traceback="",
                context={
                    "symbol": event.symbol,
                    "price": event.entry_price,
                    "volume": lot_size,
                    "error": result.get('error')
                }
            )

    def _close_positions(self, event: SignalEvent):
        """Close positions based on signal"""
        # Update position tracking first
        self._update_position_tracking()

        # Get positions for the symbol
        positions = self.open_positions.get(event.symbol, [])
        if not positions:
            self.logger.log_event(
                level="INFO",
                message=f"No positions to close for {event.symbol}",
                event_type="ORDER_CLOSE",
                component="order_manager",
                action="_close_positions",
                status="no_positions",
                details={"symbol": event.symbol, "strategy": event.strategy_name}
            )
            return

        positions_closed = 0

        for position in positions:
            # Only close positions from the same strategy
            if event.strategy_name in position.get('comment', ''):
                result = self.mt5_manager.close_position(position['ticket'])

                if result.get('result', False):
                    positions_closed += 1
                    self.logger.log_trade(
                        level="INFO",
                        message=f"Closed position #{position['ticket']} for {event.symbol}, profit: {result.get('profit', 0)}",
                        symbol=event.symbol,
                        operation="CLOSE",
                        price=result.get('close_price', 0),
                        volume=result.get('close_volume', 0),
                        order_id=position['ticket'],
                        strategy=event.strategy_name
                    )

                    # Publish order event
                    order_event = OrderEvent(
                        instrument_id=event.instrument_id,
                        symbol=event.symbol,
                        order_type="CLOSE",
                        direction="CLOSE",
                        price=result.get('close_price', 0),
                        volume=result.get('close_volume', 0),
                        order_id=position['ticket'],
                        strategy_name=event.strategy_name
                    )
                    self.event_bus.publish(order_event)
                else:
                    self.logger.log_error(
                        level="ERROR",
                        message=f"Failed to close position: {result.get('error', 'Unknown error')}",
                        exception_type="OrderError",
                        function="_close_positions",
                        traceback="",
                        context={
                            "symbol": event.symbol,
                            "ticket": position['ticket'],
                            "error": result.get('error')
                        }
                    )

        if positions_closed > 0:
            # Update position tracking after closing
            self._update_position_tracking()

            self.logger.log_event(
                level="INFO",
                message=f"Closed {positions_closed} positions for {event.symbol}",
                event_type="ORDER_CLOSE",
                component="order_manager",
                action="_close_positions",
                status="success",
                details={"symbol": event.symbol, "positions_closed": positions_closed}
            )

    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management settings.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size in lots
        """
        try:
            # Get account info
            account_info = self.mt5_manager.get_account_info()

            if not account_info:
                self.logger.log_error(
                    level="ERROR",
                    message="Failed to get account info for position sizing",
                    exception_type="AccountInfoError",
                    function="_calculate_position_size",
                    traceback="",
                    context={"symbol": symbol}
                )
                return 0.01  # Default to minimum position size

            # Extract account balance
            balance = account_info.get('balance', 0)

            # Calculate risk amount based on percentage
            risk_amount = balance * (self.base_risk_percent / 100)

            # Calculate stop loss in pips
            pip_size = 0.0001  # Default for most forex pairs

            # Adjust pip size for JPY pairs and Gold
            if symbol.endswith('JPY'):
                pip_size = 0.01
            elif symbol == 'XAUUSD':
                pip_size = 0.01

            # Calculate stop loss distance
            if stop_loss is None or entry_price is None or entry_price == 0 or stop_loss == 0:
                # If no stop loss, use default risk (0.01 lots)
                self.logger.log_event(
                    level="WARNING",
                    message=f"Missing entry or stop loss price for {symbol}, using default lot size",
                    event_type="POSITION_SIZING",
                    component="order_manager",
                    action="_calculate_position_size",
                    status="default_size",
                    details={
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss
                    }
                )
                return 0.01

            stop_distance = abs(entry_price - stop_loss)
            stop_pips = stop_distance / pip_size

            # Safety check for very small stop distance
            if stop_pips < 5:
                self.logger.log_event(
                    level="WARNING",
                    message=f"Stop loss too close ({stop_pips} pips) for {symbol}, using minimum 5 pips",
                    event_type="POSITION_SIZING",
                    component="order_manager",
                    action="_calculate_position_size",
                    status="minimum_stop",
                    details={
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "stop_pips": stop_pips
                    }
                )
                stop_pips = 5

            # Calculate position size in lots
            lot_value_per_pip = 10  # Standard lot value per pip in USD ($10 per pip for 1.0 lot)

            # Adjust for Gold which has different pip value
            if symbol == 'XAUUSD':
                lot_value_per_pip = 100  # Gold has $100 per pip for 1.0 lot

            max_risk_per_pip = risk_amount / stop_pips
            lot_size = max_risk_per_pip / lot_value_per_pip

            # Round to 2 decimal places (0.01 lot precision)
            lot_size = round(lot_size, 2)

            # Apply minimum and maximum position size limits
            lot_size = max(0.01, min(lot_size, 10.0))

            self.logger.log_event(
                level="INFO",
                message=f"Calculated position size: {lot_size} lots for {symbol}",
                event_type="POSITION_SIZING",
                component="order_manager",
                action="_calculate_position_size",
                status="calculated",
                details={
                    "symbol": symbol,
                    "balance": balance,
                    "risk_percent": self.base_risk_percent,
                    "risk_amount": risk_amount,
                    "stop_pips": stop_pips,
                    "lot_size": lot_size
                }
            )

            return lot_size

        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error calculating position size: {str(e)}",
                exception_type=type(e).__name__,
                function="_calculate_position_size",
                traceback=str(e),
                context={"symbol": symbol}
            )
            return 0.01  # Default to minimum position size on error

    def _update_position_tracking(self):
        """Update open positions tracking"""
        try:
            with self.lock:
                # Get current positions from MT5
                positions = self.mt5_manager.get_positions()

                # Reset tracking
                self.open_positions = {}

                # Update tracking dictionary
                for position in positions:
                    symbol = position.get('symbol')
                    if symbol not in self.open_positions:
                        self.open_positions[symbol] = []

                    self.open_positions[symbol].append(position)

                # Log position count
                total_positions = sum(len(pos_list) for pos_list in self.open_positions.values())

                self.logger.log_event(
                    level="INFO",
                    message=f"Trading activity: {total_positions} open positions",
                    event_type="POSITION_UPDATE",
                    component="order_manager",
                    action="_update_position_tracking",
                    status="success",
                    details={
                        "total_positions": total_positions,
                        "positions_by_symbol": {symbol: len(pos_list) for symbol, pos_list in
                                                self.open_positions.items()}
                    }
                )
        except Exception as e:
            self.logger.log_error(
                level="ERROR",
                message=f"Error updating position tracking: {str(e)}",
                exception_type=type(e).__name__,
                function="_update_position_tracking",
                traceback=str(e),
                context={}
            )