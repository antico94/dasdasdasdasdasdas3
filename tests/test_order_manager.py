import unittest
from unittest.mock import MagicMock, patch, ANY, call
from datetime import datetime, timezone

# Assuming your project structure allows these imports
# Adjust paths if necessary (e.g., if execution is in a parent folder)
# Make sure these modules/classes actually exist and are importable
try:
    from execution.order_manager import OrderManager
    # Import the *actual* SignalEvent class now that definition is known
    from Events.events import SignalEvent, OrderEvent
    from Events.event_bus import EventBus
except ImportError as e:
    print(f"Import Error: {e}. Please ensure paths are correct and modules exist.")
    # As a fallback for testing structure, define dummy classes if imports fail
    class SignalEvent:
        # This dummy reflects the provided definition
        def __init__(self, instrument_id: int, symbol: str, timeframe_id: int, timeframe_name: str,
                     direction: str, strength: float, strategy_name: str, reason: str,
                     entry_price = None, stop_loss = None, take_profit = None, timestamp = None):
            self.instrument_id = instrument_id
            self.symbol = symbol
            self.timeframe_id = timeframe_id
            self.timeframe_name = timeframe_name
            self.direction = direction
            self.strength = strength
            self.strategy_name = strategy_name
            self.reason = reason
            self.entry_price = entry_price
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.timestamp = timestamp or datetime.now(timezone.utc)

    class OrderEvent:
        def __init__(self, **kwargs):
             for k, v in kwargs.items():
                setattr(self, k, v)
    class EventBus:
        def subscribe(self, *args): pass
        def publish(self, *args): pass


# --- Mock Helpers ---
class MockSymbolInfo:
    def __init__(self, name="EURUSD", spread=5, trade_mode=1,
                 bid=1.10000, ask=1.10005, volume_min=0.01,
                 volume_max=100.0, volume_step=0.01):
        self.name = name
        self.spread = spread
        self.trade_mode = trade_mode # 1 = Enabled, 0 = Disabled
        self.bid = bid
        self.ask = ask
        self.volume_min = volume_min
        self.volume_max = volume_max
        self.volume_step = volume_step

class MockPosition:
    def __init__(self, ticket=123, symbol="EURUSD", volume=0.1, price_open=1.10000, sl=1.09900, tp=1.10100, comment=""):
        self.ticket = ticket
        self.symbol = symbol
        self.volume = volume
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.comment = comment

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key, default=None):
        return getattr(self, key, default)

# --- Dummy SignalEvent arguments based on provided definition ---
DUMMY_SIGNAL_ARGS = {
    "instrument_id": 1,
    "timeframe_id": 1,
    "timeframe_name": "M5",
    "strength": 0.75,
    "strategy_name": "TestStrategy",
    "reason": "TestReason",
    "entry_price": None,
    "stop_loss": None,
    "take_profit": None,
}

def create_test_signal(**kwargs):
    """Helper to create SignalEvent instances with defaults."""
    args = DUMMY_SIGNAL_ARGS.copy()
    args.update(kwargs)
    # Ensure timestamp is added if your SignalEvent requires it internally (it does via super().__init__)
    if 'timestamp' not in kwargs: # Only add if not explicitly provided
         args['timestamp'] = datetime.now(timezone.utc)

    # Ensure mandatory args for the helper itself are present
    if 'symbol' not in args or 'direction' not in args:
         raise ValueError("Symbol and Direction are mandatory for create_test_signal")

    try:
        # Create using the actual (or dummy) SignalEvent class
        return SignalEvent(**args)
    except TypeError as e:
        print(f"Error creating SignalEvent. Check required arguments for SignalEvent class. Provided: {args}")
        raise e
    except ValueError as e:
        print(f"Error creating SignalEvent: {e}. Provided: {args}")
        raise e


class TestOrderManager(unittest.TestCase):

    def setUp(self):
        """Set up mocks and OrderManager instance before each test."""
        self.mock_mt5_manager = MagicMock()
        self.mock_logger = MagicMock()
        self.mock_event_bus = MagicMock()
        self.mock_config_manager = MagicMock()

        # Configure Mock Defaults
        self.mock_mt5_manager.get_account_info.return_value = {
            'balance': 10000.0, 'equity': 10000.0, 'margin': 100.0,
            'margin_free': 9900.0, 'margin_level': 10000.0
        }
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo()
        self.mock_mt5_manager.get_positions.return_value = []
        self.mock_mt5_manager.place_order.return_value = {'order': 12345, 'retcode': 10009}
        self.mock_mt5_manager.close_position.return_value = {'result': True, 'profit': 10.50}

        # Instantiate OrderManager with Patched Dependencies
        with patch('execution.order_manager.ConfigManager', return_value=self.mock_config_manager), \
             patch('Config.credentials.SQL_SERVER', 'dummy_server', create=True), \
             patch('Config.credentials.SQL_DATABASE', 'dummy_db', create=True), \
             patch('Config.credentials.SQL_DRIVER', 'dummy_driver', create=True), \
             patch('Config.credentials.USE_WINDOWS_AUTH', True, create=True), \
             patch('Config.credentials.SQL_USERNAME', '', create=True), \
             patch('Config.credentials.SQL_PASSWORD', '', create=True):
            self.order_manager = OrderManager(
                mt5_manager=self.mock_mt5_manager,
                logger=self.mock_logger,
                event_bus=self.mock_event_bus
            )

        # Reset logger AFTER init to clear the init message
        self.mock_logger.reset_mock()


    def test_initialization(self):
        """Test if OrderManager initializes correctly."""
        mock_mt5 = MagicMock()
        mock_log = MagicMock()
        mock_bus = MagicMock()
        mock_cfg = MagicMock()

        with patch('execution.order_manager.ConfigManager', return_value=mock_cfg), \
             patch('Config.credentials.SQL_SERVER', 'dummy_server', create=True), \
             patch('Config.credentials.SQL_DATABASE', 'dummy_db', create=True), \
             patch('Config.credentials.SQL_DRIVER', 'dummy_driver', create=True), \
             patch('Config.credentials.USE_WINDOWS_AUTH', True, create=True), \
             patch('Config.credentials.SQL_USERNAME', '', create=True), \
             patch('Config.credentials.SQL_PASSWORD', '', create=True):
            om = OrderManager(
                    mt5_manager=mock_mt5,
                    logger=mock_log,
                    event_bus=mock_bus
                 )

        self.assertEqual(om.mt5_manager, mock_mt5)
        self.assertEqual(om.logger, mock_log)
        self.assertEqual(om.event_bus, mock_bus)
        self.assertEqual(om.base_risk_percent, 1.0)
        self.assertEqual(om.max_daily_risk_percent, 3.0)
        self.assertEqual(om.max_positions_per_symbol, 2)
        self.assertEqual(om.open_positions, {})
        self.assertEqual(om.recent_trades, [])
        mock_bus.subscribe.assert_called_once_with(SignalEvent, om._on_signal)
        mock_log.log_event.assert_called_with(
            level="INFO", message="Order Manager initialized", event_type="ORDER_MANAGER",
            component="order_manager", action="initialize", status="success",
            details={"base_risk": "1.0%", "max_daily_risk": "3.0%"}
        )

    @patch('execution.order_manager.OrderManager._process_entry_signal')
    @patch('execution.order_manager.OrderManager._process_close_signal')
    def test_on_signal_routing(self, mock_process_close, mock_process_entry): # Correct signature
        """Test signal routing in _on_signal."""
        buy_signal = create_test_signal(symbol="EURUSD", direction="BUY")
        self.order_manager._on_signal(buy_signal)
        mock_process_entry.assert_called_once_with(buy_signal)
        mock_process_close.assert_not_called()
        self.mock_logger.log_event.assert_any_call(level="INFO", message=ANY, event_type="SIGNAL_RECEIVED", component="order_manager", action="_on_signal", status="received", details=ANY)
        mock_process_entry.reset_mock()
        self.mock_logger.reset_mock()

        sell_signal = create_test_signal(symbol="EURUSD", direction="SELL")
        self.order_manager._on_signal(sell_signal)
        mock_process_entry.assert_called_once_with(sell_signal)
        mock_process_close.assert_not_called()
        self.mock_logger.log_event.assert_any_call(level="INFO", message=ANY, event_type="SIGNAL_RECEIVED", component="order_manager", action="_on_signal", status="received", details=ANY)
        mock_process_entry.reset_mock()
        self.mock_logger.reset_mock()

        close_signal = create_test_signal(symbol="EURUSD", direction="CLOSE")
        self.order_manager._on_signal(close_signal)
        mock_process_close.assert_called_once_with(close_signal)
        mock_process_entry.assert_not_called()
        self.mock_logger.log_event.assert_any_call(level="INFO", message=ANY, event_type="SIGNAL_RECEIVED", component="order_manager", action="_on_signal", status="received", details=ANY)
        mock_process_close.reset_mock()
        self.mock_logger.reset_mock()

        unknown_signal = create_test_signal(symbol="EURUSD", direction="WAIT")
        self.order_manager._on_signal(unknown_signal)
        mock_process_entry.assert_not_called()
        mock_process_close.assert_not_called()
        self.mock_logger.log_event.assert_any_call(level="WARNING", message="Unknown signal direction: WAIT", event_type="SIGNAL_ERROR", component="order_manager", action="_on_signal", status="invalid_direction", details={'direction': 'WAIT'})

    @patch('execution.order_manager.OrderManager._process_entry_signal')
    def test_on_signal_error_handling(self, mock_process_entry):
        """Test error handling within _on_signal."""
        mock_process_entry.side_effect = Exception("Test Processing Error")
        buy_signal = create_test_signal(symbol="EURUSD", direction="BUY")
        self.order_manager._on_signal(buy_signal)
        self.mock_logger.log_error.assert_called_once_with(
            level="ERROR", message="Error processing signal: Test Processing Error",
            exception_type="Exception", function="_on_signal",
            traceback="Test Processing Error", context={'signal_type': 'BUY', 'symbol': 'EURUSD'}
        )

    def test_validate_entry_signal(self):
        """Test signal validation logic."""
        valid_signal = create_test_signal(symbol="EURUSD", direction="BUY", entry_price=1.1)
        self.assertTrue(self.order_manager._validate_entry_signal(valid_signal))

        invalid_dir_signal = create_test_signal(symbol="EURUSD", direction="HOLD")
        self.assertFalse(self.order_manager._validate_entry_signal(invalid_dir_signal))
        self.mock_logger.log_event.assert_called_with(level="WARNING", message=ANY, event_type="SIGNAL_VALIDATION", status="invalid_direction", details=ANY, component=ANY, action=ANY)
        self.mock_logger.reset_mock()

        no_symbol_signal = create_test_signal(symbol="", direction="BUY")
        self.assertFalse(self.order_manager._validate_entry_signal(no_symbol_signal))
        self.mock_logger.log_event.assert_called_with(level="WARNING", message=ANY, event_type="SIGNAL_VALIDATION", status="missing_symbol", component=ANY, action=ANY)
        self.mock_logger.reset_mock()

        invalid_price_signal = create_test_signal(symbol="EURUSD", direction="SELL", entry_price=0)
        self.assertFalse(self.order_manager._validate_entry_signal(invalid_price_signal))
        self.mock_logger.log_event.assert_called_with(level="WARNING", message=ANY, event_type="SIGNAL_VALIDATION", status="invalid_price", details=ANY, component=ANY, action=ANY)

    # *** FIX: Corrected datetime mocking ***
    @patch('execution.order_manager.datetime')
    def test_is_market_open(self, mock_datetime_module):
        """Test market open/closed logic."""
        mock_now_dt = MagicMock(spec=datetime) # Use spec for better mocking
        mock_datetime_module.now.return_value = mock_now_dt
        # Ensure timezone.utc is still accessible if needed, or mock it too
        mock_datetime_module.timezone.utc = timezone.utc
        # Handle the replace call if it happens
        mock_now_dt.replace.return_value = mock_now_dt

        # Case 1: Market enabled, Monday 10:00 UTC
        mock_now_dt.hour = 10
        mock_now_dt.weekday.return_value = 0 # Monday
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo(trade_mode=1)
        self.assertTrue(self.order_manager._is_market_open("EURUSD"))
        # Check that weekday() was called if trade_mode was checked and passed
        if self.mock_mt5_manager.get_symbol_info.return_value.trade_mode != 0:
             mock_now_dt.weekday.assert_called_once()
        mock_now_dt.weekday.reset_mock()

        # Case 2: Market disabled via trade_mode
        mock_now_dt.hour = 10
        mock_now_dt.weekday.return_value = 0 # Monday
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo(trade_mode=0)
        self.assertFalse(self.order_manager._is_market_open("EURUSD"))
        mock_now_dt.weekday.assert_not_called() # Should short-circuit before weekday check
        mock_now_dt.weekday.reset_mock()

        # Case 3: Saturday
        mock_now_dt.hour = 10
        mock_now_dt.weekday.return_value = 5 # Saturday
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo(trade_mode=1)
        self.assertFalse(self.order_manager._is_market_open("EURUSD"))
        mock_now_dt.weekday.assert_called_once()
        mock_now_dt.weekday.reset_mock()

        # Case 4: Sunday before open (e.g., 16:00 UTC)
        mock_now_dt.hour = 16
        mock_now_dt.weekday.return_value = 6 # Sunday
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo(trade_mode=1)
        self.assertFalse(self.order_manager._is_market_open("EURUSD"))
        mock_now_dt.weekday.assert_called_once()
        mock_now_dt.weekday.reset_mock()

        # Case 5: Sunday after open (e.g., 22:00 UTC)
        mock_now_dt.hour = 22
        mock_now_dt.weekday.return_value = 6 # Sunday
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo(trade_mode=1)
        self.assertTrue(self.order_manager._is_market_open("EURUSD"))
        mock_now_dt.weekday.assert_called_once()
        mock_now_dt.weekday.reset_mock()

        # Case 6: Friday after close (e.g., 22:00 UTC)
        mock_now_dt.hour = 22
        mock_now_dt.weekday.return_value = 4 # Friday
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo(trade_mode=1)
        self.assertFalse(self.order_manager._is_market_open("EURUSD"))
        mock_now_dt.weekday.assert_called_once()
        mock_now_dt.weekday.reset_mock()


    def test_check_market_conditions(self):
        """Test market condition checks (openness and spread)."""
        with patch.object(self.order_manager, '_is_market_open') as mock_is_open, \
             patch.object(self.order_manager, '_get_current_spread') as mock_get_spread:
            mock_is_open.return_value = True
            mock_get_spread.return_value = 5
            self.assertTrue(self.order_manager._check_market_conditions("EURUSD"))
            self.mock_logger.log_event.assert_not_called()

            mock_is_open.return_value = False
            mock_get_spread.return_value = 5
            self.assertFalse(self.order_manager._check_market_conditions("EURUSD"))
            self.mock_logger.log_event.assert_called_with(level="WARNING", message=ANY, event_type="MARKET_CHECK", status="market_closed", details=ANY, component=ANY, action=ANY)
            self.mock_logger.reset_mock()

            mock_is_open.return_value = True
            mock_get_spread.return_value = 25
            self.assertFalse(self.order_manager._check_market_conditions("EURUSD"))
            self.mock_logger.log_event.assert_called_with(level="WARNING", message=ANY, event_type="MARKET_CHECK", status="high_spread", details=ANY, component=ANY, action=ANY)

    def test_check_risk_limits(self):
        """Test risk limit checks (max positions, daily risk)."""
        with patch.object(self.order_manager, '_get_positions_for_symbol') as mock_get_pos, \
             patch.object(self.order_manager, '_calculate_current_risk') as mock_calc_risk:
            mock_get_pos.return_value = [MockPosition()]
            mock_calc_risk.return_value = 1.5
            self.order_manager.max_positions_per_symbol = 2
            self.order_manager.max_daily_risk_percent = 3.0
            self.assertTrue(self.order_manager._check_risk_limits("EURUSD"))
            self.mock_logger.log_event.assert_not_called()

            mock_get_pos.return_value = [MockPosition(), MockPosition()]
            mock_calc_risk.return_value = 1.5
            self.order_manager.max_positions_per_symbol = 2
            self.assertFalse(self.order_manager._check_risk_limits("EURUSD"))
            self.mock_logger.log_event.assert_called_with(level="WARNING", message=ANY, event_type="RISK_CHECK", status="max_positions_reached", details=ANY, component=ANY, action=ANY)
            self.mock_logger.reset_mock()

            mock_get_pos.return_value = [MockPosition()]
            mock_calc_risk.return_value = 3.0
            self.order_manager.max_daily_risk_percent = 3.0
            self.assertFalse(self.order_manager._check_risk_limits("EURUSD"))
            self.mock_logger.log_event.assert_called_with(level="WARNING", message=ANY, event_type="RISK_CHECK", status="max_risk_reached", details=ANY, component=ANY, action=ANY)

    def test_calculate_position_size(self):
        """Test position size calculation."""
        signal = create_test_signal(symbol="EURUSD", direction="BUY", entry_price=1.10000, stop_loss=1.09900)
        risk_percent = 1.0
        self.mock_mt5_manager.get_account_info.return_value = {'balance': 10000.0}
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo(name="EURUSD", bid=1.09995, ask=1.10000, volume_min=0.01, volume_max=100, volume_step=0.01)
        expected_size = 1.0
        expected_sl = 1.09900
        pos_size, sl_price = self.order_manager._calculate_position_size(signal, risk_percent)
        self.assertAlmostEqual(pos_size, expected_size)
        self.assertAlmostEqual(sl_price, expected_sl)

        self.mock_mt5_manager.get_account_info.return_value = {'balance': 5670.0}
        expected_size_rounded = 0.57
        pos_size, _ = self.order_manager._calculate_position_size(signal, risk_percent)
        self.assertAlmostEqual(pos_size, expected_size_rounded)

        self.mock_mt5_manager.get_account_info.return_value = {'balance': 50.0}
        expected_size_min = 0.01
        pos_size, _ = self.order_manager._calculate_position_size(signal, risk_percent)
        self.assertAlmostEqual(pos_size, expected_size_min)


    def test_calculate_take_profits(self):
        """Test TP calculation based on R:R."""
        entry_price = 1.10000
        stop_loss_buy = 1.09900
        stop_loss_sell = 1.10100
        risk = 0.00100

        signal_buy = create_test_signal(symbol="EURUSD", direction="BUY", entry_price=entry_price, stop_loss=stop_loss_buy, strategy_name="trend")
        signal_sell = create_test_signal(symbol="EURUSD", direction="SELL", entry_price=entry_price, stop_loss=stop_loss_sell, strategy_name="reversion")

        expected_tp1_buy = 1.10150
        expected_tp2_buy = 1.10300
        tps_buy = self.order_manager._calculate_take_profits(signal_buy, stop_loss_buy)
        self.assertEqual(len(tps_buy), 2)
        self.assertAlmostEqual(tps_buy[0], expected_tp1_buy)
        self.assertAlmostEqual(tps_buy[1], expected_tp2_buy)

        expected_tp1_sell = 1.09900
        expected_tp2_sell = 1.09850
        tps_sell = self.order_manager._calculate_take_profits(signal_sell, stop_loss_sell)
        self.assertEqual(len(tps_sell), 2)
        self.assertAlmostEqual(tps_sell[0], expected_tp1_sell)
        self.assertAlmostEqual(tps_sell[1], expected_tp2_sell)

        signal_with_tp = create_test_signal(symbol="EURUSD", direction="BUY", entry_price=entry_price, stop_loss=stop_loss_buy, take_profit=1.10500)
        tps_provided = self.order_manager._calculate_take_profits(signal_with_tp, stop_loss_buy)
        self.assertEqual(tps_provided, [1.10500])


    @patch('execution.order_manager.OrderManager._place_order')
    def test_execute_partial_profit_strategy(self, mock_place_order):
        """Test the logic for splitting orders."""
        signal = create_test_signal(symbol="EURUSD", direction="BUY")
        stop_loss = 1.09900
        take_profits = [1.10100, 1.10200]
        self.mock_mt5_manager.get_symbol_info.return_value = MockSymbolInfo(volume_min=0.01, volume_step=0.01)

        position_size = 1.0
        size1_expected = 0.5
        size2_expected = 0.5
        self.order_manager._execute_partial_profit_strategy(signal, position_size, stop_loss, take_profits)
        expected_calls = [
            call(signal, size1_expected, stop_loss, take_profits[0], f"{signal.strategy_name}_Part1"),
            call(signal, size2_expected, stop_loss, take_profits[1], f"{signal.strategy_name}_Part2")
        ]
        mock_place_order.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(mock_place_order.call_count, 2)
        mock_place_order.reset_mock()

        position_size = 0.01
        self.order_manager._execute_partial_profit_strategy(signal, position_size, stop_loss, take_profits)
        mock_place_order.assert_called_once_with(signal, position_size, stop_loss, take_profits[0])
        mock_place_order.reset_mock()

        position_size = 0.02
        size1_expected = 0.01
        size2_expected = 0.01
        self.order_manager._execute_partial_profit_strategy(signal, position_size, stop_loss, take_profits)
        expected_calls = [
            call(signal, size1_expected, stop_loss, take_profits[0], f"{signal.strategy_name}_Part1"),
            call(signal, size2_expected, stop_loss, take_profits[1], f"{signal.strategy_name}_Part2")
        ]
        mock_place_order.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(mock_place_order.call_count, 2)


    def test_place_order_success(self):
        """Test successful order placement."""
        signal = create_test_signal(symbol="EURUSD", direction="BUY", entry_price=1.1)
        position_size = 0.5
        stop_loss = 1.09
        take_profit = 1.11
        comment = "TestComment"
        ticket = 54321
        self.mock_mt5_manager.place_order.return_value = {'order': ticket, 'retcode': 10009}

        self.order_manager._place_order(signal, position_size, stop_loss, take_profit, comment)

        self.mock_mt5_manager.place_order.assert_called_once_with(
            symbol=signal.symbol, order_type=0, volume=position_size,
            price=0.0, sl=stop_loss, tp=take_profit, comment=comment
        )
        self.mock_logger.log_event.assert_any_call(level="INFO", message=ANY, event_type="ORDER_REQUEST", status="requesting", details=ANY, component=ANY, action=ANY)
        self.mock_logger.log_event.assert_any_call(level="INFO", message=f"Order placed successfully: Ticket #{ticket}", event_type="ORDER_EXECUTION", status="success", details=ANY, component=ANY, action=ANY)
        self.mock_event_bus.publish.assert_called_once()
        published_event = self.mock_event_bus.publish.call_args[0][0]
        self.assertIsInstance(published_event, OrderEvent)
        self.assertEqual(published_event.symbol, signal.symbol)
        self.assertEqual(published_event.order_id, ticket)
        self.assertEqual(published_event.volume, position_size)
        self.assertIn(ticket, self.order_manager.open_positions)


    def test_place_order_failure(self):
        """Test failed order placement."""
        signal = create_test_signal(symbol="EURUSD", direction="SELL")
        position_size = 0.5
        stop_loss = 1.11
        take_profit = 1.09
        comment = "TestCommentFail"
        self.mock_mt5_manager.place_order.return_value = {'retcode': 10004, 'error': 'Trade timeout'}

        self.order_manager._place_order(signal, position_size, stop_loss, take_profit, comment)

        self.mock_mt5_manager.place_order.assert_called_once()
        self.mock_logger.log_event.assert_any_call(level="INFO", message=ANY, event_type="ORDER_REQUEST", status="requesting", details=ANY, component=ANY, action=ANY)
        self.mock_logger.log_event.assert_any_call(level="ERROR", message="Order placement failed: Trade timeout", event_type="ORDER_EXECUTION", status="failed", details=ANY, component=ANY, action=ANY)
        self.mock_event_bus.publish.assert_not_called()
        self.assertEqual(len(self.order_manager.open_positions), 0)


    def test_process_close_signal_success(self):
        """Test processing a close signal successfully."""
        symbol = "GBPUSD"
        strategy = "CloseStrat"
        ticket1 = 111
        ticket2 = 222
        ticket_other_strat = 333
        pos1 = MockPosition(ticket=ticket1, symbol=symbol, volume=0.1, comment=f"{strategy}_Part1")
        pos2 = MockPosition(ticket=ticket2, symbol=symbol, volume=0.2, comment=f"{strategy}_Part2")
        pos_other = MockPosition(ticket=ticket_other_strat, symbol=symbol, volume=0.3, comment="AnotherStrat")
        self.mock_mt5_manager.get_positions.return_value = [pos1, pos2, pos_other]
        self.order_manager.open_positions = {
            ticket1: {'symbol': symbol, 'volume': 0.1, 'profit': 0},
            ticket2: {'symbol': symbol, 'volume': 0.2, 'profit': 0},
            ticket_other_strat: {'symbol': symbol, 'volume': 0.3, 'profit': 0}
        }
        self.order_manager.recent_trades = [True]
        self.mock_mt5_manager.close_position.side_effect = [
            {'result': True, 'profit': 5.0}, {'result': True, 'profit': -2.0}
        ]
        close_signal = create_test_signal(symbol=symbol, direction="CLOSE", strategy_name=strategy)

        self.order_manager._process_close_signal(close_signal)

        expected_close_calls = [call(ticket1), call(ticket2)]
        self.mock_mt5_manager.close_position.assert_has_calls(expected_close_calls)
        self.assertEqual(self.mock_mt5_manager.close_position.call_count, 2)
        self.mock_logger.log_event.assert_any_call(level="INFO", message=f"Position #{ticket1} closed successfully", event_type="CLOSE_EXECUTION", status="success", details=ANY, component=ANY, action=ANY)
        self.mock_logger.log_event.assert_any_call(level="INFO", message=f"Position #{ticket2} closed successfully", event_type="CLOSE_EXECUTION", status="success", details=ANY, component=ANY, action=ANY)
        self.assertNotIn(ticket1, self.order_manager.open_positions)
        self.assertNotIn(ticket2, self.order_manager.open_positions)
        self.assertIn(ticket_other_strat, self.order_manager.open_positions)
        self.assertEqual(self.order_manager.recent_trades, [True, True, False])


    def test_process_close_signal_no_positions(self):
        """Test close signal when no positions exist for the symbol."""
        self.mock_mt5_manager.get_positions.return_value = []
        close_signal = create_test_signal(symbol="XAUUSD", direction="CLOSE", strategy_name="GoldStrat")
        self.order_manager._process_close_signal(close_signal)
        self.mock_mt5_manager.close_position.assert_not_called()
        self.mock_logger.log_event.assert_any_call(level="WARNING", message="No open positions found for XAUUSD to close", event_type="CLOSE_SIGNAL", status="no_positions", details=ANY, component=ANY, action=ANY)


    def test_process_close_signal_failure(self):
        """Test processing a close signal when the MT5 close fails."""
        symbol = "AUDUSD"
        strategy = "AussieStrat"
        ticket1 = 444
        pos1 = MockPosition(ticket=ticket1, symbol=symbol, volume=0.1, comment=strategy)
        self.mock_mt5_manager.get_positions.return_value = [pos1]
        self.order_manager.open_positions = {ticket1: {'symbol': symbol, 'volume': 0.1, 'profit': 0}}
        self.mock_mt5_manager.close_position.return_value = {'result': False, 'error': 'Requote'}
        close_signal = create_test_signal(symbol=symbol, direction="CLOSE", strategy_name=strategy)

        self.order_manager._process_close_signal(close_signal)

        self.mock_mt5_manager.close_position.assert_called_once_with(ticket1)
        self.mock_logger.log_event.assert_any_call(level="ERROR", message=f"Failed to close position #{ticket1}: Requote", event_type="CLOSE_EXECUTION", status="failed", details=ANY, component=ANY, action=ANY)
        self.assertIn(ticket1, self.order_manager.open_positions)
        self.assertEqual(self.order_manager.recent_trades, [])

    @patch('execution.order_manager.datetime')
    def test_calculate_dynamic_risk_adjustments(self, mock_datetime_module):
        """Test different adjustments in dynamic risk calculation."""
        mock_now_dt = MagicMock(spec=datetime)
        mock_datetime_module.now.return_value = mock_now_dt
        mock_datetime_module.timezone.utc = timezone.utc

        signal = create_test_signal(symbol="EURUSD", direction="BUY", strength=0.9)

        # 1. Test Session Adjustment (London/NY Overlap)
        mock_now_dt.hour = 14
        self.order_manager.recent_trades = []
        with patch.object(self.order_manager, '_calculate_current_risk', return_value=0):
            risk = self.order_manager._calculate_dynamic_risk(signal)
            self.assertAlmostEqual(risk, 1.392, places=3)

        # 2. Test Performance Adjustment (Low Win Rate)
        mock_now_dt.hour = 3
        self.order_manager.recent_trades = [False] * 8 + [True] * 2
        with patch.object(self.order_manager, '_calculate_current_risk', return_value=0):
            risk = self.order_manager._calculate_dynamic_risk(signal)
            self.assertAlmostEqual(risk, 0.8352, places=4)

        # 3. Test Max Exposure Limit
        mock_now_dt.hour = 10
        self.order_manager.recent_trades = [True] * 5 + [False] * 5
        with patch.object(self.order_manager, '_calculate_current_risk', return_value=2.5):
            risk = self.order_manager._calculate_dynamic_risk(signal)
            self.assertAlmostEqual(risk, 0.5, places=3)


# --- Boilerplate to run tests ---
if __name__ == '__main__':
    unittest.main()