# Events/events.py
from datetime import datetime
from enum import Enum
from typing import Optional

from Database.models import PriceBar


class MarketStateType(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PRE_MARKET = "PRE_MARKET"
    POST_MARKET = "POST_MARKET"


class Event:
    """Base event class"""
    def __init__(self, timestamp: Optional[datetime] = None):
        self.timestamp = timestamp or datetime.now()


class NewBarEvent(Event):
    """Event for new price bar"""
    def __init__(self,
                 instrument_id: int,
                 timeframe_id: int,
                 bar: PriceBar,
                 symbol: str,
                 timeframe_name: str,
                 timestamp: Optional[datetime] = None):
        super().__init__(timestamp)
        self.instrument_id = instrument_id
        self.timeframe_id = timeframe_id
        self.bar = bar
        self.symbol = symbol
        self.timeframe_name = timeframe_name


class MarketStateEvent(Event):
    """Event for market state changes"""
    def __init__(self,
                 instrument_id: int,
                 symbol: str,
                 state: MarketStateType,
                 timestamp: Optional[datetime] = None):
        super().__init__(timestamp)
        self.instrument_id = instrument_id
        self.symbol = symbol
        self.state = state


class SignalEvent(Event):
    """Event for trading signals"""
    def __init__(self,
                 instrument_id: int,
                 symbol: str,
                 timeframe_id: int,
                 timeframe_name: str,
                 direction: str,
                 strength: float,
                 strategy_name: str,
                 reason: str,
                 entry_price: Optional[float] = None,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None,
                 timestamp: Optional[datetime] = None):
        super().__init__(timestamp)
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


class OrderEvent(Event):
    """Event for order execution"""
    def __init__(self,
                 instrument_id: int,
                 symbol: str,
                 order_type: str,
                 direction: str,
                 price: float,
                 volume: float,
                 order_id: Optional[int] = None,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None,
                 strategy_name: Optional[str] = None,
                 timestamp: Optional[datetime] = None):
        super().__init__(timestamp)
        self.instrument_id = instrument_id
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.price = price
        self.volume = volume
        self.order_id = order_id
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy_name = strategy_name