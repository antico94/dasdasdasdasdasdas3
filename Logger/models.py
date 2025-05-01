# src/logging/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class BaseLogEntry:
    timestamp: datetime
    level: str
    message: str


@dataclass
class EventLogEntry(BaseLogEntry):
    event_type: str
    component: str
    details: Dict[str, Any] = None


@dataclass
class ErrorLogEntry(BaseLogEntry):
    exception_type: str
    function: str
    traceback: str
    context: Dict[str, Any] = None


@dataclass
class TradeLogEntry(BaseLogEntry):
    symbol: str
    operation: str
    price: float
    volume: float
    order_id: Optional[int] = None
    strategy: Optional[str] = None