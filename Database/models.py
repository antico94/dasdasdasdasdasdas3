# Database/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Instrument:
    id: int
    symbol: str
    description: str
    pip_value: float

    @classmethod
    def from_db_row(cls, row):
        return cls(
            id=row[0],
            symbol=row[1],
            description=row[2],
            pip_value=row[3]
        )


@dataclass
class Timeframe:
    id: int
    name: str
    description: str
    minutes: int

    @classmethod
    def from_db_row(cls, row):
        return cls(
            id=row[0],
            name=row[1],
            description=row[2],
            minutes=row[3]
        )


@dataclass
class PriceBar:
    instrument_id: int
    timeframe_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: Optional[float] = None
    id: Optional[int] = None

    @classmethod
    def from_db_row(cls, row):
        return cls(
            id=row[0] if len(row) > 8 else None,
            instrument_id=row[1] if len(row) > 8 else row[0],
            timeframe_id=row[2] if len(row) > 8 else row[1],
            timestamp=row[3] if len(row) > 8 else row[2],
            open=row[4] if len(row) > 8 else row[3],
            high=row[5] if len(row) > 8 else row[4],
            low=row[6] if len(row) > 8 else row[5],
            close=row[7] if len(row) > 8 else row[6],
            volume=row[8] if len(row) > 8 else row[7],
            spread=row[9] if len(row) > 9 else None
        )