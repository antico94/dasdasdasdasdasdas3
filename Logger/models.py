# src/logging/models.py
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BaseLogEntry:
    timestamp: datetime
    level: str
    message: str