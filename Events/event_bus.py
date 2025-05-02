# Events/event_bus.py
from typing import Dict, List, Any, Callable, Optional, Set, Type
import threading
import queue
import time
import sys

from Logger.logger import DBLogger


class EventBus:
    """Event bus for dispatching events to subscribers"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventBus, cls).__new__(cls)
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

        self._logger = DBLogger(
            conn_string=conn_string,
            enabled_levels={'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
            console_output=True
        )
        self._subscribers = {}
        self._event_queue = queue.Queue()
        self._running = False
        self._dispatch_thread = None
        self._stop_event = threading.Event()
        self._initialized = True

        # Start event dispatcher thread
        self.start()

    def subscribe(self, event_type: Type, callback: Callable, filters: Optional[Dict[str, Any]] = None) -> None:
        """Subscribe to events of a specific type with optional filters"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            self._subscribers[event_type].append((callback, filters))

            self._logger.log_event(
                level="INFO",
                message=f"Subscribed to event type: {event_type.__name__}",
                event_type="EVENT_SUBSCRIBE",
                component="event_bus",
                action="subscribe",
                status="success",
                details={"event_type": event_type.__name__, "filters": filters}
            )

    def unsubscribe(self, event_type: Type, callback: Callable) -> None:
        """Unsubscribe from events of a specific type"""
        with self._lock:
            if event_type not in self._subscribers:
                return

            # Find and remove the callback
            self._subscribers[event_type] = [
                (cb, f) for cb, f in self._subscribers[event_type] if cb != callback
            ]

            # Remove empty subscriber lists
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]

            self._logger.log_event(
                level="INFO",
                message=f"Unsubscribed from event type: {event_type.__name__}",
                event_type="EVENT_UNSUBSCRIBE",
                component="event_bus",
                action="unsubscribe",
                status="success",
                details={"event_type": event_type.__name__}
            )

    def publish(self, event: Any) -> None:
        """Publish an event to the queue"""
        try:
            event_type = type(event)

            # Check if anyone is subscribed to this event type
            if event_type not in self._subscribers:
                return

            # Add to event queue
            self._event_queue.put(event)

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error publishing event: {str(e)}",
                exception_type=type(e).__name__,
                function="publish",
                traceback=str(e),
                context={"event_type": type(event).__name__}
            )

    def start(self) -> bool:
        """Start event dispatcher thread"""
        if self._running:
            return True

        try:
            self._stop_event.clear()
            self._running = True
            self._dispatch_thread = threading.Thread(target=self._dispatch_loop)
            self._dispatch_thread.daemon = True
            self._dispatch_thread.start()

            self._logger.log_event(
                level="INFO",
                message="Event dispatcher started",
                event_type="EVENT_DISPATCHER_START",
                component="event_bus",
                action="start",
                status="success"
            )

            return True

        except Exception as e:
            self._running = False
            self._logger.log_error(
                level="ERROR",
                message=f"Failed to start event dispatcher: {str(e)}",
                exception_type=type(e).__name__,
                function="start",
                traceback=str(e),
                context={}
            )
            return False

    def stop(self) -> None:
        """Stop event dispatcher thread"""
        if not self._running:
            return

        try:
            self._stop_event.set()
            if self._dispatch_thread:
                self._dispatch_thread.join(timeout=10)
            self._running = False

            self._logger.log_event(
                level="INFO",
                message="Event dispatcher stopped",
                event_type="EVENT_DISPATCHER_STOP",
                component="event_bus",
                action="stop",
                status="success"
            )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error stopping event dispatcher: {str(e)}",
                exception_type=type(e).__name__,
                function="stop",
                traceback=str(e),
                context={}
            )

    def _dispatch_loop(self) -> None:
        """Main event dispatching loop"""
        while not self._stop_event.is_set():
            try:
                try:
                    # Try to get an event with timeout to check stop event periodically
                    event = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                event_type = type(event)

                # Copy subscribers list to avoid modification during iteration
                with self._lock:
                    if event_type in self._subscribers:
                        subscribers = list(self._subscribers[event_type])
                    else:
                        subscribers = []

                # Process event
                self._process_event(event, subscribers)

                # Mark task as done
                self._event_queue.task_done()

            except Exception as e:
                self._logger.log_error(
                    level="ERROR",
                    message=f"Error in event dispatch loop: {str(e)}",
                    exception_type=type(e).__name__,
                    function="_dispatch_loop",
                    traceback=str(e),
                    context={}
                )

                # Sleep a bit before retrying
                time.sleep(1)

    def _process_event(self, event: Any, subscribers: List[tuple]) -> None:
        """Process event and notify subscribers"""
        event_type = type(event)
        event_name = event_type.__name__

        for callback, filters in subscribers:
            try:
                # Check if event matches filters
                if self._matches_filters(event, filters):
                    # Call subscriber callback
                    callback(event)

            except Exception as e:
                self._logger.log_error(
                    level="ERROR",
                    message=f"Error notifying subscriber for event {event_name}: {str(e)}",
                    exception_type=type(e).__name__,
                    function="_process_event",
                    traceback=str(e),
                    context={"event_type": event_name}
                )

    def _matches_filters(self, event: Any, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if event matches the filters"""
        if not filters:
            return True

        for key, value in filters.items():
            # Check if event has the attribute
            if not hasattr(event, key):
                return False

            # Get attribute value
            attr_value = getattr(event, key)

            # Check if attribute matches filter value
            if isinstance(value, (list, tuple, set)):
                # Check if attribute value is in the filter values
                if attr_value not in value:
                    return False
            elif attr_value != value:
                return False

        return True

    def stop(self) -> None:
        """Stop event dispatcher with timeout"""
        if not self._running:
            return

        try:
            self._stop_event.set()

            # Wait for event dispatcher thread to terminate with timeout
            if self._dispatch_thread and self._dispatch_thread.is_alive():
                max_wait_seconds = 10
                for i in range(max_wait_seconds):
                    if not self._dispatch_thread.is_alive():
                        break
                    time.sleep(1)

                # If still alive after timeout, log warning
                if self._dispatch_thread.is_alive():
                    self._logger.log_event(
                        level="WARNING",
                        message=f"Event dispatcher thread didn't terminate after {max_wait_seconds} seconds",
                        event_type="EVENT_DISPATCHER_STOP",
                        component="event_bus",
                        action="stop",
                        status="timeout"
                    )
                    # Don't attempt join again as it may block indefinitely

            self._running = False

            self._logger.log_event(
                level="INFO",
                message="Event dispatcher stopped",
                event_type="EVENT_DISPATCHER_STOP",
                component="event_bus",
                action="stop",
                status="success"
            )

        except Exception as e:
            self._logger.log_error(
                level="ERROR",
                message=f"Error stopping event dispatcher: {str(e)}",
                exception_type=type(e).__name__,
                function="stop",
                traceback=str(e),
                context={}
            )
            # Mark as not running even if there was an error
            self._running = False