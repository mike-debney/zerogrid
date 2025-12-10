"""State."""

from collections import deque
from datetime import datetime, timedelta


class ControllableLoadPlanState:
    """Represents the planned state of a controllable load."""

    def __init__(self) -> None:
        """Initialize the controllable load plan state."""
        self.is_on: bool = False
        self.expected_load_amps: float = 0.0
        self.throttle_amps: float = 0.0


class PlanState:
    """Represents the overall planned state for load control."""

    def __init__(self) -> None:
        """Initialize the plan state."""
        self.available_amps: float = 0.0
        self.used_amps: float = 0.0
        self.controllable_loads: dict[str, ControllableLoadPlanState] = {}


class ControllableLoadState:
    """Represents the current state of a controllable load."""

    def __init__(self) -> None:
        """Initialize the controllable load state."""
        self.is_on: bool = False  # Actual state of the switch
        self.is_under_load_control: bool = False  # True if we have turned it on
        self.last_toggled: datetime | None = None
        self.last_throttled: datetime | None = None
        self.current_load_amps: float = 0.0
        self.expected_load_amps: float = 0.0  # What we planned for this load to consume
        self.is_switch_rate_limited: bool = False
        self.is_throttle_rate_limited: bool = False
        self.on_since: datetime | None = None
        self.can_turn_on: bool = True  # External constraint allowing turn on


class State:
    """Represents the current state the overall integration."""

    def __init__(self) -> None:
        """Initialize the state."""
        self.available_amps: float = 0.0
        self.house_consumption_amps: float = 0.0
        self.load_control_consumption_amps: float = 0.0
        self.allow_grid_import: bool = False
        self.enable_load_control: bool = False
        self.solar_generation_amps: float = 0.0
        self.controllable_loads: dict[str, ControllableLoadState] = {}
        self.overload_timestamp: datetime | None = None
        self.safety_abort_timestamp: datetime | None = None
        self.last_recalculation: datetime | None = None
        self.available_amps_history: deque[tuple[datetime, float]] = deque(maxlen=100)
        self.solar_generation_initialised: bool = False
        self.house_consumption_initialised: bool = False

    def accumulate_unallocated_amps(
        self,
        value: float,
        max_duration_seconds: float,
    ) -> None:
        """Accumulate a timestamped available amps value into the sliding window buffer.

        Args:
            value: The measured available amps value to accumulate
            max_duration_seconds: Maximum duration to keep in the buffer

        The buffer is automatically trimmed to only keep entries within max_duration_seconds
        from the most recent timestamp. This allows irregular measurement frequencies while
        maintaining accurate time-based windowing.
        """
        # Add the new value
        timestamp = datetime.now()
        self.available_amps_history.append((timestamp, value))

        # Remove entries older than max_duration_seconds from the newest entry
        cutoff_time = timestamp - timedelta(seconds=max_duration_seconds)
        while (
            self.available_amps_history
            and self.available_amps_history[0][0] < cutoff_time
        ):
            self.available_amps_history.popleft()

    def get_minimum_unallocated_amps(
        self,
        lookback_seconds: float,
    ) -> float:
        """Get the minimum available amps from the buffer within a time window.

        Args:
            current_time: The current time to measure lookback from
            lookback_seconds: How many seconds back to look for the minimum

        Returns:
            The minimum value found within the time window, or the most recent value
            if no values exist in the window, or None if the buffer is empty.
        """
        cutoff_time = datetime.now() - timedelta(seconds=lookback_seconds)
        min_value = None

        for timestamp, value in self.available_amps_history:
            if timestamp >= cutoff_time:
                if min_value is None:
                    min_value = value
                else:
                    min_value = min(min_value, value)

        # If no values found in window but buffer has data, return most recent value
        if min_value is None and self.available_amps_history:
            min_value = self.available_amps_history[-1][1]
        if min_value is None:
            min_value = 0

        return min_value
