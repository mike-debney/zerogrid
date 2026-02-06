"""The ZeroGrid integration."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import math

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_ON, Platform
from homeassistant.core import Event, HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.event import (
    EventStateChangedData,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.typing import ConfigType

from .config import Config, ControllableLoadConfig
from .const import ALLOW_GRID_IMPORT_SWITCH_ID, DOMAIN, ENABLE_LOAD_CONTROL_SWITCH_ID
from .helpers import parse_entity_domain
from .state import ControllableLoadPlanState, ControllableLoadState, PlanState, State

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.BINARY_SENSOR,
    Platform.NUMBER,
    Platform.SENSOR,
    Platform.SWITCH,
]

# Store per-entry instances keyed by entry_id
CONFIGS: dict[str, Config] = {}
STATES: dict[str, State] = {}
PLANS: dict[str, PlanState] = {}

# For backwards compatibility with existing code
CONFIG = Config()
STATE = State()
PLAN = PlanState()


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the ZeroGrid component from YAML (legacy support)."""

    # Only register the service here - config entries handle the rest
    async def handle_recalculate_load_control(call) -> None:
        """Handle the recalculate_load_control service call."""
        entry_id = call.data.get("entry_id")

        if entry_id:
            # Recalculate specific entry
            if entry_id not in CONFIGS:
                raise ServiceValidationError(
                    f"Config entry {entry_id} not found",
                    translation_domain=DOMAIN,
                    translation_key="entry_not_found",
                )
            _LOGGER.info("Manual recalculation triggered for entry %s", entry_id)
            await recalculate_load_control(hass, entry_id)
        else:
            # Recalculate all entries
            _LOGGER.info("Manual recalculation triggered for all entries")
            for entry_id in CONFIGS:
                await recalculate_load_control(hass, entry_id)

    hass.services.async_register(
        DOMAIN,
        "recalculate_load_control",
        handle_recalculate_load_control,
    )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up ZeroGrid from a config entry."""
    _LOGGER.debug("Setting up ZeroGrid from config entry: %s", entry.entry_id)

    # Create per-entry instances
    CONFIGS[entry.entry_id] = Config()
    STATES[entry.entry_id] = State()
    PLANS[entry.entry_id] = PlanState()

    # Set the global instances to this entry's instances for backwards compatibility
    global CONFIG, STATE, PLAN
    CONFIG = CONFIGS[entry.entry_id]
    STATE = STATES[entry.entry_id]
    PLAN = PLANS[entry.entry_id]

    # Merge config entry data with options (options take precedence)
    config_data = {**entry.data, **entry.options}

    # Parse configuration from config entry
    parse_config(config_data)
    initialise_state(hass)

    # Set up per-entry entity change listeners
    config = CONFIGS[entry.entry_id]
    state = STATES[entry.entry_id]
    entity_ids: list[str] = [config.house_consumption_amps_entity]

    if (
        config.allow_solar_consumption
        and config.solar_generation_amps_entity is not None
    ):
        entity_ids.append(config.solar_generation_amps_entity)

    for load_config in config.controllable_loads.values():
        entity_ids.append(load_config.load_amps_entity)
        if load_config.switch_entity is not None:
            entity_ids.append(load_config.switch_entity)
        if load_config.can_turn_on_entity is not None:
            entity_ids.append(load_config.can_turn_on_entity)

    async def state_automation_listener(event: Event[EventStateChangedData]) -> None:
        if event.event_type != "state_changed":
            return

        entity_id = event.data["entity_id"]
        new_state = event.data.get("new_state")
        if new_state is None:
            return

        if entity_id == config.house_consumption_amps_entity:
            if new_state is not None and new_state.state not in (
                "unknown",
                "unavailable",
            ):
                state.house_consumption_amps = float(new_state.state)
                state.house_consumption_initialised = True
                clear_safety_abort(hass, entry.entry_id)
                await recalculate_load_control(hass, entry.entry_id)
            elif not config.disable_consumption_unavailable_safety_abort:
                await safety_abort(hass, entry.entry_id)
            else:
                _LOGGER.warning(
                    "House consumption entity unavailable, but safety abort is disabled"
                )

        elif entity_id == config.solar_generation_amps_entity:
            if new_state is not None and new_state.state not in (
                "unknown",
                "unavailable",
            ):
                state.solar_generation_amps = float(new_state.state)
                await recalculate_load_control(hass, entry.entry_id)
            else:
                state.solar_generation_amps = 0.0

        else:
            # Check if it's a controllable load entity
            for load_config in config.controllable_loads.values():
                load = state.controllable_loads[load_config.name]

                if entity_id == load_config.switch_entity:
                    if new_state is not None and new_state.state not in (
                        "unknown",
                        "unavailable",
                    ):
                        load.is_on = new_state.state != "off"
                        # If option enabled, assume load is under control when on
                        if load.is_on and load_config.assume_always_under_load_control:
                            load.is_under_load_control = True
                            if load.on_since is None:
                                load.on_since = datetime.now()
                        _LOGGER.debug(
                            "Load %s switch changed to %s",
                            load_config.name,
                            new_state.state,
                        )
                elif entity_id == load_config.load_amps_entity:
                    if new_state is not None and new_state.state not in (
                        "unknown",
                        "unavailable",
                    ):
                        load.current_load_amps = float(new_state.state)
                elif (
                    load_config.can_turn_on_entity is not None
                    and entity_id == load_config.can_turn_on_entity
                ):
                    if new_state is not None and new_state.state not in (
                        "unknown",
                        "unavailable",
                    ):
                        load.can_turn_on = new_state.state == "on"
                        _LOGGER.debug(
                            "Load %s can_turn_on changed to %s",
                            load_config.name,
                            load.can_turn_on,
                        )
                    elif load_config.can_turn_on_ignore_unavailable:
                        load.can_turn_on = True
                        _LOGGER.debug(
                            "Load %s can_turn_on entity unavailable, ignoring safety check",
                            load_config.name,
                        )
                    else:
                        load.can_turn_on = False

    async def state_time_listener(now: datetime) -> None:
        if config.enable_automatic_recalculation:
            if (
                state.last_recalculation is None
                or state.last_recalculation
                + timedelta(seconds=config.recalculate_interval_seconds)
                < datetime.now()
            ):
                await recalculate_load_control(hass, entry.entry_id)

    # Subscribe to state changes for all relevant entities
    entry.async_on_unload(
        async_track_state_change_event(hass, entity_ids, state_automation_listener)
    )

    # Subscribe to time-based recalculation
    interval = timedelta(seconds=1)
    entry.async_on_unload(
        async_track_time_interval(hass, state_time_listener, interval)
    )

    # Store entry in hass.data for platform access
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "entry": entry,
        "config": CONFIGS[entry.entry_id],
        "state": STATES[entry.entry_id],
        "plan": PLANS[entry.entry_id],
    }

    # Register update listener for options changes
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    # Forward entry setup to platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        # Clean up this entry's data
        hass.data[DOMAIN].pop(entry.entry_id, None)

        # Clear this entry's global state
        if entry.entry_id in CONFIGS:
            CONFIGS[entry.entry_id].controllable_loads.clear()
            del CONFIGS[entry.entry_id]
        if entry.entry_id in STATES:
            STATES[entry.entry_id].controllable_loads.clear()
            STATES[entry.entry_id].available_amps_history.clear()
            del STATES[entry.entry_id]
        if entry.entry_id in PLANS:
            PLANS[entry.entry_id].controllable_loads.clear()
            del PLANS[entry.entry_id]

    return unload_ok


def parse_config(domain_config):
    """Parses the config and sets appropriates variables."""
    _LOGGER.debug(domain_config)

    CONFIG.max_total_load_amps = domain_config.get("max_total_load_amps", 0)
    CONFIG.max_grid_import_amps = domain_config.get("max_grid_import_amps", 0)
    CONFIG.max_solar_generation_amps = domain_config.get("max_solar_generation_amps", 0)
    # Sanity check - total load cannot exceed grid import + solar generation
    CONFIG.max_total_load_amps = min(
        CONFIG.max_total_load_amps,
        CONFIG.max_grid_import_amps + CONFIG.max_solar_generation_amps,
    )

    CONFIG.safety_margin_amps = domain_config.get("safety_margin_amps", 2.0)
    CONFIG.recalculate_interval_seconds = domain_config.get(
        "recalculate_interval_seconds", 10
    )
    CONFIG.enable_automatic_recalculation = domain_config.get(
        "enable_automatic_recalculation", True
    )
    CONFIG.house_consumption_amps_entity = domain_config.get(
        "house_consumption_amps_entity", None
    )
    CONFIG.disable_consumption_unavailable_safety_abort = domain_config.get(
        "disable_consumption_unavailable_safety_abort", False
    )

    CONFIG.solar_generation_amps_entity = domain_config.get(
        "solar_generation_amps_entity", None
    )
    CONFIG.allow_solar_consumption = CONFIG.solar_generation_amps_entity is not None

    control_options = domain_config.get("controllable_loads", [])
    for priority, control in enumerate(control_options):
        control_config = ControllableLoadConfig()
        control_config.name = control.get("name")
        control_config.priority = priority
        control_config.max_controllable_load_amps = control.get(
            "max_controllable_load_amps"
        )
        control_config.min_controllable_load_amps = control.get(
            "min_controllable_load_amps"
        )
        control_config.min_toggle_interval_seconds = control.get(
            "min_toggle_interval_seconds", None
        )
        control_config.min_throttle_interval_seconds = control.get(
            "min_throttle_interval_seconds", None
        )
        control_config.load_measurement_delay_seconds = control.get(
            "load_measurement_delay_seconds", 120
        )
        control_config.solar_turn_on_window_seconds = control.get(
            "solar_turn_on_window_seconds", 300
        )
        control_config.solar_turn_off_window_seconds = control.get(
            "solar_turn_off_window_seconds", 300
        )
        control_config.load_amps_entity = control.get("load_amps_entity")
        control_config.switch_entity = control.get("switch_entity")
        control_config.throttle_amps_entity = control.get("throttle_amps_entity", None)
        control_config.can_throttle = control_config.throttle_amps_entity is not None
        control_config.can_turn_on_entity = control.get("can_turn_on_entity", None)
        control_config.can_turn_on_ignore_unavailable = control.get(
            "can_turn_on_ignore_unavailable", False
        )
        control_config.assume_always_under_load_control = control.get(
            "assume_always_under_load_control", False
        )
        CONFIG.controllable_loads[control_config.name] = control_config

    _LOGGER.debug("Config successful: %s", CONFIG)


def initialise_state(hass: HomeAssistant):
    """Initialises the state of the integration."""
    if CONFIG.house_consumption_amps_entity is not None:
        state = hass.states.get(CONFIG.house_consumption_amps_entity)
        if state is not None and state.state not in ("unknown", "unavailable"):
            STATE.house_consumption_amps = float(state.state)

    if CONFIG.solar_generation_amps_entity is not None:
        state = hass.states.get(CONFIG.solar_generation_amps_entity)
        if state is not None and state.state not in ("unknown", "unavailable"):
            STATE.solar_generation_amps = float(state.state)
    else:
        STATE.solar_generation_amps = 0.0

    # Don't initialize switch states here - they will be initialized by the switch
    # entities themselves when they restore their state in async_added_to_hass.
    # The switches will update STATE.allow_grid_import and STATE.enable_load_control.

    # match to controllable loads
    for load_name in CONFIG.controllable_loads:  # pylint: disable=consider-using-dict-items
        config = CONFIG.controllable_loads[load_name]
        load_state = ControllableLoadState()

        switch_state = hass.states.get(config.switch_entity)
        if switch_state is not None and switch_state.state not in (
            "unknown",
            "unavailable",
        ):
            load_state.is_on = hass.states.is_state(config.switch_entity, STATE_ON)
            # If option enabled, assume load is under control when on
            if config.assume_always_under_load_control:
                load_state.is_under_load_control = load_state.is_on
            else:
                load_state.is_under_load_control = (
                    load_state.is_on
                )  # Assume under control initially
            if load_state.is_on and load_state.is_under_load_control:
                load_state.on_since = datetime.now()
            else:
                load_state.on_since = None

        load_amps_state = hass.states.get(config.load_amps_entity)
        if load_amps_state is not None and load_amps_state.state not in (
            "unknown",
            "unavailable",
        ):
            load_state.current_load_amps = float(load_amps_state.state)

        # Initialize can_turn_on state from entity if configured
        if config.can_turn_on_entity is not None:
            can_turn_on_state = hass.states.get(config.can_turn_on_entity)
            if can_turn_on_state is not None and can_turn_on_state.state not in (
                "unknown",
                "unavailable",
            ):
                load_state.can_turn_on = can_turn_on_state.state == STATE_ON
            elif config.can_turn_on_ignore_unavailable:
                load_state.can_turn_on = True  # Ignore unavailable, allow turn on
            else:
                load_state.can_turn_on = False  # Default to safe state
        else:
            load_state.can_turn_on = True  # No constraint configured

        STATE.controllable_loads[load_name] = load_state
        PLAN.controllable_loads[load_name] = ControllableLoadPlanState()
        _LOGGER.debug(
            "Switch entity init for %s: %s",
            load_name,
            PLAN.controllable_loads[load_name].is_on,
        )

    _LOGGER.debug("Initialised state: %s", STATE)


async def calculate_effective_available_power(
    hass: HomeAssistant,
    entry_id: str,
) -> tuple[float, float]:
    """Calculate available power including power freed by underperforming loads."""
    # Get per-entry config/state/plan
    config = hass.data[DOMAIN][entry_id]["config"]
    state = hass.data[DOMAIN][entry_id]["state"]
    plan = hass.data[DOMAIN][entry_id]["plan"]

    max_safe_total_load_amps = 0

    # Allow grid import
    grid_maximum_amps = 0.0
    if state.allow_grid_import:
        grid_maximum_amps = config.max_grid_import_amps
        max_safe_total_load_amps += config.max_grid_import_amps

    # Use solar generation amps directly
    capped_solar_generation_amps = 0.0
    if config.allow_solar_consumption and state.solar_generation_amps > 0:
        capped_solar_generation_amps = min(
            state.solar_generation_amps, config.max_solar_generation_amps
        )
        max_safe_total_load_amps += config.max_solar_generation_amps

    # Calculate total available power before accounting for loads
    now = datetime.now()
    total_available_power_amps = grid_maximum_amps + capped_solar_generation_amps
    # Safety cap to maximum total available load
    total_available_power_amps = min(
        total_available_power_amps, config.max_total_load_amps
    )

    # Subtract loads that are under load control, since we can manage those
    total_load_not_under_control = state.house_consumption_amps
    for load_name in state.controllable_loads:
        load_state = state.controllable_loads[load_name]
        load_plan = plan.controllable_loads.get(load_name)

        if load_state.is_under_load_control and load_state.is_on:
            current_load = load_state.current_load_amps
            # Determine if we should use expected load instead of measured load
            # to account for soft starts and measurement delays
            if load_state.on_since is not None and load_plan is not None:
                load_config = config.controllable_loads[load_name]
                time_since_on = (now - load_state.on_since).total_seconds()
                if time_since_on < load_config.load_measurement_delay_seconds:
                    current_load = load_plan.expected_load_amps
            total_load_not_under_control -= current_load

    # Subtract reserved current from available amps
    state.reserved_current_amps = 0.0
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "reserved_current" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        reserved_current_value = hass.data[DOMAIN][entry_id]["entities"][
            "reserved_current"
        ].native_value
        if reserved_current_value is not None:
            state.reserved_current_amps = float(max(0, reserved_current_value))

    # Now calculate total available for load control by subtracting non-controlled loads
    total_available_amps = total_available_power_amps - max(
        total_load_not_under_control, state.reserved_current_amps, 0
    )

    # Determine max window duration based on longest min_toggle_interval
    max_window_seconds = max(
        (
            lconfig.min_toggle_interval_seconds
            for lconfig in config.controllable_loads.values()
        ),
        default=60,  # Default to 60 seconds if no controllable loads configured
    )

    # Accumulate the available power for load control into history
    # This is the power available AFTER accounting for uncontrolled loads
    state.accumulate_available_amps(total_available_amps, max_window_seconds)

    _LOGGER.debug(
        "Total available power: %gA, uncontrolled load: %gA, reserved current: %gA, available for load control: %gA",
        total_available_power_amps,
        total_load_not_under_control,
        state.reserved_current_amps,
        total_available_amps,
    )

    # Update entities instead of setting state directly
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "available_load_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["available_load_sensor"].update_value(
            max(0, total_available_amps)
        )
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "uncontrolled_load_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"][
            "uncontrolled_load_sensor"
        ].update_value(total_load_not_under_control)
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "max_safe_load_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["max_safe_load_sensor"].update_value(
            max_safe_total_load_amps
        )

    return total_available_amps, max_safe_total_load_amps


def reset_load_control_state(config: Config, state: State) -> None:
    """Reset load control state when load control is disabled or re-enabled."""
    state.available_amps_history.clear()
    for control in config.controllable_loads.values():
        load = state.controllable_loads[control.name]
        load.is_under_load_control = True
        load.last_throttled = None
        load.last_toggled = None


async def recalculate_load_control(hass: HomeAssistant, entry_id: str):
    """The core of the load control algorithm.

    This function is intentionally complex as it handles the complete load planning
    algorithm including priority management, power allocation, throttling, rate limiting,
    overload protection, and reactive reallocation.
    """
    if entry_id not in hass.data.get(DOMAIN, {}):
        _LOGGER.error("Entry %s not found in hass.data", entry_id)
        return

    config = hass.data[DOMAIN][entry_id]["config"]
    state = hass.data[DOMAIN][entry_id]["state"]

    if not state.house_consumption_initialised:
        _LOGGER.debug(
            "Recalculation skipped - house consumption not initialised for entry %s",
            entry_id,
        )
        return

    if state.safety_abort_active:
        _LOGGER.debug(
            "Recalculation skipped - safety abort for entry %s",
            entry_id,
        )
        return

    now = datetime.now()
    if state.last_recalculation is not None and (
        state.last_recalculation + timedelta(seconds=1) > now
    ):
        _LOGGER.debug(
            "Recalculation skipped due to interval limit for entry %s", entry_id
        )
        return
    state.last_recalculation = now

    _LOGGER.info("Recalculating load control plan for entry %s", entry_id)
    # Check if load control is enabled
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and ENABLE_LOAD_CONTROL_SWITCH_ID in hass.data[DOMAIN][entry_id]["entities"]
    ):
        state.enable_load_control = hass.data[DOMAIN][entry_id]["entities"][
            ENABLE_LOAD_CONTROL_SWITCH_ID
        ].is_on
    if not state.enable_load_control:
        _LOGGER.debug("Load control is disabled, skipping recalculation")
        reset_load_control_state(config, state)
        return

    # Check if allow grid import is enabled
    state.allow_grid_import = False
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and ALLOW_GRID_IMPORT_SWITCH_ID in hass.data[DOMAIN][entry_id]["entities"]
    ):
        state.allow_grid_import = hass.data[DOMAIN][entry_id]["entities"][
            ALLOW_GRID_IMPORT_SWITCH_ID
        ].is_on

    new_plan = PlanState()

    # Calculate effective available power, loads that we are controlling are included
    (
        available_amps,
        max_safe_total_load_amps,
    ) = await calculate_effective_available_power(hass, entry_id)
    new_plan.available_amps = available_amps

    # Build priority list (lower number == more important)
    prioritised_loads = sorted(
        CONFIG.controllable_loads,
        key=lambda k: CONFIG.controllable_loads[k].priority,
    )
    _LOGGER.debug("Priority: %s", prioritised_loads)

    # First pass to determine if loads should be on or not
    for load_name in prioritised_loads:
        config = CONFIG.controllable_loads[load_name]
        state = STATE.controllable_loads[load_name]
        previous_plan = PLAN.controllable_loads[load_name]

        plan = new_plan.controllable_loads[load_name] = ControllableLoadPlanState()
        plan.is_on = previous_plan.is_on
        plan.expected_load_amps = 0.0

        # Determine if we are rate-limited on switching or throttling
        state.is_switch_rate_limited = (
            state.last_toggled is not None
            and state.last_toggled
            + timedelta(seconds=config.min_toggle_interval_seconds)
            > now
        )
        state.is_throttle_rate_limited = (
            config.can_throttle
            and state.last_throttled is not None
            and state.last_throttled
            + timedelta(seconds=config.min_throttle_interval_seconds)
            > now
        )

        if not state.is_under_load_control and state.is_on:
            # Load is manually turned on - we have no control
            plan.is_on = state.is_on
            _LOGGER.debug("Load %s manually turned on, skipping control", load_name)
            continue

        will_consume_amps = 0.0

        # Determine if this load should be on based on available power
        should_be_on = available_amps >= config.min_controllable_load_amps

        if not STATE.allow_grid_import:
            # Make sure we have a stable minimum available power before turning on (important for solar)
            if should_be_on and not state.is_on and not previous_plan.is_on:
                min_available_amps = STATE.get_minimum_available_amps(
                    config.solar_turn_on_window_seconds
                )
                if min_available_amps < config.min_controllable_load_amps:
                    should_be_on = False
                    _LOGGER.debug(
                        "Preventing load %s turn on due to insufficient minimum capacity of %gA over last %ds",
                        load_name,
                        min_available_amps,
                        config.solar_turn_on_window_seconds,
                    )
            # Prevent turning off loads early if available power is low for short periods
            elif state.is_on and state.is_under_load_control and not should_be_on:
                average_available_amps = STATE.get_average_available_amps(
                    config.solar_turn_off_window_seconds
                )
                if average_available_amps > config.min_controllable_load_amps:
                    should_be_on = True
                    _LOGGER.debug(
                        "Preventing load %s turn off due to average capacity of %gA over last %ds",
                        load_name,
                        average_available_amps,
                        config.solar_turn_off_window_seconds,
                    )

        # Log if external constraint is preventing turn on
        if should_be_on and not state.can_turn_on:
            should_be_on = False
            _LOGGER.debug(
                "Load %s has sufficient power but external constraint prevents turn on",
                load_name,
            )

        # Prevent toggling if rate limited
        if state.is_switch_rate_limited:
            if should_be_on != (previous_plan.is_on or state.is_on):
                if should_be_on:
                    _LOGGER.debug(
                        "Unable to turn load %s on due to switch rate limit", load_name
                    )
                else:
                    _LOGGER.debug(
                        "Unable to turn load %s off due to switch rate limit", load_name
                    )
            plan.is_on = previous_plan.is_on or state.is_on
        else:
            plan.is_on = should_be_on
            if plan.is_on != previous_plan.is_on:
                if plan.is_on:
                    _LOGGER.debug("Planning to turn load %s on", load_name)
                else:
                    _LOGGER.debug("Planning to turn load %s off", load_name)

        # Determine if we should use measured current for this load
        # (load has been on long enough that we trust the measured value)
        using_measured_current = (
            state.on_since is not None
            and state.on_since
            + timedelta(seconds=config.load_measurement_delay_seconds)
            < now
        )

        if plan.is_on:
            if config.can_throttle and state.is_throttle_rate_limited:
                # If we are unable to throttle due to rate limiting, pre-allocate previous throttle
                will_consume_amps = plan.throttle_amps = previous_plan.throttle_amps

                # Read current throttle value from entity state (not from previous plan)
                if config.throttle_amps_entity:
                    throttle_state = hass.states.get(config.throttle_amps_entity)
                    if throttle_state is not None:
                        try:
                            will_consume_amps = plan.throttle_amps = float(
                                throttle_state.state
                            )
                        except (ValueError, TypeError):
                            _LOGGER.warning(
                                "Unable to read throttle value for %s",
                                config.throttle_amps_entity,
                            )

            elif using_measured_current and not config.can_throttle:
                will_consume_amps = state.current_load_amps  # Track actual consumption
            else:
                # Allocate minimum load, regardless of throttling
                will_consume_amps = plan.throttle_amps = (
                    config.min_controllable_load_amps
                )
        else:
            will_consume_amps = 0.0

        available_amps -= will_consume_amps  # Allocate power for this load
        plan.expected_load_amps = will_consume_amps
        plan.using_measured_current = using_measured_current

    # Second pass to allocate any remaining available power by throttling loads up from their minimum
    if available_amps > 0:
        for load_name in prioritised_loads:
            config = CONFIG.controllable_loads[load_name]
            state = STATE.controllable_loads[load_name]
            previous_plan = PLAN.controllable_loads[load_name]
            plan = new_plan.controllable_loads[load_name]

            # Skip non-throttleable loads and loads that are off
            if not config.can_throttle or not plan.is_on or not state.is_on:
                continue

            if state.is_throttle_rate_limited:
                _LOGGER.debug(
                    "Skipping throttling load %s due to rate limit at %gA",
                    load_name,
                    plan.throttle_amps,
                )
                continue

            # First, give back any power we had previously allocated
            available_amps += plan.expected_load_amps

            # Give the load as much power as we can, accounting for what's currently allocated
            will_consume_amps = min(
                available_amps,
                config.max_controllable_load_amps,
            )
            will_consume_amps = max(
                math.floor(will_consume_amps), config.min_controllable_load_amps
            )
            plan.throttle_amps = plan.expected_load_amps = will_consume_amps
            available_amps -= will_consume_amps

            _LOGGER.debug(
                "Planning to throttle load %s to %gA", load_name, will_consume_amps
            )

    # Third pass to immediately cut loads if we are overloaded
    overload = False
    if (
        STATE.house_consumption_amps
        >= max_safe_total_load_amps + CONFIG.safety_margin_amps
        and max_safe_total_load_amps > 0
    ):
        if STATE.overload_timestamp is None:
            STATE.overload_timestamp = now
        if now >= STATE.overload_timestamp + timedelta(
            seconds=CONFIG.recalculate_interval_seconds
        ):
            overload = True
            _LOGGER.warning(
                "Overload detected (consumption: %gA, max: %gA, available: %gA), cutting loads in reverse priority",
                STATE.house_consumption_amps,
                max_safe_total_load_amps,
                available_amps,
            )
            for load_name in reversed(prioritised_loads):
                plan = new_plan.controllable_loads[load_name]
                state = STATE.controllable_loads[load_name]
                if not plan.is_on or not state.is_under_load_control:
                    continue  # Load will already be off or out of our control

                plan.is_on = False
                available_amps += plan.expected_load_amps
                plan.expected_load_amps = 0.0
                plan.throttle_amps = 0.0
                _LOGGER.info("Cutting load %s to reduce overload", load_name)

                # Check if we are still overloaded
                if available_amps >= 0:
                    break
    else:
        STATE.overload_timestamp = None

    # Update overload binary sensor
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "overload_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["overload_sensor"].update_state(
            overload
        )

    # Final pass to summarise plan
    new_plan.available_amps = available_amps
    for load_name in prioritised_loads:
        plan = new_plan.controllable_loads[load_name]
        new_plan.used_amps += plan.expected_load_amps
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "controlled_load_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["controlled_load_sensor"].update_value(
            new_plan.used_amps
        )

    _LOGGER.debug(
        "Planning complete: available: %gA, allocated: %gA",
        new_plan.available_amps,
        new_plan.used_amps,
    )
    for load_name in prioritised_loads:
        plan = new_plan.controllable_loads[load_name]
        state = STATE.controllable_loads[load_name]
        if plan.is_on:
            _LOGGER.debug(
                "Allocated %gA to load %s (measured: %s, measured current: %gA)",
                plan.expected_load_amps,
                load_name,
                "yes" if plan.using_measured_current else "no",
                state.current_load_amps,
            )

    await execute_plan(hass, new_plan, entry_id)


async def execute_plan(hass: HomeAssistant, plan: PlanState, entry_id: str):
    """Changes entity states to achieve load control plan."""
    now = datetime.now()

    for load_name in plan.controllable_loads:  # pylint: disable=consider-using-dict-items
        config = CONFIG.controllable_loads[load_name]
        state = STATE.controllable_loads[load_name]
        previous_plan = PLAN.controllable_loads[load_name]
        new_plan = plan.controllable_loads[load_name]

        # Turn on or off load only when we need to
        if not config.switch_entity:
            _LOGGER.error(
                "Switch entity not configured for load %s, skipping control",
                load_name,
            )
            await safety_abort(hass, entry_id, True)
            return

        switch_domain = parse_entity_domain(config.switch_entity)

        # Check if entity exists before attempting service calls
        if hass.states.get(config.switch_entity) is None:
            _LOGGER.error(
                "Switch entity %s does not exist, skipping control",
                config.switch_entity,
            )
            continue  # Skip this load and continue with the next one

        if new_plan.is_on and not state.is_on:
            _LOGGER.info("Turning on load %s", config.switch_entity)
            state.last_toggled = now
            try:
                # Use domain-specific service calls for better compatibility
                service_name = "turn_on"
                await hass.services.async_call(
                    switch_domain,
                    service_name,
                    {"entity_id": config.switch_entity},
                    blocking=True,  # Wait for completion to ensure success
                )
                state.is_under_load_control = True
                state.on_since = now  # Track when load was turned on
            except (ValueError, KeyError, RuntimeError, ServiceValidationError) as err:
                _LOGGER.error("Failed to turn on %s: %s", config.switch_entity, err)

        elif not new_plan.is_on and state.is_on:
            _LOGGER.info("Turning off load %s", config.switch_entity)
            state.last_toggled = now
            try:
                # Use domain-specific service calls for better compatibility
                service_name = "turn_off"
                await hass.services.async_call(
                    switch_domain,
                    service_name,
                    {"entity_id": config.switch_entity},
                    blocking=True,  # Wait for completion to ensure success
                )
                state.is_under_load_control = False
                state.on_since = None  # Clear on_since when turned off
            except (ValueError, KeyError, RuntimeError, ServiceValidationError) as err:
                _LOGGER.error("Failed to turn off %s: %s", config.switch_entity, err)

        if (
            config.can_throttle
            and new_plan.is_on
            and config.throttle_amps_entity
            and state.is_under_load_control
        ):
            # Check if throttle entity exists
            throttle_state = hass.states.get(config.throttle_amps_entity)
            if throttle_state is None:
                _LOGGER.error(
                    "Throttle entity %s does not exist, skipping throttling",
                    config.throttle_amps_entity,
                )
            else:
                # Read current value from the entity state
                try:
                    current_throttle_amps = float(throttle_state.state)
                except (ValueError, TypeError):
                    _LOGGER.warning(
                        "Unable to read current throttle value for %s, using previous plan value",
                        config.throttle_amps_entity,
                    )
                    current_throttle_amps = previous_plan.throttle_amps

                throttle_amps_delta = abs(
                    round(new_plan.throttle_amps) - round(current_throttle_amps)
                )
                if throttle_amps_delta > 0:
                    _LOGGER.info(
                        "Throttling load %s to %gA",
                        config.throttle_amps_entity,
                        round(new_plan.throttle_amps),
                    )
                    state.last_throttled = now
                    number_domain = parse_entity_domain(config.throttle_amps_entity)
                    try:
                        # Use domain-specific service for number entities
                        service_name = "set_value"
                        service_data = {
                            "entity_id": config.throttle_amps_entity,
                            "value": new_plan.throttle_amps,  # Don't convert to string
                        }
                        await hass.services.async_call(
                            number_domain,
                            service_name,
                            service_data,
                            blocking=True,  # Wait for completion to ensure success
                        )
                    except (
                        ValueError,
                        KeyError,
                        RuntimeError,
                        ServiceValidationError,
                    ) as err:
                        _LOGGER.error(
                            "Failed to throttle %s: %s",
                            config.throttle_amps_entity,
                            err,
                        )

    # Deep copy the controllable loads to avoid reference sharing between PLAN and new_plan
    PLAN.available_amps = plan.available_amps
    PLAN.used_amps = plan.used_amps
    PLAN.controllable_loads = {}
    for load_name, load_plan in plan.controllable_loads.items():
        PLAN.controllable_loads[load_name] = ControllableLoadPlanState()
        PLAN.controllable_loads[load_name].is_on = load_plan.is_on
        PLAN.controllable_loads[
            load_name
        ].expected_load_amps = load_plan.expected_load_amps
        PLAN.controllable_loads[load_name].throttle_amps = load_plan.throttle_amps

    _LOGGER.debug("Plan execution completed for %d loads", len(plan.controllable_loads))


def clear_safety_abort(hass: HomeAssistant, entry_id: str):
    """Clear safety abort state if system has recovered."""

    state = hass.data[DOMAIN][entry_id]["state"]
    if state.safety_abort_active:
        state.safety_abort_timestamp = None
        state.safety_abort_active = False
        if (
            entry_id in hass.data.get(DOMAIN, {})
            and "entities" in hass.data[DOMAIN][entry_id]
            and "safety_abort_sensor" in hass.data[DOMAIN][entry_id]["entities"]
        ):
            hass.data[DOMAIN][entry_id]["entities"]["safety_abort_sensor"].update_state(
                False
            )
        _LOGGER.error("Safety abort cleared for entry %s", entry_id)


async def safety_abort(hass: HomeAssistant, entry_id: str, force: bool = False):
    """Cuts all load controlled by the integration in a safety situation."""

    # Get per-entry config/state/plan
    config = hass.data[DOMAIN][entry_id]["config"]
    state = hass.data[DOMAIN][entry_id]["state"]
    plan = hass.data[DOMAIN][entry_id]["plan"]

    # Skip abort if state is not initialised yet
    if not state.house_consumption_initialised:
        return

    # Skip abort if load control is disabled
    if not state.enable_load_control and not force:
        return

    # Skip abort if already in safety abort
    if state.safety_abort_active:
        return

    # Skip abort if safety abort has not been active for long enough
    now = datetime.now()
    if state.safety_abort_timestamp is None:
        state.safety_abort_timestamp = now
    if now < state.safety_abort_timestamp + timedelta(seconds=120) and not force:
        return
    _LOGGER.error(
        "Aborting load control for safety, cutting all loads for entry %s", entry_id
    )

    # Update safety abort binary sensor
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "safety_abort_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["safety_abort_sensor"].update_state(
            True
        )
    state.safety_abort_active = True

    plan.available_amps = 0.0
    plan.used_amps = 0.0
    for load_name in config.controllable_loads:
        try:
            lconfig = config.controllable_loads[load_name]
            switch_domain = parse_entity_domain(lconfig.switch_entity)
            service_name = "turn_off"
            await hass.services.async_call(
                switch_domain,
                service_name,
                {"entity_id": lconfig.switch_entity},
                blocking=True,
            )

            lstate = state.controllable_loads[load_name]
            lstate.is_on = False
            lstate.is_under_load_control = False
            lstate.last_toggled = datetime.now()
            lstate.last_throttled = datetime.now()

            load_plan = plan.controllable_loads[load_name] = ControllableLoadPlanState()
            load_plan.is_on = False
            load_plan.expected_load_amps = 0.0
            load_plan.throttle_amps = 0.0
            _LOGGER.info("Turned off load %s for safety", config.switch_entity)
        except (ValueError, KeyError, RuntimeError) as err:
            _LOGGER.error("Failed to turn off %s: %s", config.switch_entity, err)
