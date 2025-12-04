"""The ZeroGrid integration."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

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

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR, Platform.SWITCH]

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
        _LOGGER.info("Manual recalculation triggered via service call")
        await recalculate_load_control(hass)

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
    plan = PLANS[entry.entry_id]
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
                await recalculate_load_control(hass, entry.entry_id)
            else:
                _LOGGER.error(
                    "House consumption entity is unavailable, cutting all load for safety"
                )
                await safety_abort(hass, entry.entry_id)
                return
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
    CONFIG.hysteresis_amps = domain_config.get("hysteresis_amps", 1.0)
    CONFIG.recalculate_interval_seconds = domain_config.get(
        "recalculate_interval_seconds", 10
    )
    CONFIG.load_measurement_delay_seconds = domain_config.get(
        "load_measurement_delay_seconds", 120
    )
    CONFIG.enable_automatic_recalculation = domain_config.get(
        "enable_automatic_recalculation", True
    )
    CONFIG.house_consumption_amps_entity = domain_config.get(
        "house_consumption_amps_entity", None
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
        control_config.load_amps_entity = control.get("load_amps_entity")
        control_config.switch_entity = control.get("switch_entity")
        control_config.throttle_amps_entity = control.get("throttle_amps_entity", None)
        control_config.can_throttle = control_config.throttle_amps_entity is not None
        control_config.can_turn_on_entity = control.get("can_turn_on_entity", None)
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


def _subscribe_to_entity_changes_DEPRECATED(hass: HomeAssistant):
    """DEPRECATED: This function has been replaced by per-entry listeners in async_setup_entry.

    Subscribes to required entity changes.
    """
    # This function is no longer used - listeners are now created per-entry in async_setup_entry


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

    # Determine max window duration based on longest min_toggle_interval
    max_window_seconds = max(
        (
            lconfig.min_toggle_interval_seconds
            for lconfig in config.controllable_loads.values()
        ),
        default=60,  # Default to 60 seconds if no controllable loads configured
    )

    # Accumulate the total unallocated power (before subtracting loads) into history
    state.accumulate_unallocated_amps(total_available_power_amps, max_window_seconds)

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
                time_since_on = (now - load_state.on_since).total_seconds()
                if time_since_on < CONFIG.load_measurement_delay_seconds:
                    current_load = load_plan.expected_load_amps

            total_load_not_under_control -= current_load
            _LOGGER.debug(
                "Load %s drawing %gA",
                load_name,
                current_load,
            )

    # Now calculate total available for load control by subtracting non-controlled loads
    total_available_amps = total_available_power_amps - max(
        total_load_not_under_control, 0
    )

    _LOGGER.debug(
        "Total available power: %gA, uncontrolled load: %gA, available for load control: %gA",
        total_available_power_amps,
        total_load_not_under_control,
        total_available_amps,
    )

    # Update entities instead of setting state directly
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "available_load_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["available_load_sensor"].update_value(
            total_available_amps
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

    # Clear safety abort state if system has recovered
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "safety_abort_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["safety_abort_sensor"].update_state(
            False
        )

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

        # Determine if this load should be on
        # Check both available power and external constraints
        should_be_on = (
            available_amps >= config.min_controllable_load_amps and state.can_turn_on
        )

        # Log if external constraint is preventing turn on
        if (
            available_amps >= config.min_controllable_load_amps
            and not state.can_turn_on
        ):
            _LOGGER.debug(
                "Load %s has sufficient power but external constraint prevents turn on",
                load_name,
            )

        # Make sure we have a stable minimum available power before turning on (important for solar)
        min_unallocated_amps = STATE.get_minimum_unallocated_amps(
            config.min_toggle_interval_seconds
        )
        if (
            should_be_on
            and not state.is_on
            and not previous_plan.is_on
            and not STATE.allow_grid_import
        ):
            if min_unallocated_amps < config.min_controllable_load_amps:
                _LOGGER.debug(
                    "Preventing load %s turn on due to insufficient minimum capacity of %gA over last %ds",
                    load_name,
                    min_unallocated_amps,
                    config.min_toggle_interval_seconds,
                )
                should_be_on = False

        if state.is_switch_rate_limited:
            if should_be_on != previous_plan.is_on or state.is_on:
                _LOGGER.debug(
                    "Unable to toggle load %s due to switch rate limit", load_name
                )
            plan.is_on = previous_plan.is_on or state.is_on
        else:
            plan.is_on = should_be_on
            if plan.is_on != previous_plan.is_on:
                if plan.is_on:
                    _LOGGER.debug("Planning to turn load %s on", load_name)
                else:
                    _LOGGER.debug("Planning to turn load %s off", load_name)

        if plan.is_on:
            if config.can_throttle and state.is_throttle_rate_limited:
                # If we are unable to throttle due to rate limiting, pre-allocate previous throttle
                will_consume_amps = plan.throttle_amps = previous_plan.throttle_amps
            else:
                # Allocate minimum load, regardless of throttling
                will_consume_amps = plan.throttle_amps = (
                    config.min_controllable_load_amps
                )
        else:
            will_consume_amps = 0.0

        # For loads that have been on long enough, use measured current instead of allocation
        # These loads are already subtracted in calculate_effective_available_power,
        # so we add back the allocation and don't subtract the measured value
        if (
            plan.is_on
            and state.on_since is not None
            and state.on_since
            + timedelta(seconds=CONFIG.load_measurement_delay_seconds)
            < now
        ):
            available_amps += will_consume_amps  # Free up the allocation
            will_consume_amps = state.current_load_amps  # Track actual consumption
        else:
            available_amps -= will_consume_amps  # Reserve power for this load

        plan.expected_load_amps = will_consume_amps

    # Second pass to allocate any remaining available power by throttling loads up from their minimum
    if available_amps > 0:
        for load_name in prioritised_loads:
            config = CONFIG.controllable_loads[load_name]
            state = STATE.controllable_loads[load_name]
            previous_plan = PLAN.controllable_loads[load_name]
            plan = new_plan.controllable_loads[load_name]

            # Skip non-throttleable loads and loads that are off
            if not config.can_throttle or not plan.is_on:
                continue

            if state.is_throttle_rate_limited:
                _LOGGER.debug(
                    "Skipping throttling load %s due to rate limit at %gA",
                    load_name,
                    plan.throttle_amps,
                )
                continue

            previously_allocated_amps = plan.expected_load_amps

            will_consume_amps = 0.0

            # Give the load as much power as we can, accounting for what was previously allocated
            will_consume_amps = min(
                available_amps + previously_allocated_amps,
                config.max_controllable_load_amps,
            )
            will_consume_amps = round(will_consume_amps)
            plan.throttle_amps = will_consume_amps

            available_amps += previously_allocated_amps - will_consume_amps
            plan.expected_load_amps = will_consume_amps
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
        if STATE.last_overload is None:
            STATE.last_overload = now
        if now >= STATE.last_overload + timedelta(
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
        STATE.last_overload = None

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
        "Planning complete: available=%gA, used=%gA",
        new_plan.available_amps,
        new_plan.used_amps,
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
            await safety_abort(hass, entry_id)
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
            except (ValueError, KeyError, RuntimeError, ServiceValidationError) as err:
                _LOGGER.error("Failed to turn off %s: %s", config.switch_entity, err)

                # If we fail to turn off a load, abort all load control for safety
                await safety_abort(hass, entry_id)
                return

        if (
            config.can_throttle
            and new_plan.is_on
            and config.throttle_amps_entity
            and state.is_under_load_control
        ):
            # Check if throttle entity exists
            if hass.states.get(config.throttle_amps_entity) is None:
                _LOGGER.error(
                    "Throttle entity %s does not exist, skipping throttling",
                    config.throttle_amps_entity,
                )
            else:
                throttle_amps_delta = abs(
                    new_plan.throttle_amps - previous_plan.throttle_amps
                )
                if throttle_amps_delta > CONFIG.hysteresis_amps:
                    _LOGGER.info(
                        "Throttling load %s to %gA",
                        config.throttle_amps_entity,
                        new_plan.throttle_amps,
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


async def safety_abort(hass: HomeAssistant, entry_id: str):
    """Cuts all load controlled by the integration in a safety situation."""
    _LOGGER.error("Aborting load control, cutting all loads for entry %s", entry_id)

    # Get per-entry config/state/plan
    config = hass.data[DOMAIN][entry_id]["config"]
    state = hass.data[DOMAIN][entry_id]["state"]
    plan = hass.data[DOMAIN][entry_id]["plan"]

    # Update safety abort binary sensor
    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "safety_abort_sensor" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["safety_abort_sensor"].update_state(
            True
        )

    if (
        entry_id in hass.data.get(DOMAIN, {})
        and "entities" in hass.data[DOMAIN][entry_id]
        and "enable_load_control" in hass.data[DOMAIN][entry_id]["entities"]
    ):
        hass.data[DOMAIN][entry_id]["entities"]["enable_load_control"].update_value(
            False
        )

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
