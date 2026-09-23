"""The ZeroGrid integration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
import math

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_OFF, STATE_ON, Platform
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
from .helpers import is_entity_usable, parse_amps, parse_entity_domain
from .state import ControllableLoadPlanState, ControllableLoadState, PlanState, State

_LOGGER = logging.getLogger(__name__)

# How long to let a load report the state we asked it for before the command is
# treated as lost and sent again.
SWITCH_COMMAND_TIMEOUT_SECONDS = 30

# How long to wait for a load to accept a command before counting it as failed.
# A cloud-controlled load can otherwise hold a recalculation up for as long as
# its API takes to give up.
COMMAND_CALL_TIMEOUT_SECONDS = 15

# How long to leave a load alone after each consecutive failed command. A load
# whose API is down would otherwise be sent a command on every recalculation.
COMMAND_BACKOFF_SECONDS = (30, 60, 120, 300, 600)

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

    # Create per-entry instances. Nothing is published to module level: the
    # planning code reads the entry it was given, so a second config entry
    # cannot be planned against the first entry's config and state.
    config = CONFIGS[entry.entry_id] = Config()
    state = STATES[entry.entry_id] = State()
    plan = PLANS[entry.entry_id] = PlanState()

    # Merge config entry data with options (options take precedence)
    config_data = {**entry.data, **entry.options}

    # Parse configuration from config entry
    parse_config(config, config_data)
    initialise_state(hass, config, state, plan)

    # Set up per-entry entity change listeners
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
            # A reading we cannot make a number of is no more use than no
            # reading at all, so it takes the same path.
            house_amps = parse_amps(new_state)
            if house_amps is not None:
                state.house_consumption_amps = house_amps
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
            solar_amps = parse_amps(new_state)
            if solar_amps is not None:
                state.solar_generation_amps = solar_amps
                await recalculate_load_control(hass, entry.entry_id)
            else:
                state.solar_generation_amps = 0.0

        else:
            # Check if it's a controllable load entity
            for load_config in config.controllable_loads.values():
                load = state.controllable_loads[load_config.name]

                if entity_id == load_config.switch_entity:
                    apply_switch_state(load_config, load, new_state)
                elif entity_id == load_config.load_amps_entity:
                    load_amps = parse_amps(new_state)
                    if load_amps is not None:
                        load.current_load_amps = load_amps
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
                        # Keep previous state.can_turn_on value
                        _LOGGER.debug(
                            "Load %s can_turn_on entity unavailable, keeping previous state",
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


def apply_switch_state(
    config: ControllableLoadConfig, load: ControllableLoadState, new_state
) -> None:
    """Take a load's switch state from its entity.

    A climate or humidifier entity reports its mode rather than on/off, so
    anything but "off" counts as on.
    """
    if not is_entity_usable(new_state):
        return

    load.is_on = new_state.state != STATE_OFF

    # If option enabled, assume load is under control when on
    if load.is_on and config.assume_always_under_load_control:
        load.is_under_load_control = True

    if load.is_on:
        # Start the measurement delay from the first time we see the load on.
        # A switch entity that was still unavailable when this integration set
        # up - a thermostat on a reload, say - is never turned on by us, so
        # nothing else would ever set this. Its meter would then stay untrusted
        # for as long as it ran, holding it at its configured minimum and
        # keeping that capacity from the loads below it.
        if load.on_since is None:
            load.on_since = datetime.now()
    else:
        load.on_since = None

    _LOGGER.debug("Load %s switch changed to %s", config.name, new_state.state)


def parse_config(config: Config, domain_config) -> None:
    """Parse the config entry data into the given Config."""
    _LOGGER.debug(domain_config)

    config.max_total_load_amps = domain_config.get("max_total_load_amps", 0)
    config.max_grid_import_amps = domain_config.get("max_grid_import_amps", 0)
    config.max_solar_generation_amps = domain_config.get("max_solar_generation_amps", 0)
    # Sanity check - total load cannot exceed grid import + solar generation
    config.max_total_load_amps = min(
        config.max_total_load_amps,
        config.max_grid_import_amps + config.max_solar_generation_amps,
    )

    config.safety_margin_amps = domain_config.get("safety_margin_amps", 2.0)
    config.recalculate_interval_seconds = domain_config.get(
        "recalculate_interval_seconds", 10
    )
    config.enable_automatic_recalculation = domain_config.get(
        "enable_automatic_recalculation", True
    )
    config.house_consumption_amps_entity = domain_config.get(
        "house_consumption_amps_entity", None
    )
    config.disable_consumption_unavailable_safety_abort = domain_config.get(
        "disable_consumption_unavailable_safety_abort", False
    )

    config.solar_generation_amps_entity = domain_config.get(
        "solar_generation_amps_entity", None
    )
    config.allow_solar_consumption = config.solar_generation_amps_entity is not None

    control_options = domain_config.get("controllable_loads", [])
    # Rebind rather than mutate: the attribute is declared on the class, so
    # every Config would otherwise share one dict, and reparsing after a load
    # was removed in the options would leave the removed load behind.
    config.controllable_loads = {}
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
        # These are durations, and are fed straight to timedelta by the rate
        # limit checks, so fall back to the class defaults rather than storing
        # None when a load is configured without them.
        control_config.min_toggle_interval_seconds = control.get(
            "min_toggle_interval_seconds",
            ControllableLoadConfig.min_toggle_interval_seconds,
        )
        if control_config.min_toggle_interval_seconds is None:
            control_config.min_toggle_interval_seconds = (
                ControllableLoadConfig.min_toggle_interval_seconds
            )
        control_config.min_throttle_interval_seconds = control.get(
            "min_throttle_interval_seconds",
            ControllableLoadConfig.min_throttle_interval_seconds,
        )
        if control_config.min_throttle_interval_seconds is None:
            control_config.min_throttle_interval_seconds = (
                ControllableLoadConfig.min_throttle_interval_seconds
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
        config.controllable_loads[control_config.name] = control_config

    _LOGGER.debug("Config successful: %s", config)


def initialise_state(
    hass: HomeAssistant, config: Config, state: State, plan: PlanState
) -> None:
    """Initialise the state of one config entry from the current entity states."""
    if config.house_consumption_amps_entity is not None:
        house_amps = parse_amps(hass.states.get(config.house_consumption_amps_entity))
        if house_amps is not None:
            state.house_consumption_amps = house_amps

    if config.solar_generation_amps_entity is not None:
        solar_amps = parse_amps(hass.states.get(config.solar_generation_amps_entity))
        state.solar_generation_amps = solar_amps if solar_amps is not None else 0.0
    else:
        state.solar_generation_amps = 0.0

    # Don't initialize switch states here - they will be initialized by the switch
    # entities themselves when they restore their state in async_added_to_hass.
    # The switches will update state.allow_grid_import and state.enable_load_control.

    # match to controllable loads
    for load_name in config.controllable_loads:  # pylint: disable=consider-using-dict-items
        load_config = config.controllable_loads[load_name]
        load_state = ControllableLoadState()

        switch_state = hass.states.get(load_config.switch_entity)
        if is_entity_usable(switch_state):
            # A climate entity's state is its HVAC mode (heat, cool, ...) and is
            # never "on", so treat anything but "off" as on. This matches how
            # state_automation_listener interprets later state changes.
            load_state.is_on = switch_state.state != STATE_OFF
            # If option enabled, assume load is under control when on
            if load_config.assume_always_under_load_control:
                load_state.is_under_load_control = load_state.is_on
            else:
                load_state.is_under_load_control = (
                    load_state.is_on
                )  # Assume under control initially
            if load_state.is_on and load_state.is_under_load_control:
                load_state.on_since = datetime.now()
            else:
                load_state.on_since = None

        load_amps = parse_amps(hass.states.get(load_config.load_amps_entity))
        if load_amps is not None:
            load_state.current_load_amps = load_amps

        # Initialize can_turn_on state from entity if configured
        if load_config.can_turn_on_entity is not None:
            can_turn_on_state = hass.states.get(load_config.can_turn_on_entity)
            if can_turn_on_state is not None and can_turn_on_state.state not in (
                "unknown",
                "unavailable",
            ):
                load_state.can_turn_on = can_turn_on_state.state == STATE_ON
            elif load_config.can_turn_on_ignore_unavailable:
                load_state.can_turn_on = True  # Ignore unavailable, allow turn on
            else:
                load_state.can_turn_on = False  # Default to safe state
        else:
            load_state.can_turn_on = True  # No constraint configured

        state.controllable_loads[load_name] = load_state
        plan.controllable_loads[load_name] = ControllableLoadPlanState()
        _LOGGER.debug(
            "Switch entity init for %s: %s",
            load_name,
            plan.controllable_loads[load_name].is_on,
        )

    _LOGGER.debug("Initialised state: %s", state)


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
    max_available_amps = grid_maximum_amps + capped_solar_generation_amps
    # Safety cap to maximum total available load
    max_available_amps = min(max_available_amps, config.max_total_load_amps)

    # Subtract loads that are under load control, since we can manage those
    total_load_not_under_control = state.house_consumption_amps
    # Whether this figure is solid enough to remember as a reference, and
    # whether the readings behind it describe the same instant at all.
    trust_as_reference = True
    readings_skewed = False
    for load_name in state.controllable_loads:
        load_state = state.controllable_loads[load_name]
        load_plan = plan.controllable_loads.get(load_name)

        if load_state.is_under_load_control and load_state.is_on:
            load_config = config.controllable_loads[load_name]
            if is_load_unresponsive(load_state, now):
                # It will not follow the plan, so its draw stays in the
                # uncontrolled figure, along with whatever it could still rise
                # to. That takes it off the top before priorities are applied,
                # so every other load makes room for it.
                total_load_not_under_control += (
                    unresponsive_load_amps(hass, load_config, load_state)
                    - load_state.current_load_amps
                )
                continue
            current_load = load_state.current_load_amps
            # Determine if we should use expected load instead of measured load
            # to account for soft starts and measurement delays
            if load_state.on_since is not None and load_plan is not None:
                time_since_on = (now - load_state.on_since).total_seconds()
                if time_since_on < load_config.load_measurement_delay_seconds:
                    current_load = load_plan.expected_load_amps
                    # Planned rather than measured, so do not keep it as the
                    # reference the skew handling below falls back to.
                    trust_as_reference = False
            # A throttle change takes a moment to reach the load and longer to
            # show up on its meter. Until it does, this load's reading and the
            # house reading describe different instants.
            if load_state.last_throttled is not None:
                time_since_throttled = (
                    now - load_state.last_throttled
                ).total_seconds()
                if time_since_throttled < load_config.min_throttle_interval_seconds:
                    readings_skewed = True
            total_load_not_under_control -= current_load

    # The uncontrolled figure is the house meter minus the loads' own meters,
    # which are not read at the same instant. While a load is mid-change that
    # difference is not meaningful, and acting on it is what shed a load that
    # only needed throttling: a load that had already resumed drawing but whose
    # meter still read low was counted as uncontrolled load we could not manage.
    # A negative result is the same skew seen from the other side - the loads
    # cannot draw more than the whole house - and clamping it to zero advertised
    # headroom that did not exist, which pushed a throttle up just before the
    # meter caught up. In both cases hold the last settled figure instead. A
    # genuine overload is still caught straight from the house meter by the
    # overload pass in plan_loads, which does not use this figure.
    if total_load_not_under_control < 0:
        readings_skewed = True

    if readings_skewed and state.last_settled_uncontrolled_amps is not None:
        _LOGGER.debug(
            "Load measurements unsettled (derived uncontrolled load: %gA), holding last settled value of %gA",
            total_load_not_under_control,
            state.last_settled_uncontrolled_amps,
        )
        total_load_not_under_control = state.last_settled_uncontrolled_amps
    elif not readings_skewed and trust_as_reference:
        state.last_settled_uncontrolled_amps = total_load_not_under_control

    total_load_not_under_control = max(total_load_not_under_control, 0)

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
    # and reserved current independently — reserved current limits available power on top of
    # whatever is already consumed by loads not under our control
    total_available_amps = (
        max_available_amps - total_load_not_under_control - state.reserved_current_amps
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
        max_available_amps,
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
    state.last_settled_uncontrolled_amps = None
    for control in config.controllable_loads.values():
        load = state.controllable_loads[control.name]
        load.is_under_load_control = True
        load.last_throttled = None
        load.last_toggled = None
        load.switch_command_on = None
        load.switch_command_since = None


def read_current_throttle_amps(
    hass: HomeAssistant,
    config: ControllableLoadConfig,
    fallback_amps: float,
) -> float:
    """Read the throttle setpoint a load is currently sitting at."""
    if not config.throttle_amps_entity:
        return fallback_amps

    throttle_state = hass.states.get(config.throttle_amps_entity)
    if throttle_state is None:
        return fallback_amps

    try:
        return float(throttle_state.state)
    except (ValueError, TypeError):
        _LOGGER.warning(
            "Unable to read throttle value for %s, using previous plan value",
            config.throttle_amps_entity,
        )
        return fallback_amps


def is_load_unresponsive(load: ControllableLoadState, now: datetime) -> bool:
    """Return True while a load is being left alone after a failed command.

    Once the backoff runs out the load rejoins the plan for a cycle, so the
    retry is planned and budgeted like any other command.
    """
    return load.command_retry_after is not None and now < load.command_retry_after


def unresponsive_load_amps(
    hass: HomeAssistant, config: ControllableLoadConfig, load: ControllableLoadState
) -> float:
    """Return what an unresponsive load could draw without being asked.

    Its meter says what it draws now, but nothing stops it from rising to its
    setpoint - a charger resuming after a pause - or to its rating if it cannot
    be throttled, and we could not dial it back if it did.
    """
    if config.can_throttle:
        setpoint = parse_amps(hass.states.get(config.throttle_amps_entity))
        ceiling = (
            setpoint if setpoint is not None else config.max_controllable_load_amps
        )
    else:
        ceiling = config.max_controllable_load_amps
    return max(load.current_load_amps, ceiling)


def update_unresponsive_sensor(
    hass: HomeAssistant, entry_id: str, load_name: str, unresponsive: bool
) -> None:
    """Report a load's unresponsive state on its binary sensor, if it has one."""
    sensors = (
        hass.data.get(DOMAIN, {})
        .get(entry_id, {})
        .get("entities", {})
        .get("unresponsive_sensors", {})
    )
    if load_name in sensors:
        sensors[load_name].update_state(unresponsive)


async def call_load_service(
    hass: HomeAssistant,
    entry_id: str,
    load_name: str,
    domain: str,
    service: str,
    data: dict,
) -> bool:
    """Send a command to a load and record whether it was accepted.

    Any failure is the load's, not ours: a cloud integration raises whatever its
    HTTP library does, and letting that escape stopped the plan from reaching
    the loads after this one. A failure starts a backoff during which the load
    is not sent anything and is budgeted as uncontrolled load.
    """
    state = hass.data[DOMAIN][entry_id]["state"]
    load = state.controllable_loads[load_name]
    try:
        await asyncio.wait_for(
            hass.services.async_call(domain, service, data, blocking=True),
            COMMAND_CALL_TIMEOUT_SECONDS,
        )
    except Exception as err:  # noqa: BLE001 - see docstring
        load.command_failures += 1
        backoff_seconds = COMMAND_BACKOFF_SECONDS[
            min(load.command_failures, len(COMMAND_BACKOFF_SECONDS)) - 1
        ]
        load.command_retry_after = datetime.now() + timedelta(seconds=backoff_seconds)
        # The uncontrolled figure now includes this load, so the last settled
        # one no longer describes the same thing.
        state.last_settled_uncontrolled_amps = None
        _LOGGER.warning(
            "Load %s did not accept %s.%s (%s: %s), treating it as uncontrolled and retrying in %ds",
            load_name,
            domain,
            service,
            type(err).__name__,
            err,
            backoff_seconds,
        )
        update_unresponsive_sensor(hass, entry_id, load_name, True)
        return False

    if load.command_failures > 0:
        _LOGGER.info(
            "Load %s is accepting commands again after %d failures",
            load_name,
            load.command_failures,
        )
        mark_load_responsive(hass, entry_id, load_name)
    return True


def mark_load_responsive(hass: HomeAssistant, entry_id: str, load_name: str) -> None:
    """Clear a load's failed command record."""
    state = hass.data[DOMAIN][entry_id]["state"]
    load = state.controllable_loads[load_name]
    load.command_failures = 0
    load.command_retry_after = None
    state.last_settled_uncontrolled_amps = None
    update_unresponsive_sensor(hass, entry_id, load_name, False)


async def recalculate_load_control(hass: HomeAssistant, entry_id: str):
    """Run one planning cycle, unless the previous one is still running."""
    entry_data = hass.data.get(DOMAIN, {}).get(entry_id)
    if entry_data is None:
        _LOGGER.error("Entry %s not found in hass.data", entry_id)
        return

    state = entry_data["state"]
    if state.recalculating:
        _LOGGER.debug(
            "Recalculation skipped - previous one still running for entry %s",
            entry_id,
        )
        return

    state.recalculating = True
    try:
        await _recalculate_load_control(hass, entry_id)
    finally:
        state.recalculating = False


async def _recalculate_load_control(hass: HomeAssistant, entry_id: str):
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
    plan = hass.data[DOMAIN][entry_id]["plan"]

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
        config.controllable_loads,
        key=lambda k: config.controllable_loads[k].priority,
    )
    _LOGGER.debug("Priority: %s", prioritised_loads)

    # First pass to determine if loads should be on or not
    for load_name in prioritised_loads:
        load_config = config.controllable_loads[load_name]
        load_state = state.controllable_loads[load_name]
        previous_plan = plan.controllable_loads[load_name]

        load_plan = new_plan.controllable_loads[load_name] = ControllableLoadPlanState()
        load_plan.is_on = previous_plan.is_on
        load_plan.expected_load_amps = 0.0

        # Determine if we are rate-limited on switching or throttling
        load_state.is_switch_rate_limited = (
            load_state.last_toggled is not None
            and load_state.last_toggled
            + timedelta(seconds=load_config.min_toggle_interval_seconds)
            > now
        )
        load_state.is_throttle_rate_limited = (
            load_config.can_throttle
            and load_state.last_throttled is not None
            and load_state.last_throttled
            + timedelta(seconds=load_config.min_throttle_interval_seconds)
            > now
        )

        if not load_state.is_under_load_control and load_state.is_on:
            # Load is manually turned on - we have no control
            load_plan.is_on = load_state.is_on
            _LOGGER.debug("Load %s manually turned on, skipping control", load_name)
            continue

        if is_load_unresponsive(load_state, now):
            # Plan it where it is. Its draw is already counted as uncontrolled
            # load, so it takes no share of what is left.
            load_plan.is_on = load_state.is_on
            if load_config.can_throttle:
                # Its integration is likely unavailable too, so fall back
                # quietly rather than warning on every cycle.
                setpoint = parse_amps(
                    hass.states.get(load_config.throttle_amps_entity)
                )
                load_plan.throttle_amps = load_plan.current_throttle_amps = (
                    setpoint if setpoint is not None else previous_plan.throttle_amps
                )
            _LOGGER.debug(
                "Load %s is not accepting commands, leaving it as it is until %s",
                load_name,
                load_state.command_retry_after,
            )
            continue

        will_consume_amps = 0.0

        # Determine if this load should be on based on available power
        should_be_on = available_amps >= load_config.min_controllable_load_amps

        if not state.allow_grid_import:
            # Make sure we have a stable minimum available power before turning on (important for solar)
            if should_be_on and not load_state.is_on and not previous_plan.is_on:
                min_available_amps = state.get_minimum_available_amps(
                    load_config.solar_turn_on_window_seconds
                )
                if min_available_amps < load_config.min_controllable_load_amps:
                    should_be_on = False
                    _LOGGER.debug(
                        "Preventing load %s turn on due to insufficient minimum capacity of %gA over last %ds",
                        load_name,
                        min_available_amps,
                        load_config.solar_turn_on_window_seconds,
                    )
            # Prevent turning off loads early if available power is low for short periods
            elif load_state.is_on and load_state.is_under_load_control and not should_be_on:
                average_available_amps = state.get_average_available_amps(
                    load_config.solar_turn_off_window_seconds
                )
                if average_available_amps > load_config.min_controllable_load_amps:
                    should_be_on = True
                    _LOGGER.debug(
                        "Preventing load %s turn off due to average capacity of %gA over last %ds",
                        load_name,
                        average_available_amps,
                        load_config.solar_turn_off_window_seconds,
                    )

        # Check external constraint (can_turn_on_entity)
        if load_config.can_turn_on_entity is not None:
            can_turn_on_entity_state = hass.states.get(load_config.can_turn_on_entity)
            is_unavailable = (
                can_turn_on_entity_state is None
                or can_turn_on_entity_state.state in ("unknown", "unavailable")
            )

            if is_unavailable:
                if not load_config.can_turn_on_ignore_unavailable:
                    load_state.can_turn_on = False
                # else: keep previous state.can_turn_on value
            else:
                load_state.can_turn_on = can_turn_on_entity_state.state == "on"

            if should_be_on and not load_state.can_turn_on:
                should_be_on = False
                _LOGGER.debug(
                    "Load %s has sufficient power but external constraint prevents turn on",
                    load_name,
                )

        # Prevent toggling if rate limited
        if load_state.is_switch_rate_limited:
            if should_be_on != (previous_plan.is_on or load_state.is_on):
                if should_be_on:
                    _LOGGER.debug(
                        "Unable to turn load %s on due to switch rate limit", load_name
                    )
                else:
                    _LOGGER.debug(
                        "Unable to turn load %s off due to switch rate limit", load_name
                    )
            load_plan.is_on = previous_plan.is_on or load_state.is_on
        else:
            load_plan.is_on = should_be_on
            if load_plan.is_on != previous_plan.is_on:
                if load_plan.is_on:
                    _LOGGER.debug("Planning to turn load %s on", load_name)
                else:
                    _LOGGER.debug("Planning to turn load %s off", load_name)

        # Determine if we should use measured current for this load
        # (load has been on long enough that we trust the measured value)
        using_measured_current = (
            load_state.on_since is not None
            and load_state.on_since
            + timedelta(seconds=load_config.load_measurement_delay_seconds)
            < now
        )

        if load_plan.is_on:
            if load_config.can_throttle:
                # Record where the load is actually sitting so the second pass
                # can tell an upward move (rate limited) from a downward one.
                load_plan.current_throttle_amps = read_current_throttle_amps(
                    hass, load_config, previous_plan.throttle_amps
                )
                # Reserve only the minimum for a throttleable load, even while it
                # is throttle rate limited. Reserving what it is currently drawing
                # makes it look expensive during a spike, which shed lower-priority
                # fixed loads instead of simply dialling this load back. The second
                # pass decides the real setpoint.
                will_consume_amps = load_plan.throttle_amps = (
                    load_config.min_controllable_load_amps
                )
            elif using_measured_current:
                # Track actual consumption. A thermostatic load sits on with its
                # element cycled off for long stretches, and reserving its rated
                # draw the whole time starves the loads below it. The window
                # where its meter cannot be trusted is already covered: until
                # load_measurement_delay_seconds has passed, using_measured_current
                # is false and the branch below reserves the load's minimum.
                will_consume_amps = load_state.current_load_amps
            else:
                # Allocate minimum load, regardless of throttling
                will_consume_amps = load_plan.throttle_amps = (
                    load_config.min_controllable_load_amps
                )
        else:
            will_consume_amps = 0.0

        available_amps -= will_consume_amps  # Allocate power for this load
        load_plan.expected_load_amps = will_consume_amps
        load_plan.using_measured_current = using_measured_current

    # Second pass to set the throttle setpoint of each throttleable load from the
    # power left over after the first pass. This runs even when there is nothing
    # left over: a deficit is exactly when a throttleable load has to be dialled
    # back, and skipping the pass left it sitting at its old setpoint.
    for load_name in prioritised_loads:
        load_config = config.controllable_loads[load_name]
        load_state = state.controllable_loads[load_name]
        load_plan = new_plan.controllable_loads[load_name]

        # Skip non-throttleable loads and loads that are off
        if not load_config.can_throttle or not load_plan.is_on or not load_state.is_on:
            continue
        if is_load_unresponsive(load_state, now):
            continue  # Held where it is by the first pass

        # First, give back any power we had previously allocated
        available_amps += load_plan.expected_load_amps

        # Rate limiting only guards against ramping a load up too often. Dialling
        # a load back is always allowed - it is the safe direction, and holding it
        # off is what made a spike shed other loads instead.
        ceiling_amps = load_config.max_controllable_load_amps
        if load_state.is_throttle_rate_limited:
            ceiling_amps = min(ceiling_amps, load_plan.current_throttle_amps)

        # Give the load as much power as we can, accounting for what's currently allocated
        will_consume_amps = min(
            available_amps,
            ceiling_amps,
        )
        will_consume_amps = max(
            math.floor(will_consume_amps), load_config.min_controllable_load_amps
        )
        load_plan.throttle_amps = load_plan.expected_load_amps = will_consume_amps
        available_amps -= will_consume_amps

        if load_state.is_throttle_rate_limited and will_consume_amps >= ceiling_amps:
            _LOGGER.debug(
                "Holding load %s at %gA due to throttle rate limit",
                load_name,
                will_consume_amps,
            )
        else:
            _LOGGER.debug(
                "Planning to throttle load %s to %gA", load_name, will_consume_amps
            )

    # Third pass to immediately cut loads if we are overloaded
    overload = False
    if (
        state.house_consumption_amps
        >= max_safe_total_load_amps + config.safety_margin_amps
        and max_safe_total_load_amps > 0
    ):
        if state.overload_timestamp is None:
            state.overload_timestamp = now
        if now >= state.overload_timestamp + timedelta(
            seconds=config.recalculate_interval_seconds
        ):
            overload = True
            _LOGGER.warning(
                "Overload detected (consumption: %gA, max: %gA, available: %gA), reducing loads in reverse priority",
                state.house_consumption_amps,
                max_safe_total_load_amps,
                available_amps,
            )

            # Dial throttleable loads back to their minimum before shedding
            # anything. A throttleable load can give power back without going
            # off, so it must be asked before a fixed load is cut.
            for load_name in reversed(prioritised_loads):
                load_config = config.controllable_loads[load_name]
                load_state = state.controllable_loads[load_name]
                load_plan = new_plan.controllable_loads[load_name]
                if not load_config.can_throttle or not load_plan.is_on:
                    continue
                if not load_state.is_under_load_control:
                    continue  # Out of our control
                if is_load_unresponsive(load_state, now):
                    continue  # It would not be asked, so it gives nothing back
                if load_plan.expected_load_amps <= load_config.min_controllable_load_amps:
                    continue  # Already as low as it goes

                available_amps += (
                    load_plan.expected_load_amps - load_config.min_controllable_load_amps
                )
                load_plan.throttle_amps = load_plan.expected_load_amps = (
                    load_config.min_controllable_load_amps
                )
                _LOGGER.info(
                    "Throttling load %s back to %gA to reduce overload",
                    load_name,
                    load_config.min_controllable_load_amps,
                )

            # Work out how much load still has to go. Reductions already asked
            # for but not yet reflected in the meter count towards it, so we do
            # not shed a fixed load for power a throttleable load is already
            # giving back. If the reduction never arrives we are still overloaded
            # on the next cycle and will shed then.
            excess_amps = state.house_consumption_amps - max_safe_total_load_amps
            for load_name in prioritised_loads:
                load_config = config.controllable_loads[load_name]
                load_state = state.controllable_loads[load_name]
                load_plan = new_plan.controllable_loads[load_name]
                if not load_config.can_throttle or not load_plan.is_on:
                    continue
                if not load_state.is_under_load_control:
                    continue
                if is_load_unresponsive(load_state, now):
                    continue
                excess_amps -= max(
                    0.0, load_state.current_load_amps - load_plan.expected_load_amps
                )

            for load_name in reversed(prioritised_loads):
                if excess_amps <= 0:
                    break  # Throttling back covered the overload

                load_plan = new_plan.controllable_loads[load_name]
                load_state = state.controllable_loads[load_name]
                if not load_plan.is_on or not load_state.is_under_load_control:
                    continue  # Load will already be off or out of our control
                if is_load_unresponsive(load_state, now):
                    continue  # Cutting it would free nothing

                # Cutting the load removes whatever it is really drawing, which
                # is the measured value unless it has not ramped up yet.
                excess_amps -= max(load_state.current_load_amps, load_plan.expected_load_amps)
                load_plan.is_on = False
                available_amps += load_plan.expected_load_amps
                load_plan.expected_load_amps = 0.0
                load_plan.throttle_amps = 0.0
                _LOGGER.info("Cutting load %s to reduce overload", load_name)
    else:
        state.overload_timestamp = None

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
        load_plan = new_plan.controllable_loads[load_name]
        new_plan.used_amps += load_plan.expected_load_amps
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
        load_plan = new_plan.controllable_loads[load_name]
        load_state = state.controllable_loads[load_name]
        if load_plan.is_on:
            _LOGGER.debug(
                "Allocated %gA to load %s (measured: %s, measured current: %gA)",
                load_plan.expected_load_amps,
                load_name,
                "yes" if load_plan.using_measured_current else "no",
                load_state.current_load_amps,
            )

    await execute_plan(hass, new_plan, entry_id)


async def execute_plan(hass: HomeAssistant, plan: PlanState, entry_id: str):
    """Changes entity states to achieve load control plan."""
    now = datetime.now()

    config = hass.data[DOMAIN][entry_id]["config"]
    state = hass.data[DOMAIN][entry_id]["state"]
    committed_plan = hass.data[DOMAIN][entry_id]["plan"]

    for load_name in plan.controllable_loads:  # pylint: disable=consider-using-dict-items
        load_config = config.controllable_loads[load_name]
        load_state = state.controllable_loads[load_name]
        previous_plan = committed_plan.controllable_loads[load_name]
        new_plan = plan.controllable_loads[load_name]

        # Turn on or off load only when we need to
        if not load_config.switch_entity:
            _LOGGER.error(
                "Switch entity not configured for load %s, skipping control",
                load_name,
            )
            await safety_abort(hass, entry_id, True)
            return

        # Only act on an entity that is present and reporting. A hygrostat or
        # thermostat is unavailable until its sensor first reports, and its
        # turn_on/turn_off are no-ops in that window, so switching it here would
        # leave our state out of sync with the device.
        if not is_entity_usable(hass.states.get(load_config.switch_entity)):
            _LOGGER.debug(
                "Switch entity %s is unavailable, skipping control",
                load_config.switch_entity,
            )
            continue  # Skip this load and continue with the next one

        # A load that failed its last command is left alone until its backoff
        # runs out. The plan has already budgeted it as uncontrolled load.
        if is_load_unresponsive(load_state, now):
            continue

        switch_domain = parse_entity_domain(load_config.switch_entity)
        sent_command = False

        # A switch reports its new state through the state listener, which can
        # take longer than the gap between two recalculations. Forget the
        # command once it has, and until then do not send it again: each repeat
        # used to push last_toggled and on_since forward, which extended both
        # the toggle rate limit and the window where the load's meter is not
        # trusted, for a command the load had already accepted.
        if (
            load_state.switch_command_on is not None
            and load_state.is_on == load_state.switch_command_on
        ):
            load_state.switch_command_on = None
            load_state.switch_command_since = None

        awaiting_switch = (
            load_state.switch_command_since is not None
            and (now - load_state.switch_command_since).total_seconds()
            < SWITCH_COMMAND_TIMEOUT_SECONDS
        )

        if new_plan.is_on and not load_state.is_on and awaiting_switch:
            _LOGGER.debug(
                "Load %s has not reported yet, not repeating the command",
                load_config.switch_entity,
            )
        elif not new_plan.is_on and load_state.is_on and awaiting_switch:
            _LOGGER.debug(
                "Load %s has not reported yet, not repeating the command",
                load_config.switch_entity,
            )
        elif new_plan.is_on != load_state.is_on:
            turning_on = new_plan.is_on
            _LOGGER.info(
                "Turning %s load %s",
                "on" if turning_on else "off",
                load_config.switch_entity,
            )
            sent_command = True
            # Only a command the load accepted starts the toggle rate limit. A
            # failed one changed nothing, and holding the load to its toggle
            # interval would keep it from being retried once its backoff ends.
            if await call_load_service(
                hass,
                entry_id,
                load_name,
                switch_domain,
                "turn_on" if turning_on else "turn_off",
                {"entity_id": load_config.switch_entity},
            ):
                load_state.last_toggled = now
                load_state.switch_command_on = turning_on
                load_state.switch_command_since = now
                load_state.is_under_load_control = turning_on
                # Track when the load was turned on
                load_state.on_since = now if turning_on else None

        if (
            load_config.can_throttle
            and new_plan.is_on
            and load_config.throttle_amps_entity
            and load_state.is_under_load_control
            and not is_load_unresponsive(load_state, now)
        ):
            # Check if throttle entity exists
            throttle_state = hass.states.get(load_config.throttle_amps_entity)
            if throttle_state is None:
                _LOGGER.error(
                    "Throttle entity %s does not exist, skipping throttling",
                    load_config.throttle_amps_entity,
                )
            else:
                # Read current value from the entity state
                try:
                    current_throttle_amps = float(throttle_state.state)
                except (ValueError, TypeError):
                    _LOGGER.warning(
                        "Unable to read current throttle value for %s, using previous plan value",
                        load_config.throttle_amps_entity,
                    )
                    current_throttle_amps = previous_plan.throttle_amps

                throttle_amps_delta = abs(
                    round(new_plan.throttle_amps) - round(current_throttle_amps)
                )
                if throttle_amps_delta > 0:
                    _LOGGER.info(
                        "Throttling load %s to %gA",
                        load_config.throttle_amps_entity,
                        round(new_plan.throttle_amps),
                    )
                    sent_command = True
                    if await call_load_service(
                        hass,
                        entry_id,
                        load_name,
                        parse_entity_domain(load_config.throttle_amps_entity),
                        "set_value",
                        {
                            "entity_id": load_config.throttle_amps_entity,
                            "value": new_plan.throttle_amps,  # Don't convert to string
                        },
                    ):
                        load_state.last_throttled = now

        # A load whose backoff ran out and that the plan did not need to change
        # is already where we want it, so it is no longer a problem.
        if not sent_command and load_state.command_failures > 0:
            mark_load_responsive(hass, entry_id, load_name)

    # Deep copy the controllable loads to avoid sharing references with the plan we just built
    committed_plan.available_amps = plan.available_amps
    committed_plan.used_amps = plan.used_amps
    committed_plan.controllable_loads = {}
    for load_name, load_plan in plan.controllable_loads.items():
        committed_plan.controllable_loads[load_name] = ControllableLoadPlanState()
        committed_plan.controllable_loads[load_name].is_on = load_plan.is_on
        committed_plan.controllable_loads[
            load_name
        ].expected_load_amps = load_plan.expected_load_amps
        committed_plan.controllable_loads[load_name].throttle_amps = load_plan.throttle_amps

    _LOGGER.debug("Plan execution completed for %d loads", len(plan.controllable_loads))


def clear_safety_abort(hass: HomeAssistant, entry_id: str):
    """Clear safety abort state if system has recovered."""

    state = hass.data[DOMAIN][entry_id]["state"]
    # The countdown to an abort starts on the first unusable reading, which is
    # before the abort itself becomes active. Clearing it only once an abort is
    # active left the timestamp behind whenever a blip recovered on its own,
    # and the next blip - however brief - then found its grace period already
    # spent and cut every load immediately.
    state.safety_abort_timestamp = None
    if state.safety_abort_active:
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
            if is_entity_usable(hass.states.get(lconfig.switch_entity)):
                # Sent even to a load in backoff: this is the one command worth
                # an extra call. A failure is logged and recorded, and does not
                # stop the loads after it from being turned off.
                if await call_load_service(
                    hass,
                    entry_id,
                    load_name,
                    parse_entity_domain(lconfig.switch_entity),
                    "turn_off",
                    {"entity_id": lconfig.switch_entity},
                ):
                    _LOGGER.info("Turned off load %s for safety", lconfig.switch_entity)
            else:
                _LOGGER.warning(
                    "Switch entity %s is unavailable, cannot turn it off for safety",
                    lconfig.switch_entity,
                )

            # Release the load's reservation either way - during an abort we must
            # not keep budgeting amps for a load we were unable to reach.
            lstate = state.controllable_loads[load_name]
            lstate.is_on = False
            lstate.is_under_load_control = False
            lstate.last_toggled = datetime.now()
            lstate.last_throttled = datetime.now()
            lstate.switch_command_on = None
            lstate.switch_command_since = None

            load_plan = plan.controllable_loads[load_name] = ControllableLoadPlanState()
            load_plan.is_on = False
            load_plan.expected_load_amps = 0.0
            load_plan.throttle_amps = 0.0
        except (ValueError, KeyError, RuntimeError) as err:
            _LOGGER.error("Failed to turn off load %s for safety: %s", load_name, err)
