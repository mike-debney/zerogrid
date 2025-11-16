"""The ZeroGrid integration."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

from homeassistant.const import STATE_OFF, STATE_ON, Platform
from homeassistant.core import Event, HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.discovery import async_load_platform
from homeassistant.helpers.event import (
    EventStateChangedData,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import bind_hass

from .config import Config, ControllableLoadConfig
from .const import ALLOW_GRID_IMPORT_SWITCH_ID, DOMAIN, ENABLE_LOAD_CONTROL_SWITCH_ID
from .helpers import parse_entity_domain
from .state import ControllableLoadPlanState, ControllableLoadState, PlanState, State

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR, Platform.SWITCH]

CONFIG = Config()
STATE = State()
PLAN = PlanState()


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Main entry point on startup."""
    domain_config = config[DOMAIN]
    parse_config(domain_config)
    initialise_state(hass)
    subscribe_to_entity_changes(hass)

    # Register services
    async def handle_recalculate_load_control(call) -> None:
        """Handle the recalculate_load_control service call."""
        _LOGGER.info("Manual recalculation triggered via service call")
        await recalculate_load_control(hass)

    hass.services.async_register(
        DOMAIN,
        "recalculate_load_control",
        handle_recalculate_load_control,
    )

    # Set up platforms
    hass.async_create_task(
        async_load_platform(hass, Platform.BINARY_SENSOR, DOMAIN, {}, config)
    )
    hass.async_create_task(
        async_load_platform(hass, Platform.SENSOR, DOMAIN, {}, config)
    )
    hass.async_create_task(
        async_load_platform(hass, Platform.SWITCH, DOMAIN, {}, config)
    )

    return True


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

    CONFIG.mains_voltage_entity = domain_config.get("mains_voltage_entity")
    CONFIG.solar_generation_kw_entity = domain_config.get(
        "solar_generation_kw_entity", None
    )
    CONFIG.allow_solar_consumption = (
        CONFIG.mains_voltage_entity is not None
        and CONFIG.solar_generation_kw_entity is not None
    )

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

    if CONFIG.mains_voltage_entity is not None:
        state = hass.states.get(CONFIG.mains_voltage_entity)
        if state is not None and state.state not in ("unknown", "unavailable"):
            STATE.mains_voltage = float(state.state)

    if CONFIG.solar_generation_kw_entity is not None:
        state = hass.states.get(CONFIG.solar_generation_kw_entity)
        if state is not None and state.state not in ("unknown", "unavailable"):
            STATE.solar_generation_kw = float(state.state)
    else:
        STATE.solar_generation_kw = 0.0

    # Initialize allow_grid_import from switch entity (will be set up by platform)
    grid_import_state = hass.states.get(ALLOW_GRID_IMPORT_SWITCH_ID)
    if grid_import_state is not None and grid_import_state.state not in (
        "unknown",
        "unavailable",
    ):
        STATE.allow_grid_import = grid_import_state.state == STATE_ON
    else:
        STATE.allow_grid_import = False  # Default to safe state

    # Initialize enable_load_control from switch entity (will be set up by platform)
    load_control_state = hass.states.get(ENABLE_LOAD_CONTROL_SWITCH_ID)
    if load_control_state is not None and load_control_state.state not in (
        "unknown",
        "unavailable",
    ):
        STATE.enable_load_control = load_control_state.state == STATE_ON
    else:
        STATE.enable_load_control = False  # Default to safe state

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


def subscribe_to_entity_changes(hass: HomeAssistant):
    """Subscribes to required entity changes."""
    entity_ids: list[str] = [CONFIG.house_consumption_amps_entity]
    if (
        CONFIG.allow_solar_consumption
        and CONFIG.solar_generation_kw_entity is not None
        and CONFIG.mains_voltage_entity is not None
    ):
        entity_ids.append(CONFIG.solar_generation_kw_entity)
        entity_ids.append(CONFIG.mains_voltage_entity)

    for config in CONFIG.controllable_loads.values():
        entity_ids.append(config.load_amps_entity)
        if config.switch_entity is not None:
            entity_ids.append(config.switch_entity)
        if config.can_turn_on_entity is not None:
            entity_ids.append(config.can_turn_on_entity)

    async def state_automation_listener(event: Event[EventStateChangedData]) -> None:
        if event.event_type != "state_changed":
            return

        # Update state based on entity changes
        entity_id = event.data["entity_id"]
        new_state = event.data.get("new_state")
        if new_state is None:
            return

        if entity_id == CONFIG.house_consumption_amps_entity:
            if new_state is not None and new_state.state not in (
                "unknown",
                "unavailable",
            ):
                STATE.house_consumption_amps = float(new_state.state)
                await recalculate_load_control(hass)
            else:
                _LOGGER.error(
                    "House consumption entity is unavailable, cutting all load for safety"
                )
                await safety_abort(hass)
                return

        elif entity_id == CONFIG.mains_voltage_entity:
            if new_state is not None and new_state.state not in (
                "unknown",
                "unavailable",
            ):
                STATE.mains_voltage = float(new_state.state)
            else:
                _LOGGER.error(
                    "Mains voltage entity is unavailable, cutting all load for safety"
                )
                await safety_abort(hass)
                return

        elif entity_id == CONFIG.solar_generation_kw_entity:
            if new_state is not None and new_state.state not in (
                "unknown",
                "unavailable",
            ):
                STATE.solar_generation_kw = float(new_state.state)
            else:
                STATE.solar_generation_kw = 0.0
                _LOGGER.warning(
                    "Solar generation entity is unavailable, assuming zero generation"
                )
        else:
            # match to controllable loads
            for config in CONFIG.controllable_loads.values():
                load = STATE.controllable_loads[config.name]

                if entity_id == config.switch_entity:
                    _LOGGER.debug(
                        "Switch entity %s changed to %s", entity_id, new_state.state
                    )
                    if new_state is not None and new_state.state not in (
                        "unknown",
                        "unavailable",
                    ):
                        load.is_on = new_state.state != STATE_OFF
                    else:
                        load.is_on = False
                    if load.is_on and load.is_under_load_control:
                        load.on_since = datetime.now()
                    else:
                        load.on_since = None

                elif entity_id == config.load_amps_entity:
                    if new_state is not None and new_state.state not in (
                        "unknown",
                        "unavailable",
                    ):
                        load.current_load_amps = float(new_state.state)
                    else:
                        load.current_load_amps = 0.0

                elif (
                    config.can_turn_on_entity is not None
                    and entity_id == config.can_turn_on_entity
                ):
                    # Handle changes to can_turn_on_entity
                    if new_state is not None and new_state.state not in (
                        "unknown",
                        "unavailable",
                    ):
                        load.can_turn_on = new_state.state == STATE_ON
                        _LOGGER.debug(
                            "Can turn on entity %s for load %s changed to %s (can_turn_on=%s)",
                            entity_id,
                            config.name,
                            new_state.state,
                            load.can_turn_on,
                        )
                    else:
                        load.can_turn_on = False  # Safe default
                        _LOGGER.warning(
                            "Can turn on entity %s for load %s is %s, preventing turn on",
                            entity_id,
                            config.name,
                            new_state.state,
                        )

    async def state_time_listener(now: datetime) -> None:
        if CONFIG.enable_automatic_recalculation:
            # Make sure we are recalculating at least the minimum interval
            if (
                STATE.last_recalculation is None
                or STATE.last_recalculation
                + timedelta(seconds=CONFIG.recalculate_interval_seconds)
                < datetime.now()
            ):
                await recalculate_load_control(hass)

    _LOGGER.debug("Subscribing... %s", entity_ids)
    async_track_state_change_event(hass, entity_ids, state_automation_listener)

    interval = timedelta(seconds=1)
    async_track_time_interval(hass, state_time_listener, interval)


@bind_hass
async def calculate_effective_available_power(
    hass: HomeAssistant,
) -> tuple[float, float]:
    """Calculate available power including power freed by underperforming loads."""
    max_safe_total_load_amps = 0

    # Allow grid import
    grid_maximum_amps = 0.0
    if STATE.allow_grid_import:
        grid_maximum_amps = CONFIG.max_grid_import_amps
        max_safe_total_load_amps += CONFIG.max_grid_import_amps

    # Convert solar generation to amps
    solar_generation_amps = 0.0
    if CONFIG.allow_solar_consumption and STATE.solar_generation_kw > 0:
        solar_generation_amps = (STATE.solar_generation_kw * 1000) / STATE.mains_voltage
        solar_generation_amps = min(
            solar_generation_amps, CONFIG.max_solar_generation_amps
        )
        max_safe_total_load_amps += CONFIG.max_solar_generation_amps

    # Calculate total available power before accounting for loads
    # This is what we want to track the minimum of
    now = datetime.now()
    total_available_power_amps = grid_maximum_amps + solar_generation_amps
    # Safety cap to maximum total available load
    total_available_power_amps = min(
        total_available_power_amps, CONFIG.max_total_load_amps
    )

    # Determine max window duration based on longest min_toggle_interval
    max_window_seconds = max(
        config.min_toggle_interval_seconds
        for config in CONFIG.controllable_loads.values()
    )

    # Accumulate the total unallocated power (before subtracting loads) into history
    STATE.accumulate_unallocated_amps(total_available_power_amps, max_window_seconds)

    # Subtract loads that are under load control, since we can manage those
    total_load_not_under_control = STATE.house_consumption_amps
    for load_name in STATE.controllable_loads:  # pylint: disable=consider-using-dict-items
        load_state = STATE.controllable_loads[load_name]
        load_plan = PLAN.controllable_loads.get(load_name)

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
    if DOMAIN in hass.data and "available_load_sensor" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["available_load_sensor"].update_value(total_available_amps)
    if DOMAIN in hass.data and "uncontrolled_load_sensor" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["uncontrolled_load_sensor"].update_value(
            total_load_not_under_control
        )
    if DOMAIN in hass.data and "max_safe_load_sensor" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["max_safe_load_sensor"].update_value(max_safe_total_load_amps)

    return total_available_amps, max_safe_total_load_amps


@bind_hass
async def recalculate_load_control(hass: HomeAssistant):
    """The core of the load control algorithm.

    This function is intentionally complex as it handles the complete load planning
    algorithm including priority management, power allocation, throttling, rate limiting,
    overload protection, and reactive reallocation.
    """
    _LOGGER.info("Recalculating load control plan")
    now = datetime.now()
    STATE.last_recalculation = now

    # Check if load control is enabled
    if DOMAIN in hass.data and ENABLE_LOAD_CONTROL_SWITCH_ID in hass.data[DOMAIN]:
        STATE.enable_load_control = hass.data[DOMAIN][
            ENABLE_LOAD_CONTROL_SWITCH_ID
        ].is_on
    if not STATE.enable_load_control:
        _LOGGER.debug("Load control is disabled, skipping recalculation")

        # Reset load state
        for control in CONFIG.controllable_loads.values():
            load = STATE.controllable_loads[control.name]
            load.is_under_load_control = True
            load.last_throttled = None
            load.last_toggled = None
        return

    # Clear safety abort state if system has recovered
    if DOMAIN in hass.data and "safety_abort_sensor" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["safety_abort_sensor"].update_state(False)

    # Check if allow grid import is enabled
    STATE.allow_grid_import = False
    if DOMAIN in hass.data and ALLOW_GRID_IMPORT_SWITCH_ID in hass.data[DOMAIN]:
        STATE.allow_grid_import = hass.data[DOMAIN][ALLOW_GRID_IMPORT_SWITCH_ID].is_on

    new_plan = PlanState()

    # Calculate effective available power, loads that we are controlling are included
    (
        available_amps,
        max_safe_total_load_amps,
    ) = await calculate_effective_available_power(hass)
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
            _LOGGER.debug(
                "Unable to change load %s due to switch rate limit", load_name
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

        # Account for loads that are on but may be drawing less than expected
        if (
            plan.is_on
            and state.on_since is not None
            and state.on_since
            + timedelta(seconds=CONFIG.load_measurement_delay_seconds)
            < now
        ):
            will_consume_amps = round(state.current_load_amps)

        plan.expected_load_amps = will_consume_amps
        available_amps -= will_consume_amps

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

            # Check if load has been on for a while but not drawing significant power
            if (
                state.on_since is not None
                and state.on_since
                + timedelta(seconds=CONFIG.load_measurement_delay_seconds)
                < now
                and state.current_load_amps < config.min_controllable_load_amps
            ):
                will_consume_amps = round(state.current_load_amps)
                plan.throttle_amps = config.min_controllable_load_amps

            else:
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
            seconds=CONFIG.recalculate_interval_seconds * 3
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
    if DOMAIN in hass.data and "overload_sensor" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["overload_sensor"].update_state(overload)

    # Final pass to summarise plan
    new_plan.available_amps = available_amps
    for load_name in prioritised_loads:
        plan = new_plan.controllable_loads[load_name]
        new_plan.used_amps += plan.expected_load_amps
    if DOMAIN in hass.data and "controlled_load_sensor" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["controlled_load_sensor"].update_value(new_plan.used_amps)

    _LOGGER.debug(
        "Planning complete: available=%gA, used=%gA",
        new_plan.available_amps,
        new_plan.used_amps,
    )

    await execute_plan(hass, new_plan)


@bind_hass
async def execute_plan(hass: HomeAssistant, plan: PlanState):
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
            await safety_abort(hass)
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
                await safety_abort(hass)
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


@bind_hass
async def safety_abort(hass: HomeAssistant):
    """Cuts all load controlled by the integration in a safety situation."""
    _LOGGER.error("Aborting load control, cutting all loads")

    # Update safety abort binary sensor
    if DOMAIN in hass.data and "safety_abort_sensor" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["safety_abort_sensor"].update_state(True)

    if DOMAIN in hass.data and "enable_load_control" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["enable_load_control"].update_value(False)

    plan = PlanState()
    plan.available_amps = 0.0
    plan.used_amps = 0.0
    for load_name in CONFIG.controllable_loads:  # pylint: disable=consider-using-dict-items
        try:
            config = CONFIG.controllable_loads[load_name]
            switch_domain = parse_entity_domain(config.switch_entity)
            service_name = "turn_off"
            await hass.services.async_call(
                switch_domain,
                service_name,
                {"entity_id": config.switch_entity},
                blocking=True,
            )

            state = STATE.controllable_loads[load_name]
            state.is_on = False
            state.is_under_load_control = False
            state.last_toggled = datetime.now()
            state.last_throttled = datetime.now()

            load_plan = plan.controllable_loads[load_name] = ControllableLoadPlanState()
            load_plan.is_on = False
            load_plan.expected_load_amps = 0.0
            load_plan.throttle_amps = 0.0
            _LOGGER.info("Turned off load %s for safety", config.switch_entity)
        except (ValueError, KeyError, RuntimeError) as err:
            _LOGGER.error("Failed to turn off %s: %s", config.switch_entity, err)
