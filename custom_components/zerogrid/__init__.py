"""The ZeroGrid integration."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

from homeassistant.const import Platform
from homeassistant.core import Event, HomeAssistant
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

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.SWITCH]

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
        async_load_platform(hass, Platform.SENSOR, DOMAIN, {}, config)
    )
    hass.async_create_task(
        async_load_platform(hass, Platform.SWITCH, DOMAIN, {}, config)
    )

    return True


def parse_config(domain_config):
    """Parses the config and sets appropriates variables."""
    _LOGGER.debug(domain_config)

    CONFIG.max_house_load_amps = domain_config.get("max_house_load_amps")
    CONFIG.hysteresis_amps = domain_config.get("hysteresis_amps", 2)
    CONFIG.recalculate_interval_seconds = domain_config.get(
        "recalculate_interval_seconds", 10
    )
    CONFIG.house_consumption_amps_entity = domain_config.get(
        "house_consumption_amps_entity"
    )
    CONFIG.mains_voltage_entity = domain_config.get("mains_voltage_entity")

    CONFIG.solar_generation_kw_entity = domain_config.get(
        "solar_generation_kw_entity", None
    )
    CONFIG.allow_solar_consumption = CONFIG.solar_generation_kw_entity is not None

    # Reactive power management settings
    CONFIG.variance_detection_threshold = domain_config.get(
        "variance_detection_threshold", 1.0
    )
    CONFIG.variance_detection_delay_seconds = domain_config.get(
        "variance_detection_delay_seconds", 30
    )
    CONFIG.enable_reactive_reallocation = domain_config.get(
        "enable_reactive_reallocation", True
    )
    CONFIG.enable_automatic_recalculation = domain_config.get(
        "enable_automatic_recalculation", True
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
        CONFIG.controllable_loads[control_config.name] = control_config

    _LOGGER.debug("Config successful: %s", CONFIG)


def initialise_state(hass: HomeAssistant):
    """Initialises the state of the integration."""
    if CONFIG.house_consumption_amps_entity is not None:
        state = hass.states.get(CONFIG.house_consumption_amps_entity)
        if state is not None:
            STATE.house_consumption_amps = float(state.state)

    if CONFIG.mains_voltage_entity is not None:
        state = hass.states.get(CONFIG.mains_voltage_entity)
        if state is not None:
            STATE.mains_voltage = float(state.state)

    if CONFIG.solar_generation_kw_entity is not None:
        state = hass.states.get(CONFIG.solar_generation_kw_entity)
        if state is not None:
            STATE.solar_generation_kw = float(state.state)
    else:
        STATE.solar_generation_kw = 0.0

    # Initialize allow_grid_import from switch entity (will be set up by platform)
    grid_import_state = hass.states.get(ALLOW_GRID_IMPORT_SWITCH_ID)
    if grid_import_state is not None:
        STATE.allow_grid_import = grid_import_state.state.lower() == "on"
    else:
        STATE.allow_grid_import = False  # Default to safe state

    # Initialize enable_load_control from switch entity (will be set up by platform)
    load_control_state = hass.states.get(ENABLE_LOAD_CONTROL_SWITCH_ID)
    if load_control_state is not None:
        STATE.enable_load_control = load_control_state.state.lower() == "on"
    else:
        STATE.enable_load_control = False  # Default to safe state

    # match to controllable loads
    for load_name in CONFIG.controllable_loads:  # pylint: disable=consider-using-dict-items
        config = CONFIG.controllable_loads[load_name]
        load_state = ControllableLoadState()

        switch_state = hass.states.get(config.switch_entity)
        if switch_state is not None:
            load_state.is_on = switch_state.state.lower() == "on"

        load_amps_state = hass.states.get(config.load_amps_entity)
        if load_amps_state is not None:
            load_state.current_load_amps = float(load_amps_state.state)

        STATE.controllable_loads[load_name] = load_state
        PLAN.controllable_loads[load_name] = ControllableLoadPlanState()
        _LOGGER.error(
            "Switch entity init for %s: %s",
            load_name,
            PLAN.controllable_loads[load_name].is_on,
        )

    _LOGGER.debug("Initialised state: %s", STATE)


def subscribe_to_entity_changes(hass: HomeAssistant):
    """Subscribes to required entity changes."""
    entity_ids: list[str] = [
        CONFIG.house_consumption_amps_entity,
        CONFIG.mains_voltage_entity,
    ]
    if CONFIG.solar_generation_kw_entity is not None:
        entity_ids.append(CONFIG.solar_generation_kw_entity)

    for control in CONFIG.controllable_loads.values():
        entity_ids.append(control.load_amps_entity)
        if control.switch_entity is not None:
            entity_ids.append(control.switch_entity)

    async def state_automation_listener(event: Event[EventStateChangedData]) -> None:
        if event.event_type != "state_changed":
            return

        recalculate_now = False

        # Update state based on entity changes
        entity_id = event.data["entity_id"]
        new_state = event.data.get("new_state")
        if new_state is None:
            return
        _LOGGER.debug("Entity changed: %s : %s", entity_id, new_state.state)

        if entity_id == CONFIG.house_consumption_amps_entity:
            if new_state is not None:
                STATE.house_consumption_amps = float(new_state.state)
            else:
                _LOGGER.error(
                    "House consumption entity is unavailable, cutting all load for safety"
                )
                await safety_abort(hass)
                return
            recalculate_now = True

        elif entity_id == CONFIG.mains_voltage_entity:
            if new_state is not None:
                STATE.mains_voltage = float(new_state.state)
            else:
                _LOGGER.error(
                    "Mains voltage entity is unavailable, cutting all load for safety"
                )
                await safety_abort(hass)
                return
            recalculate_now = True

        elif entity_id == CONFIG.solar_generation_kw_entity:
            if new_state is not None:
                STATE.solar_generation_kw = float(new_state.state)
            else:
                STATE.solar_generation_kw = 0.0
                _LOGGER.warning(
                    "Solar generation entity is unavailable, assuming zero generation"
                )
            recalculate_now = True

        else:
            STATE.load_control_consumption_amps = 0.0

            # match to controllable loads
            for control in CONFIG.controllable_loads.values():
                if entity_id == control.switch_entity:
                    _LOGGER.error(
                        "Switch entity change for %s: %s", control.name, new_state.state
                    )
                    if new_state is not None:
                        STATE.controllable_loads[control.name].is_on = (
                            new_state.state.lower() == "on"
                        )
                    else:
                        STATE.controllable_loads[control.name].is_on = False

                if entity_id == control.load_amps_entity:
                    if new_state is not None:
                        STATE.controllable_loads[
                            control.name
                        ].current_load_amps = float(new_state.state)
                    else:
                        STATE.controllable_loads[control.name].current_load_amps = 0.0
                STATE.load_control_consumption_amps += STATE.controllable_loads[
                    control.name
                ].current_load_amps

        if CONFIG.enable_automatic_recalculation and recalculate_now:
            await recalculate_load_control(hass)

        # # Check for consumption variance and potentially trigger reallocation
        # if CONFIG.enable_reactive_reallocation:
        #     await check_consumption_variance_and_reallocate(hass)

    async def state_time_listener(now: datetime) -> None:
        if CONFIG.enable_automatic_recalculation:
            _LOGGER.debug("Time listener fired")
            await recalculate_load_control(hass)

            # # Check for consumption variance and potentially trigger reallocation
            # if CONFIG.enable_reactive_reallocation:
            #     await check_consumption_variance_and_reallocate(hass)

    _LOGGER.debug("Subscribing... %s", entity_ids)
    async_track_state_change_event(hass, entity_ids, state_automation_listener)

    interval = timedelta(seconds=CONFIG.recalculate_interval_seconds)
    async_track_time_interval(hass, state_time_listener, interval)


@bind_hass
async def calculate_effective_available_power() -> float:
    """Calculate available power including power freed by underperforming loads."""
    # Start with the base available power calculation
    safety_margin_amps = 0.0

    # Allow grid import
    grid_maximum_amps = 0.0
    if STATE.allow_grid_import:
        grid_maximum_amps = CONFIG.max_house_load_amps

    # Convert solar generation to amps
    solar_generation_amps = 0.0
    if STATE.solar_generation_kw > 0:
        solar_generation_amps = (STATE.solar_generation_kw * 1000) / STATE.mains_voltage

    # Subtract loads that are under load control, since we can manage those
    total_load_not_under_control = STATE.house_consumption_amps
    for load_name in STATE.controllable_loads:
        load_state = STATE.controllable_loads[load_name]
        if load_state.is_on_load_control and load_state.is_on:
            total_load_not_under_control -= load_state.current_load_amps

    total_available_amps = (
        grid_maximum_amps
        + solar_generation_amps
        - max(total_load_not_under_control, 0)
        - safety_margin_amps
    )
    _LOGGER.debug(
        "Total load available for planning: %f A",
        total_available_amps,
    )
    return total_available_amps


@bind_hass
async def recalculate_load_control(hass: HomeAssistant):  # noqa: C901
    """The core of the load control algorithm.

    This function is intentionally complex as it handles the complete load planning
    algorithm including priority management, power allocation, throttling, rate limiting,
    overload protection, and reactive reallocation.
    """
    _LOGGER.debug("Recalculating load control plan...")

    # Check if load control is enabled
    if DOMAIN in hass.data and ENABLE_LOAD_CONTROL_SWITCH_ID in hass.data[DOMAIN]:
        STATE.enable_load_control = hass.data[DOMAIN][
            ENABLE_LOAD_CONTROL_SWITCH_ID
        ].is_on
    if not STATE.enable_load_control:
        _LOGGER.debug("Load control is disabled, skipping recalculation")
        return

    # Check if allow grid import is enabled
    STATE.allow_grid_import = False
    if DOMAIN in hass.data and ALLOW_GRID_IMPORT_SWITCH_ID in hass.data[DOMAIN]:
        STATE.allow_grid_import = hass.data[DOMAIN][ALLOW_GRID_IMPORT_SWITCH_ID].is_on

    now = datetime.now()
    new_plan = PlanState()

    # Calculate effective available power, loads that we are controlling are included
    available_amps = await calculate_effective_available_power()

    new_plan.available_amps = available_amps

    # Update entities instead of setting state directly
    if DOMAIN in hass.data:
        if "available_load_sensor" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["available_load_sensor"].update_value(available_amps)

    _LOGGER.info("Available amps for planning: %f", available_amps)

    # If the available load we have to play with has not changed meaningfully, do nothing
    # Exception: always recalculate if available power is zero/negative (safety)
    # Also always recalculate if reactive reallocation is enabled (to pick up variance changes)
    available_amps_delta = abs(PLAN.available_amps - available_amps)
    if (
        not CONFIG.enable_reactive_reallocation
        and available_amps_delta < CONFIG.hysteresis_amps
        and available_amps > 0
        and PLAN.available_amps > 0
    ):
        _LOGGER.debug(
            "Skipping relcalculation, available amps delta: %fA", available_amps_delta
        )
        return

    # Build priority list (lower number == more important)
    prioritised_loads = sorted(
        CONFIG.controllable_loads.copy(),
        key=lambda k: CONFIG.controllable_loads[k].priority,
    )
    _LOGGER.debug("Priority: %s", prioritised_loads)

    safety_margin_amps = 0  # CONFIG.hysteresis_amps
    overload = (
        STATE.house_consumption_amps > CONFIG.max_house_load_amps - safety_margin_amps
    )

    # Loop over controllable loads and determine if they should be on or not
    for load_index, load_name in enumerate(prioritised_loads):
        config = CONFIG.controllable_loads[load_name]
        state = STATE.controllable_loads[load_name]
        previous_plan = PLAN.controllable_loads[load_name]
        plan = new_plan.controllable_loads[load_name] = ControllableLoadPlanState()

        if not state.is_on_load_control and state.is_on:
            # Load is manually turned on - we have no control
            plan.is_on = state.is_on
            _LOGGER.debug("Load manually turned on: %s", load_name)
            continue

        plan.is_on = False
        plan.expected_load_amps = 0.0

        # Determine if we are rate-limited on switching or throttling
        is_switch_rate_limited = (
            state.last_toggled is not None
            and state.last_toggled
            + timedelta(seconds=config.min_toggle_interval_seconds)
            > now
        )
        is_throttle_rate_limited = (
            state.last_throttled is not None
            and state.last_throttled
            + timedelta(seconds=config.min_throttle_interval_seconds)
            > now
        )

        # Only allocate power if we can meet minimum requirements
        if available_amps > config.min_controllable_load_amps:
            if config.can_throttle and is_throttle_rate_limited:
                # If we are rate-limited on throttling, keep previous expected load
                will_consume_amps = previous_plan.expected_load_amps
            else:
                will_consume_amps = min(
                    available_amps, config.max_controllable_load_amps
                )
        else:
            # Not enough power available even with throttling
            will_consume_amps = 0.0

        # Determine if this load should be on
        # If we have no available budget, all loads should be off
        # If will_consume_amps is 0, the load should be off (not enough power for lower-priority loads)
        # Or if we are currently overloading the fuse
        should_be_on = False
        if new_plan.available_amps <= 0 or will_consume_amps <= 0 or overload:
            will_consume_amps = 0.0
        else:
            should_be_on = will_consume_amps <= available_amps

        _LOGGER.debug(
            "Load %s planning: should_be_on=%s, will_consume_amps=%f, available_amps=%f, previous_plan.is_on=%s",
            load_name,
            should_be_on,
            will_consume_amps,
            available_amps,
            previous_plan.is_on,
        )

        if should_be_on != previous_plan.is_on and not is_switch_rate_limited:
            plan.is_on = should_be_on
            if plan.is_on:
                _LOGGER.debug("Planning to turn load %s on", load_name)
            else:
                _LOGGER.debug("Planning to turn load %s off", load_name)
        else:
            plan.is_on = previous_plan.is_on
            if not plan.is_on:
                will_consume_amps = 0.0
            # else:
            #     will_consume_amps = state.current_load_amps
            if is_switch_rate_limited:
                _LOGGER.debug(
                    "Unable to change load %s due to switch rate limit", load_name
                )

        # Ensure that we aren't about to overload the fuse
        if plan.is_on and will_consume_amps > available_amps:
            plan.is_on = False
            will_consume_amps = 0.0

        # Account for expected load in budget
        if plan.is_on:
            plan.expected_load_amps = will_consume_amps
            new_plan.used_amps += will_consume_amps
            available_amps -= will_consume_amps

            # Set throttle value for throttleable loads
            if config.can_throttle:
                plan.throttle_amps = will_consume_amps

    # Update entity with actual planned consumption (not capped to available)
    if DOMAIN in hass.data and "controlled_load_sensor" in hass.data[DOMAIN]:
        hass.data[DOMAIN]["controlled_load_sensor"].update_value(new_plan.used_amps)

    _LOGGER.debug(
        "Planning complete: initial_available=%f, planned_used=%f",
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

        _LOGGER.debug(
            "Executing plan for load %s: new_plan.is_on=%s, previous_plan.is_on=%s, state.is_on=%s",
            load_name,
            new_plan.is_on,
            previous_plan.is_on,
            state.is_on,
        )

        # Turn on or off load only when we need to
        if not config.switch_entity:
            _LOGGER.error(
                "Switch entity not configured for load %s, skipping control",
                load_name,
            )
            continue

        switch_domain = parse_entity_domain(config.switch_entity)

        # Check if entity exists before attempting service calls
        if hass.states.get(config.switch_entity) is None:
            _LOGGER.error(
                "Switch entity %s does not exist, skipping control",
                config.switch_entity,
            )
            continue  # Skip this load and continue with the next one

        if new_plan.is_on and not state.is_on:
            _LOGGER.info("Turning on %s due to available load", config.switch_entity)
            try:
                # Use domain-specific service calls for better compatibility
                service_name = "turn_on"
                await hass.services.async_call(
                    switch_domain,
                    service_name,
                    {"entity_id": config.switch_entity},
                    blocking=True,  # Wait for completion to ensure success
                )
                state.last_toggled = now
                state.is_on_load_control = True
                _LOGGER.debug("Successfully turned on %s", config.switch_entity)
            except (ValueError, KeyError, RuntimeError) as err:
                _LOGGER.error("Failed to turn on %s: %s", config.switch_entity, err)

        elif not new_plan.is_on and state.is_on:
            _LOGGER.info("Turning off %s due to available load", config.switch_entity)
            try:
                # Use domain-specific service calls for better compatibility
                service_name = "turn_off"
                await hass.services.async_call(
                    switch_domain,
                    service_name,
                    {"entity_id": config.switch_entity},
                    blocking=True,  # Wait for completion to ensure success
                )
                state.last_toggled = now
                state.is_on_load_control = False
                _LOGGER.debug("Successfully turned off %s", config.switch_entity)
            except (ValueError, KeyError, RuntimeError) as err:
                _LOGGER.error("Failed to turn off %s: %s", config.switch_entity, err)
        else:
            _LOGGER.debug(
                "No action needed for load %s: new_plan.is_on=%s, state.is_on=%s",
                load_name,
                new_plan.is_on,
                state.is_on,
            )

        if (
            config.can_throttle and new_plan.is_on and config.throttle_amps_entity
        ):  # Removed is_on_load_control requirement
            # Check if throttle entity exists
            if hass.states.get(config.throttle_amps_entity) is None:
                _LOGGER.error(
                    "Throttle entity %s does not exist, skipping throttling",
                    config.throttle_amps_entity,
                )
            else:
                # Make sure the delta is significant enough before issuing command
                throttle_delta = state.current_load_amps - new_plan.throttle_amps
                _LOGGER.debug(
                    "Throttle delta for %s: previous=%f, new=%f, delta=%f, hysteresis=%f",
                    load_name,
                    state.current_load_amps,
                    new_plan.throttle_amps,
                    abs(throttle_delta),
                    CONFIG.hysteresis_amps,
                )
                if abs(throttle_delta) > CONFIG.hysteresis_amps:
                    _LOGGER.info(
                        "Throttling %s to %f due to available load",
                        config.throttle_amps_entity,
                        new_plan.throttle_amps,
                    )
                    number_domain = parse_entity_domain(config.throttle_amps_entity)
                    _LOGGER.debug(
                        "Service call: %s.set_value with entity_id=%s, value=%f",
                        number_domain,
                        config.throttle_amps_entity,
                        new_plan.throttle_amps,
                    )
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
                        state.last_throttled = now
                        _LOGGER.debug(
                            "Successfully throttled %s to %f",
                            config.throttle_amps_entity,
                            new_plan.throttle_amps,
                        )
                    except (ValueError, KeyError, RuntimeError) as err:
                        _LOGGER.error(
                            "Failed to throttle %s: %s",
                            config.throttle_amps_entity,
                            err,
                        )
                else:
                    # Don't update the throttle due to hysteresis, but keep the new plan value
                    _LOGGER.debug(
                        "Throttle unchanged for %s: delta %f <= hysteresis %f",
                        load_name,
                        abs(throttle_delta),
                        CONFIG.hysteresis_amps,
                    )
        elif config.can_throttle:
            _LOGGER.debug(
                "Skipping throttle for %s: new_plan.is_on=%s", load_name, new_plan.is_on
            )

        PLAN.available_amps = plan.available_amps
        PLAN.used_amps = plan.used_amps
        # Deep copy the controllable loads to avoid reference sharing between PLAN and new_plan
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

    plan = PlanState()
    plan.available_amps = 0.0
    plan.used_amps = 0.0
    for control in CONFIG.controllable_loads.values():
        load_plan = ControllableLoadPlanState()
        load_plan.is_on = False
        load_plan.expected_load_amps = 0.0
        plan.controllable_loads[control.name] = load_plan

    await execute_plan(hass, plan)
