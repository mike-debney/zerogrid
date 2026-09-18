"""Test harness for the ZeroGrid integration.

Home Assistant is not a test dependency here: it needs a newer Python than
some environments have and pulls in a very large tree for what is, in this
integration, a self-contained planning algorithm. Instead this module installs
small stand-ins for the handful of Home Assistant names the integration
imports, then drives the real `recalculate_load_control` end to end against a
fake state machine and service registry.

What that buys: the tests exercise the actual planning code, including the
rate limits, the throttle passes and the overload pass, and they can reproduce
timing faults - a meter that has not caught up with a setpoint - that are
awkward to stage against a live Home Assistant.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import sys
import types
from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# Home Assistant stand-ins
# --------------------------------------------------------------------------


def _install_homeassistant_stubs() -> None:
    """Register fake `homeassistant.*` modules in sys.modules."""
    if "homeassistant" in sys.modules:
        return

    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    ha = module("homeassistant")
    ha.__path__ = []

    const = module("homeassistant.const")
    const.STATE_ON = "on"
    const.STATE_OFF = "off"
    const.STATE_UNKNOWN = "unknown"
    const.STATE_UNAVAILABLE = "unavailable"

    class Platform(str):
        BINARY_SENSOR = "binary_sensor"
        NUMBER = "number"
        SENSOR = "sensor"
        SWITCH = "switch"

    const.Platform = Platform

    class UnitOfElectricCurrent:
        AMPERE = "A"

    const.UnitOfElectricCurrent = UnitOfElectricCurrent

    core = module("homeassistant.core")

    class State:  # noqa: D401 - mirrors homeassistant.core.State
        """An entity state."""

        def __init__(self, entity_id: str, state: str) -> None:
            self.entity_id = entity_id
            self.state = state

        def __repr__(self) -> str:
            return f"State({self.entity_id}={self.state})"

    core.State = State
    core.HomeAssistant = object
    core.Event = object
    core.callback = lambda func: func

    exceptions = module("homeassistant.exceptions")

    class HomeAssistantError(Exception):
        """Base error."""

    class ServiceValidationError(HomeAssistantError):
        """Raised when a service call is invalid."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args)

    exceptions.HomeAssistantError = HomeAssistantError
    exceptions.ServiceValidationError = ServiceValidationError

    config_entries = module("homeassistant.config_entries")
    config_entries.ConfigEntry = object

    helpers = module("homeassistant.helpers")
    helpers.__path__ = []

    helpers_event = module("homeassistant.helpers.event")
    helpers_event.EventStateChangedData = dict
    helpers_event.async_track_state_change_event = lambda *a, **k: (lambda: None)
    helpers_event.async_track_time_interval = lambda *a, **k: (lambda: None)

    helpers_typing = module("homeassistant.helpers.typing")
    helpers_typing.ConfigType = dict

    device_registry = module("homeassistant.helpers.device_registry")
    device_registry.DeviceInfo = dict


_install_homeassistant_stubs()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.zerogrid import (  # noqa: E402
    apply_switch_state,
    calculate_effective_available_power,
    execute_plan,
    initialise_state,
    parse_config,
    recalculate_load_control,
    safety_abort,
)
import custom_components.zerogrid as zerogrid  # noqa: E402
from custom_components.zerogrid.config import Config  # noqa: E402
from custom_components.zerogrid.const import (  # noqa: E402
    ALLOW_GRID_IMPORT_SWITCH_ID,
    DOMAIN,
    ENABLE_LOAD_CONTROL_SWITCH_ID,
)
from custom_components.zerogrid.state import PlanState, State  # noqa: E402

HA_STATE = sys.modules["homeassistant.core"].State


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


@dataclass
class ServiceCall:
    """A recorded service call."""

    domain: str
    service: str
    data: dict


class FakeStates:
    """Stand-in for hass.states.

    Notifies subscribers the way Home Assistant's state machine does, so a
    device reporting a new state reaches the integration by the same route it
    would in a real install.
    """

    def __init__(self) -> None:
        self._states: dict[str, HA_STATE] = {}
        self.listeners: list = []

    def get(self, entity_id: str):
        return self._states.get(entity_id)

    def set(self, entity_id: str, state, *, notify: bool = True) -> None:
        self._states[entity_id] = HA_STATE(entity_id, str(state))
        if notify:
            for listener in list(self.listeners):
                listener(entity_id, self._states[entity_id])

    def remove(self, entity_id: str) -> None:
        self._states.pop(entity_id, None)


class FakeServices:
    """Stand-in for hass.services.

    By default a call is reflected straight back into the fake state machine,
    the way a well behaved device would. `apply` can be turned off to model a
    device that has not acted on the command yet, which is the situation the
    rate limits and settling windows exist to handle.
    """

    def __init__(self, states: FakeStates) -> None:
        self._states = states
        self.calls: list[ServiceCall] = []
        self.apply = True
        self.fail_with: Exception | None = None

    async def async_call(self, domain, service, data, blocking=False) -> None:
        self.calls.append(ServiceCall(domain, service, dict(data)))
        if self.fail_with is not None:
            raise self.fail_with
        if not self.apply:
            return
        entity_id = data["entity_id"]
        if service == "turn_on":
            self._states.set(entity_id, "heat" if domain == "climate" else "on")
        elif service == "turn_off":
            self._states.set(entity_id, "off")
        elif service == "set_value":
            self._states.set(entity_id, data["value"])

    def calls_for(self, entity_id: str) -> list[ServiceCall]:
        return [c for c in self.calls if c.data.get("entity_id") == entity_id]

    def clear(self) -> None:
        self.calls.clear()


class FakeHass:
    """Stand-in for the HomeAssistant object."""

    def __init__(self) -> None:
        self.states = FakeStates()
        self.services = FakeServices(self.states)
        self.data: dict = {}


class FakeSwitchEntity:
    """Stand-in for the integration's own enable/allow switches."""

    def __init__(self, is_on: bool = True) -> None:
        self.is_on = is_on

    def update_state(self, is_on: bool) -> None:
        self.is_on = is_on


class FakeNumberEntity:
    """Stand-in for the reserved current number entity."""

    def __init__(self, value: float = 0.0) -> None:
        self.native_value = value

    def update_value(self, value: float) -> None:
        self.native_value = value


class FakeSensorEntity:
    """Stand-in for the integration's own sensors."""

    def __init__(self) -> None:
        self.value = None
        self.state = None

    def update_value(self, value) -> None:
        self.value = value

    def update_state(self, state) -> None:
        self.state = state


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def load(
    name: str,
    switch_entity: str,
    load_amps_entity: str,
    *,
    min_amps: float = 1.0,
    max_amps: float = 32.0,
    throttle_amps_entity: str | None = None,
    min_toggle_interval_seconds: int = 600,
    min_throttle_interval_seconds: int = 9,
    load_measurement_delay_seconds: int = 60,
    solar_turn_on_window_seconds: int = 300,
    solar_turn_off_window_seconds: int = 300,
    can_turn_on_entity: str | None = None,
    **extra,
) -> dict:
    """Build one controllable load config block."""
    block = {
        "name": name,
        "switch_entity": switch_entity,
        "load_amps_entity": load_amps_entity,
        "min_controllable_load_amps": min_amps,
        "max_controllable_load_amps": max_amps,
        "min_toggle_interval_seconds": min_toggle_interval_seconds,
        "min_throttle_interval_seconds": min_throttle_interval_seconds,
        "load_measurement_delay_seconds": load_measurement_delay_seconds,
        "solar_turn_on_window_seconds": solar_turn_on_window_seconds,
        "solar_turn_off_window_seconds": solar_turn_off_window_seconds,
    }
    if throttle_amps_entity is not None:
        block["throttle_amps_entity"] = throttle_amps_entity
    if can_turn_on_entity is not None:
        block["can_turn_on_entity"] = can_turn_on_entity
    block.update(extra)
    return block


@dataclass
class Harness:
    """A configured integration wired up to a fake Home Assistant."""

    hass: FakeHass
    entry_id: str
    config: Config
    state: State
    plan: PlanState
    entities: dict = field(default_factory=dict)

    def _on_entity_state(self, entity_id: str, new_state) -> None:
        """Route a switch entity report through the integration's own handler.

        This is the path a device's state takes in a real install, so the
        tests exercise the same code rather than a copy of it.
        """
        for name, cfg in self.config.controllable_loads.items():
            if entity_id == cfg.switch_entity:
                apply_switch_state(cfg, self.state.controllable_loads[name], new_state)

    # -- inputs ----------------------------------------------------------
    def set_house_amps(self, amps: float) -> None:
        """Set total house consumption, as the real listener would."""
        self.hass.states.set(self.config.house_consumption_amps_entity, amps)
        self.state.house_consumption_amps = amps
        self.state.house_consumption_initialised = True

    def set_solar_amps(self, amps: float) -> None:
        self.state.solar_generation_amps = amps

    def set_load_amps(self, name: str, amps: float) -> None:
        """Set a load's own measured current."""
        cfg = self.config.controllable_loads[name]
        self.hass.states.set(cfg.load_amps_entity, amps)
        self.state.controllable_loads[name].current_load_amps = amps

    def set_switch(self, name: str, is_on: bool) -> None:
        """Report a load's switch state back, as the real listener would."""
        cfg = self.config.controllable_loads[name]
        domain = cfg.switch_entity.split(".")[0]
        value = ("heat" if domain == "climate" else "on") if is_on else "off"
        self.hass.states.set(cfg.switch_entity, value)
        self.state.controllable_loads[name].is_on = is_on

    def set_can_turn_on(self, name: str, allowed: bool) -> None:
        cfg = self.config.controllable_loads[name]
        if cfg.can_turn_on_entity:
            self.hass.states.set(cfg.can_turn_on_entity, "on" if allowed else "off")
        self.state.controllable_loads[name].can_turn_on = allowed

    def set_throttle_setpoint(self, name: str, amps: float) -> None:
        cfg = self.config.controllable_loads[name]
        if cfg.throttle_amps_entity:
            self.hass.states.set(cfg.throttle_amps_entity, amps)

    def set_reserved_current(self, amps: float) -> None:
        self.entities["reserved_current"].native_value = amps

    def adopt(self, name: str, *, amps: float, setpoint: float | None = None) -> None:
        """Put a load into the steady on-and-under-control state."""
        load_state = self.state.controllable_loads[name]
        self.set_switch(name, True)
        self.set_load_amps(name, amps)
        load_state.is_under_load_control = True
        load_state.on_since = datetime.now() - _timedelta(seconds=3600)
        load_state.last_toggled = datetime.now() - _timedelta(seconds=3600)
        load_state.last_throttled = datetime.now() - _timedelta(seconds=3600)
        plan = self.plan.controllable_loads[name]
        plan.is_on = True
        plan.expected_load_amps = setpoint if setpoint is not None else amps
        plan.throttle_amps = setpoint if setpoint is not None else amps
        if setpoint is not None:
            self.set_throttle_setpoint(name, setpoint)

    # -- driving ---------------------------------------------------------
    def recalculate(self, *, respect_interval: bool = False) -> None:
        """Run one planning cycle."""
        if not respect_interval:
            self.state.last_recalculation = None
        _run(recalculate_load_control(self.hass, self.entry_id))

    def available_power(self) -> tuple[float, float]:
        """Run just the available-power calculation."""
        return _run(calculate_effective_available_power(self.hass, self.entry_id))

    def abort(self, force: bool = False) -> None:
        _run(safety_abort(self.hass, self.entry_id, force))

    # -- outputs ---------------------------------------------------------
    def uncontrolled_amps(self) -> float:
        return self.entities["uncontrolled_load_sensor"].value

    def available_amps(self) -> float:
        return self.entities["available_load_sensor"].value

    def setpoint_for(self, name: str) -> float | None:
        """The last throttle setpoint actually commanded for a load."""
        cfg = self.config.controllable_loads[name]
        if not cfg.throttle_amps_entity:
            return None
        calls = self.hass.services.calls_for(cfg.throttle_amps_entity)
        return calls[-1].data["value"] if calls else None

    def was_turned_off(self, name: str) -> bool:
        cfg = self.config.controllable_loads[name]
        return any(
            c.service == "turn_off"
            for c in self.hass.services.calls_for(cfg.switch_entity)
        )

    def was_turned_on(self, name: str) -> bool:
        cfg = self.config.controllable_loads[name]
        return any(
            c.service == "turn_on"
            for c in self.hass.services.calls_for(cfg.switch_entity)
        )

    def planned_on(self, name: str) -> bool:
        return self.plan.controllable_loads[name].is_on

    def clear_calls(self) -> None:
        self.hass.services.clear()


def _timedelta(**kwargs):
    from datetime import timedelta

    return timedelta(**kwargs)


def _run(coro):
    return asyncio.run(coro)


def build(
    config_data: dict,
    *,
    entry_id: str = "test_entry",
    hass: FakeHass | None = None,
) -> Harness:
    """Set the integration up against a fake Home Assistant.

    Mirrors async_setup_entry without the platform plumbing. Pass an entry_id
    and an existing hass to stand a second config entry up alongside the first.
    """
    fresh = hass is None
    hass = hass if hass is not None else FakeHass()

    # Each entry owns its config, state and plan.
    if fresh:
        zerogrid.CONFIGS.clear()
        zerogrid.STATES.clear()
        zerogrid.PLANS.clear()
    config = zerogrid.CONFIGS[entry_id] = Config()
    state = zerogrid.STATES[entry_id] = State()
    plan = zerogrid.PLANS[entry_id] = PlanState()

    # Seed entity states before parsing so initialise_state sees them.
    for block in config_data.get("controllable_loads", []):
        hass.states.set(block["switch_entity"], "off")
        hass.states.set(block["load_amps_entity"], 0)
        if block.get("throttle_amps_entity"):
            hass.states.set(block["throttle_amps_entity"], 0)
        if block.get("can_turn_on_entity"):
            hass.states.set(block["can_turn_on_entity"], "on")
    hass.states.set(config_data["house_consumption_amps_entity"], 0)
    if config_data.get("solar_generation_amps_entity"):
        hass.states.set(config_data["solar_generation_amps_entity"], 0)

    parse_config(config, config_data)
    initialise_state(hass, config, state, plan)

    entities = {
        ENABLE_LOAD_CONTROL_SWITCH_ID: FakeSwitchEntity(True),
        ALLOW_GRID_IMPORT_SWITCH_ID: FakeSwitchEntity(True),
        "reserved_current": FakeNumberEntity(0.0),
        "available_load_sensor": FakeSensorEntity(),
        "controlled_load_sensor": FakeSensorEntity(),
        "uncontrolled_load_sensor": FakeSensorEntity(),
        "max_safe_load_sensor": FakeSensorEntity(),
        "overload_sensor": FakeSensorEntity(),
        "safety_abort_sensor": FakeSensorEntity(),
    }

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry_id] = {
        "entry": None,
        "config": config,
        "state": state,
        "plan": plan,
        "entities": entities,
    }

    state.enable_load_control = True
    state.allow_grid_import = True

    harness = Harness(hass, entry_id, config, state, plan, entities)
    hass.states.listeners.append(harness._on_entity_state)
    return harness


@pytest.fixture
def make():
    """Return the harness builder."""
    return build


@pytest.fixture
def house():
    """A house resembling the one this integration was written for.

    63A main, a fixed hot water cylinder and a throttleable car charger, in
    that priority order, plus a small fixed dehumidifier below them.
    """
    return build(
        {
            "name": "ZeroGrid",
            "max_total_load_amps": 63.0,
            "max_grid_import_amps": 63.0,
            "max_solar_generation_amps": 42.0,
            "safety_margin_amps": 3.0,
            "recalculate_interval_seconds": 30,
            "house_consumption_amps_entity": "sensor.house_current",
            "controllable_loads": [
                load(
                    "Hot Water Cylinder",
                    "climate.hot_water",
                    "sensor.hot_water_current",
                    min_amps=14.0,
                    max_amps=14.0,
                    min_toggle_interval_seconds=900,
                    min_throttle_interval_seconds=10,
                    load_measurement_delay_seconds=10,
                ),
                load(
                    "Car Charger",
                    "switch.car_charger",
                    "sensor.car_charger_current",
                    min_amps=1.0,
                    max_amps=32.0,
                    throttle_amps_entity="number.car_charging_amps",
                    min_toggle_interval_seconds=1800,
                    min_throttle_interval_seconds=9,
                    load_measurement_delay_seconds=60,
                ),
                load(
                    "Dehumidifier",
                    "humidifier.dehumidifier",
                    "sensor.dehumidifier_current",
                    min_amps=1.0,
                    max_amps=1.0,
                    min_toggle_interval_seconds=900,
                    min_throttle_interval_seconds=10,
                ),
            ],
        }
    )
