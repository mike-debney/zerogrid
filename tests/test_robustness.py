"""Tests for handling entities that misbehave.

Sensors report strings, devices answer late, and meters dip. None of that
should stop the planner or hand out capacity that is not there.
"""

from datetime import datetime, timedelta

import pytest

from custom_components.zerogrid.helpers import parse_amps
import sys

State = sys.modules["homeassistant.core"].State


# -- parsing ------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "None", "n/a", "12.3.4", "unknown", "unavailable"])
def test_a_value_that_is_not_a_number_is_rejected(value):
    assert parse_amps(State("sensor.x", value)) is None


def test_a_missing_entity_is_rejected():
    assert parse_amps(None) is None


@pytest.mark.parametrize(
    ("value", "expected"), [("0", 0.0), ("13.2", 13.2), ("-1.5", -1.5), ("1e1", 10.0)]
)
def test_a_numeric_value_is_read(value, expected):
    assert parse_amps(State("sensor.x", value)) == expected


# -- the house meter ----------------------------------------------------


def test_an_unreadable_house_reading_does_not_stop_planning(house):
    """It must be treated as no reading, not raise inside the listener."""
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.hass.states.set(house.config.house_consumption_amps_entity, "")

    # The last good reading stands and planning still runs.
    house.recalculate()

    assert house.state.house_consumption_amps == 30.0


def test_an_unreadable_load_meter_leaves_the_last_reading_alone(house):
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.hass.states.set("sensor.car_charger_current", "not a number")

    house.recalculate()

    assert house.state.controllable_loads["Car Charger"].current_load_amps == 20.0


def test_startup_survives_a_sensor_reporting_nonsense(make):
    from conftest import load

    harness = make(
        {
            "name": "ZeroGrid",
            "max_total_load_amps": 63.0,
            "max_grid_import_amps": 63.0,
            "max_solar_generation_amps": 0.0,
            "house_consumption_amps_entity": "sensor.house_current",
            "controllable_loads": [
                load("Car Charger", "switch.car", "sensor.car_current")
            ],
        }
    )
    # Built without raising; nothing was read, so it stays at zero.
    assert harness.state.house_consumption_amps == 0.0


# -- a meter that dips --------------------------------------------------


def test_a_load_that_has_not_ramped_up_yet_reserves_its_minimum(house):
    """Straight after a turn on, the meter has not caught up.

    Until the measurement delay has passed the load is budgeted at its
    configured minimum rather than at whatever its meter happens to say.
    """
    house.set_house_amps(20.0)
    house.adopt("Hot Water Cylinder", amps=0.0)
    house.state.controllable_loads["Hot Water Cylinder"].on_since = datetime.now()

    house.recalculate()

    assert house.plan.controllable_loads["Hot Water Cylinder"].expected_load_amps == 14.0


def test_an_idle_thermostatic_load_frees_its_capacity(house):
    """A load that is on but not drawing must not hold its rating.

    A hot water cylinder or a heater sits on with its element cycled off for
    long stretches. Reserving its full rating throughout leaves the charger
    below it on a fraction of the power the house actually has spare.
    """
    house.set_house_amps(21.5)
    house.adopt("Hot Water Cylinder", amps=0.05)  # on, element cycled off
    house.adopt("Car Charger", amps=15.0, setpoint=15)

    house.recalculate()

    # The cylinder is budgeted at what it draws, not its 14A rating, so the
    # charger gets the rest of the budget.
    assert house.plan.controllable_loads["Hot Water Cylinder"].expected_load_amps < 1.0
    assert house.plan.controllable_loads["Car Charger"].throttle_amps == 32


# -- a switch that answers late -----------------------------------------


def test_a_switch_command_is_not_repeated_while_the_load_is_still_answering(house):
    """The device has accepted the command; it just has not reported yet."""
    house.set_house_amps(5.0)
    house.hass.services.apply = False  # the switch will not report back
    house.set_can_turn_on("Dehumidifier", True)
    house.clear_calls()

    house.recalculate()
    first = len(house.hass.services.calls_for("humidifier.dehumidifier"))
    house.recalculate()
    house.recalculate()
    after = len(house.hass.services.calls_for("humidifier.dehumidifier"))

    assert first == 1
    assert after == 1


def test_a_repeat_does_not_push_the_measurement_window_out(house):
    """Each repeat used to reset on_since, extending the settling window."""
    house.set_house_amps(5.0)
    house.hass.services.apply = False
    house.set_can_turn_on("Dehumidifier", True)
    house.recalculate()

    load_state = house.state.controllable_loads["Dehumidifier"]
    first_on_since = load_state.on_since
    first_toggled = load_state.last_toggled

    house.recalculate()

    assert load_state.on_since == first_on_since
    assert load_state.last_toggled == first_toggled


def test_a_command_is_sent_again_once_it_has_clearly_been_lost(house):
    house.set_house_amps(5.0)
    house.hass.services.apply = False
    house.set_can_turn_on("Dehumidifier", True)
    house.recalculate()
    house.clear_calls()

    load_state = house.state.controllable_loads["Dehumidifier"]
    load_state.switch_command_since = datetime.now() - timedelta(seconds=31)
    house.recalculate()

    assert len(house.hass.services.calls_for("humidifier.dehumidifier")) == 1


def test_the_command_is_forgotten_once_the_load_reports(house):
    """A pending command must not outlive the load answering it."""
    house.set_house_amps(5.0)
    house.set_can_turn_on("Dehumidifier", True)
    house.recalculate()
    assert house.state.controllable_loads["Dehumidifier"].is_on

    house.recalculate()  # the next cycle sees the load reporting on

    load_state = house.state.controllable_loads["Dehumidifier"]
    assert load_state.switch_command_on is None
    assert load_state.switch_command_since is None


def test_a_pending_command_never_blocks_the_opposite_command(house):
    """Having asked a load to turn on must not stop us shedding it later."""
    house.set_house_amps(5.0)
    house.set_can_turn_on("Dehumidifier", True)
    house.recalculate()
    house.recalculate()
    # Let everything settle: no rate limits, no pending setpoint changes.
    for load_state in house.state.controllable_loads.values():
        load_state.last_toggled = None
        load_state.last_throttled = datetime.now() - timedelta(seconds=600)
        load_state.on_since = datetime.now() - timedelta(seconds=600)
    house.clear_calls()

    # Now the house is well over budget and the load has to go.
    house.set_house_amps(70.0)
    house.recalculate()

    assert house.was_turned_off("Dehumidifier")


# -- a switch that is not ready when we start ---------------------------


def test_a_load_unavailable_at_startup_is_still_trusted_once_it_reports(house):
    """The reload case: the thermostat entity lags the integration.

    initialise_state skips a switch entity that is not reporting yet, so the
    load never gets an on_since from there, and we never turn it on ourselves
    because it is already on. Without an on_since its meter is never trusted
    and it holds its full rating for as long as it runs.
    """
    cfg = house.config.controllable_loads["Hot Water Cylinder"]
    load_state = house.state.controllable_loads["Hot Water Cylinder"]

    # As after a reload: nothing known about the load yet.
    load_state.is_on = False
    load_state.on_since = None
    load_state.is_under_load_control = True

    # The thermostat finally reports, a few seconds late.
    house.hass.states.set(cfg.switch_entity, "heat")

    assert load_state.is_on
    assert load_state.on_since is not None


def test_an_idle_load_that_started_unavailable_frees_its_capacity(house):
    """The whole point: its capacity reaches the loads below it."""
    cfg = house.config.controllable_loads["Hot Water Cylinder"]
    load_state = house.state.controllable_loads["Hot Water Cylinder"]
    load_state.is_on = False
    load_state.on_since = None
    load_state.is_under_load_control = True

    house.hass.states.set(cfg.switch_entity, "heat")
    house.set_load_amps("Hot Water Cylinder", 0.056)
    # Past the measurement delay, so its meter should be believed.
    load_state.on_since = datetime.now() - timedelta(seconds=60)

    house.set_house_amps(21.6)
    house.adopt("Car Charger", amps=15.0, setpoint=15)
    house.recalculate()

    assert house.plan.controllable_loads["Hot Water Cylinder"].expected_load_amps < 1.0
    assert house.plan.controllable_loads["Car Charger"].throttle_amps == 32


def test_a_load_reporting_off_forgets_when_it_came_on(house):
    cfg = house.config.controllable_loads["Hot Water Cylinder"]
    load_state = house.state.controllable_loads["Hot Water Cylinder"]

    house.hass.states.set(cfg.switch_entity, "heat")
    assert load_state.on_since is not None

    house.hass.states.set(cfg.switch_entity, "off")

    assert not load_state.is_on
    assert load_state.on_since is None


def test_an_unavailable_switch_report_changes_nothing(house):
    cfg = house.config.controllable_loads["Hot Water Cylinder"]
    load_state = house.state.controllable_loads["Hot Water Cylinder"]
    house.hass.states.set(cfg.switch_entity, "heat")
    was_on_since = load_state.on_since

    house.hass.states.set(cfg.switch_entity, "unavailable")

    assert load_state.is_on
    assert load_state.on_since == was_on_since
