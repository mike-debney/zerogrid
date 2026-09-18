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


def test_a_fixed_load_still_reserves_its_minimum_when_its_meter_dips(house):
    """A momentary low reading must not look like free capacity.

    The cylinder is a 14A load. If its meter reads 0 for one cycle while it is
    still on, budgeting 0 for it hands 14A to something else.
    """
    house.set_house_amps(55.4)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=23.3, setpoint=23)
    house.recalculate()
    settled_setpoint = house.plan.controllable_loads["Car Charger"].throttle_amps

    house.set_load_amps("Hot Water Cylinder", 0.0)
    house.recalculate()

    assert house.plan.controllable_loads["Hot Water Cylinder"].expected_load_amps >= 14.0
    assert house.plan.controllable_loads["Car Charger"].throttle_amps <= settled_setpoint


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
