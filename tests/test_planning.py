"""Tests for the load planning passes: priority, throttling and rate limits."""

from datetime import datetime, timedelta

import pytest


def test_higher_priority_load_is_served_first(house):
    """Priority follows the configured order, cylinder before charger."""
    house.set_house_amps(20.0)
    house.set_can_turn_on("Hot Water Cylinder", True)
    house.set_can_turn_on("Car Charger", True)

    house.recalculate()

    assert house.planned_on("Hot Water Cylinder")


def test_throttleable_load_is_given_the_power_left_over(house):
    house.set_house_amps(53.2)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=20.0, setpoint=20)

    house.recalculate()

    # 63A less 20A uncontrolled leaves 43A. The cylinder draws 13.2A and the
    # dehumidifier reserves its 1A minimum, leaving 28.8A for the charger,
    # floored to a whole amp.
    assert house.plan.controllable_loads["Car Charger"].throttle_amps == 28


def test_a_load_is_never_throttled_above_its_maximum(house):
    house.set_house_amps(15.0)
    house.adopt("Car Charger", amps=10.0, setpoint=10)

    house.recalculate()

    assert house.plan.controllable_loads["Car Charger"].throttle_amps == 32


def test_a_load_is_dialled_back_rather_than_cut_when_power_is_short(house):
    """Shedding is a last resort; a throttleable load gives power back first."""
    house.set_house_amps(55.0)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=23.0, setpoint=23)
    house.set_reserved_current(20.0)  # squeeze the budget hard
    house.clear_calls()

    house.recalculate()

    assert house.planned_on("Car Charger")
    assert not house.was_turned_off("Car Charger")
    assert house.plan.controllable_loads["Car Charger"].throttle_amps < 23


def test_a_load_may_be_dialled_back_while_throttle_rate_limited(house):
    """Rate limiting guards ramping up, not backing off.

    Holding a load at its old setpoint during a spike shed other loads instead.
    """
    house.set_house_amps(55.0)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=23.0, setpoint=23)
    house.state.controllable_loads["Car Charger"].last_throttled = datetime.now()
    house.set_reserved_current(20.0)

    house.recalculate()

    assert house.plan.controllable_loads["Car Charger"].throttle_amps < 23


def test_a_throttle_rate_limited_load_is_not_ramped_up(house):
    house.set_house_amps(53.2)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.state.controllable_loads["Car Charger"].last_throttled = datetime.now()

    house.recalculate()

    assert house.plan.controllable_loads["Car Charger"].throttle_amps == 20


def test_a_switch_rate_limited_load_is_not_turned_off(house):
    house.set_house_amps(60.0)
    house.adopt("Dehumidifier", amps=1.0)
    house.state.controllable_loads["Dehumidifier"].last_toggled = datetime.now()
    house.clear_calls()

    house.recalculate()

    assert not house.was_turned_off("Dehumidifier")


def test_a_switch_rate_limited_load_is_not_turned_on(house):
    house.set_house_amps(5.0)
    house.set_switch("Dehumidifier", False)
    house.state.controllable_loads["Dehumidifier"].last_toggled = datetime.now()
    house.clear_calls()

    house.recalculate()

    assert not house.was_turned_on("Dehumidifier")


def test_an_external_constraint_keeps_a_load_off(make):
    from conftest import load

    harness = make(
        {
            "name": "ZeroGrid",
            "max_total_load_amps": 63.0,
            "max_grid_import_amps": 63.0,
            "max_solar_generation_amps": 0.0,
            "house_consumption_amps_entity": "sensor.house_current",
            "controllable_loads": [
                load(
                    "Car Charger",
                    "switch.car",
                    "sensor.car_current",
                    can_turn_on_entity="binary_sensor.car_can_charge",
                )
            ],
        }
    )
    harness.set_house_amps(5.0)
    harness.set_can_turn_on("Car Charger", False)
    harness.clear_calls()

    harness.recalculate()

    assert not harness.planned_on("Car Charger")
    assert not harness.was_turned_on("Car Charger")


def test_a_manually_run_load_is_left_alone(house):
    """If we did not turn it on, we do not turn it off."""
    house.set_house_amps(62.0)
    house.set_switch("Car Charger", True)
    house.set_load_amps("Car Charger", 30.0)
    house.state.controllable_loads["Car Charger"].is_under_load_control = False
    house.clear_calls()

    house.recalculate()

    assert not house.was_turned_off("Car Charger")


def test_turning_a_load_on_puts_it_under_our_control(house):
    house.set_house_amps(5.0)
    house.set_can_turn_on("Dehumidifier", True)
    house.clear_calls()

    house.recalculate()

    assert house.was_turned_on("Dehumidifier")
    load_state = house.state.controllable_loads["Dehumidifier"]
    assert load_state.is_under_load_control
    assert load_state.on_since is not None


def test_recalculation_is_rate_limited_to_once_a_second(house):
    house.set_house_amps(20.0)
    house.recalculate()
    house.clear_calls()

    # A second run straight away, respecting the limiter, must do nothing.
    house.recalculate(respect_interval=True)

    assert house.hass.services.calls == []


def test_disabling_load_control_clears_the_rate_limits(house):
    from custom_components.zerogrid.const import ENABLE_LOAD_CONTROL_SWITCH_ID

    house.set_house_amps(20.0)
    house.adopt("Car Charger", amps=10.0, setpoint=10)
    house.state.controllable_loads["Car Charger"].last_toggled = datetime.now()

    house.entities[ENABLE_LOAD_CONTROL_SWITCH_ID].is_on = False
    house.recalculate()

    assert house.state.controllable_loads["Car Charger"].last_toggled is None
    assert house.state.controllable_loads["Car Charger"].last_throttled is None


def test_planning_is_skipped_until_the_house_meter_reports(house):
    house.state.house_consumption_initialised = False
    house.clear_calls()

    house.recalculate()

    assert house.hass.services.calls == []
