"""Tests for the overload pass.

This pass works straight from the house meter rather than the derived
uncontrolled figure, so it is the backstop when the derived figure is wrong.
"""

from datetime import datetime, timedelta

import pytest


def _force_overload_window(harness):
    """Overload has to persist for a recalculation interval before it acts."""
    harness.state.overload_timestamp = datetime.now() - timedelta(
        seconds=harness.config.recalculate_interval_seconds + 1
    )


def test_no_overload_below_the_limit_plus_margin(house):
    house.set_house_amps(64.0)  # 63A limit, 3A margin
    house.adopt("Car Charger", amps=30.0, setpoint=30)
    house.clear_calls()

    house.recalculate()

    assert not house.entities["overload_sensor"].state
    assert not house.was_turned_off("Car Charger")


def test_overload_must_persist_before_anything_is_cut(house):
    """A single reading over the limit is not acted on."""
    house.set_house_amps(80.0)
    house.adopt("Car Charger", amps=40.0, setpoint=32)
    house.clear_calls()

    house.recalculate()

    assert not house.was_turned_off("Car Charger")


def test_a_sustained_overload_dials_a_throttleable_load_back_first(house):
    house.set_house_amps(80.0)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=40.0, setpoint=32)
    _force_overload_window(house)
    house.clear_calls()

    house.recalculate()

    assert house.entities["overload_sensor"].state
    # The charger gives power back without going off.
    assert house.plan.controllable_loads["Car Charger"].throttle_amps == 1
    assert not house.was_turned_off("Car Charger")


def test_a_fixed_load_is_shed_when_throttling_cannot_cover_it(house):
    """With no throttleable load to dial back, something has to go."""
    house.set_house_amps(80.0)
    house.adopt("Hot Water Cylinder", amps=40.0)
    house.adopt("Dehumidifier", amps=1.0)
    _force_overload_window(house)
    house.clear_calls()

    house.recalculate()

    assert house.entities["overload_sensor"].state
    # Least important first.
    assert house.was_turned_off("Dehumidifier")


def test_shedding_starts_with_the_least_important_load(house):
    house.set_house_amps(95.0)
    house.adopt("Hot Water Cylinder", amps=14.0)
    house.adopt("Dehumidifier", amps=1.0)
    _force_overload_window(house)
    house.clear_calls()

    house.recalculate()

    assert house.was_turned_off("Dehumidifier")


def test_a_load_we_do_not_control_is_never_shed(house):
    house.set_house_amps(95.0)
    house.set_switch("Dehumidifier", True)
    house.set_load_amps("Dehumidifier", 1.0)
    house.state.controllable_loads["Dehumidifier"].is_under_load_control = False
    house.plan.controllable_loads["Dehumidifier"].is_on = True
    _force_overload_window(house)
    house.clear_calls()

    house.recalculate()

    assert not house.was_turned_off("Dehumidifier")


def test_overload_clears_once_consumption_comes_back_down(house):
    house.set_house_amps(80.0)
    house.adopt("Hot Water Cylinder", amps=40.0)
    _force_overload_window(house)
    house.recalculate()
    assert house.entities["overload_sensor"].state

    house.set_house_amps(30.0)
    house.set_load_amps("Hot Water Cylinder", 13.0)
    house.recalculate()

    assert not house.entities["overload_sensor"].state
    assert house.state.overload_timestamp is None
