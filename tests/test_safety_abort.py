"""Tests for the safety abort path.

Safety abort cuts every load the integration controls. It is reached when the
house meter stops reporting, so the grace period before it fires matters: it
is the difference between riding out a momentary sensor blip and cutting the
house's loads because of one.
"""

from datetime import datetime, timedelta

import pytest

from custom_components.zerogrid import clear_safety_abort


def test_abort_waits_out_a_brief_sensor_blip(house):
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.clear_calls()

    house.abort()  # first unusable reading starts the clock

    assert not house.state.safety_abort_active
    assert not house.was_turned_off("Car Charger")


def test_abort_fires_once_the_grace_period_has_passed(house):
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.clear_calls()

    house.abort()
    house.state.safety_abort_timestamp = datetime.now() - timedelta(seconds=121)
    house.abort()

    assert house.state.safety_abort_active
    assert house.was_turned_off("Car Charger")


def test_forced_abort_does_not_wait(house):
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.clear_calls()

    house.abort(force=True)

    assert house.state.safety_abort_active
    assert house.was_turned_off("Car Charger")


def test_abort_releases_each_load_it_could_not_reach(house):
    """A load we failed to switch must stop being budgeted for."""
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.hass.states.set("switch.car_charger", "unavailable")

    house.abort(force=True)

    load_state = house.state.controllable_loads["Car Charger"]
    assert not load_state.is_under_load_control
    assert not load_state.is_on


def test_recovery_restarts_the_grace_period(house):
    """A blip that recovers must not spend the next blip's grace period.

    The clock starts on the first bad reading, before abort is active. If a
    good reading only clears it while abort is already active, the timestamp
    from a recovered blip survives and the next blip aborts immediately.
    """
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)

    house.abort()  # blip starts the clock
    assert house.state.safety_abort_timestamp is not None

    clear_safety_abort(house.hass, house.entry_id)  # meter reports again

    assert house.state.safety_abort_timestamp is None


def test_a_later_blip_still_gets_its_full_grace_period(house):
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)

    house.abort()  # a blip, an hour ago
    house.state.safety_abort_timestamp = datetime.now() - timedelta(seconds=3600)
    clear_safety_abort(house.hass, house.entry_id)  # which recovered
    house.clear_calls()

    house.abort()  # a fresh blip now

    assert not house.state.safety_abort_active
    assert not house.was_turned_off("Car Charger")


def test_clearing_an_active_abort_resets_it(house):
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.abort(force=True)
    assert house.state.safety_abort_active

    clear_safety_abort(house.hass, house.entry_id)

    assert not house.state.safety_abort_active
    assert house.state.safety_abort_timestamp is None
    assert house.entities["safety_abort_sensor"].state is False


def test_planning_is_skipped_while_abort_is_active(house):
    house.set_house_amps(30.0)
    house.adopt("Car Charger", amps=20.0, setpoint=20)
    house.abort(force=True)
    house.clear_calls()

    house.recalculate()

    assert house.hass.services.calls == []
