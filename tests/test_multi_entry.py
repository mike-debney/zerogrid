"""Tests for keeping two config entries apart.

The planning code used to run against module level CONFIG/STATE/PLAN, which
pointed at whichever entry was set up last. Planning one entry then used the
other entry's configuration and readings.
"""

import pytest
from conftest import build, load

import custom_components.zerogrid as zerogrid


def _entry(name, switch, meter, *, limit, entry_id, hass=None, **kwargs):
    return build(
        {
            "name": name,
            "max_total_load_amps": limit,
            "max_grid_import_amps": limit,
            "max_solar_generation_amps": 0.0,
            "safety_margin_amps": 3.0,
            "recalculate_interval_seconds": 30,
            "house_consumption_amps_entity": f"sensor.{entry_id}_house",
            "controllable_loads": [load(name, switch, meter, **kwargs)],
        },
        entry_id=entry_id,
        hass=hass,
    )


@pytest.fixture
def two_entries():
    """Two entries in one Home Assistant, with different limits and loads."""
    first = _entry(
        "Garage Charger",
        "switch.garage",
        "sensor.garage_current",
        limit=63.0,
        entry_id="entry_one",
        throttle_amps_entity="number.garage_amps",
    )
    second = _entry(
        "Shed Heater",
        "switch.shed",
        "sensor.shed_current",
        limit=20.0,
        entry_id="entry_two",
        hass=first.hass,
    )
    return first, second


def test_each_entry_keeps_its_own_loads(two_entries):
    first, second = two_entries

    assert set(first.config.controllable_loads) == {"Garage Charger"}
    assert set(second.config.controllable_loads) == {"Shed Heater"}


def test_each_entry_keeps_its_own_limits(two_entries):
    first, second = two_entries

    assert first.config.max_total_load_amps == 63.0
    assert second.config.max_total_load_amps == 20.0


def test_planning_one_entry_uses_that_entry_s_budget(two_entries):
    """The entry set up last must not decide the other entry's headroom."""
    first, second = two_entries
    first.set_house_amps(10.0)
    second.set_house_amps(10.0)

    first.available_power()
    assert first.available_amps() == pytest.approx(53.0, abs=0.01)  # 63A limit

    second.available_power()
    assert second.available_amps() == pytest.approx(10.0, abs=0.01)  # 20A limit


def test_readings_do_not_leak_between_entries(two_entries):
    first, second = two_entries
    first.set_house_amps(50.0)
    second.set_house_amps(5.0)

    assert first.state.house_consumption_amps == 50.0
    assert second.state.house_consumption_amps == 5.0


def test_planning_one_entry_only_switches_that_entry_s_loads(two_entries):
    first, second = two_entries
    first.set_house_amps(5.0)
    second.set_house_amps(5.0)
    first.clear_calls()

    first.recalculate()

    assert not any(
        call.data.get("entity_id") == "switch.shed" for call in first.hass.services.calls
    )


def test_a_second_entry_does_not_disturb_the_first_s_plan(two_entries):
    first, second = two_entries
    first.set_house_amps(20.0)
    first.adopt("Garage Charger", amps=10.0, setpoint=10)
    second.set_house_amps(19.0)
    second.adopt("Shed Heater", amps=8.0)

    second.recalculate()
    first.recalculate()

    # The charger is planned against its own 63A entry, not the shed's 20A one.
    assert first.plan.controllable_loads["Garage Charger"].throttle_amps > 10
