"""Tests for the available-power calculation.

The uncontrolled figure is the house meter minus each controlled load's own
meter. Those meters are not read at the same instant, so most of what matters
here is what happens when they disagree.
"""

from datetime import datetime, timedelta

import pytest


def test_uncontrolled_load_is_the_house_minus_the_loads_we_run(house):
    house.set_house_amps(55.4)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=23.3, setpoint=23)

    available, max_safe = house.available_power()

    assert house.uncontrolled_amps() == pytest.approx(18.9, abs=0.01)
    assert available == pytest.approx(63.0 - 18.9, abs=0.01)
    assert max_safe == 63.0


def test_reserved_current_is_held_back_on_top_of_uncontrolled_load(house):
    house.set_house_amps(55.4)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=23.3, setpoint=23)
    house.set_reserved_current(7.0)

    available, _ = house.available_power()

    assert available == pytest.approx(63.0 - 18.9 - 7.0, abs=0.01)


def test_a_load_we_do_not_control_counts_against_us(house):
    """A manually run load is uncontrolled load, not headroom."""
    house.set_house_amps(30.0)
    house.set_switch("Car Charger", True)
    house.set_load_amps("Car Charger", 20.0)
    house.state.controllable_loads["Car Charger"].is_under_load_control = False

    house.available_power()

    assert house.uncontrolled_amps() == pytest.approx(30.0, abs=0.01)


def test_a_soft_starting_load_is_counted_at_what_we_planned_for_it(house):
    """A load that has not drawn its share yet must not look free."""
    house.set_house_amps(20.0)
    house.adopt("Hot Water Cylinder", amps=0.0, setpoint=14.0)
    load_state = house.state.controllable_loads["Hot Water Cylinder"]
    load_state.on_since = datetime.now()  # just switched on
    house.plan.controllable_loads["Hot Water Cylinder"].expected_load_amps = 14.0

    house.available_power()

    # 20A house less the 14A we expect the cylinder to pull once it ramps.
    assert house.uncontrolled_amps() == pytest.approx(6.0, abs=0.01)


def test_reading_is_held_while_a_load_is_still_acting_on_a_new_setpoint(house):
    """The charger's meter lags its setpoint; the house meter does not.

    Differencing them mid-change counted the charger's own draw as load we
    could not manage, which shed it instead of dialling it back.
    """
    house.set_house_amps(55.4)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=23.3, setpoint=23)
    house.available_power()
    settled = house.uncontrolled_amps()

    # The charger resumes at a higher setpoint; the house meter sees it first.
    house.state.controllable_loads["Car Charger"].last_throttled = datetime.now()
    house.set_house_amps(58.61)
    house.set_load_amps("Car Charger", 1.5)  # meter has not caught up

    house.available_power()

    assert house.uncontrolled_amps() == pytest.approx(settled, abs=0.01)


def test_loads_cannot_draw_more_than_the_whole_house(house):
    """A stale high reading must not be reported as zero uncontrolled load.

    Treating it as zero advertised headroom that did not exist.
    """
    house.set_house_amps(55.4)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=23.3, setpoint=23)
    house.available_power()
    settled = house.uncontrolled_amps()

    # The car stops drawing. The house meter drops first, so subtracting the
    # charger's stale reading gives an impossible negative.
    house.set_house_amps(33.02)

    house.available_power()

    assert house.uncontrolled_amps() == pytest.approx(settled, abs=0.01)
    assert house.available_amps() == pytest.approx(63.0 - settled, abs=0.01)


def test_a_genuine_change_in_uncontrolled_load_is_picked_up_at_once(house):
    """Holding a reading must not blind us to a real change."""
    house.set_house_amps(55.4)
    house.adopt("Hot Water Cylinder", amps=13.2)
    house.adopt("Car Charger", amps=23.3, setpoint=23)
    house.available_power()

    house.set_house_amps(80.4)  # a 25A oven comes on, nothing mid-change
    house.available_power()
    assert house.uncontrolled_amps() == pytest.approx(43.9, abs=0.01)

    house.set_house_amps(45.4)  # and goes off again
    house.available_power()
    assert house.uncontrolled_amps() == pytest.approx(8.9, abs=0.01)


def test_solar_adds_to_the_budget_only_up_to_its_cap(make):
    from conftest import load

    harness = make(
        {
            "name": "ZeroGrid",
            "max_total_load_amps": 100.0,
            "max_grid_import_amps": 63.0,
            "max_solar_generation_amps": 20.0,
            "house_consumption_amps_entity": "sensor.house_current",
            "solar_generation_amps_entity": "sensor.solar_current",
            "controllable_loads": [
                load("Car Charger", "switch.car", "sensor.car_current")
            ],
        }
    )
    harness.set_house_amps(10.0)
    harness.set_solar_amps(50.0)  # well over the cap

    available, max_safe = harness.available_power()

    assert max_safe == 83.0  # 63A grid + 20A solar cap
    assert available == pytest.approx(83.0 - 10.0, abs=0.01)


def test_no_grid_import_means_only_solar_is_available(make):
    from conftest import load
    from custom_components.zerogrid.const import ALLOW_GRID_IMPORT_SWITCH_ID

    harness = make(
        {
            "name": "ZeroGrid",
            "max_total_load_amps": 100.0,
            "max_grid_import_amps": 63.0,
            "max_solar_generation_amps": 20.0,
            "house_consumption_amps_entity": "sensor.house_current",
            "solar_generation_amps_entity": "sensor.solar_current",
            "controllable_loads": [
                load("Car Charger", "switch.car", "sensor.car_current")
            ],
        }
    )
    harness.entities[ALLOW_GRID_IMPORT_SWITCH_ID].is_on = False
    harness.state.allow_grid_import = False
    harness.set_house_amps(5.0)
    harness.set_solar_amps(12.0)

    available, _ = harness.available_power()

    assert available == pytest.approx(12.0 - 5.0, abs=0.01)
