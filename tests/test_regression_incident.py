"""End to end replay of the 2026-09-17 charger shed.

Readings are the recorded sensor values from that morning. The sequence is
the one that mattered: a charger drawing steadily, a brief pause, and the
resume that followed. Each step drives the whole planner, not just the
available-power calculation, so the assertions are about what the integration
actually did to the charger.

What went wrong, in the readings below:

  11:05:48  house 55.42A, charger 23.3A, holding at 23A
  11:05:57  charger pauses. The house meter drops to 33.02A first, so
            subtracting the charger's stale 23.3A gave a negative figure that
            was reported as no uncontrolled load at all - and 56A of headroom
            that did not exist pushed the setpoint to 32A.
  11:06:02  the charger resumes at the new setpoint. The house meter sees it
            at 58.61A while the charger's own meter still reads 1.5A, so 26A
            of the charger's draw was counted as load we could not manage.
            That left 12.16A of the 63A budget, the cylinder was inside its
            toggle interval, and the charger was cut - then locked out for its
            30 minute toggle interval.
"""

from datetime import datetime, timedelta

import pytest


@pytest.fixture
def incident(house):
    """The house as it stood just before the charger was shed."""
    house.set_house_amps(55.42)
    house.adopt("Hot Water Cylinder", amps=13.209)
    house.adopt("Car Charger", amps=23.3, setpoint=23)
    house.set_reserved_current(7.0)  # holding capacity to charge the battery
    # The cylinder came on a few minutes ago, so it cannot be shed.
    house.state.controllable_loads["Hot Water Cylinder"].last_toggled = (
        datetime.now() - timedelta(seconds=120)
    )
    house.recalculate()
    house.clear_calls()
    return house


def test_steady_state_holds_the_charger_where_it_is(incident):
    incident.recalculate()

    assert incident.planned_on("Car Charger")
    assert not incident.was_turned_off("Car Charger")


def test_a_pause_does_not_hand_out_headroom_that_does_not_exist(incident):
    """The house meter drops before the charger's own meter does."""
    incident.set_house_amps(33.02)

    incident.recalculate()

    # Nothing about the house changed, so the charger must not be ramped up.
    assert incident.plan.controllable_loads["Car Charger"].throttle_amps <= 23
    assert incident.setpoint_for("Car Charger") != 32


def test_the_charger_is_not_shed_when_its_meter_lags_behind(incident):
    """The resume: house meter high, charger's own meter still low."""
    incident.set_house_amps(33.02)
    incident.recalculate()

    incident.state.controllable_loads["Car Charger"].last_throttled = datetime.now()
    incident.set_house_amps(58.61)
    incident.set_load_amps("Car Charger", 1.5)
    incident.clear_calls()

    incident.recalculate()

    assert not incident.was_turned_off("Car Charger")
    assert incident.planned_on("Car Charger")


def test_the_whole_sequence_never_cuts_the_charger(incident):
    """Replay all three readings in order."""
    readings = [
        (55.42, 23.3, False),
        (33.02, 23.3, False),  # pause, charger meter stale high
        (58.61, 1.5, True),  # resume, charger meter stale low
    ]
    for house_amps, charger_amps, just_throttled in readings:
        if just_throttled:
            incident.state.controllable_loads["Car Charger"].last_throttled = (
                datetime.now()
            )
        incident.set_house_amps(house_amps)
        incident.set_load_amps("Car Charger", charger_amps)
        incident.recalculate()

    assert not incident.was_turned_off("Car Charger")
    # And it is still charging, not sitting at its floor.
    assert incident.plan.controllable_loads["Car Charger"].throttle_amps > 1


def test_a_real_overload_during_the_same_sequence_is_still_caught(incident):
    """Holding a stale reading must not disable the overload backstop."""
    incident.set_house_amps(80.0)  # genuinely over the 63A limit
    incident.set_load_amps("Car Charger", 40.0)
    incident.state.overload_timestamp = datetime.now() - timedelta(
        seconds=incident.config.recalculate_interval_seconds + 1
    )
    incident.clear_calls()

    incident.recalculate()

    assert incident.entities["overload_sensor"].state
    assert incident.plan.controllable_loads["Car Charger"].throttle_amps == 1


"""Second recorded incident: the charger held at 15A on 2026-09-18.

Readings from the planner's own log at 11:14:33, with a 63A limit and 3A held
back for the battery:

  uncontrolled load          21.54A
  Hot Water Cylinder   on, drawing  0.058A (rated 14A, element cycled off)
  Mike's Car Charger   on, drawing 15.1A
  Dehumidifier         on, drawing  0A     (rated 1A)
  Bedroom Fan          on, drawing  0.011A (rated 8A, not heating)

The house was pulling 37.66A of its 63A. Budgeting the cylinder and the fan at
their ratings while they sat idle reserved 22A that nothing was drawing, and
the charger was held at 15A instead of its 32A maximum.
"""


@pytest.fixture
def idle_thermostats(make):
    """The house as it stood, with two thermostatic loads on but idle."""
    from conftest import load

    harness = make(
        {
            "name": "ZeroGrid",
            "max_total_load_amps": 63.0,
            "max_grid_import_amps": 63.0,
            "max_solar_generation_amps": 42.0,
            "safety_margin_amps": 3.0,
            "recalculate_interval_seconds": 30,
            "house_consumption_amps_entity": "sensor.house_current",
            "controllable_loads": [
                load("Hot Water Cylinder", "climate.hot_water", "sensor.hot_water_current",
                     min_amps=14.0, max_amps=14.0, min_toggle_interval_seconds=900,
                     load_measurement_delay_seconds=10),
                load("Car Charger", "switch.car_charger", "sensor.car_charger_current",
                     min_amps=1.0, max_amps=32.0,
                     throttle_amps_entity="number.car_charging_amps",
                     min_toggle_interval_seconds=1800),
                load("Dehumidifier", "humidifier.dehumidifier", "sensor.dehumidifier_current",
                     min_amps=1.0, max_amps=1.0, min_toggle_interval_seconds=900),
                load("Bedroom Fan", "climate.bedroom_fan", "sensor.bedroom_fan_current",
                     min_amps=8.0, max_amps=8.0, min_toggle_interval_seconds=600),
            ],
        }
    )
    harness.set_house_amps(36.71)
    harness.adopt("Hot Water Cylinder", amps=0.058)
    harness.adopt("Car Charger", amps=15.1, setpoint=15)
    harness.adopt("Dehumidifier", amps=0.0)
    harness.adopt("Bedroom Fan", amps=0.011)
    harness.set_reserved_current(3.0)
    return harness


def test_idle_loads_do_not_hold_the_charger_down(idle_thermostats):
    idle_thermostats.recalculate()

    charger = idle_thermostats.plan.controllable_loads["Car Charger"]
    assert charger.throttle_amps == 32


def test_the_house_still_stays_inside_its_limit(idle_thermostats):
    """Handing the spare capacity over must not plan past the main."""
    idle_thermostats.recalculate()

    planned = sum(
        p.expected_load_amps
        for p in idle_thermostats.plan.controllable_loads.values()
    )
    uncontrolled = idle_thermostats.uncontrolled_amps()
    reserved = idle_thermostats.entities["reserved_current"].native_value
    assert planned + uncontrolled + reserved <= 63.0


def test_a_thermostatic_load_coming_back_throttles_the_charger_again(idle_thermostats):
    """When the element does come back on, the charger gives the power back."""
    idle_thermostats.recalculate()
    assert idle_thermostats.plan.controllable_loads["Car Charger"].throttle_amps == 32

    # The cylinder starts drawing its 14A and the house meter follows.
    idle_thermostats.set_load_amps("Hot Water Cylinder", 13.9)
    idle_thermostats.set_house_amps(36.71 + 13.9)
    idle_thermostats.state.controllable_loads["Car Charger"].last_throttled = None
    idle_thermostats.recalculate()

    assert idle_thermostats.plan.controllable_loads["Car Charger"].throttle_amps < 32
    assert not idle_thermostats.was_turned_off("Car Charger")
