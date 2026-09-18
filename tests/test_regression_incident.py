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
