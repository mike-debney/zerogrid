"""A load whose integration stops accepting commands.

On 2026-09-23 the car's cloud API timed out for eight minutes. Every command
to it raised httpx.ConnectTimeout, which escaped execute_plan: the loads after
the car were never acted on, and the throttle was sent again on nearly every
recalculation - about 17 a minute. Meanwhile the plan assumed the car had
dialled back when it was still drawing what it drew before.

A load that fails a command is now left alone on a growing backoff and
budgeted as load we do not control, so the rest of the house makes room for it.
"""

from datetime import datetime, timedelta

import pytest

import custom_components.zerogrid as zerogrid

CAR_THROTTLE = "number.car_charging_amps"
CAR_SWITCH = "switch.car_charger"


class ConnectTimeout(Exception):
    """What the car's HTTP library raised, and nothing the planner knew about."""


@pytest.fixture
def stuck(house):
    """Cylinder heating, charger flat out, dehumidifier running.

    With 12A of other load and 9A reserved, the plan wants the charger at 27A.
    """
    house.set_reserved_current(9.0)
    house.adopt("Hot Water Cylinder", amps=14.0)
    house.adopt("Car Charger", amps=32.0, setpoint=32)
    house.adopt("Dehumidifier", amps=1.0)
    house.set_house_amps(12.0 + 14.0 + 32.0 + 1.0)
    house.hass.services.fail_entities[CAR_THROTTLE] = ConnectTimeout()
    return house


def car(harness):
    return harness.state.controllable_loads["Car Charger"]


def test_a_failed_command_does_not_stop_the_loads_after_it(house):
    house.adopt("Car Charger", amps=29.2, setpoint=29)
    house.set_house_amps(25.0 + 29.2)
    house.set_reserved_current(9.0)
    house.hass.services.fail_entities[CAR_THROTTLE] = ConnectTimeout()

    house.recalculate()

    # The dehumidifier comes after the charger, and is still switched on.
    assert house.hass.services.calls_for(CAR_THROTTLE)
    assert house.was_turned_on("Dehumidifier")
    assert house.planned_on("Dehumidifier")


def test_a_failed_load_is_left_alone_during_its_backoff(stuck):
    stuck.recalculate()
    stuck.clear_calls()

    for _ in range(5):
        stuck.recalculate()

    assert not stuck.hass.services.calls_for(CAR_THROTTLE)
    assert not stuck.hass.services.calls_for(CAR_SWITCH)


def test_it_is_retried_once_its_backoff_runs_out(stuck):
    stuck.recalculate()
    car(stuck).command_retry_after = datetime.now() - timedelta(seconds=1)
    stuck.clear_calls()

    stuck.recalculate()

    assert len(stuck.hass.services.calls_for(CAR_THROTTLE)) == 1


def test_the_backoff_grows_with_each_failure(stuck):
    waits = []
    for _ in range(7):
        car(stuck).command_retry_after = None
        stuck.recalculate()
        waits.append((car(stuck).command_retry_after - datetime.now()).total_seconds())

    assert [round(w) for w in waits] == [30, 60, 120, 300, 600, 600, 600]


def test_a_stuck_load_is_budgeted_as_uncontrolled(stuck):
    stuck.recalculate()  # the throttle to 27A fails

    stuck.recalculate()

    # The charger's 32A stays with the 12A we never controlled.
    assert stuck.uncontrolled_amps() == pytest.approx(44.0)


def test_it_is_budgeted_at_its_setpoint_while_drawing_less(stuck):
    """Paused at 20A, it could resume to 32A without asking."""
    stuck.recalculate()
    stuck.set_load_amps("Car Charger", 20.0)
    stuck.set_house_amps(12.0 + 14.0 + 20.0 + 1.0)

    stuck.recalculate()

    assert stuck.uncontrolled_amps() == pytest.approx(44.0)


def test_other_loads_make_room_for_it_even_above_it_in_priority(stuck):
    stuck.recalculate()
    stuck.clear_calls()

    stuck.recalculate()

    # 63A - 44A uncontrolled - 9A reserved leaves 10A: not enough for the
    # cylinder's 14A, plenty for the dehumidifier.
    assert stuck.was_turned_off("Hot Water Cylinder")
    assert not stuck.was_turned_off("Dehumidifier")
    assert not stuck.hass.services.calls_for(CAR_THROTTLE)


def test_the_overload_pass_does_not_count_on_it_dialling_back(stuck):
    stuck.recalculate()
    stuck.state.controllable_loads["Hot Water Cylinder"].last_toggled = datetime.now()
    stuck.set_house_amps(12.0 + 14.0 + 32.0 + 1.0 + 10.0)  # 69A, over 63A + 3A
    stuck.state.overload_timestamp = datetime.now() - timedelta(seconds=60)
    stuck.clear_calls()

    stuck.recalculate()

    # The charger cannot give anything back, so fixed loads are cut instead.
    assert stuck.was_turned_off("Dehumidifier")
    assert stuck.was_turned_off("Hot Water Cylinder")
    assert not stuck.hass.services.calls_for(CAR_THROTTLE)
    assert not stuck.hass.services.calls_for(CAR_SWITCH)


def test_it_rejoins_the_plan_once_it_accepts_a_command(stuck):
    stuck.recalculate()
    assert stuck.entities["unresponsive_sensors"]["Car Charger"].state is True

    del stuck.hass.services.fail_entities[CAR_THROTTLE]
    car(stuck).command_retry_after = datetime.now() - timedelta(seconds=1)
    stuck.recalculate()

    assert car(stuck).command_failures == 0
    assert car(stuck).command_retry_after is None
    assert stuck.entities["unresponsive_sensors"]["Car Charger"].state is False
    assert stuck.setpoint_for("Car Charger") == 27


def test_it_is_cleared_when_the_plan_has_nothing_to_ask_of_it(stuck):
    stuck.recalculate()
    stuck.set_throttle_setpoint("Car Charger", 27)
    car(stuck).command_retry_after = datetime.now() - timedelta(seconds=1)
    stuck.clear_calls()

    stuck.recalculate()

    assert not stuck.hass.services.calls_for(CAR_THROTTLE)
    assert car(stuck).command_failures == 0


def test_a_command_that_hangs_counts_as_failed(stuck, monkeypatch):
    monkeypatch.setattr(zerogrid, "COMMAND_CALL_TIMEOUT_SECONDS", 0.01)
    del stuck.hass.services.fail_entities[CAR_THROTTLE]
    stuck.hass.services.hang_entities.add(CAR_THROTTLE)

    stuck.recalculate()

    assert car(stuck).command_failures == 1


def test_a_failed_turn_on_does_not_start_the_toggle_interval(house):
    house.set_house_amps(10.0)
    house.hass.services.fail_entities[CAR_SWITCH] = ConnectTimeout()

    house.recalculate()

    assert house.hass.services.calls_for(CAR_SWITCH)
    assert car(house).last_toggled is None
    assert car(house).switch_command_since is None
    assert car(house).command_failures == 1


def test_a_failed_throttle_does_not_mark_the_meters_as_skewed(stuck):
    stuck.recalculate()

    assert car(stuck).last_throttled < datetime.now() - timedelta(seconds=60)


def test_a_recalculation_is_not_started_while_one_is_running(stuck):
    stuck.hass.services.fail_entities.clear()
    stuck.state.recalculating = True

    stuck.recalculate()

    assert not stuck.hass.services.calls

    stuck.state.recalculating = False
    stuck.recalculate()

    assert stuck.hass.services.calls
    assert not stuck.state.recalculating


def test_safety_abort_turns_off_the_loads_after_one_that_fails(stuck):
    stuck.hass.services.fail_entities[CAR_SWITCH] = ConnectTimeout()

    stuck.abort(force=True)

    assert stuck.was_turned_off("Hot Water Cylinder")
    assert stuck.was_turned_off("Dehumidifier")
    assert stuck.state.safety_abort_active
