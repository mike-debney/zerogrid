"""Tests for the available-amps sliding window."""

from custom_components.zerogrid.state import State


def test_window_returns_zero_when_empty():
    state = State()
    assert state.get_minimum_available_amps(60) == 0
    assert state.get_average_available_amps(60) == 0


def test_minimum_and_average_over_the_window():
    state = State()
    for value in (10.0, 4.0, 16.0):
        state.accumulate_available_amps(value, 60)
    assert state.get_minimum_available_amps(60) == 4.0
    assert state.get_average_available_amps(60) == 10.0


def test_window_trims_entries_older_than_the_retention_period():
    state = State()
    state.accumulate_available_amps(99.0, 0)
    state.accumulate_available_amps(5.0, 0)
    # With zero retention only the newest sample survives.
    assert list(state.available_amps_history)[-1][1] == 5.0
    assert state.get_minimum_available_amps(60) == 5.0


def test_lookback_shorter_than_any_sample_falls_back_to_the_newest():
    state = State()
    state.accumulate_available_amps(7.0, 60)
    # Nothing falls inside a zero second lookback, so the latest value is used
    # rather than reporting no capacity at all.
    assert state.get_minimum_available_amps(0) == 7.0
    assert state.get_average_available_amps(0) == 7.0
