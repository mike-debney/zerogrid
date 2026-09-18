"""Tests for the small entity helpers."""

import sys

from custom_components.zerogrid.helpers import is_entity_usable, parse_entity_domain

State = sys.modules["homeassistant.core"].State


def test_parse_entity_domain_takes_the_part_before_the_dot():
    assert parse_entity_domain("switch.car_charger") == "switch"
    assert parse_entity_domain("climate.hot_water") == "climate"
    assert parse_entity_domain("humidifier.dehumidifier") == "humidifier"


def test_entity_is_unusable_when_missing_unknown_or_unavailable():
    assert is_entity_usable(None) is False
    assert is_entity_usable(State("switch.x", "unknown")) is False
    assert is_entity_usable(State("switch.x", "unavailable")) is False


def test_entity_is_usable_when_reporting_any_real_value():
    assert is_entity_usable(State("switch.x", "on")) is True
    assert is_entity_usable(State("switch.x", "off")) is True
    # A climate entity reports its HVAC mode rather than on/off.
    assert is_entity_usable(State("climate.x", "heat")) is True
