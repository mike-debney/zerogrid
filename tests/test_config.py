"""Tests for configuration parsing."""

from datetime import datetime, timedelta

import pytest
from conftest import load

import custom_components.zerogrid as zerogrid
from custom_components.zerogrid.config import Config, ControllableLoadConfig

BASE = {
    "name": "ZeroGrid",
    "max_total_load_amps": 63.0,
    "max_grid_import_amps": 63.0,
    "max_solar_generation_amps": 0.0,
    "house_consumption_amps_entity": "sensor.house_current",
}


def test_each_config_has_its_own_load_dict():
    """Two entries must not share one dict of controllable loads."""
    first, second = Config(), Config()
    first.controllable_loads["only_mine"] = ControllableLoadConfig()
    assert "only_mine" not in second.controllable_loads


def test_total_load_is_capped_by_what_the_supply_can_deliver():
    zerogrid.CONFIG = zerogrid.CONFIGS["x"] = Config()
    zerogrid.CONFIG.controllable_loads = {}
    zerogrid.parse_config(
        {**BASE, "max_total_load_amps": 100.0, "max_grid_import_amps": 40.0}
    )
    assert zerogrid.CONFIG.max_total_load_amps == 40.0


def test_intervals_fall_back_to_defaults_when_not_configured():
    """A load configured without intervals must still rate limit.

    Leaving these unset used to store None, which the rate limit checks then
    fed to timedelta.
    """
    zerogrid.CONFIG = zerogrid.CONFIGS["x"] = Config()
    zerogrid.CONFIG.controllable_loads = {}
    zerogrid.parse_config(
        {
            **BASE,
            "controllable_loads": [
                {
                    "name": "Bare",
                    "switch_entity": "switch.bare",
                    "load_amps_entity": "sensor.bare",
                    "min_controllable_load_amps": 1.0,
                    "max_controllable_load_amps": 10.0,
                }
            ],
        }
    )
    cfg = zerogrid.CONFIG.controllable_loads["Bare"]
    assert isinstance(cfg.min_toggle_interval_seconds, int)
    assert isinstance(cfg.min_throttle_interval_seconds, int)
    # The values have to be usable as a duration.
    assert datetime.now() + timedelta(seconds=cfg.min_toggle_interval_seconds)
    assert datetime.now() + timedelta(seconds=cfg.min_throttle_interval_seconds)


def test_loads_removed_from_the_options_do_not_linger(make):
    """Reconfiguring with one load dropped must forget the dropped one."""
    zerogrid.CONFIG = zerogrid.CONFIGS["x"] = Config()
    zerogrid.CONFIG.controllable_loads = {}
    zerogrid.parse_config(
        {
            **BASE,
            "controllable_loads": [
                load("Keep", "switch.keep", "sensor.keep"),
                load("Drop", "switch.drop", "sensor.drop"),
            ],
        }
    )
    assert set(zerogrid.CONFIG.controllable_loads) == {"Keep", "Drop"}

    zerogrid.parse_config(
        {**BASE, "controllable_loads": [load("Keep", "switch.keep", "sensor.keep")]}
    )
    assert set(zerogrid.CONFIG.controllable_loads) == {"Keep"}


def test_solar_consumption_follows_whether_a_solar_entity_is_set():
    zerogrid.CONFIG = zerogrid.CONFIGS["x"] = Config()
    zerogrid.CONFIG.controllable_loads = {}
    zerogrid.parse_config(BASE)
    assert zerogrid.CONFIG.allow_solar_consumption is False

    zerogrid.CONFIG = zerogrid.CONFIGS["x"] = Config()
    zerogrid.CONFIG.controllable_loads = {}
    zerogrid.parse_config({**BASE, "solar_generation_amps_entity": "sensor.solar"})
    assert zerogrid.CONFIG.allow_solar_consumption is True


def test_an_explicitly_zero_interval_is_kept():
    """Zero means no rate limit, and must not be replaced by the default."""
    zerogrid.CONFIG = zerogrid.CONFIGS["x"] = Config()
    zerogrid.parse_config(
        {
            **BASE,
            "controllable_loads": [
                load(
                    "Unlimited",
                    "switch.unlimited",
                    "sensor.unlimited",
                    min_toggle_interval_seconds=0,
                    min_throttle_interval_seconds=0,
                )
            ],
        }
    )
    cfg = zerogrid.CONFIG.controllable_loads["Unlimited"]
    assert cfg.min_toggle_interval_seconds == 0
    assert cfg.min_throttle_interval_seconds == 0
