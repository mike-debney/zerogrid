"""Sensor entities for ZeroGrid."""

from __future__ import annotations

import logging

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfElectricCurrent
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, get_device_info

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the ZeroGrid sensor platform."""
    _LOGGER.debug("Setting up ZeroGrid sensor platform")

    device_info = get_device_info(entry)

    available_load_sensor = AvailableAmpsSensor(entry, device_info)
    controlled_load_sensor = LoadControlAmpsSensor(entry, device_info)
    uncontrolled_load_sensor = UncontrolledLoadAmpsSensor(entry, device_info)
    max_safe_load_sensor = MaxSafeLoadAmpsSensor(entry, device_info)

    # Store entity references in the entry-specific data
    if "entities" not in hass.data[DOMAIN][entry.entry_id]:
        hass.data[DOMAIN][entry.entry_id]["entities"] = {}

    hass.data[DOMAIN][entry.entry_id]["entities"]["available_load_sensor"] = (
        available_load_sensor
    )
    hass.data[DOMAIN][entry.entry_id]["entities"]["controlled_load_sensor"] = (
        controlled_load_sensor
    )
    hass.data[DOMAIN][entry.entry_id]["entities"]["uncontrolled_load_sensor"] = (
        uncontrolled_load_sensor
    )
    hass.data[DOMAIN][entry.entry_id]["entities"]["max_safe_load_sensor"] = (
        max_safe_load_sensor
    )

    async_add_entities(
        [
            available_load_sensor,
            controlled_load_sensor,
            uncontrolled_load_sensor,
            max_safe_load_sensor,
        ]
    )


class AvailableAmpsSensor(SensorEntity):
    """Sensor for current available for load control."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.CURRENT
    _attr_native_unit_of_measurement = UnitOfElectricCurrent.AMPERE
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the sensor."""
        self._attr_name = "Available load"
        self._attr_unique_id = f"{entry.entry_id}_available_load"
        self._attr_native_value = 0.0
        self._attr_device_info = device_info

    @callback
    def update_value(self, amps: float) -> None:
        """Update the sensor value and notify HA."""
        self._attr_native_value = round(amps, 2)
        self.async_write_ha_state()


class LoadControlAmpsSensor(SensorEntity):
    """Sensor for total current controlled by load control."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.CURRENT
    _attr_native_unit_of_measurement = UnitOfElectricCurrent.AMPERE
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the sensor."""
        self._attr_name = "Controlled load"
        self._attr_unique_id = f"{entry.entry_id}_controlled_load"
        self._attr_native_value = 0.0
        self._attr_device_info = device_info

    @callback
    def update_value(self, amps: float) -> None:
        """Update the sensor value and notify HA."""
        self._attr_native_value = round(amps, 2)
        self.async_write_ha_state()


class UncontrolledLoadAmpsSensor(SensorEntity):
    """Sensor for total current not under load control."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.CURRENT
    _attr_native_unit_of_measurement = UnitOfElectricCurrent.AMPERE
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the sensor."""
        self._attr_name = "Uncontrolled load"
        self._attr_unique_id = f"{entry.entry_id}_uncontrolled_load"
        self._attr_native_value = 0.0
        self._attr_device_info = device_info

    @callback
    def update_value(self, amps: float) -> None:
        """Update the sensor value and notify HA."""
        self._attr_native_value = round(amps, 2)
        self.async_write_ha_state()


class MaxSafeLoadAmpsSensor(SensorEntity):
    """Sensor for maximum safe total load current."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.CURRENT
    _attr_native_unit_of_measurement = UnitOfElectricCurrent.AMPERE
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the sensor."""
        self._attr_name = "Max safe load"
        self._attr_unique_id = f"{entry.entry_id}_max_safe_load"
        self._attr_native_value = 0.0
        self._attr_device_info = device_info

    @callback
    def update_value(self, amps: float) -> None:
        """Update the sensor value and notify HA."""
        self._attr_native_value = round(amps, 2)
        self.async_write_ha_state()
