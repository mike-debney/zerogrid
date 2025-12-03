"""Binary sensor entities for ZeroGrid."""

from __future__ import annotations

import logging

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, get_device_info

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the ZeroGrid binary sensor platform."""
    _LOGGER.debug("Setting up ZeroGrid binary sensor platform")

    device_info = get_device_info(entry)

    overload_sensor = OverloadBinarySensor(entry, device_info)
    safety_abort_sensor = SafetyAbortBinarySensor(entry, device_info)

    # Store entity references in the entry-specific data
    if "entities" not in hass.data[DOMAIN][entry.entry_id]:
        hass.data[DOMAIN][entry.entry_id]["entities"] = {}

    hass.data[DOMAIN][entry.entry_id]["entities"]["overload_sensor"] = overload_sensor
    hass.data[DOMAIN][entry.entry_id]["entities"]["safety_abort_sensor"] = (
        safety_abort_sensor
    )

    async_add_entities([overload_sensor, safety_abort_sensor])


class OverloadBinarySensor(BinarySensorEntity):
    """Binary sensor indicating overload state."""

    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the binary sensor."""
        self._attr_name = "Overload"
        self._attr_unique_id = f"{entry.entry_id}_overload"
        self._attr_is_on = False
        self._attr_device_info = device_info

    @callback
    def update_state(self, is_overload: bool) -> None:
        """Update the binary sensor state and notify HA."""
        self._attr_is_on = is_overload
        self.async_write_ha_state()


class SafetyAbortBinarySensor(BinarySensorEntity):
    """Binary sensor indicating safety abort state."""

    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the binary sensor."""
        self._attr_name = "Safety abort"
        self._attr_unique_id = f"{entry.entry_id}_safety_abort"
        self._attr_is_on = False
        self._attr_device_info = device_info

    @callback
    def update_state(self, is_abort: bool) -> None:
        """Update the binary sensor state and notify HA."""
        self._attr_is_on = is_abort
        self.async_write_ha_state()
