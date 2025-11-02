"""Binary sensor entities for ZeroGrid."""

from __future__ import annotations

import logging

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the ZeroGrid binary sensor platform."""
    _LOGGER.debug("Setting up ZeroGrid binary sensor platform")

    overload_sensor = OverloadBinarySensor()

    # Store reference in hass.data for updates
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN]["overload_sensor"] = overload_sensor

    async_add_entities([overload_sensor])


class OverloadBinarySensor(BinarySensorEntity):
    """Binary sensor indicating overload state."""

    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_should_poll = False

    def __init__(self) -> None:
        """Initialize the binary sensor."""
        self._attr_name = "Overload"
        self._attr_unique_id = f"{DOMAIN}_overload"
        self._attr_is_on = False

    @callback
    def update_state(self, is_overload: bool) -> None:
        """Update the binary sensor state and notify HA."""
        self._attr_is_on = is_overload
        self.async_write_ha_state()
