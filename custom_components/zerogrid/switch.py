"""Switch entities for ZeroGrid."""

from __future__ import annotations

import logging
import sys
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from . import STATE, recalculate_load_control
from .const import ALLOW_GRID_IMPORT_SWITCH_ID, DOMAIN, ENABLE_LOAD_CONTROL_SWITCH_ID

_LOGGER = logging.getLogger(__name__)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the ZeroGrid switch platform."""
    _LOGGER.debug("Setting up ZeroGrid switch platform")

    enable_switch = EnableLoadControlSwitch()
    allow_grid_import_switch = AllowGridImportSwitch()

    # Store references in hass.data for updates
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][ENABLE_LOAD_CONTROL_SWITCH_ID] = enable_switch
    hass.data[DOMAIN][ALLOW_GRID_IMPORT_SWITCH_ID] = allow_grid_import_switch

    async_add_entities([enable_switch, allow_grid_import_switch])


class EnableLoadControlSwitch(SwitchEntity):
    """Switch to enable/disable load control."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_is_on: bool

    def __init__(self) -> None:
        """Initialize the switch."""
        self._attr_name = "Enable load control"
        self._attr_unique_id = ENABLE_LOAD_CONTROL_SWITCH_ID
        self._attr_is_on = True

    async def async_added_to_hass(self) -> None:
        """Handle entity being added to hass."""
        await super().async_added_to_hass()
        # Sync initial state to the integration
        await self._update_integration_state()

    @property
    def is_on(self) -> bool:
        """Return true if the switch is on."""
        return bool(self._attr_is_on)

    async def _update_integration_state(self) -> None:
        """Update the integration's STATE and trigger recalculation."""
        if hasattr(self.hass, "data") and DOMAIN in self.hass.data:
            STATE.enable_load_control = self._attr_is_on
            await recalculate_load_control(self.hass)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on."""
        self._attr_is_on = True
        self.async_write_ha_state()
        _LOGGER.info("Load control enabled")
        await self._update_integration_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off."""
        self._attr_is_on = False
        self.async_write_ha_state()
        _LOGGER.info("Load control disabled")
        await self._update_integration_state()

    @callback
    def update_state(self, is_on: bool) -> None:
        """Update the switch state."""
        self._attr_is_on = is_on
        self.async_write_ha_state()


class AllowGridImportSwitch(SwitchEntity):
    """Switch to allow/disallow grid import for load control."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_is_on: bool

    def __init__(self) -> None:
        """Initialize the switch."""
        self._attr_name = "Allow grid import"
        self._attr_unique_id = ALLOW_GRID_IMPORT_SWITCH_ID
        self._attr_is_on = True

    async def async_added_to_hass(self) -> None:
        """Handle entity being added to hass."""
        await super().async_added_to_hass()
        # Sync initial state to the integration
        await self._update_integration_state()

    @property
    def is_on(self) -> bool:
        """Return true if grid import is allowed."""
        return bool(self._attr_is_on)

    async def _update_integration_state(self) -> None:
        """Update the integration's STATE and trigger recalculation."""
        if hasattr(self.hass, "data") and DOMAIN in self.hass.data:
            STATE.allow_grid_import = self._attr_is_on
            await recalculate_load_control(self.hass)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Allow grid import."""
        self._attr_is_on = True
        self.async_write_ha_state()
        _LOGGER.info("Grid import allowed")
        await self._update_integration_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Disallow grid import."""
        self._attr_is_on = False
        self.async_write_ha_state()
        _LOGGER.info("Grid import disallowed")
        await self._update_integration_state()

    @callback
    def update_state(self, is_on: bool) -> None:
        """Update the switch state."""
        self._attr_is_on = is_on
        self.async_write_ha_state()
