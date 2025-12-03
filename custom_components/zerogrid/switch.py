"""Switch entities for ZeroGrid."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from . import CONFIG, STATE, recalculate_load_control
from .const import (
    ALLOW_GRID_IMPORT_SWITCH_ID,
    DOMAIN,
    ENABLE_LOAD_CONTROL_SWITCH_ID,
    get_device_info,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the ZeroGrid switch platform."""
    _LOGGER.debug("Setting up ZeroGrid switch platform")

    device_info = get_device_info(entry)

    enable_switch = EnableLoadControlSwitch(entry, device_info)
    allow_grid_import_switch = AllowGridImportSwitch(entry, device_info)

    # Store entity references in the entry-specific data
    if "entities" not in hass.data[DOMAIN][entry.entry_id]:
        hass.data[DOMAIN][entry.entry_id]["entities"] = {}

    hass.data[DOMAIN][entry.entry_id]["entities"][ENABLE_LOAD_CONTROL_SWITCH_ID] = (
        enable_switch
    )
    hass.data[DOMAIN][entry.entry_id]["entities"][ALLOW_GRID_IMPORT_SWITCH_ID] = (
        allow_grid_import_switch
    )

    async_add_entities([enable_switch, allow_grid_import_switch])


class EnableLoadControlSwitch(SwitchEntity, RestoreEntity):
    """Switch to enable/disable load control."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_is_on: bool

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the switch."""
        self.entry = entry
        self._attr_name = "Enable load control"
        self._attr_unique_id = f"{entry.entry_id}_enable_load_control"
        self._attr_device_info = device_info
        # Don't set _attr_is_on here - let it be restored in async_added_to_hass

    async def async_added_to_hass(self) -> None:
        """Handle entity being added to hass."""
        await super().async_added_to_hass()

        # Restore previous state if available
        last_state = await self.async_get_last_state()
        if last_state is not None:
            self._attr_is_on = last_state.state == "on"
            _LOGGER.info(
                "Restored enable load control switch state: %s (was: %s)",
                self._attr_is_on,
                last_state.state,
            )
        else:
            # No previous state, use default of True (enabled)
            self._attr_is_on = True
            _LOGGER.info(
                "No previous state found for enable load control switch, using default: True"
            )

        # Write the state immediately after restoration
        self.async_write_ha_state()

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
            if CONFIG.enable_automatic_recalculation:
                await recalculate_load_control(self.hass, self.entry.entry_id)

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


class AllowGridImportSwitch(SwitchEntity, RestoreEntity):
    """Switch to allow/disallow grid import for load control."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_is_on: bool

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the switch."""
        self.entry = entry
        self._attr_name = "Allow grid import"
        self._attr_unique_id = f"{entry.entry_id}_allow_grid_import"
        self._attr_device_info = device_info
        # Don't set _attr_is_on here - let it be restored in async_added_to_hass

    async def async_added_to_hass(self) -> None:
        """Handle entity being added to hass."""
        await super().async_added_to_hass()

        # Restore previous state if available
        last_state = await self.async_get_last_state()
        if last_state is not None:
            self._attr_is_on = last_state.state == "on"
            _LOGGER.info(
                "Restored allow grid import switch state: %s (was: %s)",
                self._attr_is_on,
                last_state.state,
            )
        else:
            # No previous state, use default of True (allowed)
            self._attr_is_on = True
            _LOGGER.info(
                "No previous state found for allow grid import switch, using default: True"
            )

        # Write the state immediately after restoration
        self.async_write_ha_state()

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
            if CONFIG.enable_automatic_recalculation:
                await recalculate_load_control(self.hass, self.entry.entry_id)

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
