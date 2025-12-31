"""Number entities for the ZeroGrid integration."""

from __future__ import annotations

from homeassistant.components.number import NumberMode, RestoreNumber
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfElectricCurrent
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import recalculate_load_control
from .const import DOMAIN, get_device_info


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up ZeroGrid number entities from a config entry."""
    reserved_current_number = ReservedCurrentNumber(entry, get_device_info(entry))

    # Store entity reference for updates
    hass.data[DOMAIN][entry.entry_id].setdefault("entities", {})
    hass.data[DOMAIN][entry.entry_id]["entities"]["reserved_current"] = (
        reserved_current_number
    )

    async_add_entities([reserved_current_number])


class ReservedCurrentNumber(RestoreNumber):
    """Number entity for reserved current."""

    _attr_has_entity_name = True
    _attr_translation_key = "reserved_current"
    _attr_native_unit_of_measurement = UnitOfElectricCurrent.AMPERE
    _attr_mode = NumberMode.BOX
    _attr_native_min_value = 0
    _attr_native_max_value = 100
    _attr_native_step = 1

    def __init__(self, entry: ConfigEntry, device_info) -> None:
        """Initialize the reserved current number entity."""
        self._entry = entry
        self._attr_name = "Reserved current"
        self._attr_device_info = device_info
        self._attr_unique_id = f"{entry.entry_id}_reserved_current"
        self._attr_native_value = 0.0

    async def async_added_to_hass(self) -> None:
        """Restore last state when entity is added."""
        await super().async_added_to_hass()

        # Restore previous value or default to 0
        if (last_number_data := await self.async_get_last_number_data()) is not None:
            self._attr_native_value = last_number_data.native_value

    async def async_set_native_value(self, value: float) -> None:
        """Set new value."""
        self._attr_native_value = value
        self.async_write_ha_state()

        # Trigger recalculation
        await recalculate_load_control(self.hass, self._entry.entry_id)

    @callback
    def update_value(self, value: float) -> None:
        """Update the value of the number entity."""
        self._attr_native_value = value
        self.async_write_ha_state()
