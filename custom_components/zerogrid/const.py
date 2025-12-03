"""Constants for the ZeroGrid integration."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.device_registry import DeviceInfo

DOMAIN = "zerogrid"
ENABLE_LOAD_CONTROL_SWITCH_ID = f"{DOMAIN}_enable_load_control"
ALLOW_GRID_IMPORT_SWITCH_ID = f"{DOMAIN}_allow_grid_import"


def get_device_info(entry: ConfigEntry) -> DeviceInfo:
    """Get device info for the ZeroGrid load controller."""
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name="ZeroGrid Load Controller",
        manufacturer="ZeroGrid",
        model="Load Controller",
        entry_type=None,
    )
