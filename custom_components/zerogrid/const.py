"""Constants for the ZeroGrid integration."""

from homeassistant.helpers.device_registry import DeviceInfo

DOMAIN = "zerogrid"
ENABLE_LOAD_CONTROL_SWITCH_ID = f"{DOMAIN}_enable_load_control"
ALLOW_GRID_IMPORT_SWITCH_ID = f"{DOMAIN}_allow_grid_import"

DEVICE_INFO = DeviceInfo(
    identifiers={(DOMAIN, "zerogrid_load_controller")},
    name="ZeroGrid Load Controller",
    manufacturer="ZeroGrid",
    model="Load Controller",
)
