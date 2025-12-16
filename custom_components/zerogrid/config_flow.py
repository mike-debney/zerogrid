"""Config flow for Zero Grid integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import DOMAIN

DEFAULT_MAX_TOTAL_LOAD_AMPS = 63
DEFAULT_MAX_GRID_IMPORT_AMPS = 63
DEFAULT_MAX_SOLAR_GENERATION_AMPS = 48
DEFAULT_SAFETY_MARGIN_AMPS = 2.0
DEFAULT_RECALCULATE_INTERVAL = 30
DEFAULT_LOAD_MEASUREMENT_DELAY = 120
DEFAULT_MIN_TOGGLE_INTERVAL = 600
DEFAULT_MIN_THROTTLE_INTERVAL = 10
DEFAULT_SOLAR_TURN_ON_WINDOW = 600
DEFAULT_SOLAR_TURN_OFF_WINDOW = 300


def _filter_none_values(data: dict[str, Any]) -> dict[str, Any]:
    """Remove None values from dictionary to prevent entity validation errors."""
    return {key: value for key, value in data.items() if value is not None}


class ZeroGridConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Zero Grid."""

    VERSION = 1
    MINOR_VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> ZeroGridOptionsFlow:
        """Get the options flow for this handler."""
        return ZeroGridOptionsFlow(config_entry)

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._basic_config: dict[str, Any] = {}
        self._num_loads: int = 0
        self._current_load_index: int = 0
        self._loads: list[dict[str, Any]] = []
        self._name: str = ""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step - name and monitoring."""
        errors = {}

        if user_input is not None:
            # Store name and create unique ID
            self._name = user_input["name"]

            # Set unique ID based on name to allow multiple instances
            # This allows multiple ZeroGrid instances to monitor the same sensors
            # but control different sets of loads
            await self.async_set_unique_id(self._name.lower().replace(" ", "_"))
            self._abort_if_unique_id_configured()

            # Store the monitoring and limits configuration
            self._basic_config.update(_filter_none_values(user_input))
            return await self.async_step_safety()

        schema = vol.Schema(
            {
                vol.Required("name", default="ZeroGrid"): selector.TextSelector(),
                vol.Required("house_consumption_amps_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor", "input_number"])
                ),
                vol.Optional("solar_generation_amps_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor", "input_number"])
                ),
                vol.Optional(
                    "enable_automatic_recalculation", default=True
                ): selector.BooleanSelector(),
                vol.Required(
                    "recalculate_interval_seconds", default=DEFAULT_RECALCULATE_INTERVAL
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    "load_measurement_delay_seconds",
                    default=DEFAULT_LOAD_MEASUREMENT_DELAY,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    async def async_step_safety(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the safety and timing configuration step."""
        if user_input is not None:
            # Store the safety configuration and move to loads
            self._basic_config.update(_filter_none_values(user_input))
            return await self.async_step_loads()

        schema = vol.Schema(
            {
                vol.Required(
                    "max_total_load_amps", default=DEFAULT_MAX_TOTAL_LOAD_AMPS
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    "max_grid_import_amps", default=DEFAULT_MAX_GRID_IMPORT_AMPS
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    "max_solar_generation_amps",
                    default=DEFAULT_MAX_SOLAR_GENERATION_AMPS,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    "safety_margin_amps", default=DEFAULT_SAFETY_MARGIN_AMPS
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX, step=0.1
                    )
                ),
            }
        )
        return self.async_show_form(step_id="safety", data_schema=schema)

    async def async_step_loads(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the controllable loads step."""
        if user_input is not None:
            # Get the number of loads to configure
            num_loads = user_input.get("num_loads", 0)
            if num_loads > 0:
                self._num_loads = num_loads
                self._loads = []
                self._current_load_index = 0
                return await self.async_step_load_config()
            # No loads configured, finish setup
            self._basic_config["controllable_loads"] = []
            return self.async_create_entry(title=self._name, data=self._basic_config)

        schema = vol.Schema(
            {
                vol.Required("num_loads", default=0): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, max=10, mode=selector.NumberSelectorMode.BOX
                    )
                ),
            }
        )
        return self.async_show_form(step_id="loads", data_schema=schema)

    async def async_step_load_config(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure individual load."""
        if user_input is not None:
            # Store this load configuration
            self._loads.append(_filter_none_values(user_input))
            self._current_load_index += 1

            # Check if we need to configure more loads
            if self._current_load_index < self._num_loads:
                return await self.async_step_load_config()

            # All loads configured, finish setup
            self._basic_config["controllable_loads"] = self._loads
            return self.async_create_entry(title=self._name, data=self._basic_config)

        schema = vol.Schema(
            {
                vol.Required("name"): selector.TextSelector(),
                vol.Required("max_controllable_load_amps"): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required("min_controllable_load_amps"): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required("load_amps_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor", "input_number"])
                ),
                vol.Required("switch_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["switch", "input_boolean", "climate"]
                    )
                ),
                vol.Optional(
                    "min_toggle_interval_seconds", default=DEFAULT_MIN_TOGGLE_INTERVAL
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional("throttle_amps_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["number", "input_number"])
                ),
                vol.Optional(
                    "min_throttle_interval_seconds",
                    default=DEFAULT_MIN_THROTTLE_INTERVAL,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional("can_turn_on_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["binary_sensor", "input_boolean"]
                    )
                ),
                vol.Optional(
                    "solar_turn_on_window_seconds",
                    default=DEFAULT_SOLAR_TURN_ON_WINDOW,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    "solar_turn_off_window_seconds",
                    default=DEFAULT_SOLAR_TURN_OFF_WINDOW,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="load_config",
            data_schema=schema,
            description_placeholders={
                "load_number": str(self._current_load_index + 1),
                "total_loads": str(self._num_loads),
            },
        )


class ZeroGridOptionsFlow(OptionsFlow):
    """Handle options flow for Zero Grid."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        super().__init__()
        self._monitoring_config: dict[str, Any] = {}
        self._safety_config: dict[str, Any] = {}
        self._loads: list[dict[str, Any]] = []
        self._current_load_index: int = 0
        self._edit_load_index: int | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        return await self.async_step_menu()

    async def async_step_menu(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Show main options menu."""
        return self.async_show_menu(
            step_id="menu",
            menu_options=["monitoring", "safety", "manage_loads", "reorder_loads"],
        )

    async def async_step_monitoring(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure monitoring sensors and timing."""
        if user_input is not None:
            # Save and return to menu
            current_options = {**self.config_entry.data, **self.config_entry.options}
            updated_options = {**current_options, **_filter_none_values(user_input)}
            return self.async_create_entry(title="", data=updated_options)

        current_config = {**self.config_entry.data, **self.config_entry.options}

        # Build schema with conditional default for optional entity
        schema_fields = {
            vol.Required(
                "house_consumption_amps_entity",
                default=current_config.get("house_consumption_amps_entity"),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor", "input_number"])
            ),
            vol.Optional(
                "enable_automatic_recalculation",
                default=current_config.get("enable_automatic_recalculation", True),
            ): selector.BooleanSelector(),
            vol.Required(
                "recalculate_interval_seconds",
                default=current_config.get(
                    "recalculate_interval_seconds", DEFAULT_RECALCULATE_INTERVAL
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                "load_measurement_delay_seconds",
                default=current_config.get(
                    "load_measurement_delay_seconds", DEFAULT_LOAD_MEASUREMENT_DELAY
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, mode=selector.NumberSelectorMode.BOX
                )
            ),
        }

        # Add solar entity with default only if it exists
        solar_entity = current_config.get("solar_generation_amps_entity")
        if solar_entity:
            schema_fields[
                vol.Optional("solar_generation_amps_entity", default=solar_entity)
            ] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor", "input_number"])
            )
        else:
            schema_fields[vol.Optional("solar_generation_amps_entity")] = (
                selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor", "input_number"])
                )
            )

        schema = vol.Schema(schema_fields)
        return self.async_show_form(step_id="monitoring", data_schema=schema)

    async def async_step_safety(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Configure safety and limits."""
        if user_input is not None:
            # Save and return to menu
            current_options = {**self.config_entry.data, **self.config_entry.options}
            updated_options = {**current_options, **_filter_none_values(user_input)}
            return self.async_create_entry(title="", data=updated_options)

        current_config = {**self.config_entry.data, **self.config_entry.options}

        schema = vol.Schema(
            {
                vol.Required(
                    "max_total_load_amps",
                    default=current_config.get(
                        "max_total_load_amps", DEFAULT_MAX_TOTAL_LOAD_AMPS
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    "max_grid_import_amps",
                    default=current_config.get(
                        "max_grid_import_amps", DEFAULT_MAX_GRID_IMPORT_AMPS
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    "max_solar_generation_amps",
                    default=current_config.get(
                        "max_solar_generation_amps", DEFAULT_MAX_SOLAR_GENERATION_AMPS
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    "safety_margin_amps",
                    default=current_config.get(
                        "safety_margin_amps", DEFAULT_SAFETY_MARGIN_AMPS
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX, step=0.1
                    )
                ),
            }
        )
        return self.async_show_form(step_id="safety", data_schema=schema)

    async def async_step_reorder_loads(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reorder controllable loads priority."""
        current_config = {**self.config_entry.data, **self.config_entry.options}
        current_loads = current_config.get("controllable_loads", [])

        if not current_loads:
            # No loads to reorder
            return await self.async_step_menu()

        errors = {}
        if user_input is not None:
            # Build new order based on user selections
            # Create a mapping of load names to load configs
            load_map = {load["name"]: load for load in current_loads}

            # Collect the selected order
            selected_names = []
            for i in range(len(current_loads)):
                position_key = f"position_{i}"
                if position_key in user_input:
                    selected_names.append(user_input[position_key])

            # Check for duplicates
            if len(selected_names) != len(set(selected_names)):
                errors["base"] = "duplicate_load"
            else:
                # Build reordered list
                reordered_loads = []
                for name in selected_names:
                    if name in load_map:
                        reordered_loads.append(load_map[name])

                # Ensure all loads are included (shouldn't happen but be safe)
                for load in current_loads:
                    if load not in reordered_loads:
                        reordered_loads.append(load)

                updated_options = {
                    **current_config,
                    "controllable_loads": reordered_loads,
                }
                return self.async_create_entry(title="", data=updated_options)

        # Build schema with select boxes for each position
        load_names = [load["name"] for load in current_loads]
        schema_dict = {}

        for i, load in enumerate(current_loads):
            schema_dict[vol.Required(f"position_{i}", default=load["name"])] = (
                selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=load_names,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                )
            )

        schema = vol.Schema(schema_dict)
        return self.async_show_form(
            step_id="reorder_loads",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_manage_loads(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage controllable loads."""
        current_config = {**self.config_entry.data, **self.config_entry.options}
        current_loads = current_config.get("controllable_loads", [])

        # Build menu based on whether loads exist
        menu_options = ["add_load"]
        if current_loads:
            menu_options.append("select_load_to_edit")
        menu_options.append("menu")

        return self.async_show_menu(
            step_id="manage_loads",
            menu_options=menu_options,
        )

    async def async_step_add_load(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Add a new load."""
        if user_input is not None:
            current_config = {**self.config_entry.data, **self.config_entry.options}
            loads = list(current_config.get("controllable_loads", []))
            loads.append(_filter_none_values(user_input))
            updated_options = {**current_config, "controllable_loads": loads}
            return self.async_create_entry(title="", data=updated_options)

        schema = vol.Schema(
            {
                vol.Required("name"): selector.TextSelector(),
                vol.Required(
                    "max_controllable_load_amps", default=0
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    "min_controllable_load_amps", default=0
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required("switch_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["switch", "input_boolean", "climate"]
                    )
                ),
                vol.Required(
                    "min_toggle_interval_seconds", default=DEFAULT_MIN_TOGGLE_INTERVAL
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Required("load_amps_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["sensor", "input_number"])
                ),
                vol.Optional("throttle_amps_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["number", "input_number"])
                ),
                vol.Optional(
                    "min_throttle_interval_seconds",
                    default=DEFAULT_MIN_THROTTLE_INTERVAL,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional("can_turn_on_entity"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["binary_sensor", "input_boolean"]
                    )
                ),
                vol.Optional(
                    "solar_turn_on_window_seconds",
                    default=DEFAULT_SOLAR_TURN_ON_WINDOW,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    "solar_turn_off_window_seconds",
                    default=DEFAULT_SOLAR_TURN_OFF_WINDOW,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0, mode=selector.NumberSelectorMode.BOX
                    )
                ),
            }
        )
        return self.async_show_form(step_id="add_load", data_schema=schema)

    async def async_step_select_load_to_edit(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select which load to edit."""
        current_config = {**self.config_entry.data, **self.config_entry.options}
        loads = current_config.get("controllable_loads", [])

        if user_input is not None:
            selected_load = user_input.get("load_to_edit")
            self._edit_load_index = int(selected_load)
            return await self.async_step_edit_load()

        # Create list of load names for selection
        load_options = {
            str(i): load.get("name", f"Load {i + 1}") for i, load in enumerate(loads)
        }

        schema = vol.Schema(
            {
                vol.Required("load_to_edit"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"label": name, "value": idx}
                            for idx, name in load_options.items()
                        ],
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }
        )
        return self.async_show_form(step_id="select_load_to_edit", data_schema=schema)

    async def async_step_edit_load(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit or delete a load."""
        current_config = {**self.config_entry.data, **self.config_entry.options}
        loads = list(current_config.get("controllable_loads", []))

        if self._edit_load_index is None or self._edit_load_index >= len(loads):
            return await self.async_step_manage_loads()

        current_load = loads[self._edit_load_index]

        if user_input is not None:
            if user_input.get("delete_load"):
                # Delete the load
                loads.pop(self._edit_load_index)
                self._edit_load_index = None
            else:
                # Update the load
                user_input.pop("delete_load", None)
                loads[self._edit_load_index] = _filter_none_values(user_input)
                self._edit_load_index = None

            updated_options = {**current_config, "controllable_loads": loads}
            return self.async_create_entry(title="", data=updated_options)

        # Build schema with all fields (matching config flow order)
        throttle_entity = current_load.get("throttle_amps_entity")
        can_turn_on = current_load.get("can_turn_on_entity")

        # Build schema fields conditionally based on whether optional entities have values
        schema_fields: dict[vol.Marker, Any] = {
            vol.Required(
                "name", default=current_load.get("name", "")
            ): selector.TextSelector(),
            vol.Required(
                "max_controllable_load_amps",
                default=current_load.get("max_controllable_load_amps", 0),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                "min_controllable_load_amps",
                default=current_load.get("min_controllable_load_amps", 0),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                "load_amps_entity",
                default=current_load.get("load_amps_entity"),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["sensor", "input_number"])
            ),
            vol.Required(
                "switch_entity",
                default=current_load.get("switch_entity"),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=["switch", "input_boolean", "climate"]
                )
            ),
            vol.Required(
                "min_toggle_interval_seconds",
                default=current_load.get(
                    "min_toggle_interval_seconds", DEFAULT_MIN_TOGGLE_INTERVAL
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0, mode=selector.NumberSelectorMode.BOX
                )
            ),
        }

        # Add optional fields with default only if they exist
        if throttle_entity:
            schema_fields[
                vol.Optional("throttle_amps_entity", default=throttle_entity)
            ] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["number", "input_number"])
            )
        else:
            schema_fields[vol.Optional("throttle_amps_entity")] = (
                selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["number", "input_number"])
                )
            )

        schema_fields[
            vol.Optional(
                "min_throttle_interval_seconds",
                default=current_load.get(
                    "min_throttle_interval_seconds", DEFAULT_MIN_THROTTLE_INTERVAL
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=1, mode=selector.NumberSelectorMode.BOX)
        )

        if can_turn_on:
            schema_fields[vol.Optional("can_turn_on_entity", default=can_turn_on)] = (
                selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["binary_sensor", "input_boolean"]
                    )
                )
            )
        else:
            schema_fields[vol.Optional("can_turn_on_entity")] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["binary_sensor", "input_boolean"])
            )

        schema_fields[
            vol.Optional(
                "solar_turn_on_window_seconds",
                default=current_load.get(
                    "solar_turn_on_window_seconds", DEFAULT_SOLAR_TURN_ON_WINDOW
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, mode=selector.NumberSelectorMode.BOX)
        )

        schema_fields[
            vol.Optional(
                "solar_turn_off_window_seconds",
                default=current_load.get(
                    "solar_turn_off_window_seconds", DEFAULT_SOLAR_TURN_OFF_WINDOW
                ),
            )
        ] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, mode=selector.NumberSelectorMode.BOX)
        )

        schema_fields[vol.Required("delete_load", default=False)] = (
            selector.BooleanSelector()
        )

        schema = vol.Schema(schema_fields)
        return self.async_show_form(
            step_id="edit_load",
            data_schema=schema,
            description_placeholders={"load_name": current_load.get("name", "Load")},
        )
