![ZeroGrid Logo](logo.svg)

# ZeroGrid

ZeroGrid is a Home Assistant integration that intelligently manages controllable loads to maximize use of available power from solar in order to minimise grid dependence. It can also be used with or without solar to limit grid import and prevent circuit overload.

## Features

-   **Priority-based load management** - Loads are prioritized by their order in configuration, with the first load having the highest priority
-   **Solar integration** - Automatically uses available solar power for controllable loads if configured
-   **Configurable rate limiting** - Reduces wear on contactors from rapid switching
-   **Throttleable loads** - Support for loads that can operate at variable power levels (e.g., EV chargers can be throttled between minimum and maximum amperage)
-   **External constraints** - Optional `can_turn_on_entity` allows external conditions to control whether a load can be turned on (e.g., only try to charge EV when car is plugged in)
-   **Soft start compensation** - Configurable delay period uses expected load instead of measured load to account for soft starts and measurement delays
-   **Overload detection** - Immediately sheds loads if consumption exceeds safe limits for more than `recalculate_interval_seconds`, automatically recovers once sufficient load is shed
-   **Emergency abort** - Cuts all loads if critical sensors become unavailable for more than 60s, requires manual re-enabling of load-control

## Disclaimer

**Use at Your Own Risk**

This software is provided "as is," without any warranty of any kind, express or implied. The authors and contributors are not liable for any damages or losses, including but not limited to property damage, personal injury, or financial loss, arising from the use or inability to use this software.

Controlling high-power electrical systems is inherently dangerous. It is your responsibility to ensure that your system is installed and configured safely and in compliance with all local laws and regulations.

## How It Works

### Power Allocation

ZeroGrid manages controllable loads through a multi-pass algorithm:

1. **Calculate available power** - Based on house consumption, solar generation, grid import limits, and configured safety margins.
2. **First pass - Turn loads on/off** - Evaluate loads in priority order (as listed in config). Each load is turned on if minimum power is available AND external constraints allow it (if configured), respecting rate limits
3. **Second pass - Throttle up** - Allocate any remaining available power by throttling loads up from their minimum to maximum, in priority order
4. **Third pass - Overload protection** - If total consumption exceeds safe limits, cut loads in reverse priority order until safe
5. **Execute plan** - Update switch states and throttle values to match the calculated plan

### Throttling

Loads with a `throttle_amps_entity` configured can operate at variable power levels between their min and max values. During allocation:

-   Load is first turned on at minimum amps if sufficient power is available
-   If extra power is available after all loads are at minimum, higher priority loads are throttled up first
-   Loads such as EV chargers benefit most from throttling as they can adjust charging current rapidly

For example:

1. Available power is 20A
2. EV charger requires 6-16A (min-max)
3. Load turns on at 6A initially
4. If no other loads need power, it throttles up to 16A
5. If another load needs 8A, EV charger may throttle down to make room

### External Constraints

The optional `can_turn_on_entity` allows you to prevent a load from being turned on based on external conditions:

-   When the entity state is "on", the load can be turned on if power is available
-   When the entity state is "off", the load will not be turned on even if power is available
-   Useful for scenarios like:
    -   Only charge EV when car is plugged in (`binary_sensor.car_plugged_in`)
    -   Only heat pool when home is occupied (`binary_sensor.home_occupied`)
    -   Only run heat pump during off-peak hours (`binary_sensor.off_peak_period`)

## Installation

Install via HACS or manually copy the `custom_components/zerogrid` directory to your Home Assistant configuration directory.

ZeroGrid is configured through the Home Assistant UI using a config flow. After installation, follow these steps:

1. Go to **Settings** → **Devices & Services**.
2. Click **+ Add Integration** and search for "ZeroGrid".
3. Follow the on-screen prompts to configure the integration.

The setup process is divided into several steps:

-   **Initial Setup**: Name your ZeroGrid instance and provide the primary monitoring sensors.
-   **Safety & Limits**: Define the maximum load for your system, grid import, and solar generation, along with safety margins.
-   **Load Configuration**: Specify how many controllable loads you want to manage, then configure each one with its specific entities and parameters.

You can create multiple instances of ZeroGrid to manage different sets of loads or phases independently.

### Caveats & Gotchas

-   **Multi-Phase Systems**: To manage load across multiple phases, set up a separate instance of ZeroGrid for each phase you want to manage using the appropriate entities and constraints.
-   **Off-Grid & Export Limits**: To allocate power effectively, ZeroGrid needs to know the total available solar generation. This is typically measured by placing the solar array under a continuous load, which is most easily achieved by exporting surplus power to the grid. In off-grid systems or those with strict export limits, if there is no other consistent load, ZeroGrid cannot accurately gauge the available solar power. This can lead to suboptimal power allocation, so off-grid use is not recommended.

### Configuration Options Reference

Below is a reference of all available configuration options. These are entered through the UI during setup or can be modified later by reconfiguring the integration.

#### System Settings

| Option                           | Required | Default | Description                                                                                                                      |
| -------------------------------- | -------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `max_total_load_amps`            | Yes      | -       | Maximum total load your electrical system can handle                                                                             |
| `max_grid_import_amps`           | Yes      | -       | Maximum power draw allowed from the grid                                                                                         |
| `max_solar_generation_amps`      | Yes      | -       | Maximum solar generation capacity in amps                                                                                        |
| `safety_margin_amps`             | No       | 2.0     | Safety buffer above maximum load before triggering overload protection                                                           |
| `hysteresis_amps`                | No       | 1.0     | Prevents rapid switching/oscillation                                                                                             |
| `recalculate_interval_seconds`   | No       | 30      | Periodic recalculation interval (in addition to event-driven based on house consumption readings)                                |
| `load_measurement_delay_seconds` | No       | 120     | Time in seconds to use expected load instead of measured load after turning on (accounts for soft starts and measurement delays) |
| `enable_automatic_recalculation` | No       | true    | Enable periodic recalculation of loads. If disabled, recalculation only occurs when house consumption changes.                   |

#### Monitoring Entities

| Option                          | Required | Description                                                                                     |
| ------------------------------- | -------- | ----------------------------------------------------------------------------------------------- |
| `house_consumption_amps_entity` | Yes      | Sensor measuring total house consumption in amps                                                |
| `solar_generation_amps_entity`  | No       | Sensor measuring solar generation in amps, if this is omitted solar production will not be used |

#### Controllable Loads

Each controllable load has the following configuration:
| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `name` | Yes | - | Unique name for the load |
| `max_controllable_load_amps` | Yes | - | Maximum power this load can consume |
| `min_controllable_load_amps` | Yes | - | Minimum power needed (same as max for non-throttleable loads) |
| `load_amps_entity` | Yes | - | Sensor measuring actual consumption in amps |
| `switch_entity` | Yes | - | Switch entity to control the load |
| `min_toggle_interval_seconds` | No | 600 | Minimum time between on/off switches |
| `throttle_amps_entity` | No | - | Number entity to control throttle level (enables throttling) |
| `min_throttle_interval_seconds` | No | 10 | Minimum time between throttle adjustments |
| `can_turn_on_entity` | No | - | Binary sensor or input_boolean that must be "on" for the load to be turned on |

**Note:** Loads are prioritized in the order they appear in the configuration. The first load listed has the highest priority (priority 0), the second has priority 1, and so on.

## Entities

ZeroGrid creates the following entities that indicate how the system is performing:

| Entity                                | Description                                                                                                                                      |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `sensor.zerogrid_available_load`      | Current available power for new loads (amps). Calculated from grid import + solar generation - house consumption (excluding controllable loads). |
| `sensor.zerogrid_controlled_load`     | Total power being used by ZeroGrid-controlled loads (amps). Shows the sum of expected consumption for all loads planned to be on.                |
| `sensor.zerogrid_uncontrolled_load`   | Power used by loads not under ZeroGrid control (amps). Calculated as total house consumption minus controlled loads.                             |
| `sensor.zerogrid_max_safe_load`       | Maximum safe total load (amps). Calculated from configured limits plus safety margin.                                                            |
| `binary_sensor.zerogrid_overload`     | Indicates if the system is in an overload state (on = overload detected). Clears automatically when system recovers.                             |
| `binary_sensor.zerogrid_safety_abort` | Indicates if a safety abort has occurred because critical sensor data is unavailable.                                                            |
| `switch.zerogrid_enable_load_control` | Master enable/disable for load control. When off, no loads will be automatically controlled. Cycling this also resets any throttling timers.     |
| `switch.zerogrid_allow_grid_import`   | Enable/disable grid import. When off, only solar power can be used for controllable loads.                                                       |
