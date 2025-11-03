# ZeroGrid Integration

ZeroGrid is a Home Assistant integration that intelligently manages controllable loads to maximize use of available power from solar and/or grid while preventing circuit overload. It dynamically allocates power to multiple loads based on priority and measured consumption.

## Features
- **Priority-based load management** - Loads are prioritized by their order in configuration, with the first load having the highest priority
- **Solar integration** - Automatically uses available solar power for controllable loads if configured
- **Throttleable loads** - Support for loads that can operate at variable power levels (e.g., EV chargers can be throttled between minimum and maximum amperage)
- **Overload detection** - Immediately sheds loads if consumption exceeds safe limits for more than `recalculate_interval_seconds * 3`, automatically recovers once sufficient load is shed
- **Emergency abort** - Cuts all loads if critical sensors become unavailable, requires manual re-enabling of load-control
- **Rate limiting** - Reduces wear on contactors from rapid switching

## How It Works
### Power Allocation
ZeroGrid manages controllable loads through a multi-pass algorithm:
1. **Calculate available power** - Based on house consumption, solar generation, grid import limits, and configured safety margins
2. **First pass - Turn loads on/off** - Evaluate loads in priority order (as listed in config). Each load is turned on if minimum power is available, respecting rate limits
3. **Second pass - Throttle up** - Allocate any remaining available power by throttling loads up from their minimum to maximum, in priority order
4. **Third pass - Overload protection** - If total consumption exceeds safe limits, cut loads in reverse priority order until safe
5. **Execute plan** - Update switch states and throttle values to match the calculated plan

### Throttling
Loads with a `throttle_amps_entity` configured can operate at variable power levels between their min and max values. During allocation:
- Load is first turned on at minimum amps if sufficient power is available
- If extra power is available after all loads are at minimum, higher priority loads are throttled up first
- Loads such as EV chargers benefit most from throttling as they can adjust charging current rapidly

For example:
1. Available power is 20A
2. EV charger requires 6-16A (min-max)
3. Load turns on at 6A initially
4. If no other loads need power, it throttles up to 16A
5. If another load needs 8A, EV charger may throttle down to make room


## Installation
Install via HACS or manually copy the `custom_components/zerogrid` directory to your Home Assistant configuration directory.

## Configuration
### Configuration Options

#### System Settings
| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `max_total_load_amps` | Yes | - | Maximum total load your electrical system can handle |
| `max_grid_import_amps` | Yes | - | Maximum power draw allowed from the grid |
| `max_solar_generation_amps` | Yes | - | Maximum solar generation capacity in amps |
| `safety_margin_amps` | No | 2.0 | Safety buffer above maximum load before triggering overload protection |
| `hysteresis_amps` | No | 1.0 | Prevents rapid switching/oscillation |
| `recalculate_interval_seconds` | No | 10 | Periodic recalculation interval (in addition to event-driven based on house consumption readings) |

#### Monitoring Entities
| Option | Required | Description |
|--------|----------|-------------|
| `house_consumption_amps_entity` | Yes | Sensor measuring total house consumption in amps |
| `solar_generation_kw_entity` | No | Sensor measuring solar generation in kW, if this is omitted solar production will not be used |
| `mains_voltage_entity` | No | Sensor measuring mains voltage, if this is omitted solar production will not be used |

#### Controllable Loads
Each controllable load has the following configuration:
| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `name` | Yes | - | Unique name for the load |
| `max_controllable_load_amps` | Yes | - | Maximum power this load can consume |
| `min_controllable_load_amps` | Yes | - | Minimum power needed (same as max for non-throttleable loads) |
| `load_amps_entity` | Yes | - | Sensor measuring actual consumption in amps |
| `switch_entity` | Yes | - | Switch entity to control the load |
| `min_toggle_interval_seconds` | No | 60 | Minimum time between on/off switches |
| `throttle_amps_entity` | No | - | Number entity to control throttle level (enables throttling) |
| `min_throttle_interval_seconds` | No | 10 | Minimum time between throttle adjustments |

**Note:** Loads are prioritized in the order they appear in the configuration. The first load listed has the highest priority (priority 0), the second has priority 1, and so on.

### Example Configuration
Add to your `configuration.yaml`:

```yaml
zerogrid:
    # Required: System limits
    max_total_load_amps: 63          # Maximum total load your system can handle
    max_grid_import_amps: 63         # Maximum draw from grid
    max_solar_generation_amps: 20    # Maximum solar generation capacity

    # Required: Monitoring entities
    house_consumption_amps_entity: "sensor.house_consumption"

    # Optional: Solar generation
    solar_generation_kw_entity: "sensor.solar_generation"
    mains_voltage_entity: "sensor.mains_voltage"

    # Optional: Safety and timing
    safety_margin_amps: 2.0          # Safety buffer above max load before triggering overload protection (default: 2.0)
    hysteresis_amps: 1.0             # Prevents oscillation (default: 1.0)
    recalculate_interval_seconds: 10 # Periodic recalculation interval (default: 10)
  
    controllable_loads:
        # First load = highest priority
        # Non-throttleable: min = max (hot water heater is either fully on or off)
        - name: "hot_water_heater"
          max_controllable_load_amps: 10
          min_controllable_load_amps: 10
          min_toggle_interval_seconds: 300
          load_amps_entity: "sensor.hot_water_current"
          switch_entity: "switch.hot_water_heater"
        
        # Second load = lower priority
        # Throttleable: min < max, includes throttle_amps_entity
        - name: "ev_charger"
          max_controllable_load_amps: 16
          min_controllable_load_amps: 6
          min_toggle_interval_seconds: 60
          min_throttle_interval_seconds: 30
          load_amps_entity: "sensor.ev_charger_current"
          switch_entity: "switch.ev_charger"
          throttle_amps_entity: "number.ev_charger_max_current"
```

## Entities
ZeroGrid creates the following entities that indicate how the system is performing:

| Entity | Description |
|--------|-------------|
| `sensor.zerogrid_available_load` | Current available power for new loads (amps). Calculated from grid import + solar generation - house consumption (excluding controllable loads). |
| `sensor.zerogrid_controlled_load` | Total power being used by ZeroGrid-controlled loads (amps). Shows the sum of expected consumption for all loads planned to be on. |
| `sensor.zerogrid_uncontrolled_load` | Power used by loads not under ZeroGrid control (amps). Calculated as total house consumption minus controlled loads. |
| `sensor.zerogrid_max_safe_load` | Maximum safe total load (amps). Calculated from configured limits plus safety margin. |
| `binary_sensor.zerogrid_overload` | Indicates if the system is in an overload state (on = overload detected). Clears automatically when system recovers. |
| `binary_sensor.zerogrid_safety_abort` | Indicates if a safety abort has occurred because critical sensor data is unavailable. |
| `switch.zerogrid_enable_load_control` | Master enable/disable for load control. When off, no loads will be automatically controlled. Cycling this also resets any throttling timers. |
| `switch.zerogrid_allow_grid_import` | Enable/disable grid import. When off, only solar power can be used for controllable loads. |
