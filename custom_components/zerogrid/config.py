"""Configuration."""


class ControllableLoadConfig:
    """Configuration for a controllable load."""

    name: str = ""
    priority: int = 0
    max_controllable_load_amps: float = 0
    min_controllable_load_amps: float = 0
    min_toggle_interval_seconds: int = 60
    min_throttle_interval_seconds: int = 10
    solar_turn_on_window_seconds: int = 300
    solar_turn_off_window_seconds: int = 300
    load_measurement_delay_seconds: int = 120
    load_amps_entity: str
    switch_entity: str
    throttle_amps_entity: str | None = None
    can_throttle: bool = False
    can_turn_on_entity: str | None = None
    can_turn_on_ignore_unavailable: bool = False
    assume_always_under_load_control: bool = False


class Config:
    """Configuration for Zero Grid integration."""

    max_total_load_amps: float
    max_grid_import_amps: float
    max_solar_generation_amps: float
    safety_margin_amps: float = 2.0
    recalculate_interval_seconds: int = 10
    house_consumption_amps_entity: str
    solar_generation_amps_entity: str | None = None
    allow_solar_consumption: bool = False
    enable_automatic_recalculation: bool = True
    disable_consumption_unavailable_safety_abort: bool = False
    controllable_loads: dict[str, ControllableLoadConfig] = {}
