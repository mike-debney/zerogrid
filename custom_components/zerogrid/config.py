"""Configuration."""


class ControllableLoadConfig:
    """Configuration for a controllable load."""

    name: str = ""
    priority: int = 0
    max_controllable_load_amps: float = 0
    min_controllable_load_amps: float = 0
    min_toggle_interval_seconds: int = 60
    min_throttle_interval_seconds: int = 10
    load_amps_entity: str
    switch_entity: str
    throttle_amps_entity: str | None = None
    can_throttle: bool = False


class Config:
    """Configuration for Zero Grid integration."""

    max_total_load_amps: float
    max_grid_import_amps: float
    max_solar_generation_amps: float
    safety_margin_amps: float = 2.0
    hysteresis_amps: float = 1.0
    recalculate_interval_seconds: int = 10
    house_consumption_amps_entity: str
    mains_voltage_entity: str | None = None
    solar_generation_kw_entity: str | None = None
    allow_solar_consumption: bool = False
    enable_automatic_recalculation: bool = True
    controllable_loads: dict[str, ControllableLoadConfig] = {}
