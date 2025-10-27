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

    max_house_load_amps: float
    hysteresis_amps: float
    recalculate_interval_seconds: int = 10
    house_consumption_amps_entity: str
    mains_voltage_entity: str
    solar_generation_kw_entity: str | None = None
    allow_solar_consumption: bool = False
    controllable_loads: dict[str, ControllableLoadConfig] = {}
    enable_automatic_recalculation: bool = True
    enable_reactive_reallocation: bool = True
    # Minimum variance in amps to trigger reallocation
    variance_detection_threshold: float = 1.0
    # How long to wait before considering variance stable
    variance_detection_delay_seconds: int = 30
