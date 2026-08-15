"""Helper functions for Zero Grid integration."""

from typing import TypeIs

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import State


def parse_entity_domain(entity_id: str) -> str:
    """Extract the domain from an entity ID.

    Args:
        entity_id: The entity ID (e.g., 'switch.my_switch')

    Returns:
        The domain portion of the entity ID (e.g., 'switch')
    """
    return entity_id.split(".")[0]


def is_entity_usable(state: State | None) -> TypeIs[State]:
    """Return True if an entity exists and reports a real value.

    Args:
        state: The entity state, or None when the entity does not exist.

    Returns:
        False when the entity is missing, unknown, or unavailable.
    """
    return state is not None and state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE)
