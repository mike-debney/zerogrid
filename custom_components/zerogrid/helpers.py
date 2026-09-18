"""Helper functions for Zero Grid integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import State

if TYPE_CHECKING:
    # TypeIs is only in typing from Python 3.13. Importing it under
    # TYPE_CHECKING, with annotations postponed, keeps the hint for type
    # checkers without requiring it at runtime.
    from typing_extensions import TypeIs


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


def parse_amps(state: State | None) -> float | None:
    """Return an entity's value as a number, or None if it is not one.

    Entities report strings, and an entity can report something that is
    neither a number nor one of the two states we check for - an empty string
    while a device is starting up, for instance. Converting that directly
    raised inside the state listener, which stopped the recalculation it was
    supposed to trigger.
    """
    if not is_entity_usable(state):
        return None
    try:
        return float(state.state)
    except (ValueError, TypeError):
        return None
