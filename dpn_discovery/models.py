"""
Formal data structures for the DPN Discovery Pipeline.

Defines all core types from Section 2 of the specification:
  - Event Log & Traces (§2.1)
  - Extended Finite State Machine (§2.2)
  - Data Petri Net (§2.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import z3


# ---------------------------------------------------------------------------
# §2.1  Event Log & Traces
# ---------------------------------------------------------------------------

DataPayload = dict[str, float | int | str]
"""A data vector  d ∈ D  mapping variable names to values."""


@dataclass(frozen=True, slots=True)
class Event:
    """An event  e = (a, d⃗)  with an activity label and a data vector."""

    activity: str
    payload: DataPayload = field(default_factory=dict)


@dataclass(slots=True)
class Trace:
    """A trace  σ = ⟨e₁, e₂, …, eₙ⟩  — an ordered sequence of events."""

    events: list[Event] = field(default_factory=list)


@dataclass(slots=True)
class EventLog:
    """An event log  L  — a multiset of traces.

    Also stores the globally extracted metadata (activity set Σ,
    variable set V) produced by the preprocessing step.
    """

    traces: list[Trace] = field(default_factory=list)
    activities: set[str] = field(default_factory=set)
    variables: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# §2.2  Extended Finite State Machine (EFSM)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Transition:
    """An EFSM transition  t = (s_src, s_tgt, a, g, u).

    Attributes:
        source_id:     Identifier of the source state.
        target_id:     Identifier of the target state.
        activity:      Activity label  a ∈ Σ.
        data_samples:  Collected data observations on this edge
                       (list of {var: value} dicts).  Populated
                       during PTA construction and consumed by
                       guard / postcondition synthesis.
        pre_post_pairs: List of (pre-state, post-state) observation
                       pairs from the original traces.  The pre-state
                       is the payload of the preceding event (or
                       empty dict for the first event in a trace).
                       Populated during PTA construction and consumed
                       by postcondition synthesis.
        guard_formula: Z3 Boolean formula  g(V) → {0,1}.
                       ``None`` means *True* (unconditional).
        update_rule:   Mapping from variable name to its Z3
                       update expression.  ``None`` means identity.
    """

    source_id: str
    target_id: str
    activity: str
    data_samples: list[dict[str, Any]] = field(default_factory=list)
    pre_post_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)
    guard_formula: Optional[z3.BoolRef] = None
    update_rule: Optional[dict[str, z3.ExprRef]] = None


@dataclass(slots=True)
class EFSM:
    """Extended Finite State Machine  M = (S, s₀, Σ, V, T).

    States are represented as string identifiers.  The actual
    transition list stores all structural / data information.
    """

    states: set[str] = field(default_factory=set)
    initial_state: str = "q0"
    alphabet: set[str] = field(default_factory=set)
    variables: set[str] = field(default_factory=set)
    transitions: list[Transition] = field(default_factory=list)
    accepting_states: set[str] = field(default_factory=set)

    # ----- convenience helpers -------------------------------------------

    def outgoing(self, state_id: str) -> list[Transition]:
        """Return all transitions leaving *state_id*."""
        return [t for t in self.transitions if t.source_id == state_id]

    def incoming(self, state_id: str) -> list[Transition]:
        """Return all transitions entering *state_id*."""
        return [t for t in self.transitions if t.target_id == state_id]

    def outgoing_by_activity(self, state_id: str) -> dict[str, list[Transition]]:
        """Group outgoing transitions of *state_id* by activity label."""
        groups: dict[str, list[Transition]] = {}
        for t in self.outgoing(state_id):
            groups.setdefault(t.activity, []).append(t)
        return groups

    def successor_states(self, state_id: str) -> set[str]:
        """Return the set of states reachable in one step from *state_id*."""
        return {t.target_id for t in self.outgoing(state_id)}

    def deep_copy(self) -> EFSM:
        """Return a deep copy of this EFSM (structural only; Z3 refs shared)."""
        return EFSM(
            states=set(self.states),
            initial_state=self.initial_state,
            alphabet=set(self.alphabet),
            variables=set(self.variables),
            transitions=[
                Transition(
                    source_id=t.source_id,
                    target_id=t.target_id,
                    activity=t.activity,
                    data_samples=[dict(d) for d in t.data_samples],
                    pre_post_pairs=[
                        (dict(pre), dict(post))
                        for pre, post in t.pre_post_pairs
                    ],
                    guard_formula=t.guard_formula,
                    update_rule=dict(t.update_rule) if t.update_rule else None,
                )
                for t in self.transitions
            ],
            accepting_states=set(self.accepting_states),
        )


# ---------------------------------------------------------------------------
# §2.3  Data Petri Net (DPN)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Place:
    """A place  p ∈ P  in the Data Petri Net."""

    name: str


@dataclass(slots=True)
class DPNTransition:
    """A DPN transition with optional guard and update annotations.

    Attributes:
        name:          Human-readable identifier (often the activity label).
        guard:         Z3 Boolean guard  g(V).
        update_rule:   Mapping variable → Z3 update expression.
        input_places:  Set of input place names  (•t).
        output_places: Set of output place names  (t•).
    """

    name: str
    guard: Optional[z3.BoolRef] = None
    update_rule: Optional[dict[str, z3.ExprRef]] = None
    input_places: set[str] = field(default_factory=set)
    output_places: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Normalize default sets.

        :param self: Transition instance.
        :return: None.
        """
        if self.input_places is None:
            self.input_places = set()
        if self.output_places is None:
            self.output_places = set()


@dataclass(slots=True)
class DataPetriNet:
    """Data Petri Net  N = (P, T, F, V, G, U).

    The flow relation F is implicit in ``DPNTransition.input_places``
    and ``DPNTransition.output_places``.
    """

    places: list[Place] = field(default_factory=list)
    transitions: list[DPNTransition] = field(default_factory=list)
    variables: set[str] = field(default_factory=set)
    initial_marking: set[str] = field(default_factory=set)

    # ----- helpers --------------------------------------------------------

    def place_names(self) -> set[str]:
        """Return the set of all place names."""
        return {p.name for p in self.places}

    def __init__(
        self,
        places: Optional[list[Place]] = None,
        transitions: Optional[list[DPNTransition]] = None,
        variables: Optional[set[str]] = None,
        initial_marking: Optional[set[str]] = None,
    ) -> None:
        """
        Create a DataPetriNet.

        :param places: Places.
        :param transitions: Transitions.
        :param variables: Variable names.
        :param initial_marking: Set of place names in the initial marking.
        :return: None.
        """
        self.places = places or []
        self.transitions = transitions or []
        self.variables = variables or set()
        self.initial_marking = initial_marking or set()