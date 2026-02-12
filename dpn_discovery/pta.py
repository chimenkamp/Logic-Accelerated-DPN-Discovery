"""
Step 2 — Prefix Tree Acceptor (PTA) Construction.

Builds a tree-shaped EFSM from an ``EventLog`` where every unique
trace prefix becomes a distinct state and edges carry the data
payload  d⃗  of the corresponding event.

Reference: §4 Steps 1–2 of the specification.
"""

from __future__ import annotations

from typing import Any

from dpn_discovery.models import EFSM, EventLog, Transition


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pta(log: EventLog) -> EFSM:
    """Construct the Prefix Tree Acceptor from *log*.

    Algorithm (per specification):
      • Root state  s₀ = "q0".
      • For each trace  σ = ⟨e₁, …, eₙ⟩ :
          – Traverse from s₀.  If an edge labelled eᵢ.a exists,
            follow it.
          – Otherwise create a new state  s_new  and a new edge
            (s_curr, s_new, eᵢ.a).
          – **Crucially**, store eᵢ.d⃗  on the edge.  This data
            collection is required for Steps 3 and 4.
    """
    initial = "q0"
    states: set[str] = {initial}
    transitions: list[Transition] = []
    accepting: set[str] = set()
    _counter = _StateCounter()

    for trace in log.traces:
        current_state = initial
        accumulated_state: dict[str, Any] = {}  # full state across the trace

        for event in trace.events:
            # Look for an existing edge from current_state with this activity.
            next_state: str | None = None
            matched_transition: Transition | None = None

            for t in transitions:
                if t.source_id == current_state and t.activity == event.activity:
                    next_state = t.target_id
                    matched_transition = t
                    break

            post_payload = dict(event.payload)

            # Pre-state is the accumulated state *before* this event.
            pre_snapshot = dict(accumulated_state)

            if next_state is not None and matched_transition is not None:
                # Edge exists — follow it, append data sample and pre/post pair.
                matched_transition.data_samples.append(post_payload)
                matched_transition.pre_post_pairs.append(
                    (pre_snapshot, post_payload)
                )
                current_state = next_state
            else:
                # Create fresh state and edge.
                new_state = _counter.next()
                states.add(new_state)
                transitions.append(
                    Transition(
                        source_id=current_state,
                        target_id=new_state,
                        activity=event.activity,
                        data_samples=[post_payload],
                        pre_post_pairs=[(pre_snapshot, post_payload)],
                    )
                )
                current_state = new_state

            # Update accumulated state with values written by this event.
            accumulated_state.update(post_payload)

        # The last state in every trace is accepting.
        accepting.add(current_state)

    return EFSM(
        states=states,
        initial_state=initial,
        alphabet=set(log.activities),
        variables=set(log.variables),
        transitions=transitions,
        accepting_states=accepting,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StateCounter:
    """Monotonic state-id generator producing "q1", "q2", …"""

    def __init__(self) -> None:
        self._n: int = 0

    def next(self) -> str:
        self._n += 1
        return f"q{self._n}"
