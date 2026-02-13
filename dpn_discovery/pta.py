"""
Step 2 — Prefix Tree Acceptor (PTA) Construction.

Builds a tree-shaped EFSM from an ``EventLog`` where every unique
trace prefix becomes a distinct state and edges carry the data
payload  d⃗  of the corresponding event.

In **MINT mode** (Walkinshaw et al. 2013, Algorithm 3 line 4),
prefix sharing additionally requires that the trained classifiers
produce identical predictions for every data configuration in the
prefix.  This makes the PTA the most specific EFSM consistent
with the traces.

Reference: §4 Steps 1–2 of the specification;
           Walkinshaw et al. (2013), Algorithm 3 line 4.
"""

from __future__ import annotations

from typing import Any

from dpn_discovery.classifiers import Classifiers, predict_next_label
from dpn_discovery.models import EFSM, EventLog, Transition


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pta(
    log: EventLog,
    *,
    classifiers: Classifiers | None = None,
    variables: list[str] | None = None,
) -> EFSM:
    """Construct the Prefix Tree Acceptor from *log*.

    Parameters
    ----------
    log : EventLog
        The parsed event log.
    classifiers : Classifiers | None
        Trained per-label classifiers.  When provided (MINT mode),
        prefix sharing requires that classifier predictions match
        at every position.
    variables : list[str] | None
        Ordered variable names for classifier feature vectors.
        Required when *classifiers* is given.

    Algorithm (per specification + Walkinshaw Algorithm 3, line 4):
      • Root state  s₀ = "q0".
      • For each trace  σ = ⟨e₁, …, eₙ⟩ :
          – Traverse from s₀.  If an edge labelled eᵢ.a exists
            **and** (in MINT mode) the classifier prediction for
            the current data matches the predictions already
            stored on that edge → follow it.
          – Otherwise create a new state  s_new  and a new edge
            (s_curr, s_new, eᵢ.a).
          – **Crucially**, store eᵢ.d⃗  on the edge.
    """
    use_classifiers = classifiers is not None and variables is not None

    initial = "q0"
    states: set[str] = {initial}
    transitions: list[Transition] = []
    accepting: set[str] = set()
    _counter = _StateCounter()

    # Per-transition cache of classifier predictions (for MINT prefix check).
    _transition_predictions: dict[int, str | None] = {}

    for trace in log.traces:
        current_state = initial
        accumulated_state: dict[str, Any] = {}  # full state across the trace

        events = trace.events
        for idx, event in enumerate(events):
            # Compute the classifier prediction for this position
            # (what the classifier says the *next* event should be).
            this_prediction: str | None = None
            if use_classifiers and idx < len(events) - 1:
                data_for_classifier = {
                    v: event.payload.get(v, 0) for v in variables  # type: ignore[union-attr]
                }
                this_prediction = predict_next_label(
                    classifiers, event.activity, data_for_classifier, variables,  # type: ignore[arg-type]
                )

            # Look for an existing edge from current_state with this activity.
            next_state: str | None = None
            matched_transition: Transition | None = None

            for t in transitions:
                if t.source_id == current_state and t.activity == event.activity:
                    if use_classifiers:
                        # MINT mode: only share prefix if the classifier
                        # prediction matches what was recorded for this edge.
                        existing_pred = _transition_predictions.get(id(t))
                        if existing_pred != this_prediction:
                            continue
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
                new_t = Transition(
                    source_id=current_state,
                    target_id=new_state,
                    activity=event.activity,
                    data_samples=[post_payload],
                    pre_post_pairs=[(pre_snapshot, post_payload)],
                )
                transitions.append(new_t)
                # Cache classifier prediction for MINT prefix sharing.
                _transition_predictions[id(new_t)] = this_prediction
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
