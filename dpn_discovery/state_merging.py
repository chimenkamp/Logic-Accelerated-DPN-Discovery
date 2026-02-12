"""
Step 3 — Data-Driven State Merging (Walkinshaw et al., 2013).

Generalises the Prefix Tree Acceptor into a compact EFSM by
iteratively merging compatible states using the Blue-Fringe /
Evidence-Driven State-Merging (EDSM) algorithm.

Fold-closure compatibility
--------------------------
Per Walkinshaw et al. a merge of  q  and  q'  is valid **only**
if the merge is fold-closed: merging those two states AND all
pairs of successor states reachable via the same activity must
not create data-contradictory transitions.  Concretely, when a
folded state would have ≥ 2 outgoing transitions on the same
activity going to distinct effective targets, their data samples
must be linearly separable (so a guard can later distinguish
them).

Reference
---------
  Walkinshaw, N., Taylor, R., Derrick, J.  *Inferring Extended
  Finite State Machine Models from Software Executions* (2013).
"""

from __future__ import annotations

from collections import deque

import z3

from dpn_discovery.models import EFSM, Transition


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_state_merging(pta: EFSM) -> EFSM:
    """Execute the Blue-Fringe state-merging loop on *pta*.

    Colour scheme (Blue-Fringe):
      • **Red** states: finalised (part of the hypothesis).
      • **Blue** states: immediate successors of red that have
        not yet been promoted or merged.

    At each iteration the lowest-ordered blue state is picked.
    We attempt to merge it with every red state; the first
    fold-compatible red state wins.  If no red state is
    compatible the blue state is promoted to red.
    """
    efsm = pta.deep_copy()

    red: set[str] = {efsm.initial_state}
    blue: deque[str] = deque(
        sorted(_direct_successors(efsm, efsm.initial_state) - red)
    )

    while blue:
        candidate = blue.popleft()
        if candidate not in efsm.states:
            continue

        merged = False
        for target in sorted(red):
            if target not in efsm.states:
                continue
            if _is_fold_compatible(efsm, target, candidate):
                efsm = _merge_states(efsm, target, candidate)
                # Recalculate blue frontier.
                blue = deque(sorted(
                    s for s in _all_successors_of_set(efsm, red) - red
                    if s in efsm.states
                ))
                merged = True
                break

        if not merged:
            red.add(candidate)
            new_blue = _direct_successors(efsm, candidate) - red - set(blue)
            blue.extend(sorted(new_blue))

    return efsm


# ═══════════════════════════════════════════════════════════════════════════
# Fold-closure compatibility test  (Walkinshaw et al.)
# ═══════════════════════════════════════════════════════════════════════════

def _is_fold_compatible(
    efsm: EFSM,
    q: str,
    q_prime: str,
    _visited: set[tuple[str, str]] | None = None,
) -> bool:
    """Return ``True`` iff  q  and  q'  can be merged in a
    fold-closed manner.

    Rules (following Walkinshaw et al. 2013 closely):

      1. Merging an accepting and a non-accepting state is
         forbidden (fundamentally different process endings).

      2. For every shared activity  a  whose outgoing edges lead
         to *different* effective targets: the data samples must
         be linearly separable (so a guard can later distinguish
         them).  If either side has no data samples we consider
         the pair **not** separable (missing data ≠ proof of
         separability).

      3. **Fold-closure**: for every shared activity  a  where
         both states have outgoing transitions going to distinct
         targets, the respective target states must *also* be
         fold-compatible (recursively).

      4. **Incoming-activity consistency**: states reached by
         completely different activities AND that also have
         symmetrically different outgoing alphabets represent
         truly different process positions and must not be merged.
         (Prevents collapsing stages like "after Submit" with
         "after Approve" while allowing loop-entry merges.)
    """
    if _visited is None:
        _visited = set()

    pair = (min(q, q_prime), max(q, q_prime))
    if pair in _visited:
        return True
    _visited.add(pair)

    # Rule 1: accept / non-accept clash.
    q_acc = q in efsm.accepting_states
    qp_acc = q_prime in efsm.accepting_states
    if q_acc != qp_acc:
        return False

    out_q = _group_by_activity(efsm, q)
    out_q_prime = _group_by_activity(efsm, q_prime)

    acts_q = set(out_q)
    acts_qp = set(out_q_prime)
    shared = acts_q & acts_qp

    # Rule 4: incoming-activity consistency.
    # States reached exclusively by different activities *and*
    # that also have symmetrically different outgoing alphabets
    # represent truly different process positions.
    in_q = _incoming_activities(efsm, q)
    in_qp = _incoming_activities(efsm, q_prime)
    if in_q and in_qp and in_q.isdisjoint(in_qp):
        only_out_q = acts_q - acts_qp
        only_out_qp = acts_qp - acts_q
        if only_out_q and only_out_qp:
            # Both sides have exclusive incoming AND exclusive outgoing
            # activities — clearly different process stages.
            return False

    # Rule 4b: initial-state protection.
    # The initial state (no incoming edges) should only be merged
    # with another state that shares at least one outgoing
    # activity.  Without shared outgoing activities the merge
    # would collapse the first process step with a later stage.
    if (not in_q) != (not in_qp):
        # Exactly one of the two is the initial state (no incoming).
        if not shared:
            return False

    # Rule 3: data separability for shared activities.
    for activity in shared:
        targets_q = _unique_targets(out_q[activity])
        targets_qp = _unique_targets(out_q_prime[activity])
        if targets_q != targets_qp:
            samples_q = _collect_samples(out_q[activity])
            samples_qp = _collect_samples(out_q_prime[activity])
            # Both sides must have data for a separability check
            # to be meaningful.  Missing data is NOT evidence of
            # separability.
            if not samples_q or not samples_qp:
                return False
            if not _data_separable(samples_q, samples_qp, efsm.variables):
                return False

    # Rule 4: fold-closure (recursive).
    for activity in shared:
        tgts_q = sorted(_unique_targets(out_q[activity]))
        tgts_qp = sorted(_unique_targets(out_q_prime[activity]))
        # Recursively check each pair of targets that would be
        # folded together.  For single-target activities this is
        # the standard Walkinshaw check.  For multi-target we pair
        # them in sorted order.
        for sq, sqp in zip(tgts_q, tgts_qp):
            if sq != sqp:
                if not _is_fold_compatible(efsm, sq, sqp, _visited):
                    return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Data separability via Z3
# ═══════════════════════════════════════════════════════════════════════════

def _data_separable(
    samples_a: list[dict],
    samples_b: list[dict],
    variables: set[str],
) -> bool:
    """Check whether two data sets can be separated by a linear
    predicate   h(d) = Σ cᵢ·dᵢ + c₀  ≷ 0.

    Uses Z3:
      ∀ d ∈ A :  h(d) ≥ ε
      ∀ d ∈ B :  h(d) ≤ −ε
    """
    if not samples_a or not samples_b:
        return True

    numeric_vars = _numeric_variable_names(samples_a + samples_b, variables)
    if not numeric_vars:
        return True

    solver = z3.Solver()
    solver.set("timeout", 2000)

    coeffs = {v: z3.Real(f"c_{v}") for v in numeric_vars}
    bias = z3.Real("c_bias")
    eps = z3.RealVal(1)

    for sample in samples_a:
        expr = sum(
            coeffs[v] * _to_z3(sample.get(v, 0)) for v in numeric_vars
        ) + bias
        solver.add(expr >= eps)

    for sample in samples_b:
        expr = sum(
            coeffs[v] * _to_z3(sample.get(v, 0)) for v in numeric_vars
        ) + bias
        solver.add(expr <= -eps)

    return solver.check() == z3.sat


# ---------------------------------------------------------------------------
# Merge operation
# ---------------------------------------------------------------------------

def _merge_states(efsm: EFSM, keep: str, remove: str) -> EFSM:
    """Merge *remove* into *keep*: redirect all edges and union data.

    Returns a **new** EFSM instance (immutable-style update).
    """
    new_transitions: list[Transition] = []

    for t in efsm.transitions:
        src = keep if t.source_id == remove else t.source_id
        tgt = keep if t.target_id == remove else t.target_id

        # Check for duplicate edge (same src, tgt, activity) — if so,
        # union the data samples instead of creating a second edge.
        existing = next(
            (
                nt
                for nt in new_transitions
                if nt.source_id == src
                and nt.target_id == tgt
                and nt.activity == t.activity
            ),
            None,
        )
        if existing is not None:
            existing.data_samples.extend(t.data_samples)
            existing.pre_post_pairs.extend(t.pre_post_pairs)
        else:
            new_transitions.append(
                Transition(
                    source_id=src,
                    target_id=tgt,
                    activity=t.activity,
                    data_samples=list(t.data_samples),
                    pre_post_pairs=list(t.pre_post_pairs),
                    guard_formula=t.guard_formula,
                    update_rule=dict(t.update_rule) if t.update_rule else None,
                )
            )

    new_states = {s for s in efsm.states if s != remove}
    new_accepting = {keep if s == remove else s for s in efsm.accepting_states}

    return EFSM(
        states=new_states,
        initial_state=efsm.initial_state,
        alphabet=set(efsm.alphabet),
        variables=set(efsm.variables),
        transitions=new_transitions,
        accepting_states=new_accepting,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_by_activity(efsm: EFSM, state_id: str) -> dict[str, list[Transition]]:
    groups: dict[str, list[Transition]] = {}
    for t in efsm.transitions:
        if t.source_id == state_id:
            groups.setdefault(t.activity, []).append(t)
    return groups


def _collect_samples(transitions: list[Transition]) -> list[dict]:
    """Collect pre-state samples for data-separability checking.

    Guards evaluate the state *before* a transition fires, so the
    separability test must use pre-state snapshots.  Falls back to
    ``data_samples`` when ``pre_post_pairs`` are unavailable.
    """
    samples: list[dict] = []
    for t in transitions:
        if t.pre_post_pairs:
            samples.extend(pre for pre, _post in t.pre_post_pairs if pre)
        else:
            samples.extend(s for s in t.data_samples if s)
    return samples


def _unique_targets(transitions: list[Transition]) -> set[str]:
    return {t.target_id for t in transitions}


def _incoming_activities(efsm: EFSM, state_id: str) -> set[str]:
    """Return the set of activity labels on edges *entering* *state_id*."""
    return {t.activity for t in efsm.transitions if t.target_id == state_id}


def _direct_successors(efsm: EFSM, state_id: str) -> set[str]:
    return {t.target_id for t in efsm.transitions if t.source_id == state_id}


def _all_successors_of_set(efsm: EFSM, states: set[str]) -> set[str]:
    result: set[str] = set()
    for s in states:
        result |= _direct_successors(efsm, s)
    return result


def _numeric_variable_names(
    samples: list[dict], variables: set[str]
) -> list[str]:
    """Return variable names that carry numeric values in *samples*."""
    numeric: set[str] = set()
    for sample in samples:
        for var in variables:
            val = sample.get(var)
            if isinstance(val, (int, float)):
                numeric.add(var)
    return sorted(numeric)


def _to_z3(value: int | float | str) -> z3.ArithRef:
    """Convert a Python numeric value to a Z3 rational literal."""
    match value:
        case int():
            return z3.RealVal(value)
        case float():
            return z3.RealVal(value)
        case _:
            return z3.RealVal(0)
