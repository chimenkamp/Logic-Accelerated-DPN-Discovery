"""
Step 3 — State Merging  (Walkinshaw et al., 2013).

Implements **two** merging strategies from the paper:

  • **Blue-Fringe** (Algorithms 1 & 2) — control-flow-only merging
    with Evidence-Driven State-Merging (EDSM) scoring.
  • **MINT** (Algorithm 3) — Blue-Fringe extended with trained data
    classifiers for transition equivalence, non-determinism
    resolution, and post-merge consistency validation.

Both strategies maintain the red/blue colouring scheme.  At each
iteration, *all* (red, blue) candidate pairs are scored; the pair
with the **highest EDSM score** (≥ *k*) is selected.  If no pair
meets the threshold, the blue candidate is promoted to red.

Key differences from the previous implementation
-------------------------------------------------
  • Removed the Z3-based linear-hyperplane separability check.
  • Added EDSM scoring (``_compute_edsm_score``).
  • Added classifier-based transition equivalence
    (``_equivalent_transitions``).
  • Added recursive non-determinism resolution inside merge.
  • Added post-merge ``_is_consistent`` validation (MINT only).
  • Added a ``Failed`` set to skip rejected merge pairs.
  • Added ``MergeStrategy`` enum support: ``NONE`` / ``BLUE_FRINGE``
    / ``MINT``.

Reference
---------
  Walkinshaw, N., Taylor, R., Derrick, J.  *Inferring Extended
  Finite State Machine Models from Software Executions* (2013).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from dpn_discovery.classifiers import Classifiers, predict_next_label
from dpn_discovery.models import EFSM, MergeStrategy, Transition

logger = logging.getLogger(__name__)

# Type alias: mapping  transition-id → list-of-data-dicts.
# Each data dict corresponds to one trace element that traversed
# the transition.  Used for classifier queries during merging.
Vars = dict[int, list[dict[str, Any]]]


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_state_merging(
    pta: EFSM,
    *,
    strategy: MergeStrategy = MergeStrategy.BLUE_FRINGE,
    classifiers: Classifiers | None = None,
    variables: list[str] | None = None,
    k: int = 0,
) -> EFSM:
    """Execute the state-merging loop on *pta*.

    Parameters
    ----------
    pta : EFSM
        The Prefix Tree Acceptor built from the event log.
    strategy : MergeStrategy
        ``NONE`` — return PTA unchanged.
        ``BLUE_FRINGE`` — control-flow-only (Algorithms 1 & 2).
        ``MINT`` — data-aware with classifiers (Algorithm 3).
    classifiers : Classifiers | None
        Trained per-label classifiers (required for MINT).
    variables : list[str] | None
        Ordered variable names for classifier feature vectors.
        Required for MINT.
    k : int
        Minimum EDSM score for a merge to be considered
        (Algorithm 2, parameter *k*).  Default is 0.

    Returns
    -------
    EFSM
        The generalised state machine after merging.
    """
    if strategy is MergeStrategy.NONE:
        logger.info("Merging disabled (strategy=NONE)")
        return pta.deep_copy()

    if strategy is MergeStrategy.MINT:
        if classifiers is None or variables is None:
            raise ValueError(
                "MINT strategy requires trained classifiers and "
                "an ordered variables list."
            )

    efsm = pta.deep_copy()
    vars_map = _build_vars_map(efsm)
    failed: set[tuple[str, str]] = set()

    red: set[str] = {efsm.initial_state}
    blue: deque[str] = deque(
        sorted(_direct_successors(efsm, efsm.initial_state) - red)
    )

    while blue:
        candidate = blue.popleft()
        if candidate not in efsm.states:
            continue

        # --- choosePairs: score all (red, blue) candidates ----------------
        best_target: str | None = None
        best_score: int = k - 1  # must exceed k

        for target in sorted(red):
            if target not in efsm.states:
                continue
            pair = _canon(target, candidate)
            if pair in failed:
                continue

            score = _compute_edsm_score(
                efsm, target, candidate,
                classifiers=classifiers,
                variables=variables,
                strategy=strategy,
            )
            if score > best_score:
                best_score = score
                best_target = target

        # --- attempt merge with best-scoring pair -------------------------
        merged = False
        if best_target is not None and best_score >= k:
            backup = efsm.deep_copy()
            backup_vars = dict(vars_map)

            efsm, vars_map = _merge_states(
                efsm, best_target, candidate, vars_map,
                classifiers=classifiers,
                variables=variables,
                strategy=strategy,
            )

            # MINT post-merge consistency check (Algorithm 3, line 7).
            if strategy is MergeStrategy.MINT:
                if not _is_consistent(efsm, classifiers, variables, vars_map):  # type: ignore[arg-type]
                    # Reject merge.
                    logger.debug(
                        "Merge %s ← %s rejected (inconsistent)",
                        best_target, candidate,
                    )
                    failed.add(_canon(best_target, candidate))
                    efsm = backup
                    vars_map = backup_vars
                else:
                    merged = True
            else:
                merged = True

            if merged:
                logger.debug(
                    "Merged %s ← %s  (score=%d)", best_target, candidate, best_score
                )
                # Recalculate the blue frontier.
                blue = deque(sorted(
                    s for s in _all_successors_of_set(efsm, red) - red
                    if s in efsm.states
                ))

        if not merged:
            red.add(candidate)
            new_blue = _direct_successors(efsm, candidate) - red - set(blue)
            blue.extend(sorted(new_blue))

    return efsm


# ═══════════════════════════════════════════════════════════════════════════
# EDSM Scoring  (Algorithm 2 — calculateScore)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_edsm_score(
    efsm: EFSM,
    s1: str,
    s2: str,
    *,
    classifiers: Classifiers | None,
    variables: list[str] | None,
    strategy: MergeStrategy,
    _visited: set[tuple[str, str]] | None = None,
) -> int:
    """Recursively compute the EDSM compatibility score for merging
    *s1* and *s2*.

    The score counts the number of outgoing-transition pairs that
    share the same label (and in MINT mode, produce the same
    classifier predictions for all attached data).

    A return value of ``-1`` signals incompatibility (merging
    would produce a contradiction).

    Corresponds to ``calculateScore`` in Algorithm 2 of
    Walkinshaw et al. (2013).
    """
    if _visited is None:
        _visited = set()

    pair = _canon(s1, s2)
    if pair in _visited:
        return 0
    _visited.add(pair)

    # Accept / non-accept clash → incompatible.
    if (s1 in efsm.accepting_states) != (s2 in efsm.accepting_states):
        return -1

    out1 = _group_by_activity(efsm, s1)
    out2 = _group_by_activity(efsm, s2)
    shared = set(out1) & set(out2)

    score = 0

    for activity in shared:
        ts1 = out1[activity]
        ts2 = out2[activity]

        if strategy is MergeStrategy.MINT and classifiers and variables:
            # In MINT mode, transitions on the same label are only
            # "matching" if the classifier gives the same predictions
            # for all their data.
            if _transitions_classifier_equivalent(
                ts1, ts2, activity, classifiers, variables
            ):
                score += 1
            else:
                # Classifier distinguishes them — they are
                # data-deterministic and do NOT need merging.
                # This is fine, not incompatible.
                score += 1
        else:
            # Blue-Fringe (CF only): transitions match by label.
            score += 1

        # Recurse into successor states.
        targets1 = sorted({t.target_id for t in ts1})
        targets2 = sorted({t.target_id for t in ts2})
        for t1, t2 in zip(targets1, targets2):
            if t1 != t2:
                sub = _compute_edsm_score(
                    efsm, t1, t2,
                    classifiers=classifiers,
                    variables=variables,
                    strategy=strategy,
                    _visited=_visited,
                )
                if sub < 0:
                    return -1
                score += sub

    return score


# ═══════════════════════════════════════════════════════════════════════════
# Merge + recursive non-determinism resolution  (Algorithm 3, lines 14–25)
# ═══════════════════════════════════════════════════════════════════════════

def _merge_states(
    efsm: EFSM,
    keep: str,
    remove: str,
    vars_map: Vars,
    *,
    classifiers: Classifiers | None,
    variables: list[str] | None,
    strategy: MergeStrategy,
) -> tuple[EFSM, Vars]:
    """Merge *remove* into *keep*, then recursively resolve
    non-determinism.

    In Blue-Fringe mode, non-determinism is detected by label only.
    In MINT mode, ``_equivalent_transitions`` uses classifiers to
    decide whether two same-label transitions are distinguishable
    by data (Algorithm 3, lines 26–33).

    Returns the updated ``(efsm, vars_map)``.
    """
    # --- Redirect all edges from *remove* to *keep* -----------------------
    new_transitions: list[Transition] = []

    for t in efsm.transitions:
        src = keep if t.source_id == remove else t.source_id
        tgt = keep if t.target_id == remove else t.target_id

        # Union data when a duplicate edge arises.
        existing = next(
            (
                nt for nt in new_transitions
                if nt.source_id == src
                and nt.target_id == tgt
                and nt.activity == t.activity
            ),
            None,
        )
        if existing is not None:
            existing.data_samples.extend(t.data_samples)
            existing.pre_post_pairs.extend(t.pre_post_pairs)
            # Merge vars_map entries.
            eid = id(existing)
            tid = id(t)
            if tid in vars_map:
                vars_map.setdefault(eid, []).extend(vars_map.pop(tid))
        else:
            nt = Transition(
                source_id=src,
                target_id=tgt,
                activity=t.activity,
                data_samples=list(t.data_samples),
                pre_post_pairs=list(t.pre_post_pairs),
                guard_formula=t.guard_formula,
                update_rule=dict(t.update_rule) if t.update_rule else None,
            )
            new_transitions.append(nt)
            # Transfer vars_map.
            tid = id(t)
            if tid in vars_map:
                vars_map[id(nt)] = vars_map.pop(tid)

    new_states = {s for s in efsm.states if s != remove}
    new_accepting = {keep if s == remove else s for s in efsm.accepting_states}

    efsm = EFSM(
        states=new_states,
        initial_state=efsm.initial_state,
        alphabet=set(efsm.alphabet),
        variables=set(efsm.variables),
        transitions=new_transitions,
        accepting_states=new_accepting,
    )

    # --- Recursive non-determinism resolution (findNonDeterminism) --------
    efsm, vars_map = _resolve_non_determinism(
        efsm, keep, vars_map,
        classifiers=classifiers,
        variables=variables,
        strategy=strategy,
    )

    return efsm, vars_map


def _resolve_non_determinism(
    efsm: EFSM,
    state: str,
    vars_map: Vars,
    *,
    classifiers: Classifiers | None,
    variables: list[str] | None,
    strategy: MergeStrategy,
    _visited: set[str] | None = None,
) -> tuple[EFSM, Vars]:
    """Recursively merge non-deterministic successor states.

    For each activity leaving *state*, if there are ≥ 2 transitions
    going to **different** targets that are considered *equivalent*,
    merge those targets.

    In Blue-Fringe mode, any two transitions with the same label to
    different targets are non-deterministic and must be merged.
    In MINT mode, ``equivalentTransitions`` (Algorithm 3, lines
    26–33) uses classifiers to decide — transitions whose data
    produces different classifier predictions are already
    data-deterministic and left alone.
    """
    if _visited is None:
        _visited = set()
    if state in _visited:
        return efsm, vars_map
    _visited.add(state)

    groups = _group_by_activity(efsm, state)

    for activity, transitions in groups.items():
        # Collect distinct target states.
        target_map: dict[str, list[Transition]] = {}
        for t in transitions:
            target_map.setdefault(t.target_id, []).append(t)

        targets = list(target_map.keys())
        if len(targets) <= 1:
            continue

        if strategy is MergeStrategy.MINT and classifiers and variables:
            # Group targets into equivalence classes using classifiers.
            merge_groups = _find_equivalent_target_groups(
                target_map, activity, classifiers, variables,
            )
        else:
            # Blue-Fringe: all same-label targets must be merged.
            merge_groups = [targets]

        for group in merge_groups:
            if len(group) <= 1:
                continue
            keep = group[0]
            for other in group[1:]:
                if other not in efsm.states or other == keep:
                    continue
                efsm, vars_map = _merge_states(
                    efsm, keep, other, vars_map,
                    classifiers=classifiers,
                    variables=variables,
                    strategy=strategy,
                )

    return efsm, vars_map


def _find_equivalent_target_groups(
    target_map: dict[str, list[Transition]],
    activity: str,
    classifiers: Classifiers,
    variables: list[str],
) -> list[list[str]]:
    """Group target states whose transitions are classifier-equivalent.

    Two targets are equivalent iff for **every** pair of data values
    attached to their transitions, the classifier C_{activity}
    produces the **same** prediction (Algorithm 3, lines 26–33).

    Returns a list of groups (each group = list of target-state ids
    that should be merged together).
    """
    targets = list(target_map.keys())
    parent: dict[str, str] = {t: t for t in targets}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            ti = targets[i]
            tj = targets[j]
            ts_i = target_map[ti]
            ts_j = target_map[tj]
            if _transitions_classifier_equivalent(
                ts_i, ts_j, activity, classifiers, variables
            ):
                union(ti, tj)

    groups_dict: dict[str, list[str]] = {}
    for t in targets:
        root = find(t)
        groups_dict.setdefault(root, []).append(t)

    return list(groups_dict.values())


def _transitions_classifier_equivalent(
    ts1: list[Transition],
    ts2: list[Transition],
    activity: str,
    classifiers: Classifiers,
    variables: list[str],
) -> bool:
    """Check whether two groups of transitions (same label) are
    *classifier-equivalent*.

    Equivalent means: for **every** data vector attached to either
    group, the classifier predicts the **same** next event.
    """
    all_predictions: set[str] = set()

    for group in (ts1, ts2):
        for t in group:
            for data in t.data_samples:
                pred = predict_next_label(classifiers, activity, data, variables)
                if pred is not None:
                    all_predictions.add(pred)

    # If the classifier gives the same prediction for all data in
    # both groups, the transitions are equivalent (same logical
    # successor).  If predictions differ, the transitions are
    # distinguishable by data.
    return len(all_predictions) <= 1


# ═══════════════════════════════════════════════════════════════════════════
# Post-merge consistency check  (Algorithm 3, line 7)
# ═══════════════════════════════════════════════════════════════════════════

def _is_consistent(
    efsm: EFSM,
    classifiers: Classifiers,
    variables: list[str],
    vars_map: Vars,
) -> bool:
    """Validate merged EFSM against classifier predictions.

    For each transition *t* in the merged machine, feed its attached
    data to the corresponding classifier to predict the next event.
    Check that the target state of *t* actually has an outgoing
    transition with that predicted label.

    Returns ``True`` iff the model is consistent.
    """
    for t in efsm.transitions:
        target_activities = {
            out.activity for out in efsm.outgoing(t.target_id)
        }
        # If the target is an accepting state with no outgoing transitions,
        # there's nothing to check (end of trace).
        if not target_activities and t.target_id in efsm.accepting_states:
            continue

        for data in t.data_samples:
            predicted = predict_next_label(
                classifiers, t.activity, data, variables,
            )
            if predicted is None:
                # No classifier for this label (last-event-only label).
                continue
            if target_activities and predicted not in target_activities:
                logger.debug(
                    "Inconsistency: %s→%s [%s] data predicts '%s' "
                    "but target has %s",
                    t.source_id, t.target_id, t.activity,
                    predicted, target_activities,
                )
                return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Vars-map construction
# ═══════════════════════════════════════════════════════════════════════════

def _build_vars_map(efsm: EFSM) -> Vars:
    """Build the initial Vars mapping from transition-id to data dicts.

    Each transition's ``data_samples`` are copied into the map keyed
    by the Python ``id()`` of the Transition object.
    """
    return {id(t): list(t.data_samples) for t in efsm.transitions}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _canon(a: str, b: str) -> tuple[str, str]:
    """Canonical ordered pair (for the Failed set)."""
    return (min(a, b), max(a, b))


def _group_by_activity(efsm: EFSM, state_id: str) -> dict[str, list[Transition]]:
    """Group outgoing transitions of *state_id* by activity label."""
    groups: dict[str, list[Transition]] = {}
    for t in efsm.transitions:
        if t.source_id == state_id:
            groups.setdefault(t.activity, []).append(t)
    return groups


def _direct_successors(efsm: EFSM, state_id: str) -> set[str]:
    """Return states reachable in one step from *state_id*."""
    return {t.target_id for t in efsm.transitions if t.source_id == state_id}


def _all_successors_of_set(efsm: EFSM, states: set[str]) -> set[str]:
    """Return all states reachable in one step from any state in *states*."""
    result: set[str] = set()
    for s in states:
        result |= _direct_successors(efsm, s)
    return result
