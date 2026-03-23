"""
Step 3 — State Merging  (Walkinshaw et al., 2013).

Implements **two** merging strategies from the paper:

  • **Blue-Fringe** (Algorithms 1 & 2) — control-flow-only merging
    with Evidence-Driven State-Merging (EDSM) scoring.  Non-
    determinism is resolved **globally** (Algorithm 1, line 11:
    ``findNonDeterminism(S, T)``).
  • **MINT** (Algorithm 3) — Blue-Fringe scoring extended with
    trained data classifiers for transition equivalence and post-
    merge consistency validation.  Non-determinism is resolved
    **locally** at the merge target using
    ``equivalentTransitions`` with classifiers, cascading
    downward recursively.

When MINT is selected it runs directly on the PTA — it is a
**self-contained** merging algorithm that replaces Blue-Fringe,
not a post-processing step on top of it.

Both strategies maintain the red/blue colouring scheme.  At each
iteration, *all* (red, blue) candidate pairs are scored; the pair
with the **highest EDSM score** (≥ *k*) is selected.  If no pair
meets the threshold, the blue candidate is promoted to red.

Key design decisions aligned with the paper
--------------------------------------------
  • **No accept/non-accept incompatibility** — the paper
    explicitly considers only positive traces (§3.1).  Blocking
    merges between accepting and non-accepting states would
    prevent leaf-to-intermediate merges, leaving the model
    unnecessarily large.
  • EDSM scoring (``_compute_edsm_score``) is **classifier-
    aware** in MINT mode — a shared-label pair only contributes
    to the score if the transitions are classifier-equivalent
    (Algorithm 3, lines 26–33, **existential** check: at least
    one shared classifier prediction between the two groups).
  • Leaf-state bonus: two accepting states with no outgoing
    transitions score +1 so they are preferred merge partners
    (prevents proliferation of distinct final states).
  • Recursive non-determinism resolution inside merge
    (Algorithm 3, lines 19–24).
  • Post-merge ``_is_consistent`` validation (MINT only,
    Algorithm 3, line 7) — **binary**: any inconsistency
    rejects the merge.
  • ``Failed`` set to skip rejected merge pairs.

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

        # MINT (Algorithm 3): uses Blue-Fringe scoring but adds
        # data-aware merge validation (equivalentTransitions +
        # isConsistent).  This REPLACES Blue-Fringe rather than
        # running after it — Algorithm 3 is a self-contained
        # merging procedure.
        logger.info("  MINT merging (data-aware, Algorithm 3)")
        efsm = _run_merge_loop(
            pta.deep_copy(), MergeStrategy.MINT, classifiers, variables, k,
        )
    elif strategy is MergeStrategy.BLUE_FRINGE:
        logger.warning(
            "  Blue-Fringe merging is available but MINT is recommended. "
            "Blue-Fringe does not use classifiers and may under-merge."
        )
        efsm = _run_merge_loop(
            pta.deep_copy(), strategy, None, None, k,
        )
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")

    return efsm


# ═══════════════════════════════════════════════════════════════════════════
# Core merge loop  (Blue-Fringe / MINT)
# ═══════════════════════════════════════════════════════════════════════════

def _run_merge_loop(
    efsm: EFSM,
    strategy: MergeStrategy,
    classifiers: Classifiers | None,
    variables: list[str] | None,
    k: int,
) -> EFSM:
    """Execute a single Blue-Fringe / MINT merge pass.

    This is the inner loop of Algorithm 1 (Blue-Fringe) or
    Algorithm 3 (MINT) from Walkinshaw et al. (2013).

    Parameters
    ----------
    efsm : EFSM
        The current machine (modified in place and returned).
    strategy : MergeStrategy
        BLUE_FRINGE or MINT.
    classifiers / variables :
        Required only for MINT.
    k : int
        Minimum EDSM score for a merge.
    """
    vars_map = _build_vars_map(efsm)
    failed: set[tuple[str, str]] = set()

    red: set[str] = {efsm.initial_state}
    blue: deque[str] = deque(
        sorted(_direct_successors(efsm, efsm.initial_state) - red)
    )

    initial_states = len(efsm.states)
    merges_done = 0
    iterations = 0

    while blue:
        iterations += 1
        if iterations % 50 == 0:
            logger.info(
                "    merge loop: iter=%d  |red|=%d  |blue|=%d  "
                "states=%d  merges=%d  failed=%d",
                iterations, len(red), len(blue),
                len(efsm.states), merges_done, len(failed),
            )
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
            # Per the paper, consistent(S,T) is a **binary** predicate:
            # the merged model must have zero classifier mismatches.
            if strategy is MergeStrategy.MINT:
                if not _is_consistent(
                    efsm, classifiers, variables, vars_map,  # type: ignore[arg-type]
                ):
                    logger.debug(
                        "Merge %s \u2190 %s (score=%d) REJECTED "
                        "(post-merge model inconsistent)",
                        best_target, candidate, best_score,
                    )
                    failed.add(_canon(best_target, candidate))
                    efsm = backup
                    vars_map = backup_vars
                else:
                    merged = True
            else:
                merged = True

            if merged:
                merges_done += 1
                logger.debug(
                    "Merged %s \u2190 %s  (score=%d)",
                    best_target, candidate, best_score,
                )
                # Purge stale red entries (some red states may have
                # been removed by cascading non-determinism merges)
                # and recalculate the blue frontier.
                red = {s for s in red if s in efsm.states}
                blue = deque(sorted(
                    s for s in _all_successors_of_set(efsm, red) - red
                    if s in efsm.states
                ))

        if not merged:
            red.add(candidate)
            new_blue = _direct_successors(efsm, candidate) - red - set(blue)
            blue.extend(sorted(new_blue))

    logger.info(
        "  Merge loop done: %d iterations, %d merges, "
        "%d → %d states",
        iterations, merges_done, initial_states, len(efsm.states),
    )
    return efsm


# ═══════════════════════════════════════════════════════════════════════════
# EDSM Scoring  (Algorithm 2 — calculateScore)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_edsm_score(
    efsm: EFSM,
    s1: str,
    s2: str,
    *,
    classifiers: Classifiers | None = None,
    variables: list[str] | None = None,
    _visited: set[tuple[str, str]] | None = None,
) -> int:
    """Recursively compute the EDSM compatibility score for merging
    *s1* and *s2*.

    The score counts the number of outgoing-transition pairs that
    share the same activity label and recurses into their
    successor states.

    When *classifiers* and *variables* are provided (MINT mode,
    Algorithm 3), a shared-label pair only contributes to the
    score if the two transition groups are **classifier-equivalent**
    (Algorithm 3, lines 26–33).  This avoids inflating the score
    for pairs that would be rejected during the merge anyway.

    A return value of ``-1`` signals incompatibility (merging
    would produce a contradiction).

    Corresponds to ``calculateScore`` in Algorithm 2 of
    Walkinshaw et al. (2013), extended by Algorithm 3 for MINT.
    """
    if _visited is None:
        _visited = set()

    pair = _canon(s1, s2)
    if pair in _visited:
        return 0
    _visited.add(pair)

    # NOTE: No accept / non-accept incompatibility check.
    # Walkinshaw et al. (2013) explicitly state that only positive
    # traces are considered (§3.1).  Blocking merges between
    # accepting and non-accepting states would prevent the merging
    # of leaf nodes (end of trace) with intermediate nodes, leaving
    # the model unnecessarily large.

    out1 = _group_by_activity(efsm, s1)
    out2 = _group_by_activity(efsm, s2)
    shared = set(out1) & set(out2)

    score = 0

    # Leaf-state bonus: if both states are accepting leaves with
    # no outgoing transitions, they are structurally equivalent
    # end-of-trace states.  Give them a score of +1 so they are
    # preferred merge partners (avoids proliferation of distinct
    # final states in process mining settings).
    if not out1 and not out2:
        if s1 in efsm.accepting_states and s2 in efsm.accepting_states:
            return 1
        return 0

    for activity in shared:
        ts1 = out1[activity]
        ts2 = out2[activity]

        # MINT mode (Algorithm 3, lines 26–33): only count this
        # shared label if the transitions are classifier-equivalent.
        if classifiers is not None and variables is not None:
            if not _transitions_classifier_equivalent(
                ts1, ts2, activity, classifiers, variables,
            ):
                continue

        # Algorithm 2: classifier-compatible shared label contributes +1.
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
                    _visited=_visited,
                )
                if sub < 0:
                    return -1
                score += sub

    return score


# ═══════════════════════════════════════════════════════════════════════════
# Merge + non-determinism resolution  (Algorithms 1 & 3)
# ═══════════════════════════════════════════════════════════════════════════

def _redirect_edges(
    efsm: EFSM,
    keep: str,
    remove: str,
    vars_map: Vars,
) -> tuple[EFSM, Vars]:
    """Redirect all edges from/to *remove* to *keep* and de-duplicate.

    This is the pure structural merge — it does NOT resolve any
    non-determinism.  The caller is responsible for calling the
    appropriate resolution strategy afterwards.
    """
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
    return efsm, vars_map


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
    """Merge *remove* into *keep*, then resolve non-determinism.

    In **Blue-Fringe** mode (Algorithm 1), non-determinism is
    resolved **globally** via ``findNonDeterminism(S, T)`` — we
    scan ALL states for same-label transitions to different
    targets and merge those targets, looping to a fixpoint.

    In **MINT** mode (Algorithm 3, lines 19–24),
    ``equivalentTransitions`` only checks outgoing transitions
    of the keep-state, cascading downward recursively.

    Returns the updated ``(efsm, vars_map)``.
    """
    efsm, vars_map = _redirect_edges(efsm, keep, remove, vars_map)

    if strategy is MergeStrategy.MINT and classifiers and variables:
        # Algorithm 3: local resolution at keep, cascading downward.
        efsm, vars_map = _resolve_non_determinism_local(
            efsm, keep, vars_map,
            classifiers=classifiers,
            variables=variables,
            strategy=strategy,
        )
    else:
        # Algorithm 1: global fixpoint resolution.
        efsm, vars_map = _resolve_non_determinism_global(
            efsm, vars_map,
        )

    return efsm, vars_map


# -----------------------------------------------------------------------
# Algorithm 1 — global non-determinism resolution (findNonDeterminism)
# -----------------------------------------------------------------------

def _find_any_non_determinism(efsm: EFSM) -> tuple[str, str] | None:
    """Scan the entire EFSM for a non-deterministic transition pair.

    Returns ``(keep_target, remove_target)`` — the two distinct
    targets of same-label transitions from the same source state.
    Returns ``None`` when the EFSM is deterministic.
    """
    for state_id in sorted(efsm.states):
        groups = _group_by_activity(efsm, state_id)
        for activity in sorted(groups):
            targets = sorted({t.target_id for t in groups[activity]})
            if len(targets) >= 2:
                return (targets[0], targets[1])
    return None


def _resolve_non_determinism_global(
    efsm: EFSM,
    vars_map: Vars,
) -> tuple[EFSM, Vars]:
    """Algorithm 1 fixpoint: find and merge ALL non-deterministic
    transition targets until the machine is deterministic.

    Each iteration finds one pair of targets that share the same
    source state and label, redirects edges to merge them, and
    repeats.  Because every merge reduces the state count by one,
    the loop always terminates.
    """
    while True:
        pair = _find_any_non_determinism(efsm)
        if pair is None:
            break
        keep_t, remove_t = pair
        if remove_t not in efsm.states:
            continue
        efsm, vars_map = _redirect_edges(efsm, keep_t, remove_t, vars_map)
    return efsm, vars_map


# -----------------------------------------------------------------------
# Algorithm 3 — local non-determinism resolution (equivalentTransitions)
# -----------------------------------------------------------------------

def _resolve_non_determinism_local(
    efsm: EFSM,
    state: str,
    vars_map: Vars,
    *,
    classifiers: Classifiers,
    variables: list[str],
    strategy: MergeStrategy,
    _visited: set[str] | None = None,
) -> tuple[EFSM, Vars]:
    """Resolve non-determinism at *state* only, cascading downward.

    MINT mode (Algorithm 3, lines 19–24): for each activity leaving
    *state*, group transitions by classifier equivalence and merge
    the targets within each equivalence class.  Merging is
    recursive so non-determinism at the merged targets is also
    resolved (cascading downward through the graph).
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

        # Group targets into equivalence classes using classifiers.
        merge_groups = _find_equivalent_target_groups(
            target_map, activity, classifiers, variables,
        )

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

    Two targets are equivalent iff there **exists** at least one pair
    of data values (one from each group) for which the classifier
    C_{activity} produces the **same** prediction (Algorithm 3,
    lines 26–33, existential check).

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

    Per Walkinshaw et al. (2013), Algorithm 3, lines 26–33:
    equivalentTransitions uses an **existential** check — return
    True if there exists ANY pair of data vectors (one from each
    group) for which the classifier predicts the same next event.
    """
    preds_1: set[str] = set()
    preds_2: set[str] = set()

    for t in ts1:
        for data in t.data_samples:
            pred = predict_next_label(classifiers, activity, data, variables)
            if pred is not None:
                preds_1.add(pred)

    for t in ts2:
        for data in t.data_samples:
            pred = predict_next_label(classifiers, activity, data, variables)
            if pred is not None:
                preds_2.add(pred)

    # Existential: equivalent iff the two prediction sets share
    # at least one common prediction (Algorithm 3, lines 26–33).
    # If either group has no predictions, fall back to True
    # (no evidence to distinguish them).
    if not preds_1 or not preds_2:
        return True
    return bool(preds_1 & preds_2)


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


def _count_inconsistencies(
    efsm: EFSM,
    classifiers: Classifiers,
    variables: list[str],
    vars_map: Vars,
) -> int:
    """Count the number of data-sample/classifier mismatches.

    Used to compare consistency before and after a merge: a merge
    is rejected only if it **increases** the mismatch count (i.e.
    makes the model worse).  This is necessary because classifiers
    are imperfect approximations and the PTA itself may already
    contain mismatches.
    """
    count = 0
    for t in efsm.transitions:
        target_activities = {
            out.activity for out in efsm.outgoing(t.target_id)
        }
        if not target_activities and t.target_id in efsm.accepting_states:
            continue
        for data in t.data_samples:
            predicted = predict_next_label(
                classifiers, t.activity, data, variables,
            )
            if predicted is None:
                continue
            if target_activities and predicted not in target_activities:
                count += 1
    return count


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


# ═══════════════════════════════════════════════════════════════════════════
# Post-merge activity-based state deduplication
# ═══════════════════════════════════════════════════════════════════════════

def _deduplicate_states(efsm: EFSM) -> EFSM:
    """Merge states with identical outgoing transition signatures.

    Two states are considered duplicates if they have exactly the
    same set of ``(activity, target_state)`` pairs on their outgoing
    transitions.  Merging such states ensures that the resulting DPN
    has no redundant places for the same activity pattern.

    The pass iterates to a fixpoint because merging two states may
    make other states equivalent.
    """
    changed = True
    while changed:
        changed = False
        sig_map: dict[frozenset[tuple[str, str]], list[str]] = {}
        for state in sorted(efsm.states):
            if state == efsm.initial_state:
                continue
            sig = frozenset(
                (t.activity, t.target_id)
                for t in efsm.transitions
                if t.source_id == state
            )
            sig_map.setdefault(sig, []).append(state)

        for group in sig_map.values():
            if len(group) <= 1:
                continue
            keep = group[0]
            for remove in group[1:]:
                if remove not in efsm.states:
                    continue
                efsm = _simple_merge(efsm, keep, remove)
                changed = True

    return efsm


def bisimulation_reduction(efsm: EFSM) -> EFSM:
    """Compute the coarsest bisimulation quotient of *efsm*.

    Two states are *bisimilar* iff they have the same set of
    outgoing activity labels AND each label leads to bisimilar
    successors.  This is strictly stronger than the existing
    ``_deduplicate_states`` which only checks one level of
    outgoing ``(activity, target_id)`` pairs.

    Algorithm — partition-refinement (Kanellakis–Smolka):
      1. Start with a single partition block containing all states.
      2. Refine: split each block so that states in the same sub-
         block have identical ``{activity → block-of-target}``
         signatures.
      3. Iterate to fixpoint (no block changes between iterations).
      4. For each block with >1 state, merge all states into a
         single representative via ``_simple_merge``.

    This is purely language-preserving — it never introduces
    assumptions beyond what the EFSM already encodes.

    Parameters
    ----------
    efsm : EFSM
        The EFSM to reduce (not modified; returns a deep copy).

    Returns
    -------
    EFSM
        The reduced EFSM.  Structurally smaller or equal.
    """
    efsm = efsm.deep_copy()
    states = sorted(efsm.states)

    if len(states) <= 1:
        return efsm

    # ── 1. Initial partition: all states in one block ────────────────────
    # We keep the initial state distinguishable from the start so it
    # never gets merged away.
    block_of: dict[str, int] = {}
    block_id = 0
    # Separate initial state into its own block.
    block_of[efsm.initial_state] = block_id
    block_id += 1
    for s in states:
        if s != efsm.initial_state:
            block_of[s] = block_id
    block_id += 1

    # ── 2. Partition-refinement loop ─────────────────────────────────────
    # Pre-compute outgoing transitions grouped by source.
    outgoing_map: dict[str, list[Transition]] = {s: [] for s in states}
    for t in efsm.transitions:
        if t.source_id in outgoing_map:
            outgoing_map[t.source_id].append(t)

    changed = True
    while changed:
        changed = False
        new_block_of: dict[str, int] = {}
        next_id = 0
        # Group states by (current_block, signature) where signature
        # encodes outgoing activities and the block of each target.
        sig_to_block: dict[tuple[int, frozenset[tuple[str, int]]], int] = {}

        for s in states:
            sig = frozenset(
                (t.activity, block_of[t.target_id])
                for t in outgoing_map.get(s, [])
                if t.target_id in block_of
            )
            key = (block_of[s], sig)
            if key not in sig_to_block:
                sig_to_block[key] = next_id
                next_id += 1
            new_block_of[s] = sig_to_block[key]

        if new_block_of != block_of:
            changed = True
            block_of = new_block_of

    # ── 3. Merge states within each partition block ──────────────────────
    blocks: dict[int, list[str]] = {}
    for s, bid in block_of.items():
        blocks.setdefault(bid, []).append(s)

    merges = 0
    for group in blocks.values():
        if len(group) <= 1:
            continue
        # Choose representative: prefer the initial state, else first
        # alphabetically so results are deterministic.
        if efsm.initial_state in group:
            keep = efsm.initial_state
        else:
            keep = sorted(group)[0]
        for remove in sorted(group):
            if remove == keep or remove not in efsm.states:
                continue
            efsm = _simple_merge(efsm, keep, remove)
            merges += 1

    if merges:
        logger.info(
            "  Bisimulation reduction: %d merges → %d states",
            merges, len(efsm.states),
        )

    return efsm


def _simple_merge(efsm: EFSM, keep: str, remove: str) -> EFSM:
    """Merge *remove* into *keep* without non-determinism resolution.

    Used by ``_deduplicate_states`` where the two states are already
    known to be structurally equivalent.
    """
    new_transitions: list[Transition] = []

    for t in efsm.transitions:
        src = keep if t.source_id == remove else t.source_id
        tgt = keep if t.target_id == remove else t.target_id

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
