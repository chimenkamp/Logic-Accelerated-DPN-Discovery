"""
Region-Based Petri Net Synthesis from Transition Systems.

Implements the theory of regions from:

    Cortadella, Kishinevsky, Lavagno, Yakovlev.
    "Deriving Petri Nets from Finite Transition Systems."
    IEEE Transactions on Computers, 1998.

The EFSM's states become TS states, activity labels become TS events,
and EFSM edges become TS transitions.  Regions (subsets of states with
uniform event crossing behaviour, Def 2.2) become DPN places.  After
synthesis, DPN transitions are annotated by matching their activity
label back to the original EFSM edges to recover guards and updates.

Top-level entry point: ``region_based_efsm_to_dpn(efsm)``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional

from dpn_discovery.models import (
    EFSM,
    DPNTransition,
    DataPetriNet,
    Place,
    Transition,
)
from dpn_discovery.smt import get_solver

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Transition System extraction from an EFSM
# ===========================================================================
# The EFSM is treated as a labeled transition system TS = (S, E, T, s_in)
# where S = efsm.states, E = set of activity labels, T = set of
# (source, activity, target) triples, s_in = efsm.initial_state.
# We keep the data close to the EFSM and do not introduce new types.


def _extract_ts(efsm: EFSM) -> tuple[
    set[str],                             # S  -- states
    set[str],                             # E  -- events (activity labels)
    set[tuple[str, str, str]],            # T  -- transitions (src, evt, tgt)
    str,                                  # s_in
]:
    """Extract a plain transition system from the EFSM.

    Returns (S, E, T, s_in) directly from the EFSM fields.
    """
    states = set(efsm.states)
    events: set[str] = set()
    ts_transitions: set[tuple[str, str, str]] = set()

    for t in efsm.transitions:
        events.add(t.activity)
        ts_transitions.add((t.source_id, t.activity, t.target_id))

    return states, events, ts_transitions, efsm.initial_state


# ===========================================================================
# 2. Region predicates  (Def 2.1, Cortadella et al.)
# ===========================================================================


def _enter(
    event: str,
    region: frozenset[str],
    ts_transitions: set[tuple[str, str, str]],
) -> bool:
    """Def 2.1 -- enter(e, S'):  exists(s,e,s')  in  T : s  not in  S'  and  s'  in  S'."""
    for src, evt, tgt in ts_transitions:
        if evt == event and src not in region and tgt in region:
            return True
    return False


def _exit(
    event: str,
    region: frozenset[str],
    ts_transitions: set[tuple[str, str, str]],
) -> bool:
    """Def 2.1 -- exit(e, S'):  exists(s,e,s')  in  T : s  in  S'  and  s'  not in  S'."""
    for src, evt, tgt in ts_transitions:
        if evt == event and src in region and tgt not in region:
            return True
    return False


def _in(
    event: str,
    region: frozenset[str],
    ts_transitions: set[tuple[str, str, str]],
) -> bool:
    """Def 2.1 -- in(e, S'):  exists(s,e,s')  in  T : s  in  S'  and  s'  in  S'."""
    for src, evt, tgt in ts_transitions:
        if evt == event and src in region and tgt in region:
            return True
    return False


def _out(
    event: str,
    region: frozenset[str],
    ts_transitions: set[tuple[str, str, str]],
) -> bool:
    """Def 2.1 -- out(e, S'):  exists(s,e,s')  in  T : s  not in  S'  and  s'  not in  S'."""
    for src, evt, tgt in ts_transitions:
        if evt == event and src not in region and tgt not in region:
            return True
    return False


def _is_region(
    candidate: frozenset[str],
    events: set[str],
    ts_transitions: set[tuple[str, str, str]],
) -> bool:
    """Def 2.2 -- Check whether *candidate* is a region of the TS.

    A set of states r <= S is a region iff for each event e  in  E,
    exactly one crossing type holds.  Concretely:
      1) enter(e,r) => not in(e,r)  and  not out(e,r)  and  not exit(e,r)
      2) exit(e,r)  => not in(e,r)  and  not out(e,r)  and  not enter(e,r)
      3) in(e,r) and out(e,r) cannot both hold (non-uniform internal/external)
    """
    for e in events:
        en = _enter(e, candidate, ts_transitions)
        ex = _exit(e, candidate, ts_transitions)
        i = _in(e, candidate, ts_transitions)
        o = _out(e, candidate, ts_transitions)

        # Condition 1: if enter then no in, no out, no exit
        if en and (i or o or ex):
            return False
        # Condition 2: if exit then no in, no out, no enter
        if ex and (i or o or en):
            return False
        # Condition 3 (Def 2.2): in and out are mutually exclusive.
        # If some e-transitions stay inside r AND some stay outside r,
        # the event has non-uniform crossing behaviour.
        if i and o:
            return False
    return True


# ===========================================================================
# 3. Excitation & Switching Regions  (Def 3.1, Cortadella et al.)
# ===========================================================================


def _excitation_region(
    event: str,
    ts_transitions: set[tuple[str, str, str]],
) -> frozenset[str]:
    """Def 3.1 -- ER(a): maximal set of states where event a is enabled.

    ER(a) = {s | existss' : (s, a, s')  in  T}.
    """
    return frozenset(src for src, evt, _ in ts_transitions if evt == event)


def _switching_region(
    event: str,
    ts_transitions: set[tuple[str, str, str]],
) -> frozenset[str]:
    """Def 3.1 -- SR(a): maximal set of states reached after event a.

    SR(a) = {s' | existss : (s, a, s')  in  T}.
    """
    return frozenset(tgt for _, evt, tgt in ts_transitions if evt == event)


# ===========================================================================
# 4. Minimal Pre-Region Generation  (S4.1, Lemma 4.1 / 4.2, Fig. 10)
# ===========================================================================


def _find_violating_event(
    candidate: frozenset[str],
    events: set[str],
    ts_transitions: set[tuple[str, str, str]],
) -> Optional[tuple[str, int]]:
    """Lemma 4.1 -- Find an event that violates the region conditions.

    Returns (event, violation_type) or None if *candidate* is a region.

    Violation types (Lemma 4.1 + Def 2.2):
      1) in(e,r)  and  [enter(e,r)  or  exit(e,r)]
      2) enter(e,r)  and  exit(e,r)
      3) out(e,r)  and  enter(e,r)
      4) out(e,r)  and  exit(e,r)
      5) in(e,r)  and  out(e,r)  (non-uniform, Def 2.2)
    """
    for e in events:
        en = _enter(e, candidate, ts_transitions)
        ex = _exit(e, candidate, ts_transitions)
        i = _in(e, candidate, ts_transitions)
        o = _out(e, candidate, ts_transitions)

        # Type 1: internal + crossing
        if i and (en or ex):
            return (e, 1)
        # Type 2: both enter and exit
        if en and ex:
            return (e, 2)
        # Type 3: external + enter
        if o and en:
            return (e, 3)
        # Type 4: external + exit
        if o and ex:
            return (e, 4)
        # Type 5: in + out (non-uniform, Def 2.2)
        if i and o:
            return (e, 5)
    return None


def _states_to_add(
    candidate: frozenset[str],
    event: str,
    violation_type: int,
    ts_transitions: set[tuple[str, str, str]],
) -> list[frozenset[str]]:
    """Lemma 4.2 -- Compute the set(s) of states that must be added.

    For violation types 1 and 2 there is a single deterministic
    expansion.  For types 3 and 4 there are two branching options
    (binary search tree, Fig. 12).

    Returns a list of expansion sets (1 element for types 1/2,
    2 elements for types 3/4 -- each branch of the search tree).
    """
    if violation_type == 1:
        # Lemma 4.2 case 1: in(e,r)  and  [enter(e,r)  or  exit(e,r)]
        # Add all s outside r that are connected to some s' in r via e.
        add = set()
        for src, evt, tgt in ts_transitions:
            if evt == event:
                if src in candidate or tgt in candidate:
                    if src not in candidate:
                        add.add(src)
                    if tgt not in candidate:
                        add.add(tgt)
        return [frozenset(add)]

    if violation_type == 2:
        # Lemma 4.2 case 2: enter(e,r)  and  exit(e,r)
        # Add sources of entering transitions AND targets of exiting
        # transitions that are outside r.
        add = set()
        for src, evt, tgt in ts_transitions:
            if evt == event:
                # entering: src  not in  r, tgt  in  r  ->  add src
                if src not in candidate and tgt in candidate:
                    add.add(src)
                # exiting:  src  in  r, tgt  not in  r  ->  add tgt
                if src in candidate and tgt not in candidate:
                    add.add(tgt)
        return [frozenset(add)]

    if violation_type == 3:
        # Lemma 4.2 case 3: out(e,r)  and  enter(e,r) -- two branches:
        #   Branch A: add sources of entering e-transitions
        #   Branch B: add targets of exiting e-transitions
        # (Note: we know exit doesn't hold when we reach type 3 check,
        #  but we still need to gather all sources of entering arcs
        #  and potentially some targets to fully resolve.)
        branch_a = set()  # sources of entering transitions
        branch_b = set()  # targets of entering transitions (i.e., make them all external)
        for src, evt, tgt in ts_transitions:
            if evt == event:
                if src not in candidate and tgt in candidate:
                    branch_a.add(src)
                # For branch B: add targets of exits -- but if no exit,
                # we add the sources of the "out" transitions to absorb them
        # Actually, per the paper: branch A adds {s | existss' in r : (s,e,s') in T} <= r_bar
        # branch B adds {s | existss' not in r : (s',e,s) in T} ... but more precisely:
        #   branch A: all sources of transitions entering r  -> add them to r
        #   branch B: all states in r that have an entering arc -> remove? no, we only expand
        # The paper says:
        #   3) out(e,r)  and  enter(e,r) =>
        #      [{s|existss' in r: (s,e,s') in T} <= r]  or  [{s|existss' not in r: (s',e,s) in T} <= r]
        # Since we only expand, branch A = add all sources of entering transitions,
        # branch B = add all targets of external transitions (that are outside r).
        for src, evt, tgt in ts_transitions:
            if evt == event and src not in candidate and tgt not in candidate:
                branch_b.add(src)
                branch_b.add(tgt)
        # Correction: re-reading Lemma 4.2 more carefully:
        # For case 3: out(e,r)  and  enter(e,r):
        #   Branch A: add {s | existss' in r : (s,e,s')  in  T, s not in r}  (sources of entering)
        #   Branch B: add {s' | existss not in r : (s,e,s')  in  T, s' not in r} (targets of external,
        #             making them internal or entering -- actually we need to also
        #             capture the enter sources)
        # The simplest correct reading: for each external transition (s,e,s'),
        # to legalize we must either make it enter (add s' -> branch A already covers enter sources)
        # or make it internal by adding both endpoints.
        # Per paper: the two options for legalization are:
        #   A: {s | existss' in r : (s,e,s') in T  and  s not in r} must be <= r (i.e., add them)
        #   B: {s' | existss in r : (s,e,s') in T  and  s' not in r} -- wait, there's no exit here.
        # Let me just implement both branches correctly:
        branch_a_set = set()
        branch_b_set = set()
        for src, evt, tgt in ts_transitions:
            if evt == event:
                # Entering transitions: src not in r, tgt in r -> to "un-enter", add src
                if src not in candidate and tgt in candidate:
                    branch_a_set.add(src)
                # External transitions: src not in r, tgt not in r -> to eliminate out:
                # make them enter (add tgt to r) or internal (add both)
                if src not in candidate and tgt not in candidate:
                    branch_b_set.add(tgt)

        branches = []
        if branch_a_set:
            branches.append(frozenset(branch_a_set))
        if branch_b_set:
            branches.append(frozenset(branch_b_set))
        if not branches:
            # Fallback: add all external endpoints
            fallback = set()
            for src, evt, tgt in ts_transitions:
                if evt == event:
                    if src not in candidate:
                        fallback.add(src)
                    if tgt not in candidate:
                        fallback.add(tgt)
            branches.append(frozenset(fallback))
        return branches

    if violation_type == 4:
        # Lemma 4.2 case 4: out(e,r)  and  exit(e,r) -- two branches,
        # symmetric to case 3.
        branch_a_set = set()  # targets of exiting transitions
        branch_b_set = set()  # sources of external transitions
        for src, evt, tgt in ts_transitions:
            if evt == event:
                if src in candidate and tgt not in candidate:
                    branch_a_set.add(tgt)
                if src not in candidate and tgt not in candidate:
                    branch_b_set.add(src)
        branches = []
        if branch_a_set:
            branches.append(frozenset(branch_a_set))
        if branch_b_set:
            branches.append(frozenset(branch_b_set))
        if not branches:
            fallback = set()
            for src, evt, tgt in ts_transitions:
                if evt == event:
                    if src not in candidate:
                        fallback.add(src)
                    if tgt not in candidate:
                        fallback.add(tgt)
            branches.append(frozenset(fallback))
        return branches

    if violation_type == 5:
        # Def 2.2 violation: in(e,r) and out(e,r).
        # Some e-transitions are internal (both endpoints in r) while
        # others are external (both endpoints outside r).  To legalise,
        # add the external endpoints into r (making them internal).
        add = set()
        for src, evt, tgt in ts_transitions:
            if evt == event:
                if src not in candidate and tgt not in candidate:
                    add.add(src)
                    add.add(tgt)
        return [frozenset(add)]

    return []


def _generate_min_preregions(
    event: str,
    states: set[str],
    events: set[str],
    ts_transitions: set[tuple[str, str, str]],
) -> set[frozenset[str]]:
    """S4.1 / Fig. 10 -- Generate all minimal pre-regions of *event*.

    Starts from ER(event) and iteratively expands by legalising
    violating events (Lemma 4.1/4.2).  Uses a branch-and-bound
    search tree (Fig. 12) with pruning:
      - Duplicate elimination
      - Region bounding (no minimal region below another region)

    Proposition 4.1 guarantees this finds all minimal pre-regions
    that are predecessors of some TS event.
    """
    er = _excitation_region(event, ts_transitions)
    if not er:
        return set()

    all_states = frozenset(states)
    found_regions: set[frozenset[str]] = set()
    visited: set[frozenset[str]] = set()

    # BFS/DFS work-list: each item is a candidate set of states.
    worklist: list[frozenset[str]] = [er]

    while worklist:
        candidate = worklist.pop()

        # Duplicate elimination.
        if candidate in visited:
            continue
        visited.add(candidate)

        # If the candidate covers all states, it's the trivial region -- skip.
        if candidate == all_states:
            continue

        # Check if already a region.
        if _is_region(candidate, events, ts_transitions):
            # Region bounding: only keep if no existing found region is
            # a proper subset (i.e., this one is not minimal).
            is_minimal = True
            to_remove = set()
            for existing in found_regions:
                if existing < candidate:
                    # A smaller region already found -- this one isn't minimal.
                    is_minimal = False
                    break
                if candidate < existing:
                    # This new region is smaller -- remove the old one.
                    to_remove.add(existing)
            if is_minimal:
                found_regions -= to_remove
                found_regions.add(candidate)
            continue

        # Find a violating event and expand (Lemma 4.1 / 4.2).
        violation = _find_violating_event(candidate, events, ts_transitions)
        if violation is None:
            # Shouldn't happen if _is_region returned False, but be safe.
            continue

        viol_event, viol_type = violation
        branches = _states_to_add(candidate, viol_event, viol_type, ts_transitions)

        for branch in branches:
            if not branch:
                continue
            expanded = candidate | branch
            if expanded not in visited:
                # Prune: don't expand beyond a known region.
                pruned = False
                for existing in found_regions:
                    if existing <= expanded:
                        pruned = True
                        break
                if not pruned:
                    worklist.append(expanded)

    return found_regions


# ===========================================================================
# 5. Excitation Closure Check  (Def 3.2, Cortadella et al.)
# ===========================================================================


def _preregions_of(
    event: str,
    regions: set[frozenset[str]],
    ts_transitions: set[tuple[str, str, str]],
) -> set[frozenset[str]]:
    """Return the set of pre-regions of *event* ('e).

    A region r is a pre-region of e iff some e-transition exits r
    (Def following Def 2.2).
    """
    return {r for r in regions if _exit(event, r, ts_transitions)}


def _postregions_of(
    event: str,
    regions: set[frozenset[str]],
    ts_transitions: set[tuple[str, str, str]],
) -> set[frozenset[str]]:
    """Return the set of post-regions of *event* (e').

    A region r is a post-region of e iff some e-transition enters r.
    Prop 2.2: r is a region iff S-r is a region, so post-regions
    are complements of pre-regions of e, but we compute directly.
    """
    return {r for r in regions if _enter(event, r, ts_transitions)}


def _check_excitation_closure(
    events: set[str],
    regions: set[frozenset[str]],
    ts_transitions: set[tuple[str, str, str]],
) -> dict[str, bool]:
    """Def 3.2 -- Check the excitation closure condition for each event.

    A.4') For each event a:   & {r  in  'a} r = ER(a)
    A.5') For each event a:  'a != {}

    Returns a dict mapping event -> True if excitation-closed for that event.
    """
    result: dict[str, bool] = {}
    for e in events:
        er = _excitation_region(e, ts_transitions)
        pre = _preregions_of(e, regions, ts_transitions)
        if not pre:
            # A.5' violated: no pre-region exists.
            result[e] = False
            continue
        intersection = frozenset.intersection(*pre) if pre else frozenset()
        result[e] = (intersection == er)
    return result


# ===========================================================================
# 6. Label Splitting  (S4.3, Cortadella et al.)
# ===========================================================================


def _split_labels(
    states: set[str],
    events: set[str],
    ts_transitions: set[tuple[str, str, str]],
    s_in: str,
    all_regions: set[frozenset[str]],
) -> tuple[
    set[str],                          # new events
    set[tuple[str, str, str]],         # new transitions
    dict[str, str],                    # split_label -> original_label
]:
    """S4.3 -- Split event labels to achieve excitation closure.

    When  & {r  in  'a} r != ER(a), split the offending events into
    sub-events so the TS becomes excitation-closed.  The strategy:

      1. For each non-excitation-closed event a, find the
         intersection of its pre-regions (call it I_a).
      2. Partition the a-transitions so that within each partition
         the source states form a subset that can become a region.
         The simplest correct partition: group a-transitions by
         whether their source is in ER(a) - I_a or in I_a - ER(a).
      3. Assign fresh labels a_1, a_2, ... to each partition group.

    Returns the updated (events, transitions, label_map).
    The label_map maps every split label back to its original activity.
    """
    closure = _check_excitation_closure(events, all_regions, ts_transitions)
    label_map: dict[str, str] = {e: e for e in events}

    new_events = set(events)
    new_transitions = set(ts_transitions)

    split_counter = 0

    for event in sorted(events):
        if closure.get(event, True):
            continue

        er = _excitation_region(event, ts_transitions)
        pre = _preregions_of(event, all_regions, ts_transitions)

        if not pre:
            # A.5' violated: no pre-region exists for this event.
            # Per Cortadella et al. §4.3, we must split the event
            # so that each sub-event has a smaller excitation region
            # that can yield non-trivial pre-regions.
            # Strategy: partition transitions by target state.
            event_transitions = [
                (s, e, t) for s, e, t in new_transitions if e == event
            ]
            target_groups: dict[str, list[tuple[str, str, str]]] = {}
            for s, e, t in event_transitions:
                target_groups.setdefault(t, []).append((s, e, t))

            if len(target_groups) <= 1:
                # All transitions go to the same target — try splitting
                # by source state instead.
                source_groups: dict[str, list[tuple[str, str, str]]] = {}
                for s, e, t in event_transitions:
                    source_groups.setdefault(s, []).append((s, e, t))
                if len(source_groups) <= 1:
                    continue  # truly unsplittable
                target_groups = source_groups

            # Remove old transitions for this event.
            for tr in event_transitions:
                new_transitions.discard(tr)

            first_group = True
            for _key, group in sorted(target_groups.items()):
                if first_group:
                    # Keep original label for the first group.
                    for s, _, t in group:
                        new_transitions.add((s, event, t))
                    first_group = False
                else:
                    split_counter += 1
                    new_label = f"{event}__split_{split_counter}"
                    new_events.add(new_label)
                    label_map[new_label] = label_map.get(event, event)
                    for s, _, t in group:
                        new_transitions.add((s, new_label, t))

            logger.info(
                "  Label split (A.5'): %s -> %d sub-events",
                event, len(target_groups),
            )
            continue

        intersection = frozenset.intersection(*pre)

        # States in the intersection but NOT in ER -- these are the
        # problematic states where event is not actually enabled.
        extra_states = intersection - er

        if not extra_states:
            continue

        # Partition transitions of this event based on whether source
        # is in ER(event) only vs in the "extra" states.
        # Per S4.3: we group transitions to minimise region violations.
        event_transitions = [
            (s, e, t) for s, e, t in new_transitions if e == event
        ]

        # Group by source state membership in ER.
        group_in_er = [(s, e, t) for s, e, t in event_transitions if s in er]
        group_extra = [(s, e, t) for s, e, t in event_transitions if s not in er]

        if not group_extra:
            continue

        # Remove old transitions.
        for tr in event_transitions:
            new_transitions.discard(tr)

        # Keep original label for the main ER group.
        for s, _, t in group_in_er:
            new_transitions.add((s, event, t))

        # Split the extra group into a new label.
        split_counter += 1
        new_label = f"{event}__split_{split_counter}"
        new_events.add(new_label)
        label_map[new_label] = event

        for s, _, t in group_extra:
            new_transitions.add((s, new_label, t))

        logger.info(
            "  Label split: %s -> %s (%d transitions moved)",
            event, new_label, len(group_extra),
        )

    return new_events, new_transitions, label_map


# ===========================================================================
# 7. Irredundant Region Cover  (S4.2, Cortadella et al.)
# ===========================================================================


def _find_irredundant_cover(
    events: set[str],
    all_preregions: set[frozenset[str]],
    ts_transitions: set[tuple[str, str, str]],
) -> set[frozenset[str]]:
    """S4.2 -- Find an irredundant set of regions (Thm 3.5).

    Step 1: Identify essential regions -- regions uniquely needed for
            some excitation-closure constraint.
    Step 2: Formulate a set-cover problem over non-essential regions.
    Step 3: Solve with cost |-p| + |p-| + 1 (arc+place minimisation).

    Theorem 3.5: If I is an irredundant set of regions, then N_I
    is place-irredundant.
    """
    if not all_preregions:
        return set()

    # For each event, determine which regions are pre-regions and
    # what "coverage obligations" exist.
    # Obligation: for each event e and each state s  in  ( &  'e) - ER(e)... wait,
    # the condition for excitation closure is  & {r in 'e} r = ER(e).
    # So we need: for each event e, for each state s  not in  ER(e),
    # at least one pre-region of e must NOT contain s.
    # Equivalently: for each (e, s) where s  not in  ER(e), we need some
    # r  in  'e with s  not in  r.  Those are the "covering obligations".

    obligations: list[tuple[str, str, set[frozenset[str]]]] = []
    # Each obligation: (event, state, set of regions that can cover it)

    for e in events:
        er = _excitation_region(e, ts_transitions)
        pre = _preregions_of(e, all_preregions, ts_transitions)
        if not pre:
            continue

        # All states in the union of pre-regions that are NOT in ER.
        all_pre_states = frozenset.union(*pre) if pre else frozenset()
        problematic_states = all_pre_states - er

        for s in problematic_states:
            # Which pre-regions exclude s?
            covering = {r for r in pre if s not in r}
            if covering:
                obligations.append((e, s, covering))

    # Step 1: Identify essential regions.
    essential: set[frozenset[str]] = set()
    for e, s, covering in obligations:
        if len(covering) == 1:
            essential.update(covering)

    # Start with essential regions and check what's still uncovered.
    selected = set(essential)
    remaining_obligations = []
    for e, s, covering in obligations:
        # Check if any selected region already covers this obligation.
        if covering & selected:
            continue
        remaining_obligations.append((e, s, covering))

    # Step 2: Greedy set-cover for remaining obligations.
    # Cost = |-p| + |p-| + 1  (number of input/output arcs + 1).
    while remaining_obligations:
        # Score each candidate region by how many obligations it covers.
        candidate_scores: dict[frozenset[str], int] = defaultdict(int)
        for _, _, covering in remaining_obligations:
            for r in covering:
                candidate_scores[r] += 1

        if not candidate_scores:
            break

        def _region_cost(r: frozenset[str]) -> float:
            """S4.2 cost: |-p| + |p-| + 1."""
            n_pre = sum(1 for e in events if _exit(e, r, ts_transitions))
            n_post = sum(1 for e in events if _enter(e, r, ts_transitions))
            return n_pre + n_post + 1

        # Select region with best coverage/cost ratio.
        best_region = max(
            candidate_scores.keys(),
            key=lambda r: candidate_scores[r] / _region_cost(r),
        )
        selected.add(best_region)

        # Remove covered obligations.
        remaining_obligations = [
            (e, s, cov)
            for e, s, cov in remaining_obligations
            if best_region not in cov
        ]

    # Also include all regions that are pre-regions of events that
    # have NO other pre-region in selected (event effectiveness A.5').
    for e in events:
        pre = _preregions_of(e, all_preregions, ts_transitions)
        if pre and not (pre & selected):
            # No selected region is a pre-region of e -- pick cheapest.
            cheapest = min(pre, key=lambda r: len(r))
            selected.add(cheapest)

    # Include post-regions needed for the flow relation to be complete.
    # We also need post-regions to build proper arcs.  Include all
    # post-regions that are also pre-regions of some event (already
    # in the set) plus any additional post-regions needed.
    # Per the paper, the saturated net uses ALL regions.  The
    # irredundant cover only removes regions while maintaining
    # excitation closure.  Post-regions that are not pre-regions
    # of any event can be safely included -- they don't affect
    # excitation closure but may be needed for correct token flow.
    all_postregions: set[frozenset[str]] = set()
    for e in events:
        all_postregions.update(_postregions_of(e, all_preregions, ts_transitions))

    # Add post-regions that are NOT already in selected and not
    # pre-regions of any event.  These are "pure post-regions".
    # Per the paper, we should include them for correct behaviour.
    for r in all_postregions:
        if r not in selected:
            # Check if this region is a pre-region of any event.
            is_prereg = any(_exit(e, r, ts_transitions) for e in events)
            if not is_prereg:
                selected.add(r)

    return selected


# ===========================================================================
# 8. Saturated PN Synthesis  (S2.5.1, Thm 3.4, Cortadella et al.)
# ===========================================================================


def _build_dpn_from_regions(
    efsm: EFSM,
    events: set[str],
    ts_transitions: set[tuple[str, str, str]],
    irredundant_regions: set[frozenset[str]],
    label_map: dict[str, str],
    s_in: str,
) -> DataPetriNet:
    """S2.5.1 / Thm 3.4 -- Construct a DPN from the irredundant region set.

    Algorithm (saturated PN synthesis, adapted for irredundant cover):
      1. For each event e  in  E, generate a DPN transition labeled e.
      2. For each region r_i  in  I, generate a place.
      3. Place r_i is marked iff s_in  in  r_i.
      4. Flow: (r, e)  in  F iff r is a pre-region of e;
              (e, r)  in  F iff r is a post-region of e.

    Data annotations (guards, updates) are recovered by matching
    each DPN transition's activity label back to the original EFSM
    edges via the label_map.
    """
    # Assign stable names to regions.
    region_list = sorted(irredundant_regions, key=lambda r: tuple(sorted(r)))
    region_to_place: dict[frozenset[str], str] = {}
    for idx, region in enumerate(region_list):
        # Name from constituent states for traceability.
        state_part = "_".join(sorted(region)[:3])
        if len(region) > 3:
            state_part += f"_+{len(region) - 3}"
        region_to_place[region] = f"p_r{idx}_{state_part}"

    # 2. Places.
    places = [Place(name=name) for name in sorted(region_to_place.values())]

    # 3. Initial marking: regions containing s_in.
    initial_marking: set[str] = set()
    for region, pname in region_to_place.items():
        if s_in in region:
            initial_marking.add(pname)

    # Pre-compute pre-regions and post-regions per event.
    pre_per_event: dict[str, set[frozenset[str]]] = {}
    post_per_event: dict[str, set[frozenset[str]]] = {}
    for e in events:
        pre_per_event[e] = {
            r for r in irredundant_regions
            if _exit(e, r, ts_transitions)
        }
        post_per_event[e] = {
            r for r in irredundant_regions
            if _enter(e, r, ts_transitions)
        }

    # Also find "self-loop" regions: where the event is internal.
    internal_per_event: dict[str, set[frozenset[str]]] = {}
    for e in events:
        internal_per_event[e] = {
            r for r in irredundant_regions
            if _in(e, r, ts_transitions)
            and not _exit(e, r, ts_transitions)
            and not _enter(e, r, ts_transitions)
        }

    # Build an index: original_activity -> list of EFSM edges.
    efsm_edges_by_activity: dict[str, list[Transition]] = defaultdict(list)
    for t in efsm.transitions:
        efsm_edges_by_activity[t.activity].append(t)

    # 1 & 4. Transitions + flow.
    dpn_transitions: list[DPNTransition] = []
    smt = get_solver()
    counter = 0

    for event in sorted(events):
        counter += 1
        original_activity = label_map.get(event, event)

        # Input places: pre-regions (+ self-loop regions).
        input_places: set[str] = set()
        for r in pre_per_event.get(event, set()):
            input_places.add(region_to_place[r])
        for r in internal_per_event.get(event, set()):
            input_places.add(region_to_place[r])

        # Output places: post-regions (+ self-loop regions).
        output_places: set[str] = set()
        for r in post_per_event.get(event, set()):
            output_places.add(region_to_place[r])
        for r in internal_per_event.get(event, set()):
            output_places.add(region_to_place[r])

        # Recover guard & update from EFSM edges with matching activity.
        matching_edges = efsm_edges_by_activity.get(original_activity, [])
        guard = _combine_guards(matching_edges, smt)
        update = _combine_updates(matching_edges)

        tname = f"t_{original_activity}_{counter}"
        dpn_transitions.append(DPNTransition(
            name=tname,
            guard=guard,
            update_rule=update,
            input_places=input_places,
            output_places=output_places,
        ))

    return DataPetriNet(
        places=places,
        transitions=dpn_transitions,
        variables=set(efsm.variables),
        initial_marking=initial_marking,
    )


def _combine_guards(edges: list[Transition], smt) -> Optional:
    """Combine guards from multiple EFSM edges via disjunction.

    When a DPN transition corresponds to multiple EFSM edges
    (same activity label), their guards are OR-ed together.
    """
    guards = [e.guard_formula for e in edges if e.guard_formula is not None]
    if not guards:
        return None
    if len(guards) == 1:
        return guards[0]
    combined = guards[0]
    for g in guards[1:]:
        combined = smt.Or(combined, g)
    return combined


def _combine_updates(edges: list[Transition]) -> Optional[dict]:
    """Combine update rules from multiple EFSM edges.

    If all edges have compatible updates (same variables, same
    expressions), return that update.  Otherwise, return the first
    non-None update found (conservative -- guards will discriminate).
    """
    updates = [e.update_rule for e in edges if e.update_rule is not None]
    if not updates:
        return None
    if len(updates) == 1:
        return dict(updates[0])
    # Check compatibility: same keys and string representations.
    ref = updates[0]
    for u in updates[1:]:
        if set(u.keys()) != set(ref.keys()):
            return dict(ref)
        for var in ref:
            if str(u[var]) != str(ref[var]):
                return dict(ref)
    return dict(ref)


# ===========================================================================
# 8b. Flower model for single-state EFSM
# ===========================================================================


def _build_flower_dpn(
    efsm: EFSM,
    states: set[str],
    events: set[str],
    ts_transitions: set[tuple[str, str, str]],
    s_in: str,
) -> DataPetriNet:
    """Build a 'flower' Petri net for a single-state EFSM.

    When the EFSM has only one state, the resulting net has one place
    and every transition is a self-loop on that place.  Guards and
    updates are recovered from the EFSM edges via activity labels.
    """
    place_name = f"p_{s_in}"
    places = [Place(name=place_name)]

    # Build activity -> EFSM edges index.
    activity_edges: dict[str, list[Transition]] = defaultdict(list)
    for t in efsm.transitions:
        activity_edges[t.activity].append(t)

    smt = get_solver()
    dpn_transitions: list[DPNTransition] = []
    counter = 0

    for activity in sorted(events):
        edges = activity_edges.get(activity, [])
        counter += 1
        tname = f"t_{activity}_{counter}"

        guard = _combine_guards(edges, smt) if edges else None
        update = _combine_updates(edges) if edges else None

        dpn_transitions.append(DPNTransition(
            name=tname,
            guard=guard,
            update_rule=update,
            input_places={place_name},
            output_places={place_name},
        ))

    return DataPetriNet(
        places=places,
        transitions=dpn_transitions,
        variables=set(efsm.variables),
        initial_marking={place_name},
    )


# ===========================================================================
# 9. Top-Level: Region-Based EFSM -> DPN  (Fig. 10, Cortadella et al.)
# ===========================================================================


def region_based_efsm_to_dpn(efsm: EFSM) -> DataPetriNet:
    """Fig. 10 -- Full Petri Net synthesis algorithm.

    Top-level procedure from the paper:

        function Petri_Net_synthesis(TS):
          repeat
            for each event e in E:
              generate_min_preregions(e)
            find_irredundant_cover()
            if not excitation_closed(TS):
              split_labels(TS)
          until excitation_closed(TS)
          map_to_PN()

    After PN construction, DPN transitions are annotated with guards
    and updates by matching activity labels back to the original EFSM.
    """
    states, events, ts_transitions, s_in = _extract_ts(efsm)

    logger.info("  Region synthesis: |S| = %d, |E| = %d, |T| = %d",
                len(states), len(events), len(ts_transitions))

    # ---- Degenerate case: single-state TS (flower model) ----------------
    # When the EFSM has only one state with all self-loops, the theory
    # of regions cannot produce non-trivial regions (Def 2.2 is vacuously
    # satisfied only by the full set {s} which gives a single place with
    # all transitions as self-loops).  This produces a "flower" Petri net.
    if len(states) == 1:
        logger.info("  Single-state EFSM: producing flower Petri net.")
        return _build_flower_dpn(efsm, states, events, ts_transitions, s_in)

    label_map: dict[str, str] = {e: e for e in events}

    max_iterations = 10  # Safety bound for the split-labels loop.

    for iteration in range(max_iterations):
        # -- Step 1: Generate minimal pre-regions for each event ------
        all_preregions: set[frozenset[str]] = set()
        for event in sorted(events):
            preregions = _generate_min_preregions(
                event, states, events, ts_transitions,
            )
            all_preregions.update(preregions)
            logger.debug("    Event '%s': %d minimal pre-regions",
                         event, len(preregions))

        # Also generate post-regions via complement (Prop 2.2):
        # r is a region <=> S - r is a region.
        all_states_fs = frozenset(states)
        complements: set[frozenset[str]] = set()
        for r in all_preregions:
            comp = all_states_fs - r
            if comp and comp != all_states_fs:
                if _is_region(comp, events, ts_transitions):
                    complements.add(comp)
        all_regions = all_preregions | complements

        logger.info("    Iteration %d: %d pre-regions + %d complements = %d total regions",
                     iteration + 1, len(all_preregions), len(complements), len(all_regions))

        # -- Step 2: Check excitation closure (Def 3.2) ---------------
        closure = _check_excitation_closure(events, all_regions, ts_transitions)
        all_closed = all(closure.values())

        if all_closed:
            logger.info("    Excitation closure satisfied.")
            break

        # -- Step 3: Split labels (S4.3) ------------------------------
        non_closed = [e for e, ok in closure.items() if not ok]
        logger.info("    Excitation closure violated for: %s -- splitting labels",
                     non_closed)

        events, ts_transitions, new_label_map = _split_labels(
            states, events, ts_transitions, s_in, all_regions,
        )
        # Update the label_map to chain through splits.
        for lbl, orig in new_label_map.items():
            label_map[lbl] = label_map.get(orig, orig)

    else:
        logger.warning("  Region synthesis: max iterations reached without full excitation closure.")

    # -- Step 4: Find irredundant cover (S4.2, Thm 3.5) --------------
    irredundant = _find_irredundant_cover(events, all_regions, ts_transitions)
    logger.info("    Irredundant cover: %d regions (from %d total)",
                len(irredundant), len(all_regions))

    # -- Step 5: Build DPN (S2.5.1 / Thm 3.4) ------------------------
    dpn = _build_dpn_from_regions(
        efsm, events, ts_transitions, irredundant, label_map, s_in,
    )
    logger.info("    DPN: %d places, %d transitions",
                len(dpn.places), len(dpn.transitions))

    return dpn
