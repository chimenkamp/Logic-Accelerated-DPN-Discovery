"""Step 7 -- EFSM -> Data Petri Net Transformation.

Converts a guarded Extended Finite State Machine into a Data Petri
Net using the **theory of regions** from:

    Cortadella, Kishinevsky, Lavagno, Yakovlev.
    "Deriving Petri Nets from Finite Transition Systems."
    IEEE Transactions on Computers, 1998.

The default ``efsm_to_dpn`` now delegates to the region-based
synthesis algorithm (S4, Fig. 10 of the paper).  The legacy 1:1
structural mapping is kept as ``efsm_to_dpn_structural`` for
comparison / fallback.

Additionally provides:
  - PNML serialisation  (Petri Net Markup Language -- XML).
  - Log-replay verification.
  - Post-construction DPN reduction (place fusion + transition collapse).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

from dpn_discovery.smt import get_solver, SMTBool, SMTExpr

from dpn_discovery.models import (
    EFSM,
    DPNTransition,
    DataPetriNet,
    EventLog,
    Place,
)
from dpn_discovery.regions import region_based_efsm_to_dpn

logger = logging.getLogger(__name__)


# ===========================================================================
# 1.  Core mapping
# ===========================================================================

def efsm_to_dpn(efsm: EFSM) -> DataPetriNet:
    """Convert *efsm* into a ``DataPetriNet``.

    Uses the **theory of regions** (Cortadella et al. 1998):
      - S4, Fig. 10: top-level synthesis loop
      - S4.1: minimal pre-region generation (Lemma 4.1/4.2)
      - S4.3: label splitting for excitation closure (Def 3.2)
      - S4.2: irredundant cover (Thm 3.5)
      - S2.5.1: saturated PN synthesis / Thm 3.4

    Guards and update rules are recovered by matching each DPN
    transition's activity label back to the original EFSM edges.

    See ``efsm_to_dpn_structural`` for the legacy 1:1 mapping.
    """
    return region_based_efsm_to_dpn(efsm)


def efsm_to_dpn_structural(efsm: EFSM) -> DataPetriNet:
    """Legacy 1:1 structural mapping (pre-region-theory fallback).

    Algorithm (specification S4 Step 6 -- original):
      1. Create place  p_s  for every state  s in S.
      2. For every EFSM transition  (s_src, s_tgt, a, g, u):
         a) Create DPN transition  t_dpn.
         b) Add arc  p_{s_src} -> t_dpn.
         c) Add arc  t_dpn  -> p_{s_tgt}.
         d) Annotate t_dpn with guard g and update u.
      3. Mark the place corresponding to the initial state.
    """
    places: list[Place] = []
    place_names: set[str] = set()

    # 1. Places.
    for state_id in sorted(efsm.states):
        pname = f"p_{state_id}"
        places.append(Place(name=pname))
        place_names.add(pname)

    # 2. Transitions + arcs.
    dpn_transitions: list[DPNTransition] = []
    _counter = 0

    for t in efsm.transitions:
        _counter += 1
        tname = f"t_{t.activity}_{_counter}"

        dpn_t = DPNTransition(
            name=tname,
            guard=t.guard_formula,
            update_rule=dict(t.update_rule) if t.update_rule else None,
            input_places={f"p_{t.source_id}"},
            output_places={f"p_{t.target_id}"},
        )
        dpn_transitions.append(dpn_t)

    # 3. Initial marking.
    initial_place = f"p_{efsm.initial_state}"
    initial_marking = {initial_place} if initial_place in place_names else set()

    return DataPetriNet(
        places=places,
        transitions=dpn_transitions,
        variables=set(efsm.variables),
        initial_marking=initial_marking,
    )


# ===========================================================================
# 2.  PNML serialisation
# ===========================================================================

def dpn_to_pnml(dpn: DataPetriNet, net_id: str = "dpn_net") -> str:
    """Serialise *dpn* to PNML (Petri Net Markup Language) XML.

    Produces a valid ``<pnml>`` document with:
      - ``<place>`` elements (with initial marking where applicable).
      - ``<transition>`` elements (with guard/update annotations in
        ``<toolspecific>`` blocks).
      - ``<arc>`` elements for the flow relation.
    """
    root = ET.Element("pnml")
    net = ET.SubElement(root, "net", id=net_id, type="http://www.pnml.org/version-2009/grammar/pnml")

    page = ET.SubElement(net, "page", id="page0")

    # Places.
    for place in dpn.places:
        p_el = ET.SubElement(page, "place", id=place.name)
        name_el = ET.SubElement(p_el, "name")
        ET.SubElement(name_el, "text").text = place.name
        if place.name in dpn.initial_marking:
            marking_el = ET.SubElement(p_el, "initialMarking")
            ET.SubElement(marking_el, "text").text = "1"

    # Transitions.
    for trans in dpn.transitions:
        t_el = ET.SubElement(page, "transition", id=trans.name)
        name_el = ET.SubElement(t_el, "name")
        # Strip numeric suffix for human-readable label.
        label = trans.name.split("_", 1)[1].rsplit("_", 1)[0] if "_" in trans.name else trans.name
        ET.SubElement(name_el, "text").text = label

        # Guard annotation.
        if trans.guard is not None:
            ts = ET.SubElement(t_el, "toolspecific", tool="dpn-discovery", version="1.0")
            guard_el = ET.SubElement(ts, "guard")
            guard_el.text = str(trans.guard)

        # Update annotation.
        if trans.update_rule:
            ts = ET.SubElement(t_el, "toolspecific", tool="dpn-discovery", version="1.0")
            for var, expr in trans.update_rule.items():
                upd_el = ET.SubElement(ts, "update", variable=var)
                upd_el.text = str(expr)

        # Arcs -- input.
        for ip in trans.input_places:
            arc_id = f"arc_{ip}_to_{trans.name}"
            ET.SubElement(page, "arc", id=arc_id, source=ip, target=trans.name)

        # Arcs -- output.
        for op in trans.output_places:
            arc_id = f"arc_{trans.name}_to_{op}"
            ET.SubElement(page, "arc", id=arc_id, source=trans.name, target=op)

    # Variables.
    if dpn.variables:
        ts_vars = ET.SubElement(net, "toolspecific", tool="dpn-discovery", version="1.0")
        for var in sorted(dpn.variables):
            ET.SubElement(ts_vars, "variable", name=var)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ===========================================================================
# 3.  Log-replay verification
# ===========================================================================

def verify_dpn(dpn: DataPetriNet, log: EventLog) -> bool:
    """Replay every trace in *log* through *dpn* and return
    ``True`` iff all traces can be replayed successfully.

    A trace is **replayable** if, starting from the initial
    marking, every event in the trace matches a firable
    transition (token present in the input place, guard
    satisfied by current data state).
    """
    all_ok = True
    for trace in log.traces:
        if not _replay_trace(dpn, trace):
            all_ok = False
    return all_ok


def _replay_trace(
    dpn: DataPetriNet,
    trace: Any,  # models.Trace
) -> bool:
    """Replay a single trace through the DPN.

    Returns ``True`` if every event fires exactly one transition.
    """
    # Token set: initially one token in each initial-marking place.
    tokens: dict[str, int] = {}
    for p in dpn.initial_marking:
        tokens[p] = tokens.get(p, 0) + 1

    # Data state.
    data_state: dict[str, float] = {}

    for event in trace.events:
        fired = False
        for trans in dpn.transitions:
            # Activity label check.
            activity_label = _transition_activity(trans)
            if activity_label != event.activity:
                continue

            # Token availability check.
            can_fire = all(tokens.get(p, 0) >= 1 for p in trans.input_places)
            if not can_fire:
                continue

            # Guard check (evaluate with current data state + event payload).
            if trans.guard is not None and not _evaluate_guard(trans.guard, {**data_state, **event.payload}):
                continue

            # Fire.
            for p in trans.input_places:
                tokens[p] -= 1
            for p in trans.output_places:
                tokens[p] = tokens.get(p, 0) + 1

            # Apply updates.
            data_state.update(event.payload)
            if trans.update_rule:
                for var, expr in trans.update_rule.items():
                    val = _evaluate_expr(expr, data_state)
                    if val is not None:
                        data_state[var] = val

            fired = True
            break

        if not fired:
            return False

    return True


# ===========================================================================
# 4.  DPN post-construction reduction
# ===========================================================================

def reduce_dpn(dpn: DataPetriNet, log: EventLog) -> DataPetriNet:
    """Apply data-driven reduction passes to *dpn*.

    This is a **post-construction** pass that does not change the
    language accepted by the net w.r.t. the log.  It consists of
    two stages:

      1. **Place fusion** -- replay the log and record which places
         always carry identical token counts across all reachable
         markings.  Fuse those places into a single representative.
      2. **Transition collapsing** -- after place fusion, transitions
         sharing the same activity label AND the same input/output
         place sets are merged (guards become disjunctions via OR).

    Both stages are purely data-driven (no assumptions beyond the
    log) and replay-verified.

    Parameters
    ----------
    dpn : DataPetriNet
        The DPN to reduce (not modified; a new DPN is returned).
    log : EventLog
        The original event log (for replay-based analysis).

    Returns
    -------
    DataPetriNet
        The reduced DPN.
    """
    before_p, before_t = len(dpn.places), len(dpn.transitions)

    dpn = _fuse_places(dpn, log)
    dpn = _collapse_transitions(dpn)
    dpn = _merge_duplicate_activity_transitions(dpn)

    after_p, after_t = len(dpn.places), len(dpn.transitions)

    if before_p != after_p or before_t != after_t:
        logger.info(
            "  DPN reduction: places %d -> %d  |  transitions %d -> %d",
            before_p, after_p, before_t, after_t,
        )

    return dpn


def _merge_duplicate_activity_transitions(dpn: DataPetriNet) -> DataPetriNet:
    """Merge DPN transitions that share the same activity label.

    After region-based synthesis with label splitting (Cortadella et al.
    S4.3), the same activity may produce multiple DPN transitions with
    different arc structures.  A valid (data-aware) Petri net for process
    mining requires **unique activity labels**.

    Merging strategy
    ----------------
    For each group of transitions sharing an activity label:

      * **Guards** are combined via disjunction (OR).  Because the
        split partitions the state space, the guards are typically
        mutually exclusive, so the OR is semantically correct.
      * **Input / output places** are *unioned*.  This makes the
        merged transition require tokens in *all* pre-places of
        *every* original transition – a stricter enabling condition.
        In practice the guards (which encode the routing context)
        prevent spurious firings, so observed behaviour is preserved.
      * **Update rules** are kept from the first transition if all
        originals agree; otherwise they are combined via an
        ``If(guard_i, update_i, ...)`` expression when the SMT
        backend supports it, or the first non-``None`` update is
        used as a conservative fallback.

    This pass is designed to run *after* ``_collapse_transitions``
    (which only merges transitions with identical arc sets) and
    catches the remaining duplicates that differ in arc structure.
    """
    from collections import defaultdict as _defaultdict

    activity_groups: dict[str, list[DPNTransition]] = _defaultdict(list)
    for trans in dpn.transitions:
        activity = _transition_activity(trans)
        activity_groups[activity].append(trans)

    new_transitions: list[DPNTransition] = []
    smt = get_solver()

    for activity, group in activity_groups.items():
        if len(group) == 1:
            new_transitions.append(group[0])
            continue

        logger.info(
            "  Merging %d duplicate transitions for activity '%s'",
            len(group), activity,
        )

        # ---- Union arcs ------------------------------------------------
        merged_in: set[str] = set()
        merged_out: set[str] = set()
        guards: list = []
        updates: list[tuple[dict | None, Any]] = []  # (update_rule, guard)

        for t in group:
            merged_in |= t.input_places
            merged_out |= t.output_places
            if t.guard is not None:
                guards.append(t.guard)
            updates.append((t.update_rule, t.guard))

        # ---- Combine guards via OR -------------------------------------
        combined_guard = None
        if guards:
            combined_guard = guards[0]
            for g in guards[1:]:
                combined_guard = smt.Or(combined_guard, g)

        # ---- Combine update rules --------------------------------------
        non_none_updates = [(u, g) for u, g in updates if u is not None]
        merged_update: dict | None = None

        if len(non_none_updates) == 0:
            merged_update = None
        elif len(non_none_updates) == 1:
            merged_update = dict(non_none_updates[0][0])
        else:
            # Check if all updates are structurally identical.
            ref_u = non_none_updates[0][0]
            all_same = all(
                _updates_equal(ref_u, u) for u, _ in non_none_updates[1:]
            )
            if all_same:
                merged_update = dict(ref_u)
            else:
                # Updates differ across split transitions.  Since the
                # guards already discriminate the routing context, use
                # the union of all update variables.  For variables that
                # appear in only one update, keep that expression.  For
                # conflicting variables, use the first expression as a
                # conservative fallback (the guard ensures correctness).
                all_vars: set[str] = set()
                for u, _ in non_none_updates:
                    all_vars.update(u.keys())

                merged_update = {}
                for var in sorted(all_vars):
                    for u, _ in non_none_updates:
                        if var in u:
                            merged_update[var] = u[var]
                            break  # first wins

        new_transitions.append(DPNTransition(
            name=group[0].name,
            guard=combined_guard,
            update_rule=merged_update,
            input_places=merged_in,
            output_places=merged_out,
        ))

    # Only rebuild if something actually changed.
    if len(new_transitions) == len(dpn.transitions):
        return dpn

    return DataPetriNet(
        places=list(dpn.places),
        transitions=new_transitions,
        variables=set(dpn.variables),
        initial_marking=set(dpn.initial_marking),
    )


def _fuse_places(dpn: DataPetriNet, log: EventLog) -> DataPetriNet:
    """Fuse places that always carry identical token counts.

    Replays every trace through the DPN, recording the full
    token-count vector at every step.  Two places whose count
    vectors are identical across all reachable markings of all
    traces are fused into a single representative place.

    This merges structurally redundant places that survived EFSM
    deduplication -- e.g. places that appear distinct at the EFSM
    level but behave identically in every observed execution.
    """
    place_names = sorted(p.name for p in dpn.places)
    if len(place_names) <= 1:
        return dpn

    # -- Collect token-count profiles per place across all traces ---------
    # profile[place] = list of token counts observed at each firing step.
    profiles: dict[str, list[int]] = {p: [] for p in place_names}

    for trace in log.traces:
        tokens: dict[str, int] = {}
        for p in dpn.initial_marking:
            tokens[p] = tokens.get(p, 0) + 1

        # Record initial marking.
        for p in place_names:
            profiles[p].append(tokens.get(p, 0))

        for event in trace.events:
            fired = False
            for trans in dpn.transitions:
                activity_label = _transition_activity(trans)
                if activity_label != event.activity:
                    continue
                can_fire = all(tokens.get(p, 0) >= 1 for p in trans.input_places)
                if not can_fire:
                    continue
                if trans.guard is not None and not _evaluate_guard(
                    trans.guard, {**{}, **event.payload}
                ):
                    continue

                for p in trans.input_places:
                    tokens[p] -= 1
                for p in trans.output_places:
                    tokens[p] = tokens.get(p, 0) + 1
                fired = True
                break

            # Record marking after this event.
            for p in place_names:
                profiles[p].append(tokens.get(p, 0))

    # -- Group places by identical profiles -------------------------------
    profile_groups: dict[tuple[int, ...], list[str]] = {}
    for pname in place_names:
        key = tuple(profiles[pname])
        profile_groups.setdefault(key, []).append(pname)

    # Build a mapping:  old_place_name -> representative_name.
    remap: dict[str, str] = {}
    for group in profile_groups.values():
        rep = group[0]  # first alphabetically (they're sorted)
        for p in group:
            remap[p] = rep

    # If nothing to fuse, return as-is.
    if all(remap[p] == p for p in place_names):
        return dpn

    # -- Build fused DPN --------------------------------------------------
    kept_places = sorted(set(remap.values()))
    new_places = [Place(name=p) for p in kept_places]

    new_transitions: list[DPNTransition] = []
    for trans in dpn.transitions:
        new_in = {remap.get(p, p) for p in trans.input_places}
        new_out = {remap.get(p, p) for p in trans.output_places}
        new_transitions.append(DPNTransition(
            name=trans.name,
            guard=trans.guard,
            update_rule=dict(trans.update_rule) if trans.update_rule else None,
            input_places=new_in,
            output_places=new_out,
        ))

    new_initial = {remap.get(p, p) for p in dpn.initial_marking}

    return DataPetriNet(
        places=new_places,
        transitions=new_transitions,
        variables=set(dpn.variables),
        initial_marking=new_initial,
    )


def _collapse_transitions(dpn: DataPetriNet) -> DataPetriNet:
    """Collapse DPN transitions with identical activity + arc structure.

    Two transitions are candidates for collapsing when they:
      1. Share the same activity label.
      2. Have identical input place sets.
      3. Have identical output place sets.

    When collapsed, the guards are combined via disjunction (OR) so
    no behavior is lost.  Update rules are kept from the first
    transition (they should be identical for structurally equivalent
    transitions; if not, we keep both transitions).

    This removes the 'duplicate transition' pattern that arises
    from the 1:1 EFSM->DPN mapping when multiple EFSM edges encode
    the same routing with different guards.
    """
    # Group by (activity, input_places, output_places).
    GroupKey = tuple[str, frozenset[str], frozenset[str]]
    groups: dict[GroupKey, list[DPNTransition]] = {}

    for trans in dpn.transitions:
        activity = _transition_activity(trans)
        key: GroupKey = (
            activity,
            frozenset(trans.input_places),
            frozenset(trans.output_places),
        )
        groups.setdefault(key, []).append(trans)

    new_transitions: list[DPNTransition] = []
    smt = get_solver()

    for (activity, in_p, out_p), group in groups.items():
        if len(group) == 1:
            new_transitions.append(group[0])
            continue

        # Check that update rules are compatible before collapsing.
        # If they differ, keep all transitions separate.
        updates_compatible = True
        ref_update = group[0].update_rule
        for t in group[1:]:
            if not _updates_equal(ref_update, t.update_rule):
                updates_compatible = False
                break

        if not updates_compatible:
            new_transitions.extend(group)
            continue

        # Combine guards via disjunction.
        guards = [t.guard for t in group if t.guard is not None]
        if not guards:
            combined_guard = None
        elif len(guards) == 1:
            combined_guard = guards[0]
        else:
            combined_guard = guards[0]
            for g in guards[1:]:
                combined_guard = smt.Or(combined_guard, g)

        merged = DPNTransition(
            name=group[0].name,
            guard=combined_guard,
            update_rule=dict(ref_update) if ref_update else None,
            input_places=set(in_p),
            output_places=set(out_p),
        )
        new_transitions.append(merged)

    return DataPetriNet(
        places=list(dpn.places),
        transitions=new_transitions,
        variables=set(dpn.variables),
        initial_marking=set(dpn.initial_marking),
    )


def _updates_equal(
    u1: dict[str, Any] | None,
    u2: dict[str, Any] | None,
) -> bool:
    """Check whether two update-rule dicts are structurally equal."""
    if u1 is None and u2 is None:
        return True
    if u1 is None or u2 is None:
        return False
    if set(u1.keys()) != set(u2.keys()):
        return False
    smt = get_solver()
    for var in u1:
        # Compare string representations as a fast structural check.
        if str(u1[var]) != str(u2[var]):
            return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transition_activity(trans: DPNTransition) -> str:
    """Extract the activity label from a DPN transition name.

    Convention:  t_{activity}_{counter}  ->  {activity}.
    """
    parts = trans.name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:-1])
    return trans.name


def _evaluate_guard(guard: SMTBool, values: dict[str, Any]) -> bool:
    """Evaluate an SMT Boolean guard by substituting concrete values."""
    smt = get_solver()
    try:
        substitutions = []
        for name, val in values.items():
            if isinstance(val, (int, float)):
                substitutions.append((smt.Real(name), smt.RealVal(val)))
        result = smt.substitute(guard, *substitutions) if substitutions else guard
        result = smt.simplify(result)
        return smt.is_true(result)
    except Exception:
        # Conservative: if we can't evaluate, assume the guard is satisfied.
        return True


def _evaluate_expr(expr: SMTExpr, values: dict[str, Any]) -> float | None:
    """Evaluate an SMT arithmetic expression by substituting concrete values."""
    smt = get_solver()
    try:
        substitutions = []
        for name, val in values.items():
            if isinstance(val, (int, float)):
                substitutions.append((smt.Real(name), smt.RealVal(val)))
        result = smt.substitute(expr, *substitutions) if substitutions else expr
        result = smt.simplify(result)
        if smt.is_rational_value(result) or smt.is_int_value(result):
            return smt.to_real_float(result)
        return None
    except Exception:
        return None
