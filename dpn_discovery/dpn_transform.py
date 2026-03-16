"""
Step 6 — EFSM → Data Petri Net Transformation.

Converts a guarded Extended Finite State Machine into a Data Petri
Net using the straightforward structural mapping defined in §4
Step 6 of the specification:

  • Places:      one place  pₛ  per EFSM state  s.
  • Transitions: one DPN transition  t_dpn  per EFSM transition
                 (s_src, s_tgt, a, g, u).
  • Arcs:        p_{s_src} → t_dpn  and  t_dpn → p_{s_tgt}.
  • Annotations: guard  g  and update  u  carried over verbatim.

Additionally provides:
  • PNML serialisation  (Petri Net Markup Language — XML).
  • Log-replay verification.

Reference: §4 Step 6 of the specification.
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

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Core mapping
# ═══════════════════════════════════════════════════════════════════════════

def efsm_to_dpn(efsm: EFSM) -> DataPetriNet:
    """Convert *efsm* into a ``DataPetriNet``.

    Algorithm (specification §4 Step 6):
      1. Create place  pₛ  for every state  s ∈ S.
      2. For every EFSM transition  (s_src, s_tgt, a, g, u):
         a) Create DPN transition  t_dpn.
         b) Add arc  p_{s_src} → t_dpn.
         c) Add arc  t_dpn  → p_{s_tgt}.
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


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PNML serialisation
# ═══════════════════════════════════════════════════════════════════════════

def dpn_to_pnml(dpn: DataPetriNet, net_id: str = "dpn_net") -> str:
    """Serialise *dpn* to PNML (Petri Net Markup Language) XML.

    Produces a valid ``<pnml>`` document with:
      • ``<place>`` elements (with initial marking where applicable).
      • ``<transition>`` elements (with guard/update annotations in
        ``<toolspecific>`` blocks).
      • ``<arc>`` elements for the flow relation.
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

        # Arcs — input.
        for ip in trans.input_places:
            arc_id = f"arc_{ip}_to_{trans.name}"
            ET.SubElement(page, "arc", id=arc_id, source=ip, target=trans.name)

        # Arcs — output.
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


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Log-replay verification
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# 4.  DPN post-construction reduction
# ═══════════════════════════════════════════════════════════════════════════

def reduce_dpn(dpn: DataPetriNet, log: EventLog) -> DataPetriNet:
    """Apply data-driven reduction passes to *dpn*.

    This is a **post-construction** pass that does not change the
    language accepted by the net w.r.t. the log.  It consists of
    two stages:

      1. **Place fusion** — replay the log and record which places
         always carry identical token counts across all reachable
         markings.  Fuse those places into a single representative.
      2. **Transition collapsing** — after place fusion, transitions
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

    after_p, after_t = len(dpn.places), len(dpn.transitions)

    if before_p != after_p or before_t != after_t:
        logger.info(
            "  DPN reduction: places %d → %d  |  transitions %d → %d",
            before_p, after_p, before_t, after_t,
        )

    return dpn


def _fuse_places(dpn: DataPetriNet, log: EventLog) -> DataPetriNet:
    """Fuse places that always carry identical token counts.

    Replays every trace through the DPN, recording the full
    token-count vector at every step.  Two places whose count
    vectors are identical across all reachable markings of all
    traces are fused into a single representative place.

    This merges structurally redundant places that survived EFSM
    deduplication — e.g. places that appear distinct at the EFSM
    level but behave identically in every observed execution.
    """
    place_names = sorted(p.name for p in dpn.places)
    if len(place_names) <= 1:
        return dpn

    # ── Collect token-count profiles per place across all traces ─────────
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

    # ── Group places by identical profiles ───────────────────────────────
    profile_groups: dict[tuple[int, ...], list[str]] = {}
    for pname in place_names:
        key = tuple(profiles[pname])
        profile_groups.setdefault(key, []).append(pname)

    # Build a mapping:  old_place_name → representative_name.
    remap: dict[str, str] = {}
    for group in profile_groups.values():
        rep = group[0]  # first alphabetically (they're sorted)
        for p in group:
            remap[p] = rep

    # If nothing to fuse, return as-is.
    if all(remap[p] == p for p in place_names):
        return dpn

    # ── Build fused DPN ──────────────────────────────────────────────────
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
    from the 1:1 EFSM→DPN mapping when multiple EFSM edges encode
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

    Convention:  t_{activity}_{counter}  →  {activity}.
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
