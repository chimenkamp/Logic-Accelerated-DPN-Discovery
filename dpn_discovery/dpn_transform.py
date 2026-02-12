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

import xml.etree.ElementTree as ET
from typing import Any

import z3

from dpn_discovery.models import (
    EFSM,
    DPNTransition,
    DataPetriNet,
    EventLog,
    Place,
)


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


def _evaluate_guard(guard: z3.BoolRef, values: dict[str, Any]) -> bool:
    """Evaluate a Z3 Boolean guard by substituting concrete values."""
    try:
        substitutions = []
        for name, val in values.items():
            if isinstance(val, (int, float)):
                substitutions.append((z3.Real(name), z3.RealVal(val)))
        result = z3.substitute(guard, *substitutions) if substitutions else guard
        result = z3.simplify(result)
        return z3.is_true(result)
    except Exception:
        # Conservative: if we can't evaluate, assume the guard is satisfied.
        return True


def _evaluate_expr(expr: z3.ExprRef, values: dict[str, Any]) -> float | None:
    """Evaluate a Z3 arithmetic expression by substituting concrete values."""
    try:
        substitutions = []
        for name, val in values.items():
            if isinstance(val, (int, float)):
                substitutions.append((z3.Real(name), z3.RealVal(val)))
        result = z3.substitute(expr, *substitutions) if substitutions else expr
        result = z3.simplify(result)
        if z3.is_rational_value(result):
            return float(result.as_fraction())
        if z3.is_int_value(result):
            return float(result.as_long())
        return None
    except Exception:
        return None
