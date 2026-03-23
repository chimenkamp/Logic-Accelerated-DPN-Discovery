"""
Alternative DPN Discovery Pipeline — ILP Miner + Heuristic Annotations.

A lightweight, **heuristic-based** baseline for Data Petri Net
discovery.  Trades the formal guarantees of the main pipeline
(region theory, PHOG-accelerated guard synthesis, SMT-based
postcondition abduction) for **speed and simplicity**.

Pipeline steps
--------------
  1. **Control flow** — pm4py's ILP miner discovers a sound Petri net
     from the event log.
  2. **Log replay** — alignment-based replay collects per-transition
     data observations (pre/post state pairs).
  3. **Guard synthesis** — PHOG-weighted A* search over the SyGuS
     guard grammar (§4 Step 4) with partition verification.
  4. **Postcondition synthesis** — enumerative abduction
     (GetAbductUCL, Reynolds et al.) over the SyGuS expression
     grammar (§4 Step 5).

All discovered guards and updates are compiled into the same SMT
expression format (``SMTBool`` / ``SMTExpr``) used by the main
pipeline, so the output ``DataPetriNet`` is fully compatible with
the existing visualiser, PNML serialiser and log-replay verifier.

Entry points
~~~~~~~~~~~~
  - ``run_ilp_pipeline(log)``  → ``DataPetriNet``
  - ``run_ilp_pipeline_full(log)`` → ``(pm4py_net, DataPetriNet)``
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any
import time

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking

from dpn_discovery.models import (
    DPNTransition,
    DataPetriNet,
    EventLog,
    Place,
)
from dpn_discovery.smt import get_solver, SMTBool, SMTArith, SMTExpr
from dpn_discovery.guard_synthesis import (
    GuardNode,
    PHOGModel,
    _search_guard,
    _verify_partition_smt,
)
from dpn_discovery.postcondition_synthesis import _get_abduct_ucl

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Control-flow discovery  (pm4py ILP miner)
# ═══════════════════════════════════════════════════════════════════════════

def _eventlog_to_pm4py_log(log: EventLog) -> pd.DataFrame:
    """Convert our ``EventLog`` to a pm4py-compatible ``DataFrame``.

    Generates synthetic case IDs and timestamps so that the ILP
    miner can operate.  Data variable columns are preserved for
    replay.
    """
    rows: list[dict[str, Any]] = []
    for case_idx, trace in enumerate(log.traces):
        for event_idx, event in enumerate(trace.events):
            row: dict[str, Any] = {
                "case:concept:name": f"case_{case_idx}",
                "concept:name": event.activity,
                "time:timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=event_idx),
            }
            # Carry data variables through.
            for var, val in event.payload.items():
                row[var] = val
            rows.append(row)

    df = pd.DataFrame(rows)
    df = pm4py.format_dataframe(df, case_id="case:concept:name",
                                activity_key="concept:name",
                                timestamp_key="time:timestamp")
    return df


def _discover_control_flow(
    df: pd.DataFrame,
    alpha: float = 1.0,
) -> tuple[PetriNet, Marking, Marking]:
    """Run pm4py's ILP miner and return the Petri net + markings.

    Parameters
    ----------
    alpha : float
        The ILP miner's noise threshold (0..1).  Lower values
        allow more behaviour (less fitting, more general).
    """
    net, im, fm = pm4py.discover_petri_net_ilp(df, alpha=alpha)
    logger.info(
        "  ILP miner: %d places, %d transitions, %d arcs",
        len(net.places), len(net.transitions), len(net.arcs),
    )
    return net, im, fm


# ═══════════════════════════════════════════════════════════════════════════
# 2.  pm4py Petri net → DataPetriNet conversion
# ═══════════════════════════════════════════════════════════════════════════

def _pm4py_to_dpn(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    variables: set[str],
) -> DataPetriNet:
    """Convert a pm4py ``PetriNet`` to our ``DataPetriNet`` format.

    Silent transitions (label ``None``) are assigned names prefixed
    with ``tau_``.
    """
    # Places.
    place_map: dict[str, str] = {}  # pm4py name → our name
    places: list[Place] = []
    for idx, p in enumerate(sorted(net.places, key=lambda p: str(p))):
        pname = f"p_{idx}"
        place_map[str(p)] = pname
        places.append(Place(name=pname))

    # Initial marking.
    initial_marking: set[str] = set()
    for p in im:
        initial_marking.add(place_map[str(p)])

    # Transitions.
    dpn_transitions: list[DPNTransition] = []
    activity_counter: dict[str, int] = defaultdict(int)
    tau_counter = 0

    for t in sorted(net.transitions, key=lambda t: str(t)):
        if t.label is None:
            # Silent transition.
            tau_counter += 1
            tname = f"tau_{tau_counter}"
        else:
            activity_counter[t.label] += 1
            tname = f"t_{t.label}_{activity_counter[t.label]}"

        input_places: set[str] = set()
        output_places: set[str] = set()

        for arc in net.arcs:
            if arc.target == t:
                input_places.add(place_map[str(arc.source)])
            elif arc.source == t:
                output_places.add(place_map[str(arc.target)])

        dpn_transitions.append(DPNTransition(
            name=tname,
            guard=None,
            update_rule=None,
            input_places=input_places,
            output_places=output_places,
        ))

    return DataPetriNet(
        places=places,
        transitions=dpn_transitions,
        variables=variables,
        initial_marking=initial_marking,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2b. Remove ILP-miner tau transitions (start / sink)
# ═══════════════════════════════════════════════════════════════════════════

def _remove_tau_transitions(dpn: DataPetriNet) -> DataPetriNet:
    """Remove silent (tau) transitions introduced by the ILP miner.

    The ILP miner typically inserts:
      * **start-tau**: source_place → tau → intermediate_place.
        Fix: promote the tau's output places into the initial marking
        and drop the tau (and its now-orphaned source place).
      * **sink-tau**: intermediate_place → tau → sink_place.
        Fix: drop the tau AND its output places (the "children").

    Any remaining tau transitions that are neither pure start nor
    pure sink (i.e. they participate in routing) are kept.
    """
    taus = [t for t in dpn.transitions if t.name.startswith("tau_")]
    if not taus:
        return dpn

    visible = [t for t in dpn.transitions if not t.name.startswith("tau_")]
    visible_input_places = {p for t in visible for p in t.input_places}
    visible_output_places = {p for t in visible for p in t.output_places}

    new_initial_marking = set(dpn.initial_marking)
    places_to_remove: set[str] = set()
    taus_to_remove: set[str] = set()

    for tau in taus:
        # --- Start-tau: input ⊆ initial_marking, outputs feed visible ---
        is_start = tau.input_places.issubset(dpn.initial_marking)

        # --- Sink-tau: outputs NOT consumed by any visible transition ---
        outputs_unused = not tau.output_places.intersection(visible_input_places)
        is_sink = outputs_unused and len(tau.output_places) > 0

        if is_start:
            # Promote tau's output places into the initial marking.
            new_initial_marking |= tau.output_places
            # The tau's input places are no longer needed as initial
            # (unless another visible transition also uses them).
            for p in tau.input_places:
                if p not in visible_input_places and p not in visible_output_places:
                    places_to_remove.add(p)
            taus_to_remove.add(tau.name)
            logger.info("  Removing start-tau '%s'", tau.name)

        elif is_sink:
            # Remove the tau and its output places (the children).
            places_to_remove |= tau.output_places
            taus_to_remove.add(tau.name)
            logger.info("  Removing sink-tau '%s' and %d sink places",
                        tau.name, len(tau.output_places))

        # Otherwise: routing tau — keep it.

    if not taus_to_remove:
        return dpn

    # Remove initial-marking entries for removed places.
    new_initial_marking -= places_to_remove

    new_transitions = [
        t for t in dpn.transitions if t.name not in taus_to_remove
    ]
    new_places = [
        p for p in dpn.places if p.name not in places_to_remove
    ]

    logger.info(
        "  After tau removal: %d places, %d transitions",
        len(new_places), len(new_transitions),
    )

    return DataPetriNet(
        places=new_places,
        transitions=new_transitions,
        variables=set(dpn.variables),
        initial_marking=new_initial_marking,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Log replay — collect per-transition data observations
# ═══════════════════════════════════════════════════════════════════════════

def _replay_and_collect(
    dpn: DataPetriNet,
    log: EventLog,
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Replay the log through the DPN and collect (pre, post) pairs.

    For each trace, greedily fire transitions matching the current
    event's activity (token-game replay).  Records:
      - pre-state:  data state *before* the event
      - post-state: data state *after* the event (= event payload)

    Returns a dict mapping transition name → list of (pre, post) pairs.
    """
    observations: dict[str, list[tuple[dict, dict]]] = defaultdict(list)

    replayed = 0
    failed = 0

    for trace in log.traces:
        tokens: dict[str, int] = {}
        for p in dpn.initial_marking:
            tokens[p] = tokens.get(p, 0) + 1

        data_state: dict[str, Any] = {}
        trace_ok = True

        for event in trace.events:
            pre_state = dict(data_state)
            fired = False

            for trans in dpn.transitions:
                # Skip silent transitions for activity matching.
                if trans.name.startswith("tau_"):
                    continue

                activity = _transition_activity(trans)
                if activity != event.activity:
                    continue

                # Token check.
                can_fire = all(tokens.get(p, 0) >= 1 for p in trans.input_places)
                if not can_fire:
                    continue

                # Fire.
                for p in trans.input_places:
                    tokens[p] -= 1
                for p in trans.output_places:
                    tokens[p] = tokens.get(p, 0) + 1

                # Record observation.
                post_state = dict(event.payload)
                observations[trans.name].append((pre_state, post_state))

                # Update data state.
                data_state.update(event.payload)
                fired = True
                break

            if not fired:
                # Try firing silent transitions to advance the token game.
                fired_silent = True
                while fired_silent:
                    fired_silent = False
                    for trans in dpn.transitions:
                        if not trans.name.startswith("tau_"):
                            continue
                        can_fire = all(tokens.get(p, 0) >= 1 for p in trans.input_places)
                        if can_fire:
                            for p in trans.input_places:
                                tokens[p] -= 1
                            for p in trans.output_places:
                                tokens[p] = tokens.get(p, 0) + 1
                            fired_silent = True
                            break

                # Retry the activity after firing silent transitions.
                for trans in dpn.transitions:
                    if trans.name.startswith("tau_"):
                        continue
                    activity = _transition_activity(trans)
                    if activity != event.activity:
                        continue
                    can_fire = all(tokens.get(p, 0) >= 1 for p in trans.input_places)
                    if not can_fire:
                        continue
                    for p in trans.input_places:
                        tokens[p] -= 1
                    for p in trans.output_places:
                        tokens[p] = tokens.get(p, 0) + 1
                    post_state = dict(event.payload)
                    observations[trans.name].append((pre_state, post_state))
                    data_state.update(event.payload)
                    fired = True
                    break

                if not fired:
                    trace_ok = False
                    break

        if trace_ok:
            replayed += 1
        else:
            failed += 1

    logger.info(
        "  Replay: %d/%d traces replayed successfully (%d failed)",
        replayed, replayed + failed, failed,
    )
    return dict(observations)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Heuristic guard synthesis  (Decision Tree)
# ═══════════════════════════════════════════════════════════════════════════

def _find_choice_groups(dpn: DataPetriNet) -> dict[str, list[DPNTransition]]:
    """Find groups of transitions that share at least one input place.

    These are "choice points" where the token game must decide
    which transition fires.  Guards discriminate between them.
    """
    # Map place → transitions that consume from it.
    place_consumers: dict[str, list[DPNTransition]] = defaultdict(list)
    for trans in dpn.transitions:
        if trans.name.startswith("tau_"):
            continue
        for p in trans.input_places:
            place_consumers[p].append(trans)

    # Group transitions that share an input place.
    groups: dict[str, list[DPNTransition]] = {}
    for place, consumers in place_consumers.items():
        if len(consumers) > 1:
            groups[place] = consumers

    return groups


def _discriminative_variables(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    all_variables: list[str],
) -> list[str]:
    """Return variables that carry numeric data AND differ between
    positive / negative samples.

    A variable that has exactly the same value set in both groups
    cannot discriminate and would just bloat the threshold pool.
    """
    disc: list[str] = []
    for v in all_variables:
        pos_vals = {s[v] for s in positive if v in s and isinstance(s[v], (int, float))}
        neg_vals = {s[v] for s in negative if v in s and isinstance(s[v], (int, float))}
        # Need numeric values in at least one group.
        if not pos_vals and not neg_vals:
            continue
        # Variable discriminates if value sets are not identical.
        if pos_vals != neg_vals:
            disc.append(v)
    return disc


def _synthesise_guards_grammar(
    dpn: DataPetriNet,
    observations: dict[str, list[tuple[dict, dict]]],
    variables: list[str],
) -> DataPetriNet:
    """Discover guards using the SyGuS grammar + A* search.

    For each choice group (transitions sharing an input place),
    uses one-vs-rest synthesis with disjointness constraints,
    following the same grammar and search strategy as the
    region-based pipeline (PHOG-weighted A*, §4 Step 4):

      Guard → (var ≤ c) | (var > c) | (var = c)
            | Guard ∧ Guard | Guard ∨ Guard | ¬Guard | ⊤

    Transitions 1..k−1 are synthesised via A* search; the last
    transition receives ¬(φ₁ ∨ … ∨ φₖ₋₁) for an exhaustive
    partition.
    """
    phog = PHOGModel()
    smt = get_solver()
    choice_groups = _find_choice_groups(dpn)

    if not choice_groups:
        logger.info("  No choice points found — all guards are True.")
        return dpn

    # Collect all transitions that need a guard.
    guarded_transitions: set[str] = set()
    for place, group in choice_groups.items():
        if len(group) > 1:
            for t in group:
                guarded_transitions.add(t.name)

    logger.info(
        "  Guard synthesis: %d choice points, %d transitions need guards",
        len(choice_groups), len(guarded_transitions),
    )

    trans_guards: dict[str, SMTBool] = {}

    for place, group in choice_groups.items():
        t_group_start = time.perf_counter()
        # Skip groups where we have no observations.
        group_with_obs = [
            t for t in group
            if t.name in observations and observations[t.name]
        ]
        if len(group_with_obs) < 2:
            continue

        # ── Group-level feasibility pre-screen ───────────────────────
        # Pool all samples in the group and compute thresholds.
        # If no single atom can achieve >60% accuracy for any
        # one-vs-rest split, the group is likely unseparable.
        all_group_samples: list[dict[str, Any]] = []
        per_trans_samples: dict[str, list[dict[str, Any]]] = {}
        for t in group_with_obs:
            pres = [pre for pre, _ in observations.get(t.name, [])]
            per_trans_samples[t.name] = pres
            all_group_samples.extend(pres)

        skip_group = True
        if all_group_samples:
            from dpn_discovery.guard_synthesis import (
                _compute_thresholds,
                _enumerate_atomic_candidates,
            )
            group_thresholds = _compute_thresholds(all_group_samples, variables)
            group_atoms = _enumerate_atomic_candidates(variables, group_thresholds)
            for t in group_with_obs:
                positive = per_trans_samples[t.name]
                negative = [s for tn, sl in per_trans_samples.items()
                            if tn != t.name for s in sl]
                if not positive or not negative:
                    continue
                for atom in group_atoms:
                    tp = sum(1 for s in positive if atom.evaluate(s) is True)
                    tn = sum(1 for s in negative
                             if atom.evaluate(s) in (False, None))
                    score = (tp + tn) / (len(positive) + len(negative))
                    if score > 0.8:
                        skip_group = False
                        break
                if not skip_group:
                    break

        if skip_group:
            logger.info(
                "    Choice group '%s': %d transitions — skipped (no promising atoms)",
                place, len(group_with_obs),
            )
            continue

        synthesized_nodes: list[GuardNode] = []
        all_guard_nodes: list[GuardNode] = []
        consecutive_failures = 0
        # Allow more failures in larger groups before giving up,
        # but cap at 4 to avoid wasting time on unseparable data.
        max_consecutive_failures = min(4, max(3, len(group_with_obs) - 1))

        for idx, trans in enumerate(group_with_obs):
            is_last = (idx == len(group_with_obs) - 1)
            all_previous_ok = (len(synthesized_nodes) == idx)

            # ── Last transition: assign ¬(φ₁ ∨ … ∨ φₖ₋₁) ───────────
            if is_last and all_previous_ok and synthesized_nodes:
                if len(synthesized_nodes) == 1:
                    neg_node = GuardNode(
                        kind="not", children=(synthesized_nodes[0],),
                    )
                else:
                    or_node = GuardNode(
                        kind="or", children=tuple(synthesized_nodes),
                    )
                    neg_node = GuardNode(
                        kind="not", children=(or_node,),
                    )

                smt_vars = {v: smt.Real(v) for v in variables}
                trans_guards[trans.name] = neg_node.to_smt(smt_vars, smt)
                all_guard_nodes.append(neg_node)
                logger.debug(
                    "    Last-branch negation guard for '%s': %s",
                    trans.name, neg_node.pretty(),
                )
                continue

            # ── One-vs-rest with disjointness constraints ───────────
            positive: list[dict[str, Any]] = [
                pre for (pre, _) in observations.get(trans.name, [])
            ]
            negative: list[dict[str, Any]] = []
            for other in group_with_obs:
                if other.name != trans.name:
                    negative.extend(
                        pre for (pre, _) in observations.get(other.name, [])
                    )

            if not positive or not negative:
                continue

            # Restrict to variables that actually discriminate.
            disc_vars = _discriminative_variables(positive, negative, variables)
            if not disc_vars:
                continue

            guard_node = _search_guard(
                positive, negative, disc_vars, phog,
                max_candidates=500,
                disjoint_from=synthesized_nodes if synthesized_nodes else None,
            )
            logger.info(
                "    [%s] pos=%d neg=%d disc_vars=%d → %s",
                trans.name, len(positive), len(negative), len(disc_vars),
                guard_node.pretty() if guard_node else "None",
            )

            if guard_node is not None:
                consecutive_failures = 0
                synthesized_nodes.append(guard_node)
                all_guard_nodes.append(guard_node)
                smt_vars = {v: smt.Real(v) for v in variables}
                trans_guards[trans.name] = guard_node.to_smt(smt_vars, smt)
                logger.debug(
                    "    Guard for '%s': %s",
                    trans.name, guard_node.pretty(),
                )
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.debug(
                        "    Skipping rest of group '%s' after %d failures",
                        place, consecutive_failures,
                    )
                    break
                # Fallback: unconstrained guard (best-effort).
                logger.debug(
                    "    No separating guard found for '%s' — assigning ⊤",
                    trans.name,
                )

        # ── Post-synthesis partition verification ────────────────────
        if len(all_guard_nodes) >= 2:
            _verify_partition_smt(
                all_guard_nodes, variables, f"choice@{place}",
            )
        t_group_end = time.perf_counter()
        logger.info(
            "    Choice group '%s': %d transitions, %.1fs",
            place, len(group_with_obs), t_group_end - t_group_start,
        )

    # Apply guards to transitions.
    new_transitions: list[DPNTransition] = []
    for trans in dpn.transitions:
        if trans.name in trans_guards:
            new_transitions.append(DPNTransition(
                name=trans.name,
                guard=trans_guards[trans.name],
                update_rule=trans.update_rule,
                input_places=set(trans.input_places),
                output_places=set(trans.output_places),
            ))
        else:
            new_transitions.append(trans)

    return DataPetriNet(
        places=list(dpn.places),
        transitions=new_transitions,
        variables=set(dpn.variables),
        initial_marking=set(dpn.initial_marking),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Postcondition synthesis  (SyGuS enumerative abduction)
# ═══════════════════════════════════════════════════════════════════════════

def _fast_path_update(
    var: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[str, float | None] | None:
    """O(n) check for trivial update patterns before SMT.

    Returns ``("identity", None)``, ``("constant", value)``, or
    ``("increment", delta)`` when the pattern is unambiguous.
    Returns ``None`` to fall through to full abduction.
    """
    numeric: list[tuple[float, float]] = []
    for pre, post in pairs:
        pv, qv = pre.get(var), post.get(var)
        if isinstance(qv, (int, float)) and isinstance(pv, (int, float)):
            numeric.append((float(pv), float(qv)))

    if not numeric:
        return None

    # Identity: post == pre  for all pairs.
    if all(abs(p - q) < 1e-9 for p, q in numeric):
        return ("identity", None)

    # Constant: post == c  for all pairs.
    unique_posts = {round(q, 6) for _, q in numeric}
    if len(unique_posts) == 1:
        return ("constant", numeric[0][1])

    # Increment: post == pre + delta  for all pairs.
    deltas = {round(q - p, 6) for p, q in numeric}
    if len(deltas) == 1:
        d = numeric[0][1] - numeric[0][0]
        return ("identity", None) if abs(d) < 1e-9 else ("increment", d)

    return None


def _relevant_variables(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    all_variables: list[str],
) -> list[str]:
    """Return the subset of *all_variables* that carry numeric
    values in at least one observation pair.

    Passing only relevant variables to ``_get_abduct_ucl`` shrinks
    the depth-2 candidate space from O(V²) to O(R²) where
    R ≪ V in practice.
    """
    relevant: set[str] = set()
    for pre, post in pairs:
        for v in all_variables:
            if isinstance(pre.get(v), (int, float)) or isinstance(post.get(v), (int, float)):
                relevant.add(v)
    return sorted(relevant)


def _synthesise_postconditions_grammar(
    dpn: DataPetriNet,
    observations: dict[str, list[tuple[dict, dict]]],
    variables: list[str],
    use_symbolic_pruning: bool = False,
) -> DataPetriNet:
    """Discover update rules using SyGuS grammar + enumerative abduction.

    For each variable on each transition, first tries a cheap O(n)
    fast-path (identity / constant / increment).  Only falls through
    to the full GetAbductUCL search (Reynolds et al.) when the
    pattern is non-trivial.

    To keep the depth-2 candidate space manageable, only the
    variables that actually carry numeric data on a given
    transition are passed to the grammar enumerator.

    Grammar (§4 Step 5 of spec)::

        Expr → Const | Var | Expr + Expr | Const * Var + Const
    """
    smt = get_solver()

    new_transitions: list[DPNTransition] = []
    total = len(dpn.transitions)

    for t_idx, trans in enumerate(dpn.transitions, 1):
        if trans.name.startswith("tau_"):
            new_transitions.append(trans)
            continue

        pairs = observations.get(trans.name, [])
        if not pairs:
            new_transitions.append(trans)
            continue

        # Restrict grammar to variables present in the data.
        rel_vars = _relevant_variables(pairs, variables)

        # Further restrict to variables that actually *change* on
        # this transition (pre ≠ post for at least one pair).  This
        # dramatically reduces the candidate space for transitions
        # where most variables pass through unchanged.
        changing_vars: list[str] = []
        for v in rel_vars:
            for pre, post in pairs:
                pv, qv = pre.get(v), post.get(v)
                if (isinstance(pv, (int, float)) and isinstance(qv, (int, float))
                        and abs(float(pv) - float(qv)) > 1e-9):
                    changing_vars.append(v)
                    break
        # Use changing_vars for abduction grammar, but keep rel_vars
        # for the fast-path check (which needs to see all variables).
        abduct_vars = changing_vars if changing_vars else rel_vars

        logger.info(
            "  [%d/%d] %s: %d obs, %d relevant vars, %d changing vars",
            t_idx, total, trans.name, len(pairs), len(rel_vars), len(abduct_vars),
        )

        update_rule: dict[str, SMTExpr] = {}

        for var in rel_vars:
            # Skip variables without numeric post-state observations.
            has_numeric_post = any(
                isinstance(post.get(var), (int, float))
                for _, post in pairs
            )
            if not has_numeric_post:
                continue

            # ── Fast path (O(n)) ─────────────────────────────────────
            fp = _fast_path_update(var, pairs)
            if fp is not None:
                kind, val = fp
                if kind == "identity":
                    continue  # no explicit identity update
                if kind == "constant" and val is not None:
                    update_rule[var] = smt.RealVal(val)
                    continue
                if kind == "increment" and val is not None:
                    update_rule[var] = smt.Add(smt.Real(var), smt.RealVal(val))
                    continue
                continue

            # ── Full SyGuS abduction (expensive) ─────────────────────
            t_abd_start = time.perf_counter()
            abduct = _get_abduct_ucl(
                var,
                pairs,
                abduct_vars,
                guard_formula=trans.guard,
                use_symbolic_pruning=use_symbolic_pruning,
            )
            t_abd_end = time.perf_counter()
            if abduct is not None:
                smt_vars = {v: smt.Real(v) for v in abduct_vars}
                update_rule[var] = abduct.to_smt(smt_vars, smt)
                logger.debug(
                    "    abduction for %s: %.1fs → %s",
                    var, t_abd_end - t_abd_start, abduct.pretty(),
                )
            elif t_abd_end - t_abd_start > 1.0:
                logger.debug(
                    "    abduction for %s: %.1fs → None",
                    var, t_abd_end - t_abd_start,
                )

        new_transitions.append(DPNTransition(
            name=trans.name,
            guard=trans.guard,
            update_rule=update_rule if update_rule else None,
            input_places=set(trans.input_places),
            output_places=set(trans.output_places),
        ))

    return DataPetriNet(
        places=list(dpn.places),
        transitions=new_transitions,
        variables=set(dpn.variables),
        initial_marking=set(dpn.initial_marking),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Cleanup — remove orphaned places
# ═══════════════════════════════════════════════════════════════════════════

def _remove_orphaned_places(dpn: DataPetriNet) -> DataPetriNet:
    """Remove places that are not connected to any transition."""
    used_places: set[str] = set()
    for trans in dpn.transitions:
        used_places |= trans.input_places
        used_places |= trans.output_places
    # Always keep initial marking places.
    used_places |= dpn.initial_marking

    new_places = [p for p in dpn.places if p.name in used_places]
    return DataPetriNet(
        places=new_places,
        transitions=list(dpn.transitions),
        variables=set(dpn.variables),
        initial_marking=set(dpn.initial_marking),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Top-level pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_ilp_pipeline(
    log: EventLog,
    alpha: float = 1.0,
) -> DataPetriNet:
    """Discover a Data Petri Net using ILP miner + grammar-based annotations.

    Parameters
    ----------
    log : EventLog
        The event log to mine.
    alpha : float
        ILP miner noise threshold (0..1).

    Returns
    -------
    DataPetriNet
        The discovered DPN with SyGuS-grammar guards and updates.
    """
    _, dpn = run_ilp_pipeline_full(log, alpha=alpha)
    return dpn


def run_ilp_pipeline_full(
    log: EventLog,
    alpha: float = 1.0,
) -> tuple[tuple[PetriNet, Marking, Marking], DataPetriNet]:
    """Full ILP pipeline returning intermediate artefacts.

    Returns
    -------
    tuple
        ``((pm4py_net, im, fm), dpn)``
    """
    variables = sorted(log.variables)

    # -- Step 1: Discover control flow via ILP miner ----------------------
    logger.info("Step 1  >  ILP miner: discovering control flow")
    df = _eventlog_to_pm4py_log(log)
    net, im, fm = _discover_control_flow(df, alpha=alpha)

    # -- Step 2: Convert pm4py net to our DPN format ----------------------
    logger.info("Step 2  >  Converting pm4py net to DataPetriNet")
    dpn = _pm4py_to_dpn(net, im, fm, set(variables))
    logger.info(
        "         %d places, %d transitions (before tau removal)",
        len(dpn.places), len(dpn.transitions),
    )

    # -- Step 2b: Remove ILP-miner tau transitions ------------------------
    logger.info("Step 2b >  Removing ILP-miner tau transitions")
    dpn = _remove_tau_transitions(dpn)

    # -- Step 3: Replay log and collect data observations -----------------
    logger.info("Step 3  >  Replaying log to collect data observations")
    observations = _replay_and_collect(dpn, log)
    total_obs = sum(len(v) for v in observations.values())
    logger.info(
        "         %d transitions with observations, %d total data points",
        len(observations), total_obs,
    )

    # -- Step 4: Grammar-based guard synthesis (A* + PHOG) ----------------
    logger.info("Step 4  >  Grammar-based guard synthesis (A* + PHOG)")
    t0 = time.perf_counter()
    dpn = _synthesise_guards_grammar(dpn, observations, variables)
    t1 = time.perf_counter()

    n_guarded = sum(1 for t in dpn.transitions if t.guard is not None)
    logger.info("         %d/%d transitions have guards (%.1fs)", n_guarded, len(dpn.transitions), t1 - t0)

    # -- Step 5: Grammar-based postcondition synthesis (abduction) --------
    logger.info("Step 5  >  Grammar-based postcondition synthesis (enumerative abduction)")
    t2 = time.perf_counter()
    dpn = _synthesise_postconditions_grammar(dpn, observations, variables)
    t3 = time.perf_counter()

    n_updated = sum(1 for t in dpn.transitions if t.update_rule)
    logger.info("         %d/%d transitions have update rules (%.1fs)", n_updated, len(dpn.transitions), t3 - t2)

    # -- Step 6: Cleanup --------------------------------------------------
    dpn = _remove_orphaned_places(dpn)

    return (net, im, fm), dpn


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _transition_activity(trans: DPNTransition) -> str:
    """Extract the activity label from a DPN transition name."""
    parts = trans.name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[1:-1])
    return trans.name
