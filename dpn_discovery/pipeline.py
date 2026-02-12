"""
Pipeline Orchestrator — chains the 6 discovery steps.

    1. Log Preprocessing
    2. PTA Construction
    3. State Merging       (Walkinshaw et al.)
    4. Guard Synthesis     (PHOG-accelerated SAT, Lee et al.)
    5. Postcondition Synth (Abduction, Reynolds et al.)
    6. EFSM → DPN Mapping
"""

from __future__ import annotations

import logging
from pathlib import Path

from dpn_discovery.dpn_transform import dpn_to_pnml, efsm_to_dpn, verify_dpn
from dpn_discovery.guard_synthesis import synthesise_guards
from dpn_discovery.models import DataPetriNet, EFSM, EventLog
from dpn_discovery.postcondition_synthesis import synthesise_postconditions
from dpn_discovery.preprocessing import load_event_log
from dpn_discovery.pta import build_pta
from dpn_discovery.state_merging import run_state_merging

logger = logging.getLogger(__name__)


def run_pipeline(log_source: str | Path | EventLog) -> DataPetriNet:
    """Execute the full DPN-discovery pipeline.

    Parameters
    ----------
    log_source : str | Path | EventLog
        Either a path to an event-log file (.xes, .csv, .json)
        or an already-parsed ``EventLog`` instance.

    Returns
    -------
    DataPetriNet
        The discovered Data-Aware Petri Net.
    """
    # ── Step 1: Preprocessing ────────────────────────────────────────────
    if isinstance(log_source, EventLog):
        logger.info("Step 1  ▸  Using pre-loaded EventLog")
        log = log_source
    else:
        logger.info("Step 1  ▸  Preprocessing event log from %s", log_source)
        log = load_event_log(log_source)
    logger.info(
        "         Activities Σ = %s  |  Variables V = %s  |  Traces = %d",
        log.activities,
        log.variables,
        len(log.traces),
    )

    # ── Step 2: PTA Construction ─────────────────────────────────────────
    logger.info("Step 2  ▸  Building Prefix Tree Acceptor")
    pta: EFSM = build_pta(log)
    logger.info(
        "         States = %d  |  Transitions = %d",
        len(pta.states),
        len(pta.transitions),
    )

    # ── Step 3: State Merging  (Walkinshaw et al.) ───────────────────────
    logger.info("Step 3  ▸  Data-driven state merging")
    merged_efsm: EFSM = run_state_merging(pta)
    logger.info(
        "         States = %d → %d  |  Transitions = %d → %d",
        len(pta.states),
        len(merged_efsm.states),
        len(pta.transitions),
        len(merged_efsm.transitions),
    )

    # ── Step 4: Guard Synthesis  (Lee et al.) ────────────────────────────
    logger.info("Step 4  ▸  Synthesising guards (PHOG-accelerated SAT)")
    guarded_efsm: EFSM = synthesise_guards(merged_efsm)
    _log_guards(guarded_efsm)

    # ── Step 5: Postcondition Synthesis  (Reynolds et al.) ───────────────
    logger.info("Step 5  ▸  Synthesising postconditions (abduction)")
    full_efsm: EFSM = synthesise_postconditions(guarded_efsm)
    _log_updates(full_efsm)

    # ── Step 6: EFSM → DPN ──────────────────────────────────────────────
    logger.info("Step 6  ▸  Transforming EFSM → Data Petri Net")
    dpn: DataPetriNet = efsm_to_dpn(full_efsm)
    logger.info(
        "         Places = %d  |  Transitions = %d",
        len(dpn.places),
        len(dpn.transitions),
    )

    # ── Verification ────────────────────────────────────────────────────
    ok = verify_dpn(dpn, log)
    if ok:
        logger.info("✓  Log replay verification PASSED")
    else:
        logger.warning("✗  Log replay verification FAILED — DPN may be incomplete")

    return dpn



def run_pipeline_full(
    log_source: str | Path | EventLog,
    case_sampling_ratio: float = 1.0,
) -> tuple[EFSM, EFSM, DataPetriNet]:
    """Execute the full pipeline and return intermediate artefacts.

    Returns
    -------
    tuple[EFSM, EFSM, DataPetriNet]
        ``(pta, final_efsm, dpn)`` — the raw PTA, the fully
        annotated EFSM (after merging + guards + updates), and the
        resulting Data Petri Net.
    """
    # ── Step 1: Preprocessing ────────────────────────────────────────────
    if isinstance(log_source, EventLog):
        log = log_source
    else:
        log = load_event_log(log_source)

    if case_sampling_ratio < 1.0:
        log = sample_event_log(log, case_sampling_ratio)

    # ── Step 2: PTA Construction  (Walkinshaw et al.) 
    pta: EFSM = build_pta(log)

    # ── Step 3: State Merging (Walkinshaw et al.)
    merged_efsm: EFSM = run_state_merging(pta)

    # ── Step 4: Guard Synthesis (Lee et al.)
    guarded_efsm: EFSM = synthesise_guards(merged_efsm)

    # ── Step 5: Postcondition Synthesis (Reynolds et al.)
    full_efsm: EFSM = synthesise_postconditions(guarded_efsm)

    # ── Step 6: EFSM → DPN 
    dpn: DataPetriNet = efsm_to_dpn(full_efsm)

    return pta, full_efsm, dpn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_guards(efsm: EFSM) -> None:
    for t in efsm.transitions:
        if t.guard_formula is not None:
            logger.info(
                "         %s → %s [%s]  guard = %s",
                t.source_id,
                t.target_id,
                t.activity,
                t.guard_formula,
            )


def _log_updates(efsm: EFSM) -> None:
    for t in efsm.transitions:
        if t.update_rule:
            for var, expr in t.update_rule.items():
                logger.info(
                    "         %s → %s [%s]  %s := %s",
                    t.source_id,
                    t.target_id,
                    t.activity,
                    var,
                    expr,
                )


def sample_event_log(log: EventLog, case_sampling_ratio: float) -> EventLog:
    """Sample a fraction of cases from the event log."""
    if case_sampling_ratio >= 1.0:
        return log

    num_cases = int(len(log.traces) * case_sampling_ratio)
    sampled_traces = log.traces[:num_cases]
    return EventLog(traces=sampled_traces, activities=log.activities, variables=log.variables)