"""
Pipeline Orchestrator — chains the 6 discovery steps.

    1. Log Preprocessing
    2. Classifier Training  (MINT only)
    3. PTA Construction
    4. State Merging       (Walkinshaw et al.)
    5. Guard Synthesis     (PHOG-accelerated SAT, Lee et al.)
    6. Postcondition Synth (Abduction, Reynolds et al.)
    7. EFSM → DPN Mapping
"""

from __future__ import annotations

import logging
from pathlib import Path

from dpn_discovery.classifiers import (
    ClassifierAlgorithm,
    Classifiers,
    infer_classifiers,
    prepare_data_traces,
)
from dpn_discovery.dpn_transform import dpn_to_pnml, efsm_to_dpn, verify_dpn
from dpn_discovery.guard_synthesis import synthesise_guards
from dpn_discovery.models import DataPetriNet, EFSM, EventLog, MergeStrategy
from dpn_discovery.postcondition_synthesis import synthesise_postconditions
from dpn_discovery.preprocessing import load_event_log
from dpn_discovery.pta import build_pta
from dpn_discovery.state_merging import run_state_merging

logger = logging.getLogger(__name__)


def run_pipeline(
    log_source: str | Path | EventLog,
    use_symbolic_pruning: bool = False,
    merge_strategy: MergeStrategy = MergeStrategy.BLUE_FRINGE,
    classifier_algorithm: ClassifierAlgorithm = ClassifierAlgorithm.DECISION_TREE,
    min_merge_score: int = 0,
) -> DataPetriNet:
    """Execute the full DPN-discovery pipeline.

    Parameters
    ----------
    log_source : str | Path | EventLog
        Either a path to an event-log file (.xes, .csv, .json)
        or an already-parsed ``EventLog`` instance.
    use_symbolic_pruning : bool
        When ``True``, use guard formulas to symbolically prune
        equivalent postcondition candidates (Step 5).  Default
        is ``False``.
    merge_strategy : MergeStrategy
        ``NONE`` — skip merging (return PTA as-is).
        ``BLUE_FRINGE`` — control-flow-only (Walkinshaw Alg. 1 & 2).
        ``MINT`` — data-aware with classifiers (Walkinshaw Alg. 3).
    classifier_algorithm : ClassifierAlgorithm
        Which scikit-learn algorithm to use for MINT classifiers.
        Ignored when *merge_strategy* is not ``MINT``.
    min_merge_score : int
        Minimum EDSM score *k* for a merge to be attempted.

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

    ordered_vars = sorted(log.variables)

    # ── Step 2: Classifier Training (MINT only) ──────────────────────────
    classifiers: Classifiers | None = None
    if merge_strategy is MergeStrategy.MINT:
        logger.info(
            "Step 2  ▸  Training classifiers (algorithm=%s)",
            classifier_algorithm.name,
        )
        data_traces = prepare_data_traces(log, log.variables)
        classifiers = infer_classifiers(
            data_traces, ordered_vars, algorithm=classifier_algorithm,
        )
        logger.info("         Classifiers trained for %d labels", len(classifiers))
    else:
        logger.info("Step 2  ▸  Classifier training skipped (strategy=%s)", merge_strategy.name)

    # ── Step 3: PTA Construction ─────────────────────────────────────────
    logger.info("Step 3  ▸  Building Prefix Tree Acceptor")
    pta: EFSM = build_pta(
        log,
        classifiers=classifiers,
        variables=ordered_vars if classifiers else None,
    )
    logger.info(
        "         States = %d  |  Transitions = %d",
        len(pta.states),
        len(pta.transitions),
    )

    # ── Step 4: State Merging  (Walkinshaw et al.) ───────────────────────
    logger.info("Step 4  ▸  State merging (strategy=%s)", merge_strategy.name)
    merged_efsm: EFSM = run_state_merging(
        pta,
        strategy=merge_strategy,
        classifiers=classifiers,
        variables=ordered_vars if classifiers else None,
        k=min_merge_score,
    )
    logger.info(
        "         States = %d → %d  |  Transitions = %d → %d",
        len(pta.states),
        len(merged_efsm.states),
        len(pta.transitions),
        len(merged_efsm.transitions),
    )

    # ── Step 5: Guard Synthesis  (Lee et al.) ────────────────────────────
    logger.info("Step 5  ▸  Synthesising guards (PHOG-accelerated SAT)")
    guarded_efsm: EFSM = synthesise_guards(merged_efsm)
    _log_guards(guarded_efsm)

    # ── Step 6: Postcondition Synthesis  (Reynolds et al.) ───────────────
    logger.info("Step 6  ▸  Synthesising postconditions (abduction)")
    full_efsm: EFSM = synthesise_postconditions(
        guarded_efsm, use_symbolic_pruning=use_symbolic_pruning,
    )
    _log_updates(full_efsm)

    # ── Step 7: EFSM → DPN ──────────────────────────────────────────────
    logger.info("Step 7  ▸  Transforming EFSM → Data Petri Net")
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
    use_symbolic_pruning: bool = False,
    merge_strategy: MergeStrategy = MergeStrategy.BLUE_FRINGE,
    classifier_algorithm: ClassifierAlgorithm = ClassifierAlgorithm.DECISION_TREE,
    min_merge_score: int = 0,
) -> tuple[EFSM, EFSM, DataPetriNet]:
    """Execute the full pipeline and return intermediate artefacts.

    Parameters
    ----------
    log_source : str | Path | EventLog
        Either a path to an event-log file or a parsed ``EventLog``.
    case_sampling_ratio : float
        Fraction of traces to sample (1.0 = all).
    use_symbolic_pruning : bool
        When ``True``, use guard formulas to symbolically prune
        equivalent postcondition candidates (Step 5).
    merge_strategy : MergeStrategy
        ``NONE`` — skip merging.
        ``BLUE_FRINGE`` — control-flow-only.
        ``MINT`` — data-aware with classifiers.
    classifier_algorithm : ClassifierAlgorithm
        Which scikit-learn algorithm to use for MINT classifiers.
    min_merge_score : int
        Minimum EDSM score *k* for a merge to be attempted.

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

    ordered_vars = sorted(log.variables)

    # ── Step 2: Classifier Training (MINT only) ──────────────────────────
    classifiers: Classifiers | None = None
    if merge_strategy is MergeStrategy.MINT:
        data_traces = prepare_data_traces(log, log.variables)
        classifiers = infer_classifiers(
            data_traces, ordered_vars, algorithm=classifier_algorithm,
        )

    # ── Step 3: PTA Construction ─────────────────────────────────────────
    pta: EFSM = build_pta(
        log,
        classifiers=classifiers,
        variables=ordered_vars if classifiers else None,
    )

    # ── Step 4: State Merging (Walkinshaw et al.) ────────────────────────
    merged_efsm: EFSM = run_state_merging(
        pta,
        strategy=merge_strategy,
        classifiers=classifiers,
        variables=ordered_vars if classifiers else None,
        k=min_merge_score,
    )

    # ── Step 5: Guard Synthesis (Lee et al.) ─────────────────────────────
    guarded_efsm: EFSM = synthesise_guards(merged_efsm)

    # ── Step 6: Postcondition Synthesis (Reynolds et al.) ────────────────
    full_efsm: EFSM = synthesise_postconditions(
        guarded_efsm, use_symbolic_pruning=use_symbolic_pruning,
    )

    # ── Step 7: EFSM → DPN ──────────────────────────────────────────────
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