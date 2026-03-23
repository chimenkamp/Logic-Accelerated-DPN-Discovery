"""
Pipeline Orchestrator -- chains the 7 discovery steps.

    1. Log Preprocessing
    2. Classifier Training  (MINT only)
    3. PTA Construction
    4. State Merging       (Walkinshaw et al.)
    5. Guard Synthesis     (PHOG-accelerated SAT, Lee et al.)
    6. Postcondition Synth (Abduction, Reynolds et al.)
    7. EFSM -> DPN Mapping  (Theory of Regions, Cortadella et al.)
"""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Optional

import pm4py

from dpn_discovery.classifiers import (
    ClassifierAlgorithm,
    Classifiers,
    infer_classifiers,
    prepare_data_traces,
)
from dpn_discovery.dpn_transform import dpn_to_pnml, efsm_to_dpn, reduce_dpn, verify_dpn
from dpn_discovery.guard_synthesis import synthesise_guards
from dpn_discovery.models import DataPetriNet, EFSM, EventLog, MergeStrategy
from dpn_discovery.postcondition_synthesis import synthesise_postconditions
from dpn_discovery.preprocessing import load_event_log
from dpn_discovery.pta import build_pta
from dpn_discovery.state_merging import bisimulation_reduction, run_state_merging

logger = logging.getLogger(__name__)


def run_pipeline(
    log_source: str | Path | EventLog,
    use_symbolic_pruning: bool = False,
    merge_strategy: MergeStrategy = MergeStrategy.MINT,
    classifier_algorithm: ClassifierAlgorithm = ClassifierAlgorithm.DECISION_TREE,
    min_merge_score: int = 0,
    use_bisimulation: bool = True,
    use_dpn_reduction: bool = True,
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
        ``NONE`` -- skip merging (return PTA as-is).
        ``BLUE_FRINGE`` -- control-flow-only (Walkinshaw Alg. 1 & 2).
        ``MINT`` -- data-aware with classifiers (Walkinshaw Alg. 3).
    classifier_algorithm : ClassifierAlgorithm
        Which scikit-learn algorithm to use for MINT classifiers.
        Ignored when *merge_strategy* is not ``MINT``.
    min_merge_score : int
        Minimum EDSM score *k* for a merge to be attempted.
    use_bisimulation : bool
        When ``True``, apply bisimulation-based state reduction
        after merging (before guard synthesis).  Reduces EFSM
        states without introducing assumptions.  Default ``True``.
    use_dpn_reduction : bool
        When ``True``, apply replay-based place fusion and
        transition collapsing after the DPN is constructed.
        Default ``True``.

    Returns
    -------
    DataPetriNet
        The discovered Data-Aware Petri Net.
    """
    # -- Step 1: Preprocessing --------------------------------------------
    if isinstance(log_source, EventLog):
        logger.info("Step 1  >  Using pre-loaded EventLog")
        log = log_source
    else:
        logger.info("Step 1  >  Preprocessing event log from %s", log_source)
        log = load_event_log(log_source)
    logger.info(
        "         Activities E = %s  |  Variables V = %s  |  Traces = %d",
        log.activities,
        log.variables,
        len(log.traces),
    )

    ordered_vars = sorted(log.variables)

    # -- Step 2: Classifier Training (MINT only) --------------------------
    classifiers: Classifiers | None = None
    if merge_strategy is MergeStrategy.MINT:
        logger.info(
            "Step 2  >  Training classifiers (algorithm=%s)",
            classifier_algorithm.name,
        )
        data_traces = prepare_data_traces(log, log.variables)
        classifiers = infer_classifiers(
            data_traces, ordered_vars, algorithm=classifier_algorithm,
        )
        logger.info("         Classifiers trained for %d labels", len(classifiers))
    else:
        logger.info("Step 2  >  Classifier training skipped (strategy=%s)", merge_strategy.name)

    # -- Step 3: PTA Construction -----------------------------------------
    logger.info("Step 3  >  Building Prefix Tree Acceptor")
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

    # -- Step 4: State Merging  (Walkinshaw et al.) -----------------------
    logger.info("Step 4  >  State merging (strategy=%s)", merge_strategy.name)
    merged_efsm: EFSM = run_state_merging(
        pta,
        strategy=merge_strategy,
        classifiers=classifiers,
        variables=ordered_vars if classifiers else None,
        k=min_merge_score,
    )
    logger.info(
        "         States = %d -> %d  |  Transitions = %d -> %d",
        len(pta.states),
        len(merged_efsm.states),
        len(pta.transitions),
        len(merged_efsm.transitions),
    )

    # -- Step 4b: Bisimulation Reduction (optional) -----------------------
    if use_bisimulation:
        logger.info("Step 4b >  Bisimulation-based state reduction")
        pre_bisim = len(merged_efsm.states)
        merged_efsm = bisimulation_reduction(merged_efsm)
        logger.info(
            "         States = %d -> %d  |  Transitions = %d",
            pre_bisim, len(merged_efsm.states), len(merged_efsm.transitions),
        )

    # -- Step 5: Guard Synthesis  (Lee et al.) ----------------------------
    logger.info("Step 5  >  Synthesising guards (PHOG-accelerated SAT)")
    guarded_efsm: EFSM = synthesise_guards(merged_efsm)
    _log_guards(guarded_efsm)

    # -- Step 6: Postcondition Synthesis  (Reynolds et al.) ---------------
    logger.info("Step 6  >  Synthesising postconditions (abduction)")
    full_efsm: EFSM = synthesise_postconditions(
        guarded_efsm, use_symbolic_pruning=use_symbolic_pruning,
    )
    _log_updates(full_efsm)

    # -- Step 7: EFSM -> DPN  (Cortadella et al. S4 Fig. 10) --------------
    #   Region-based Petri Net synthesis: theory of regions derives
    #   places from subsets of EFSM states with uniform event crossing
    #   behaviour (Def 2.2).  The irredundant cover (S4.2, Thm 3.5)
    #   yields a place-irredundant net whose RG is bisimilar to the TS.
    logger.info("Step 7  >  Region-based EFSM -> DPN (Cortadella et al. S4)")
    dpn: DataPetriNet = efsm_to_dpn(full_efsm)
    logger.info(
        "         Places = %d  |  Transitions = %d",
        len(dpn.places),
        len(dpn.transitions),
    )

    # -- Step 7b: DPN Reduction (optional) --------------------------------
    #   Transition collapsing may still re-merge split labels.
    if use_dpn_reduction:
        logger.info("Step 7b >  Post-synthesis DPN reduction (transition collapse)")
        pre_p, pre_t = len(dpn.places), len(dpn.transitions)
        dpn = reduce_dpn(dpn, log)
        logger.info(
            "         Places = %d -> %d  |  Transitions = %d -> %d",
            pre_p, len(dpn.places), pre_t, len(dpn.transitions),
        )

    # -- Verification ----------------------------------------------------
    ok = verify_dpn(dpn, log)
    if ok:
        logger.info("[ok]  Log replay verification PASSED")
    else:
        logger.warning("[FAIL]  Log replay verification FAILED -- DPN may be incomplete")

    return dpn



def run_pipeline_full(
    log_source: str | Path | EventLog,
    use_symbolic_pruning: bool = False,
    merge_strategy: MergeStrategy = MergeStrategy.MINT,
    classifier_algorithm: ClassifierAlgorithm = ClassifierAlgorithm.DECISION_TREE,
    min_merge_score: int = 0,
    case_sampling_ratio: Optional[float] = None,
    use_bisimulation: bool = True,
    use_dpn_reduction: bool = True,
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
        ``NONE`` -- skip merging.
        ``BLUE_FRINGE`` -- control-flow-only.
        ``MINT`` -- data-aware with classifiers.
    classifier_algorithm : ClassifierAlgorithm
        Which scikit-learn algorithm to use for MINT classifiers.
    min_merge_score : int
        Minimum EDSM score *k* for a merge to be attempted.
    use_bisimulation : bool
        When ``True``, apply bisimulation-based state reduction
        after merging.  Default ``True``.
    use_dpn_reduction : bool
        When ``True``, apply replay-based DPN place fusion and
        transition collapsing.  Default ``True``.

    Returns
    -------
    tuple[EFSM, EFSM, DataPetriNet]
        ``(pta, final_efsm, dpn)`` -- the raw PTA, the fully
        annotated EFSM (after merging + guards + updates), and the
        resulting Data Petri Net.
    """
    # -- Step 1: Preprocessing --------------------------------------------
    if isinstance(log_source, EventLog):
        logger.info("Step 1  >  Using pre-loaded EventLog")
        log = log_source
    else:
        logger.info("Step 1  >  Preprocessing event log from %s", log_source)
        log = load_event_log(log_source)

    if case_sampling_ratio:
        original_size = len(log.traces)
        log = log.sample(case_sampling_ratio)
        new_size = len(log.traces)
        logger.info("         Sampling: %d -> %d traces (%.1f%%)", original_size, new_size, (new_size / original_size) * 100)

    logger.info("         Activities = %s  |  Variables = %s  |  Traces = %d",
                log.activities, log.variables, len(log.traces))

    ordered_vars = sorted(log.variables)

    # -- Step 2: Classifier Training (MINT only) --------------------------
    classifiers: Classifiers | None = None
    if merge_strategy is MergeStrategy.MINT:
        logger.info("Step 2  >  Training classifiers (algorithm=%s)", classifier_algorithm.name)
        data_traces = prepare_data_traces(log, log.variables)
        classifiers = infer_classifiers(
            data_traces, ordered_vars, algorithm=classifier_algorithm,
        )
        logger.info("         Classifiers trained for %d labels", len(classifiers))
    else:
        logger.info("Step 2  >  Classifier training skipped (strategy=%s)", merge_strategy.name)

    # -- Step 3: PTA Construction -----------------------------------------
    logger.info("Step 3  >  Building Prefix Tree Acceptor")
    pta: EFSM = build_pta(
        log,
        classifiers=classifiers,
        variables=ordered_vars if classifiers else None,
    )
    logger.info("         States = %d  |  Transitions = %d",
                len(pta.states), len(pta.transitions))

    # -- Step 4: State Merging (Walkinshaw et al.) ------------------------
    logger.info("Step 4  >  State merging (strategy=%s)", merge_strategy.name)
    merged_efsm: EFSM = run_state_merging(
        pta,
        strategy=merge_strategy,
        classifiers=classifiers,
        variables=ordered_vars if classifiers else None,
        k=min_merge_score,
    )
    logger.info("         States = %d -> %d  |  Transitions = %d -> %d",
                len(pta.states), len(merged_efsm.states),
                len(pta.transitions), len(merged_efsm.transitions))

    # -- Step 4b: Bisimulation Reduction (optional) -----------------------
    if use_bisimulation:
        logger.info("Step 4b >  Bisimulation-based state reduction")
        pre_bisim = len(merged_efsm.states)
        merged_efsm = bisimulation_reduction(merged_efsm)
        logger.info("         States = %d -> %d  |  Transitions = %d",
                    pre_bisim, len(merged_efsm.states), len(merged_efsm.transitions))

    # -- Step 5: Guard Synthesis (Lee et al.) -----------------------------
    logger.info("Step 5  >  Synthesising guards (PHOG-accelerated SAT)")
    guarded_efsm: EFSM = synthesise_guards(merged_efsm)
    logger.info("         Guards synthesised")

    # -- Step 6: Postcondition Synthesis (Reynolds et al.) ----------------
    logger.info("Step 6  >  Synthesising postconditions (abduction)")
    full_efsm: EFSM = synthesise_postconditions(
        guarded_efsm, use_symbolic_pruning=use_symbolic_pruning,
    )
    logger.info("         Postconditions synthesised")

    # -- Step 7: EFSM -> DPN  (Cortadella et al. S4 Fig. 10) --------------
    logger.info("Step 7  >  Region-based EFSM -> DPN (Cortadella et al. S4)")
    dpn: DataPetriNet = efsm_to_dpn(full_efsm)
    logger.info("         Places = %d  |  Transitions = %d",
                len(dpn.places), len(dpn.transitions))

    # -- Step 7b: DPN Reduction (optional) --------------------------------
    if use_dpn_reduction:
        logger.info("Step 7b >  Post-synthesis DPN reduction (transition collapse)")
        pre_p, pre_t = len(dpn.places), len(dpn.transitions)
        dpn = reduce_dpn(dpn, log)
        logger.info("         Places = %d -> %d  |  Transitions = %d -> %d",
                    pre_p, len(dpn.places), pre_t, len(dpn.transitions))

    return pta, full_efsm, dpn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_guards(efsm: EFSM) -> None:
    for t in efsm.transitions:
        if t.guard_formula is not None:
            logger.info(
                "         %s -> %s [%s]  guard = %s",
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
                    "         %s -> %s [%s]  %s := %s",
                    t.source_id,
                    t.target_id,
                    t.activity,
                    var,
                    expr,
                )
