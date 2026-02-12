"""
End-to-end tests for the DPN Discovery Pipeline.

Uses the loan-approval running example from the pipeline diagram
(dpn_discovery.drawio) as the ground-truth reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import z3

from dpn_discovery.dpn_transform import dpn_to_pnml, efsm_to_dpn, verify_dpn
from dpn_discovery.guard_synthesis import synthesise_guards
from dpn_discovery.models import EFSM, DataPetriNet, EventLog, Transition
from dpn_discovery.pipeline import run_pipeline
from dpn_discovery.postcondition_synthesis import synthesise_postconditions
from dpn_discovery.preprocessing import load_event_log, parse_event_log
from dpn_discovery.pta import build_pta
from dpn_discovery.state_merging import run_state_merging


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOAN_LOG = DATA_DIR / "loan_example.json"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Preprocessing
# ═══════════════════════════════════════════════════════════════════════════


class TestPreprocessing:
    def test_load_log(self) -> None:
        log = load_event_log(LOAN_LOG)
        assert len(log.traces) == 8

    def test_activity_extraction(self) -> None:
        log = load_event_log(LOAN_LOG)
        assert log.activities == {"Submit", "AutoApprove", "Review", "Approve", "Reject"}

    def test_variable_extraction(self) -> None:
        log = load_event_log(LOAN_LOG)
        assert log.variables == {"amount", "counter"}

    def test_events_populated(self) -> None:
        log = load_event_log(LOAN_LOG)
        first_trace = log.traces[0]
        assert len(first_trace.events) == 2
        assert first_trace.events[0].activity == "Submit"
        assert first_trace.events[0].payload["amount"] == 500


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: PTA Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestPTA:
    def test_pta_structure(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        # Root + at least the branching states.
        assert "q0" in pta.states
        assert len(pta.states) >= 4
        assert len(pta.transitions) >= 4

    def test_pta_initial_state(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        assert pta.initial_state == "q0"

    def test_data_on_edges(self) -> None:
        """Crucial: data payloads must be stored on edges (spec §4 Step 2)."""
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        for t in pta.transitions:
            assert len(t.data_samples) > 0, f"Edge {t.source_id}→{t.target_id} [{t.activity}] has no data"

    def test_pta_alphabet(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        assert pta.alphabet == {"Submit", "AutoApprove", "Review", "Approve", "Reject"}


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: State Merging
# ═══════════════════════════════════════════════════════════════════════════


class TestStateMerging:
    def test_merging_preserves_initial(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        assert merged.initial_state == "q0"

    def test_merging_does_not_increase_states(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        assert len(merged.states) <= len(pta.states)

    def test_merged_data_samples_preserved(self) -> None:
        """Data samples must be unioned, not dropped (spec §4 Step 3)."""
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        total_samples_pta = sum(len(t.data_samples) for t in pta.transitions)
        total_samples_merged = sum(len(t.data_samples) for t in merged.transitions)
        assert total_samples_merged >= total_samples_pta


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Guard Synthesis
# ═══════════════════════════════════════════════════════════════════════════


class TestGuardSynthesis:
    def test_all_transitions_get_guard(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        guarded = synthesise_guards(merged)
        for t in guarded.transitions:
            assert t.guard_formula is not None, (
                f"Transition {t.source_id}→{t.target_id} [{t.activity}] has no guard"
            )

    def test_guard_is_z3_bool(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        guarded = synthesise_guards(merged)
        for t in guarded.transitions:
            assert isinstance(t.guard_formula, z3.BoolRef)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Postcondition Synthesis
# ═══════════════════════════════════════════════════════════════════════════


class TestPostconditionSynthesis:
    def test_update_rules_discovered(self) -> None:
        """At least some transitions should have update rules."""
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        guarded = synthesise_guards(merged)
        full = synthesise_postconditions(guarded)
        has_update = any(t.update_rule for t in full.transitions)
        assert has_update, "No update rules discovered — expected counter := counter + 1"


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: EFSM → DPN
# ═══════════════════════════════════════════════════════════════════════════


class TestDPNTransform:
    def test_places_match_states(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        guarded = synthesise_guards(merged)
        full = synthesise_postconditions(guarded)
        dpn = efsm_to_dpn(full)
        assert len(dpn.places) == len(full.states)

    def test_transitions_match(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        guarded = synthesise_guards(merged)
        full = synthesise_postconditions(guarded)
        dpn = efsm_to_dpn(full)
        assert len(dpn.transitions) == len(full.transitions)

    def test_initial_marking(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        guarded = synthesise_guards(merged)
        full = synthesise_postconditions(guarded)
        dpn = efsm_to_dpn(full)
        assert "p_q0" in dpn.initial_marking

    def test_pnml_serialisation(self) -> None:
        log = load_event_log(LOAN_LOG)
        pta = build_pta(log)
        merged = run_state_merging(pta)
        guarded = synthesise_guards(merged)
        full = synthesise_postconditions(guarded)
        dpn = efsm_to_dpn(full)
        pnml = dpn_to_pnml(dpn)
        assert "<pnml>" in pnml
        assert "<place" in pnml
        assert "<transition" in pnml
        assert "<arc" in pnml


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_end_to_end(self) -> None:
        dpn = run_pipeline(LOAN_LOG)
        assert isinstance(dpn, DataPetriNet)
        assert len(dpn.places) > 0
        assert len(dpn.transitions) > 0
        assert dpn.variables == {"amount", "counter"}
