"""
Root entry point for the DPN Discovery Pipeline.

Runs all three test cases (loop, loan_application, fibonacci),
saves the visualizations to output/<case>/, and prints the
discovered guards and update rules for manual comparison
against the reference petri-net.png in each data folder.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dpn_discovery.dpn_transform import dpn_to_pnml
from dpn_discovery.models import EFSM, DataPetriNet
from dpn_discovery.pipeline import run_pipeline_full
from dpn_discovery.visualization import DPNVisualizer, VisualizerSettings

DATAPATH = Path(__file__).resolve().parent / "data"


@dataclass
class TestCase:
    name: str
    csv_path: Path
    ref_png: Path


TEST_CASES: list[TestCase] = [
    TestCase("loop",             DATAPATH / "loop/data.csv",             DATAPATH / "loop/petri-net.png"),
    TestCase("loan_application", DATAPATH / "loan_application/data.csv", DATAPATH / "loan_application/petri-net.png"),
    TestCase("fibonacci",        DATAPATH / "fibonacci/data.csv",        DATAPATH / "fibonacci/petri-net.png"),
]


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def _print_model_summary(name: str, pta: EFSM, efsm: EFSM, dpn: DataPetriNet) -> None:
    """Print a human-readable summary of the discovered model."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {name.upper()} — Model Summary")
    print(sep)
    print(f"  PTA   : {len(pta.states)} states, {len(pta.transitions)} transitions")
    print(f"  EFSM  : {len(efsm.states)} states, {len(efsm.transitions)} transitions")
    print(f"  DPN   : {len(dpn.places)} places, {len(dpn.transitions)} transitions")
    print()

    print("  Transitions:")
    for t in dpn.transitions:
        activity = t.name
        if activity.startswith("t_"):
            parts = activity[2:].rsplit("_", 1)
            activity = parts[0] if len(parts) == 2 and parts[1].isdigit() else activity[2:]

        guard_str = str(t.guard) if t.guard is not None else "True"
        print(f"    {activity:20s}  guard = {guard_str}")

        if t.update_rule:
            for var, expr in sorted(t.update_rule.items()):
                print(f"    {'':20s}  {var}' := {expr}")
        print()

    print(sep)


def run_on_log(tc: TestCase) -> None:
    """Run the full pipeline on a single log and print the discovered model."""
    logger.info("=" * 60)
    logger.info("  Running: %s  (%s)", tc.name, tc.csv_path)
    logger.info("=" * 60)

    # Run the full discovery pipeline.
    pta, efsm, dpn = run_pipeline_full(str(tc.csv_path), case_sampling_ratio=0.2)

    # Print model summary for manual comparison.
    _print_model_summary(tc.name, pta, efsm, dpn)

    # Save PNML.
    out_dir = Path(f"output/{tc.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pnml = dpn_to_pnml(dpn)
    (out_dir / "dpn.pnml").write_text(pnml, encoding="utf-8")

    # Save visualizations.
    viz.save_efsm(pta,  str(out_dir / "pta"),  title=f"{tc.name} — PTA")
    viz.save_efsm(efsm, str(out_dir / "efsm"), title=f"{tc.name} — Merged EFSM")
    viz.save_dpn(dpn,   str(out_dir / "dpn"),   title=f"{tc.name} — Discovered DPN")
    viz.save_comparison(efsm, dpn, str(out_dir / "comparison"),
                        efsm_title=f"{tc.name} — EFSM",
                        dpn_title=f"{tc.name} — DPN")

    logger.info("  Output saved to %s/", out_dir)
    logger.info("  Reference model: %s", tc.ref_png)
    logger.info("")


sepsis_log = TestCase("Sepsis", Path("/Users/christianimenkamp/Documents/Data-Repository/Community/sepsis/Sepsis Cases - Event Log.xes"), DATAPATH / "sepsis/petri-net.png")

if __name__ == "__main__":
    settings = VisualizerSettings(output_format="png", rankdir="LR")
    viz = DPNVisualizer(settings)

    # for tc in TEST_CASES:
    #     run_on_log(tc)

    run_on_log(sepsis_log)