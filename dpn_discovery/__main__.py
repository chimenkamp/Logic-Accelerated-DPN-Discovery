"""
CLI entry point for the DPN Discovery Pipeline.

Usage::

    python -m dpn_discovery <log_file.json> [--output dpn.pnml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dpn_discovery.dpn_transform import dpn_to_pnml
from dpn_discovery.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dpn-discover",
        description="Discover a Data-Aware Petri Net from an event log.",
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to the JSON event-log file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path for the PNML output file (default: stdout).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    if not args.log_file.exists():
        logging.error("File not found: %s", args.log_file)
        sys.exit(1)

    dpn = run_pipeline(args.log_file)
    pnml = dpn_to_pnml(dpn)

    if args.output:
        args.output.write_text(pnml, encoding="utf-8")
        logging.info("PNML written to %s", args.output)
    else:
        print(pnml)


if __name__ == "__main__":
    main()
