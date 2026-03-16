"""
SMT Solver Abstraction Layer.

Provides a solver-agnostic interface for creating and manipulating
SMT expressions and checking satisfiability.  The default backend
is Z3, but alternative solvers can be registered via ``set_solver()``.

Usage::

    from dpn_discovery.smt import get_solver, SatResult

    smt = get_solver()
    x = smt.Real("x")
    ctx = smt.create_context(timeout_ms=1000)
    ctx.add(smt.GT(x, smt.RealVal(0)))
    if ctx.check() == SatResult.SAT:
        ...
"""

from __future__ import annotations

from .base import (
    SMTArith,
    SMTBool,
    SMTContext,
    SMTExpr,
    SMTModel,
    SMTSolver,
    SatResult,
)

__all__ = [
    "SMTArith",
    "SMTBool",
    "SMTContext",
    "SMTExpr",
    "SMTModel",
    "SMTSolver",
    "SatResult",
    "get_solver",
    "set_solver",
]

_solver: SMTSolver | None = None


def get_solver() -> SMTSolver:
    """Return the currently configured SMT solver instance.

    On first call (or after ``set_solver(None)``), creates a
    default ``Z3SMTSolver``.
    """
    global _solver
    if _solver is None:
        from .z3_solver import Z3SMTSolver

        _solver = Z3SMTSolver()
    return _solver


def set_solver(solver: SMTSolver | None) -> None:
    """Set the global SMT solver instance.

    Pass ``None`` to reset to the default (Z3 on next ``get_solver()``).
    """
    global _solver
    _solver = solver
