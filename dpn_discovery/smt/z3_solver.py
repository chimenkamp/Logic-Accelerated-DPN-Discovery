"""
Z3 SMT Solver Backend.

Implements the ``SMTSolver`` interface using Microsoft's Z3
theorem prover (``z3-solver`` Python package).
"""

from __future__ import annotations

from typing import Any

import z3

from .base import (
    SMTArith,
    SMTBool,
    SMTContext,
    SMTExpr,
    SMTModel,
    SMTSolver,
    SatResult,
)


# ---------------------------------------------------------------------------
# Z3 Model
# ---------------------------------------------------------------------------

class Z3Model(SMTModel):
    """Z3 implementation of ``SMTModel``."""

    def __init__(self, model: z3.ModelRef) -> None:
        self._model = model

    def evaluate(self, expr: SMTExpr, model_completion: bool = False) -> SMTExpr:
        return self._model.eval(expr, model_completion=model_completion)


# ---------------------------------------------------------------------------
# Z3 Solver Context
# ---------------------------------------------------------------------------

class Z3Context(SMTContext):
    """Z3 implementation of ``SMTContext``."""

    def __init__(self, timeout_ms: int | None = None) -> None:
        self._solver = z3.Solver()
        if timeout_ms is not None:
            self._solver.set("timeout", timeout_ms)

    def add(self, *assertions: SMTBool) -> None:
        self._solver.add(*assertions)

    def check(self) -> SatResult:
        result = self._solver.check()
        if result == z3.sat:
            return SatResult.SAT
        if result == z3.unsat:
            return SatResult.UNSAT
        return SatResult.UNKNOWN

    def reset(self) -> None:
        """Remove all assertions, returning the solver to its initial state."""
        self._solver.reset()

    def model(self) -> Z3Model:
        return Z3Model(self._solver.model())


# ---------------------------------------------------------------------------
# Z3 SMT Solver
# ---------------------------------------------------------------------------

class Z3SMTSolver(SMTSolver):
    """Z3 implementation of ``SMTSolver``."""

    # --- Variable / constant creation ------------------------------------

    def Real(self, name: str) -> SMTArith:
        return z3.Real(name)

    def RealVal(self, val: int | float) -> SMTArith:
        return z3.RealVal(val)

    def BoolVal(self, val: bool) -> SMTBool:
        return z3.BoolVal(val)

    # --- Boolean connectives ---------------------------------------------

    def And(self, *args: SMTBool) -> SMTBool:
        return z3.And(*args)

    def Or(self, *args: SMTBool) -> SMTBool:
        return z3.Or(*args)

    def Not(self, expr: SMTBool) -> SMTBool:
        return z3.Not(expr)

    # --- Comparison operators --------------------------------------------

    def LE(self, lhs: SMTArith, rhs: SMTArith) -> SMTBool:
        return lhs <= rhs  # type: ignore[return-value]

    def GT(self, lhs: SMTArith, rhs: SMTArith) -> SMTBool:
        return lhs > rhs  # type: ignore[return-value]

    def Eq(self, lhs: SMTExpr, rhs: SMTExpr) -> SMTBool:
        return lhs == rhs  # type: ignore[return-value]

    def NEq(self, lhs: SMTExpr, rhs: SMTExpr) -> SMTBool:
        return lhs != rhs  # type: ignore[return-value]

    # --- Arithmetic operators --------------------------------------------

    def Add(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        return lhs + rhs  # type: ignore[return-value]

    def Sub(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        return lhs - rhs  # type: ignore[return-value]

    def Mul(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        return lhs * rhs  # type: ignore[return-value]

    # --- Expression manipulation -----------------------------------------

    def substitute(
        self, expr: SMTExpr, *substitutions: tuple[SMTExpr, SMTExpr],
    ) -> SMTExpr:
        if not substitutions:
            return expr
        return z3.substitute(expr, *substitutions)

    def simplify(self, expr: SMTExpr) -> SMTExpr:
        return z3.simplify(expr)

    # --- Expression predicates -------------------------------------------

    def is_true(self, expr: SMTExpr) -> bool:
        return z3.is_true(expr)

    def is_rational_value(self, expr: SMTExpr) -> bool:
        return z3.is_rational_value(expr)

    def is_int_value(self, expr: SMTExpr) -> bool:
        return z3.is_int_value(expr)

    def to_real_float(self, expr: SMTExpr) -> float:
        if z3.is_rational_value(expr):
            return float(expr.as_fraction())
        if z3.is_int_value(expr):
            return float(expr.as_long())
        raise ValueError(f"Cannot convert expression to float: {expr}")

    # --- Expression display ----------------------------------------------

    def expr_to_string(self, expr: SMTExpr) -> str:
        return str(expr)

    # --- Type predicates -------------------------------------------------

    def is_bool_expr(self, expr: Any) -> bool:
        return isinstance(expr, z3.BoolRef)

    # --- Solver context --------------------------------------------------

    def create_context(self, timeout_ms: int | None = None) -> Z3Context:
        return Z3Context(timeout_ms)
