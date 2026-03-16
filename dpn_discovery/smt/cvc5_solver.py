"""
CVC5 SMT Solver Backend.

Implements the ``SMTSolver`` interface using the cvc5 theorem prover's
Pythonic API (``cvc5.pythonic``), which provides a Z3Py-compatible
interface.

Install via::

    pip install cvc5
"""

from __future__ import annotations

from typing import Any

from cvc5.pythonic import (
    And as cvc5_And,
    BoolRef as CVC5BoolRef,
    BoolVal as cvc5_BoolVal,
    Not as cvc5_Not,
    Or as cvc5_Or,
    Real as cvc5_Real,
    RealVal as cvc5_RealVal,
    Solver as CVC5SolverClass,
    is_int_value as cvc5_is_int_value,
    is_rational_value as cvc5_is_rational_value,
    is_true as cvc5_is_true,
    sat as cvc5_sat,
    simplify as cvc5_simplify,
    substitute as cvc5_substitute,
    unsat as cvc5_unsat,
)

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
# CVC5 Model
# ---------------------------------------------------------------------------

class CVC5Model(SMTModel):
    """CVC5 implementation of ``SMTModel``."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def evaluate(self, expr: SMTExpr, model_completion: bool = False) -> SMTExpr:
        return self._model.eval(expr, model_completion=model_completion)


# ---------------------------------------------------------------------------
# CVC5 Solver Context
# ---------------------------------------------------------------------------

class CVC5Context(SMTContext):
    """CVC5 implementation of ``SMTContext``."""

    def __init__(self, timeout_ms: int | None = None) -> None:
        self._solver = CVC5SolverClass()
        if timeout_ms is not None:
            self._solver.set("tlimit-per", timeout_ms)

    def add(self, *assertions: SMTBool) -> None:
        self._solver.add(*assertions)

    def check(self) -> SatResult:
        result = self._solver.check()
        if result == cvc5_sat:
            return SatResult.SAT
        if result == cvc5_unsat:
            return SatResult.UNSAT
        return SatResult.UNKNOWN

    def model(self) -> CVC5Model:
        return CVC5Model(self._solver.model())


# ---------------------------------------------------------------------------
# CVC5 SMT Solver
# ---------------------------------------------------------------------------

class CVC5SMTSolver(SMTSolver):
    """CVC5 implementation of ``SMTSolver``."""

    # --- Variable / constant creation ------------------------------------

    def Real(self, name: str) -> SMTArith:
        return cvc5_Real(name)

    def RealVal(self, val: int | float) -> SMTArith:
        return cvc5_RealVal(val)

    def BoolVal(self, val: bool) -> SMTBool:
        return cvc5_BoolVal(val)

    # --- Boolean connectives ---------------------------------------------

    def And(self, *args: SMTBool) -> SMTBool:
        return cvc5_And(*args)

    def Or(self, *args: SMTBool) -> SMTBool:
        return cvc5_Or(*args)

    def Not(self, expr: SMTBool) -> SMTBool:
        return cvc5_Not(expr)

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
        return cvc5_substitute(expr, *substitutions)

    def simplify(self, expr: SMTExpr) -> SMTExpr:
        return cvc5_simplify(expr)

    # --- Expression predicates -------------------------------------------

    def is_true(self, expr: SMTExpr) -> bool:
        return cvc5_is_true(expr)

    def is_rational_value(self, expr: SMTExpr) -> bool:
        return cvc5_is_rational_value(expr)

    def is_int_value(self, expr: SMTExpr) -> bool:
        return cvc5_is_int_value(expr)

    def to_real_float(self, expr: SMTExpr) -> float:
        if cvc5_is_rational_value(expr):
            return float(expr.as_fraction())
        if cvc5_is_int_value(expr):
            return float(expr.as_long())
        raise ValueError(f"Cannot convert expression to float: {expr}")

    # --- Expression display ----------------------------------------------

    def expr_to_string(self, expr: SMTExpr) -> str:
        return str(expr)

    # --- Type predicates -------------------------------------------------

    def is_bool_expr(self, expr: Any) -> bool:
        return isinstance(expr, CVC5BoolRef)

    # --- Solver context --------------------------------------------------

    def create_context(self, timeout_ms: int | None = None) -> CVC5Context:
        return CVC5Context(timeout_ms)
