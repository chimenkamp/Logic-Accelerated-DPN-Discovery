"""
Abstract SMT Solver Interface.

Defines the abstract base classes that decouple the rest of the
pipeline from any specific SMT solver implementation (e.g. Z3,
CVC5, Yices2).

To add a new solver backend:
  1. Subclass ``SMTSolver``, ``SMTContext``, and ``SMTModel``.
  2. Implement all abstract methods.
  3. Register via ``set_solver()`` in ``dpn_discovery.smt``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Opaque expression type aliases
# ---------------------------------------------------------------------------
# These are intentionally ``Any`` because the concrete type depends on
# the solver backend.  They serve as *documentation* in type hints,
# making it clear what kind of expression is expected.

SMTBool = Any
"""A solver-specific Boolean expression (e.g. ``z3.BoolRef``)."""

SMTArith = Any
"""A solver-specific arithmetic expression (e.g. ``z3.ArithRef``)."""

SMTExpr = Any
"""A solver-specific expression of any sort (e.g. ``z3.ExprRef``)."""


# ---------------------------------------------------------------------------
# Satisfiability result
# ---------------------------------------------------------------------------

class SatResult(Enum):
    """Result of a satisfiability check."""

    SAT = auto()
    UNSAT = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Abstract model (satisfying assignment)
# ---------------------------------------------------------------------------

class SMTModel(ABC):
    """Abstract interface for a satisfying model returned by the solver."""

    @abstractmethod
    def evaluate(self, expr: SMTExpr, model_completion: bool = False) -> SMTExpr:
        """Evaluate *expr* under this model.

        Parameters
        ----------
        expr : SMTExpr
            The expression to evaluate.
        model_completion : bool
            When ``True``, assign default values to any
            uninterpreted constants not in the model.
        """
        ...


# ---------------------------------------------------------------------------
# Abstract solver context (assertion set + check)
# ---------------------------------------------------------------------------

class SMTContext(ABC):
    """Abstract solver context — an assertion set with satisfiability checking.

    Supports the context-manager protocol for deterministic cleanup
    of native resources (important for Yices2)::

        with smt.create_context() as ctx:
            ctx.add(formula)
            result = ctx.check()
    """

    @abstractmethod
    def add(self, *assertions: SMTBool) -> None:
        """Add one or more assertions to the context."""
        ...

    @abstractmethod
    def check(self) -> SatResult:
        """Check satisfiability of the current assertion set."""
        ...

    @abstractmethod
    def model(self) -> SMTModel:
        """Return the satisfying model (only valid after ``check() == SAT``)."""
        ...

    def dispose(self) -> None:
        """Release native resources held by this context.

        The default implementation is a no-op.  Backends that hold
        native handles (e.g. Yices2) should override this.
        """

    def reset(self) -> None:
        """Remove all assertions from this context.

        After calling ``reset()``, the context is empty and ready
        to accept new assertions — equivalent to a freshly created
        context but *without* the overhead of allocating / freeing
        native handles.

        The default implementation raises ``NotImplementedError``.
        Backends must override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support reset()"
        )

    def __enter__(self) -> SMTContext:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()


# ---------------------------------------------------------------------------
# Abstract SMT solver
# ---------------------------------------------------------------------------

class SMTSolver(ABC):
    """Abstract SMT solver interface.

    Provides factory methods for creating expressions, performing
    operations on them, and creating solver contexts for
    satisfiability checking.

    All expression objects are *opaque* — they should only be
    manipulated through this interface, never introspected directly.
    """

    # --- Variable / constant creation 

    @abstractmethod
    def Real(self, name: str) -> SMTArith:
        """Create a real-valued symbolic variable."""
        ...

    @abstractmethod
    def RealVal(self, val: int | float) -> SMTArith:
        """Create a real-valued constant."""
        ...

    @abstractmethod
    def BoolVal(self, val: bool) -> SMTBool:
        """Create a Boolean constant (``True`` or ``False``)."""
        ...

    # --- Boolean connectives 

    @abstractmethod
    def And(self, *args: SMTBool) -> SMTBool:
        """Logical conjunction of one or more Boolean expressions."""
        ...

    @abstractmethod
    def Or(self, *args: SMTBool) -> SMTBool:
        """Logical disjunction of one or more Boolean expressions."""
        ...

    @abstractmethod
    def Not(self, expr: SMTBool) -> SMTBool:
        """Logical negation."""
        ...

    # --- Comparison operators 

    @abstractmethod
    def LE(self, lhs: SMTArith, rhs: SMTArith) -> SMTBool:
        """Less-than-or-equal  (lhs ≤ rhs)."""
        ...

    @abstractmethod
    def GT(self, lhs: SMTArith, rhs: SMTArith) -> SMTBool:
        """Greater-than  (lhs > rhs)."""
        ...

    @abstractmethod
    def Eq(self, lhs: SMTExpr, rhs: SMTExpr) -> SMTBool:
        """Equality  (lhs = rhs)."""
        ...

    @abstractmethod
    def NEq(self, lhs: SMTExpr, rhs: SMTExpr) -> SMTBool:
        """Disequality  (lhs ≠ rhs)."""
        ...

    # --- Arithmetic operators 

    @abstractmethod
    def Add(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        """Addition  (lhs + rhs)."""
        ...

    @abstractmethod
    def Sub(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        """Subtraction  (lhs − rhs)."""
        ...

    @abstractmethod
    def Mul(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        """Multiplication  (lhs × rhs)."""
        ...

    # --- Expression manipulation 

    @abstractmethod
    def substitute(
        self, expr: SMTExpr, *substitutions: tuple[SMTExpr, SMTExpr],
    ) -> SMTExpr:
        """Simultaneously replace variables in *expr*.

        Each element of *substitutions* is a ``(from, to)`` pair.
        If *substitutions* is empty, returns *expr* unchanged.
        """
        ...

    @abstractmethod
    def simplify(self, expr: SMTExpr) -> SMTExpr:
        """Simplify *expr* (constant folding, algebraic rewrites, …)."""
        ...

    # --- Expression predicates 

    @abstractmethod
    def is_true(self, expr: SMTExpr) -> bool:
        """Return ``True`` if *expr* is syntactically the constant ``True``."""
        ...

    @abstractmethod
    def is_rational_value(self, expr: SMTExpr) -> bool:
        """Return ``True`` if *expr* is a concrete rational number."""
        ...

    @abstractmethod
    def is_int_value(self, expr: SMTExpr) -> bool:
        """Return ``True`` if *expr* is a concrete integer."""
        ...

    @abstractmethod
    def to_real_float(self, expr: SMTExpr) -> float:
        """Convert a concrete numeric expression to a Python ``float``.

        Raises ``ValueError`` if *expr* is not a concrete number.
        """
        ...

    # --- Expression display 

    @abstractmethod
    def expr_to_string(self, expr: SMTExpr) -> str:
        """Return a human-readable string representation of *expr*."""
        ...

    # --- Type predicates 

    @abstractmethod
    def is_bool_expr(self, expr: Any) -> bool:
        """Return ``True`` if *expr* is a Boolean expression from this solver."""
        ...

    # --- Solver context

    @abstractmethod
    def create_context(self, timeout_ms: int | None = None) -> SMTContext:
        """Create a fresh solver context (assertion set).

        Parameters
        ----------
        timeout_ms : int | None
            Timeout in milliseconds.  ``None`` means no timeout.
        """
        ...
