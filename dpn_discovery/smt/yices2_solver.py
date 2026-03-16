"""
Yices2 SMT Solver Backend.

Implements the ``SMTSolver`` interface using the Yices2 theorem prover
via its native Python bindings (``yices`` package).

Unlike Z3 and CVC5, Yices2 represents terms as plain integers.
This backend wraps them into thin Python objects so that operator
overloading (``+``, ``-``, ``*``, ``<=``, ``>``, ``==``, ``!=``)
works transparently with the rest of the pipeline.

Prerequisites
~~~~~~~~~~~~~
1. Install the Yices2 *native library* – see https://yices.csl.sri.com/
2. Install the Python bindings::

       pip install yices

.. note::

   The Yices2 native library (``libyices``) is loaded **lazily** –
   merely importing this module does **not** trigger a CDLL load.
   This avoids ``DYLD_LIBRARY_PATH`` conflicts with the Z3 native
   library on macOS (Homebrew installs both under ``/opt/homebrew/lib``).
"""

from __future__ import annotations

import atexit
import os
import sys
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

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
# Lazy yices module references – populated by _init_yices()
# ---------------------------------------------------------------------------

_yl = SimpleNamespace(
    Terms=None,
    Types=None,
    Config=None,
    Context=None,
    Model=None,
    Status=None,
    Yices=None,
    Constructor=None,
)


# ---------------------------------------------------------------------------
# Yices lifecycle
# ---------------------------------------------------------------------------

_YICES_LIB_SEARCH_PATHS: list[str] = [
    # Homebrew on Apple Silicon
    "/opt/homebrew/lib",
    # Homebrew on Intel Mac / manual install
    "/usr/local/lib",
]


class _YicesState:
    initialised: bool = False


def _init_yices() -> None:
    """Ensure Yices is initialised exactly once.

    On first call this will:

    1. Locate the native ``libyices`` shared library and temporarily
       adjust ``DYLD_LIBRARY_PATH`` (macOS) or ``LD_LIBRARY_PATH`` so
       that the ``yices`` Python package can find it via ``ctypes.CDLL``.
    2. Import the ``yices`` Python package (which triggers ``yices_api``
       module-level ``loadYices()``).
    3. Restore the original library path to avoid interfering with other
       native libraries (notably Z3).
    """
    if _YicesState.initialised:
        return

    # -- Step 1: discover the native library directory -------------------
    lib_dir = _find_yices_lib_dir()

    # -- Step 2: temporarily widen the library search path ---------------
    path_var = (
        "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    )
    orig_val = os.environ.get(path_var)
    if lib_dir is not None:
        parts = [lib_dir]
        if orig_val:
            parts.append(orig_val)
        os.environ[path_var] = os.pathsep.join(parts)

    try:
        # -- Step 3: import yices (triggers loadYices inside yices_api) --
        from yices import (  # noqa: E501
            Config,
            Context,
            Model,
            Status,
            Terms,
            Types,
            Yices,
        )
        from yices.Constructors import Constructor

        _yl.Terms = Terms
        _yl.Types = Types
        _yl.Config = Config
        _yl.Context = Context
        _yl.Model = Model
        _yl.Status = Status
        _yl.Yices = Yices
        _yl.Constructor = Constructor

        Yices.init()
        atexit.register(Yices.exit)
        _YicesState.initialised = True
    finally:
        # -- Step 4: restore the original value --------------------------
        if orig_val is not None:
            os.environ[path_var] = orig_val
        else:
            os.environ.pop(path_var, None)


def _find_yices_lib_dir() -> str | None:
    """Return the directory containing ``libyices``, or *None*."""
    import ctypes.util

    # 1. Check if find_library already knows (e.g. LD path or ldconfig)
    found = ctypes.util.find_library("yices")
    if found is not None:
        import pathlib

        p = pathlib.Path(found)
        if p.is_absolute() and p.exists():
            return str(p.parent)

    # 2. Probe well-known directories
    ext = "dylib" if sys.platform == "darwin" else "so"
    for d in _YICES_LIB_SEARCH_PATHS:
        candidate = os.path.join(d, f"libyices.{ext}")
        if os.path.isfile(candidate):
            return d

    # 3. Check YICES_LIB_DIR env var (user override)
    env_dir = os.environ.get("YICES_LIB_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    return None


# ---------------------------------------------------------------------------
# Thin term wrappers – give Yices integer-terms operator overloading
# ---------------------------------------------------------------------------
#
# NOTE: All methods below that call _yl.Terms.xxx() are only invoked
# after _init_yices() has run (i.e. _yl.Terms is never None at that
# point).  The wrappers are only created by Yices2SMTSolver methods.
#

class _YicesExpr:
    """Base wrapper around a Yices term id (``int``)."""

    __slots__ = ("_term",)

    def __init__(self, term: int) -> None:
        self._term = term

    @property
    def term(self) -> int:
        return self._term

    def __repr__(self) -> str:
        return _yl.Terms.to_string(self._term)

    def __str__(self) -> str:
        return _yl.Terms.to_string(self._term)

    # Equality / hashing by wrapped term id
    def __hash__(self) -> int:
        return hash(self._term)

    # Structural equality (Python-level, not SMT)
    def __eq__(self, other: object) -> _YicesBool | bool:  # type: ignore[override]
        if isinstance(other, _YicesExpr):
            return _YicesBool(
                _yl.Terms.arith_eq_atom(self._term, other._term),
            )
        if isinstance(other, (int, float)):
            return _YicesBool(
                _yl.Terms.arith_eq_atom(self._term, _num_to_term(other)),
            )
        return NotImplemented

    def __ne__(self, other: object) -> _YicesBool | bool:  # type: ignore[override]
        if isinstance(other, _YicesExpr):
            return _YicesBool(
                _yl.Terms.arith_neq_atom(self._term, other._term),
            )
        if isinstance(other, (int, float)):
            return _YicesBool(
                _yl.Terms.arith_neq_atom(self._term, _num_to_term(other)),
            )
        return NotImplemented


class _YicesArith(_YicesExpr):
    """Arithmetic term with operator overloading."""

    # --- comparison -------------------------------------------------------

    def __le__(self, other: _YicesArith | int | float) -> _YicesBool:
        return _YicesBool(
            _yl.Terms.arith_leq_atom(self._term, _coerce(other)),
        )

    def __lt__(self, other: _YicesArith | int | float) -> _YicesBool:
        return _YicesBool(
            _yl.Terms.arith_lt_atom(self._term, _coerce(other)),
        )

    def __ge__(self, other: _YicesArith | int | float) -> _YicesBool:
        return _YicesBool(
            _yl.Terms.arith_geq_atom(self._term, _coerce(other)),
        )

    def __gt__(self, other: _YicesArith | int | float) -> _YicesBool:
        return _YicesBool(
            _yl.Terms.arith_gt_atom(self._term, _coerce(other)),
        )

    # --- arithmetic -------------------------------------------------------

    def __add__(self, other: _YicesArith | int | float) -> _YicesArith:
        return _YicesArith(_yl.Terms.add(self._term, _coerce(other)))

    def __radd__(self, other: int | float) -> _YicesArith:
        return _YicesArith(_yl.Terms.add(_num_to_term(other), self._term))

    def __sub__(self, other: _YicesArith | int | float) -> _YicesArith:
        return _YicesArith(_yl.Terms.sub(self._term, _coerce(other)))

    def __rsub__(self, other: int | float) -> _YicesArith:
        return _YicesArith(_yl.Terms.sub(_num_to_term(other), self._term))

    def __mul__(self, other: _YicesArith | int | float) -> _YicesArith:
        return _YicesArith(_yl.Terms.mul(self._term, _coerce(other)))

    def __rmul__(self, other: int | float) -> _YicesArith:
        return _YicesArith(_yl.Terms.mul(_num_to_term(other), self._term))

    def __neg__(self) -> _YicesArith:
        return _YicesArith(_yl.Terms.neg(self._term))


class _YicesBool(_YicesExpr):
    """Boolean term wrapper (no arithmetic ops)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_to_term(val: int | float) -> int:
    """Convert a Python number to a Yices rational term."""
    if isinstance(val, float):
        frac = Fraction(val).limit_denominator()
        return _yl.Terms.rational(frac.numerator, frac.denominator)
    return _yl.Terms.rational(int(val), 1)


def _coerce(obj: _YicesExpr | int | float) -> int:
    """Return raw Yices term id, converting Python numbers as needed."""
    if isinstance(obj, _YicesExpr):
        return obj.term
    return _num_to_term(obj)


def _wrap(term: int) -> _YicesArith | _YicesBool:
    """Wrap a raw Yices term id in the correct Python wrapper."""
    if _yl.Terms.is_bool(term):
        return _YicesBool(term)
    return _YicesArith(term)


# ---------------------------------------------------------------------------
# Yices2 Model
# ---------------------------------------------------------------------------

class Yices2Model(SMTModel):
    """Yices2 implementation of ``SMTModel``."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def evaluate(self, expr: SMTExpr, model_completion: bool = False) -> SMTExpr:
        raw = _coerce(expr)
        # get_value_as_term returns a Yices constant term representing
        # the value assigned in the model.
        result_term = self._model.get_value_as_term(raw)
        return _wrap(result_term)


# ---------------------------------------------------------------------------
# Yices2 Solver Context
# ---------------------------------------------------------------------------

class Yices2Context(SMTContext):
    """Yices2 implementation of ``SMTContext``.

    Parameters
    ----------
    timeout_ms : int | None
        Solver timeout in milliseconds.
    logic : str
        SMT-LIB logic string (default ``"QF_LRA"``).
        Use ``"QF_NRA"`` when non-linear arithmetic is required.
    """

    def __init__(
        self,
        timeout_ms: int | None = None,
        logic: str = "QF_LRA",
    ) -> None:
        cfg = _yl.Config()
        cfg.default_config_for_logic(logic)
        try:
            self._ctx = _yl.Context(cfg)
            # Yices returns NULL (→ None) when it cannot allocate a
            # context, but the Python binding only checks for -1.
            if self._ctx.context is None:
                raise RuntimeError(
                    f"Yices2 failed to create context for logic '{logic}' "
                    "(native handle is NULL — possible resource exhaustion)"
                )
        finally:
            # Config is copied by yices_new_context; free it immediately.
            cfg.dispose()

        self._timeout: float | None = None
        if timeout_ms is not None:
            # Yices check_context accepts timeout in *seconds* (float).
            self._timeout = timeout_ms / 1000.0

    def add(self, *assertions: SMTBool) -> None:
        for a in assertions:
            self._ctx.assert_formula(_coerce(a))

    def check(self) -> SatResult:
        status = self._ctx.check_context(timeout=self._timeout)
        if status == _yl.Status.SAT:
            return SatResult.SAT
        if status == _yl.Status.UNSAT:
            return SatResult.UNSAT
        return SatResult.UNKNOWN

    def model(self) -> Yices2Model:
        raw_model = _yl.Model.from_context(self._ctx, 1)
        return Yices2Model(raw_model)

    def dispose(self) -> None:
        """Free the underlying Yices context to avoid resource exhaustion."""
        if self._ctx is not None and self._ctx.context is not None:
            self._ctx.dispose()
            self._ctx = None

    def reset(self) -> None:
        """Remove all assertions, returning the context to its initial state."""
        if self._ctx is not None:
            self._ctx.reset_context()


# ---------------------------------------------------------------------------
# Yices2 SMT Solver
# ---------------------------------------------------------------------------

class Yices2SMTSolver(SMTSolver):
    """Yices2 implementation of ``SMTSolver``."""

    def __init__(self) -> None:
        _init_yices()
        self._real_type = _yl.Types.real_type()
        self._bool_type = _yl.Types.bool_type()

    # --- Variable / constant creation ------------------------------------

    def Real(self, name: str) -> SMTArith:
        term = _yl.Terms.new_uninterpreted_term(self._real_type, name)
        return _YicesArith(term)

    def RealVal(self, val: int | float) -> SMTArith:
        return _YicesArith(_num_to_term(val))

    def BoolVal(self, val: bool) -> SMTBool:
        return _YicesBool(_yl.Terms.true() if val else _yl.Terms.false())

    # --- Boolean connectives ---------------------------------------------

    def And(self, *args: SMTBool) -> SMTBool:
        raw = [_coerce(a) for a in args]
        if not raw:
            return _YicesBool(_yl.Terms.true())
        return _YicesBool(_yl.Terms.yand(raw))

    def Or(self, *args: SMTBool) -> SMTBool:
        raw = [_coerce(a) for a in args]
        if not raw:
            return _YicesBool(_yl.Terms.false())
        return _YicesBool(_yl.Terms.yor(raw))

    def Not(self, expr: SMTBool) -> SMTBool:
        return _YicesBool(_yl.Terms.ynot(_coerce(expr)))

    # --- Comparison operators --------------------------------------------

    def LE(self, lhs: SMTArith, rhs: SMTArith) -> SMTBool:
        return _YicesBool(
            _yl.Terms.arith_leq_atom(_coerce(lhs), _coerce(rhs)),
        )

    def GT(self, lhs: SMTArith, rhs: SMTArith) -> SMTBool:
        return _YicesBool(
            _yl.Terms.arith_gt_atom(_coerce(lhs), _coerce(rhs)),
        )

    def Eq(self, lhs: SMTExpr, rhs: SMTExpr) -> SMTBool:
        return _YicesBool(_yl.Terms.eq(_coerce(lhs), _coerce(rhs)))

    def NEq(self, lhs: SMTExpr, rhs: SMTExpr) -> SMTBool:
        return _YicesBool(_yl.Terms.neq(_coerce(lhs), _coerce(rhs)))

    # --- Arithmetic operators --------------------------------------------

    def Add(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        return _YicesArith(_yl.Terms.add(_coerce(lhs), _coerce(rhs)))

    def Sub(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        return _YicesArith(_yl.Terms.sub(_coerce(lhs), _coerce(rhs)))

    def Mul(self, lhs: SMTArith, rhs: SMTArith) -> SMTArith:
        return _YicesArith(_yl.Terms.mul(_coerce(lhs), _coerce(rhs)))

    # --- Expression manipulation -----------------------------------------

    def substitute(
        self, expr: SMTExpr, *substitutions: tuple[SMTExpr, SMTExpr],
    ) -> SMTExpr:
        if not substitutions:
            return expr
        variables = [_coerce(old) for old, _ in substitutions]
        replacements = [_coerce(new) for _, new in substitutions]
        result = _yl.Terms.subst(variables, replacements, _coerce(expr))
        return _wrap(result)

    def simplify(self, expr: SMTExpr) -> SMTExpr:
        # Yices2 does not expose a public simplify API.
        # Return the expression unchanged.
        return expr

    # --- Expression predicates -------------------------------------------

    def is_true(self, expr: SMTExpr) -> bool:
        raw = _coerce(expr)
        return raw == _yl.Terms.true()

    def is_rational_value(self, expr: SMTExpr) -> bool:
        raw = _coerce(expr)
        if not _yl.Terms.is_arithmetic(raw):
            return False
        try:
            constructor = _yl.Terms.constructor(raw)
            return constructor == _yl.Constructor.ARITH_CONSTANT
        except Exception:
            return False

    def is_int_value(self, expr: SMTExpr) -> bool:
        if not self.is_rational_value(expr):
            return False
        raw = _coerce(expr)
        frac = self._get_fraction_value(raw)
        return frac is not None and frac.denominator == 1

    @staticmethod
    def _get_fraction_value(raw_term: int) -> Fraction | None:
        """Try to extract a Fraction from a constant arithmetic term.

        Yices2 doesn't expose a convenient Python-level API for reading
        rational constant values.  We fall back to parsing the string
        representation which is always ``"N"`` or ``"N/D"``.
        """
        try:
            s = _yl.Terms.to_string(raw_term)
            if "/" in s:
                num, den = s.split("/", 1)
                return Fraction(int(num), int(den))
            # Could be a plain integer
            return Fraction(int(s), 1)
        except Exception:
            return None

    def to_real_float(self, expr: SMTExpr) -> float:
        raw = _coerce(expr)
        frac = self._get_fraction_value(raw)
        if frac is not None:
            return float(frac)
        raise ValueError(f"Cannot convert expression to float: {expr}")

    # --- Expression display ----------------------------------------------

    def expr_to_string(self, expr: SMTExpr) -> str:
        return _yl.Terms.to_string(_coerce(expr))

    # --- Type predicates -------------------------------------------------

    def is_bool_expr(self, expr: Any) -> bool:
        if isinstance(expr, _YicesBool):
            return True
        if isinstance(expr, _YicesExpr):
            return _yl.Terms.is_bool(expr.term)
        return False

    # --- Solver context --------------------------------------------------

    def create_context(
        self,
        timeout_ms: int | None = None,
        logic: str = "QF_LRA",
    ) -> Yices2Context:
        return Yices2Context(timeout_ms, logic=logic)
