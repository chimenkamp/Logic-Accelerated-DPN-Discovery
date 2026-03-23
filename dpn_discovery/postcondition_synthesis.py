"""
Step 5 — Postcondition Synthesis via Abduction.

Infers data-update functions for every transition by solving an
abductive synthesis problem: given pre-state / post-state
observation pairs, find the simplest expression  Op  such that

    ∀ σ .  Pre(σ) ∧ Op(σ, σ') ⟹ Post(σ')

The algorithm follows Reynolds et al. (IJCAR 2020):
  • Enumerate candidate expressions from a typed grammar in
    size-increasing order (simplest hypothesis first).
  • For each candidate, check consistency against all observed
    (dᵢₙ, dₒᵤₜ) pairs using Z3.
  • De-duplicate observation pairs before solving to reduce
    redundant Z3 constraints.

Reference
---------
  Reynolds, A., Barbosa, H., Nötzli, A., Barrett, C., Tinelli, C.
  *Scalable Algorithms for Abduction via Enumerative Syntax-Guided
  Synthesis* (IJCAR 2020).

  Specification §4 Step 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from dpn_discovery.smt import get_solver, SMTSolver, SMTBool, SMTArith, SMTExpr, SatResult

from dpn_discovery.models import EFSM, Transition

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Expression AST for update-rule candidates
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ExprNode:
    """AST node for a candidate update expression."""
    kind: str           # "const" | "var" | "add" | "mul" | "sub"
    value: int | float | None = None      # for "const"
    var_name: str | None = None           # for "var"
    children: tuple[ExprNode, ...] = ()   # for composite nodes

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def to_smt(self, smt_vars: dict[str, SMTArith], solver: SMTSolver) -> SMTArith:
        """Compile to an SMT arithmetic expression."""
        match self.kind:
            case "const":
                return solver.RealVal(self.value)
            case "var":
                return smt_vars[self.var_name]
            case "add":
                return solver.Add(
                    self.children[0].to_smt(smt_vars, solver),
                    self.children[1].to_smt(smt_vars, solver),
                )
            case "sub":
                return solver.Sub(
                    self.children[0].to_smt(smt_vars, solver),
                    self.children[1].to_smt(smt_vars, solver),
                )
            case "mul":
                return solver.Mul(
                    self.children[0].to_smt(smt_vars, solver),
                    self.children[1].to_smt(smt_vars, solver),
                )
            case _:
                raise ValueError(f"Unknown expression kind: {self.kind}")

    def pretty(self) -> str:
        match self.kind:
            case "const":
                return str(self.value)
            case "var":
                return str(self.var_name)
            case "add":
                return f"({self.children[0].pretty()} + {self.children[1].pretty()})"
            case "sub":
                return f"({self.children[0].pretty()} - {self.children[1].pretty()})"
            case "mul":
                return f"({self.children[0].pretty()} * {self.children[1].pretty()})"
            case _:
                return f"?{self.kind}"

    def evaluate(self, sample: dict[str, Any]) -> float | None:
        """Evaluate the expression under a concrete variable assignment.

        Returns the numeric result, or ``None`` if a referenced
        variable is missing or non-numeric in *sample*.
        """
        match self.kind:
            case "const":
                return float(self.value) if self.value is not None else None
            case "var":
                v = sample.get(self.var_name)
                return float(v) if isinstance(v, (int, float)) else None
            case "add":
                a = self.children[0].evaluate(sample)
                b = self.children[1].evaluate(sample)
                return (a + b) if (a is not None and b is not None) else None
            case "sub":
                a = self.children[0].evaluate(sample)
                b = self.children[1].evaluate(sample)
                return (a - b) if (a is not None and b is not None) else None
            case "mul":
                a = self.children[0].evaluate(sample)
                b = self.children[1].evaluate(sample)
                return (a * b) if (a is not None and b is not None) else None
            case _:
                return None


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Expression grammar enumeration
# ═══════════════════════════════════════════════════════════════════════════

def _enumerate_candidates(
    variables: list[str],
    max_depth: int = 2,
    data_driven_constants: set[int | float] | None = None,
    target_variable: str | None = None,
) -> list[ExprNode]:
    """Enumerate candidate update expressions up to *max_depth*.

    Grammar (§4 Step 5 of spec):
        Expr → Const | Var | Expr + Expr | Const * Var + Const

    Enumeration order mirrors Reynolds et al.'s size-increasing
    strategy so the simplest (most general) hypothesis is tried
    first.

    Level 0: constants  0, 1, −1  (+ data-driven deltas)
    Level 1: variables  x, y, …
    Level 2: x+1, x−1, 2*x, x+y, …

    When *target_variable* is provided, depth-2 candidates that
    reference the target variable are placed before those that do
    not (self-referencing preference).  This avoids coincidental
    cross-variable matches when a same-variable expression also
    fits the data.
    """
    base_const_values = {0, 1, -1}
    if data_driven_constants:
        base_const_values |= data_driven_constants
    # Sort for deterministic enumeration.
    const_values_sorted = sorted(base_const_values, key=lambda x: (abs(x), x))

    constants = [ExprNode(kind="const", value=v) for v in const_values_sorted]

    var_nodes = [ExprNode(kind="var", var_name=v) for v in variables]

    # Depth-0 candidates.
    candidates: list[ExprNode] = list(constants)

    # Depth-1 candidates: target variable first (if specified).
    if target_variable:
        target_vars = [v for v in var_nodes if v.var_name == target_variable]
        other_vars = [v for v in var_nodes if v.var_name != target_variable]
    else:
        target_vars = var_nodes
        other_vars = []

    candidates.extend(target_vars)

    if max_depth < 2:
        candidates.extend(other_vars)
        return candidates

    # Depth-2 candidates: binary combinations.
    atoms = constants + var_nodes

    # Separate self-referencing (uses target_variable) from cross-variable.
    self_ref: list[ExprNode] = []
    cross_ref: list[ExprNode] = []

    for lhs in atoms:
        for rhs in atoms:
            if lhs.kind == "const" and rhs.kind == "const":
                continue  # const ⊕ const is just another const — skip.
            for op in ("add", "sub", "mul"):
                node = ExprNode(kind=op, children=(lhs, rhs))
                if target_variable and _expr_references_var(node, target_variable):
                    self_ref.append(node)
                else:
                    cross_ref.append(node)

    # Enumeration order (self-referencing preference):
    #   1. constants          (depth 0)
    #   2. target variable    (depth 1, identity check)
    #   3. self-ref depth-2   (e.g. y+1, y-1 — preferred)
    #   4. other variables    (depth 1, cross-variable)
    #   5. cross-ref depth-2  (e.g. x+1, x*2)
    candidates.extend(self_ref)
    candidates.extend(other_vars)
    candidates.extend(cross_ref)

    return candidates


def _expr_references_var(node: ExprNode, var_name: str) -> bool:
    """Return True if *node* contains a reference to *var_name*."""
    if node.kind == "var" and node.var_name == var_name:
        return True
    return any(_expr_references_var(c, var_name) for c in node.children)


def _compute_data_driven_constants(
    var: str,
    pre_post_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> set[int | float]:
    """Compute candidate constants from observed pre/post deltas.

    For each observation pair where both pre and post values for
    *var* are numeric, compute  delta = post[var] - pre[var].
    The set of unique non-trivial deltas (excluding 0, 1, −1 which
    are already in the base pool) is returned.

    To avoid blowing up the search space, at most 10 unique delta
    values are kept (the most frequently occurring ones).
    """
    from collections import Counter
    delta_counts: Counter[int | float] = Counter()

    for pre, post in pre_post_pairs:
        pre_val = pre.get(var)
        post_val = post.get(var)
        if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)):
            delta = post_val - pre_val
            if delta not in (0, 1, -1):
                # Keep as int if possible.
                if isinstance(delta, float) and delta == int(delta):
                    delta = int(delta)
                delta_counts[delta] += 1

    # Return the top-10 most frequent deltas.
    return {d for d, _ in delta_counts.most_common(10)}


# ═══════════════════════════════════════════════════════════════════════════
# 3.  GetAbductUCL — enumerative abduction with UNSAT-core learning
#     (Reynolds et al., §4)
# ═══════════════════════════════════════════════════════════════════════════

def _deduplicate_pairs(
    var: str,
    pre_post_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    variables: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """De-duplicate observation pairs to their unique projections.

    Many traces visit the same transition with identical data values.
    Since the abduction check is purely equational, duplicate pairs
    add redundant Z3 constraints without changing the outcome.
    De-duplicating dramatically reduces solver work.
    """
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for pre, post in pre_post_pairs:
        # Build a hashable key from the relevant variable values.
        key_parts: list[tuple[str, Any]] = []
        for v in variables:
            key_parts.append((f"pre_{v}", pre.get(v)))
        key_parts.append((f"post_{var}", post.get(var)))
        key = tuple(key_parts)

        if key not in seen:
            seen.add(key)
            unique.append((pre, post))

    return unique


# ═══════════════════════════════════════════════════════════════════════════
# 3b.  Symbolic guard-based candidate pruning
# ═══════════════════════════════════════════════════════════════════════════

def _is_trivial_guard(guard: SMTBool | None) -> bool:
    """Return True if *guard* is ``None`` or syntactically ``True``.

    A trivial guard provides no symbolic constraints, so pruning
    would yield no benefit.
    """
    if guard is None:
        return True
    smt = get_solver()
    return smt.is_true(smt.simplify(guard))


def _are_equivalent_under_guard(
    e1: ExprNode,
    e2: ExprNode,
    guard: SMTBool,
    variables: list[str],
    timeout_ms: int = 500,
) -> bool:
    """Check whether *e1* and *e2* produce identical values whenever *guard* holds.

    Formally, returns ``True`` iff:

        ¬( g(V) → e₁(V) = e₂(V) )  is UNSAT

    i.e.  g(V) ⊨ e₁(V) = e₂(V).
    """
    smt = get_solver()
    smt_vars = {v: smt.Real(v) for v in variables}
    expr1 = e1.to_smt(smt_vars, smt)
    expr2 = e2.to_smt(smt_vars, smt)

    with smt.create_context(timeout_ms=timeout_ms) as ctx:
        # Assert: guard holds AND the two expressions differ.
        # If UNSAT → they are always equal under the guard.
        ctx.add(guard)
        ctx.add(smt.NEq(expr1, expr2))

        return ctx.check() == SatResult.UNSAT


def _prune_candidates_with_guard(
    candidates: list[ExprNode],
    guard: SMTBool,
    variables: list[str],
) -> list[ExprNode]:
    """Remove candidates that are equivalent under *guard* to an earlier one.

    Iterates *candidates* in order (assumed size-increasing) and keeps
    only the **first representative** of each equivalence class.  Two
    candidates belong to the same class iff they produce identical
    output values whenever the guard holds.

    This leverages the synthesised guard as symbolic background
    knowledge to collapse the search space before the expensive
    per-observation consistency checks.
    """
    kept: list[ExprNode] = []

    for candidate in candidates:
        is_redundant = False
        for representative in kept:
            if _are_equivalent_under_guard(
                candidate, representative, guard, variables
            ):
                is_redundant = True
                break
        if not is_redundant:
            kept.append(candidate)

    n_pruned = len(candidates) - len(kept)
    if n_pruned > 0:
        logger.debug(
            "Symbolic pruning: %d / %d candidates removed (guard: %s)",
            n_pruned,
            len(candidates),
            guard,
        )

    return kept


def _get_abduct_ucl(
    var: str,
    pre_post_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    variables: list[str],
    guard_formula: SMTBool | None = None,
    use_symbolic_pruning: bool = False,
) -> ExprNode | None:
    """Enumerative abduction (GetAbductUCL, Reynolds et al. §4).

    For each candidate expression ``e`` in size-increasing order,
    check whether  ∀ (dᵢₙ, dₒᵤₜ) : dₒᵤₜ[var] == e(dᵢₙ)  holds.

    Note on UNSAT-core learning
    ---------------------------
    Reynolds et al.'s UNSAT-core pruning (§4.2) is designed for a
    quantified synthesis setting where different candidates can
    produce structurally different constraint sets.  In our ground
    instantiation every candidate is checked against the *same*
    observation indices, so a core recorded from one candidate
    would incorrectly prune *all* subsequent candidates.

    Instead we apply two optimisations that achieve comparable
    speedups without unsound pruning:

    1. **De-duplication** of observation pairs (often 700 → 14
       unique pairs), which reduces each Z3 call dramatically.
    2. **Size-increasing enumeration** — returning the first
       satisfying candidate guarantees a simplest-first result.
    """
    smt = get_solver()

    # De-duplicate observations to speed up solver calls.
    unique_pairs = _deduplicate_pairs(var, pre_post_pairs, variables)

    # ── Data-driven constant inference ───────────────────────────────
    # Compute pre/post deltas for the target variable and add them
    # to the expression grammar so that updates like  var + 100  or
    # var - 50  become reachable.
    data_constants = _compute_data_driven_constants(var, unique_pairs)

    candidates = _enumerate_candidates(
        variables,
        max_depth=2,
        data_driven_constants=data_constants,
        target_variable=var,
    )

    # ── Symbolic pruning (optional) ──────────────────────────────────
    # When enabled and a non-trivial guard is available, collapse
    # candidates that are equivalent under the guard into a single
    # representative.  This reduces the number of expensive
    # per-observation solver checks.
    if use_symbolic_pruning and not _is_trivial_guard(guard_formula):
        candidates = _prune_candidates_with_guard(
            candidates, guard_formula, variables  # type: ignore[arg-type]
        )

    # ── Pre-check: skip if no observation has a numeric post-state ────
    # If no pair has a numeric post-state value for *var*, every
    # candidate trivially satisfies the (empty) abduction problem.
    # This is logically sound: an empty set of constraints is
    # vacuously satisfiable, so there is nothing to abduce.
    has_numeric_post = any(
        isinstance(post.get(var), (int, float))
        for _, post in unique_pairs
    )
    if not has_numeric_post:
        return None

    # ── Context-reuse loop ───────────────────────────────────────────
    # Try each candidate in size-increasing order.  First attempt a
    # pure-Python evaluation (O(n) per candidate, no SMT).  Only
    # fall back to SMT if a variable is missing from a sample.
    smt_vars = {v: smt.Real(v) for v in variables}

    for candidate in candidates:
        # ── Fast path: pure-Python evaluation ────────────────────
        fast_ok = True
        fast_has_constraint = False
        use_smt_fallback = False

        for pre, post in unique_pairs:
            if var not in post:
                continue
            if not isinstance(post[var], (int, float)):
                fast_ok = False
                break

            fast_has_constraint = True
            result = candidate.evaluate(pre)
            if result is None:
                # Variable missing → need SMT fallback for this candidate.
                use_smt_fallback = True
                break
            if abs(result - float(post[var])) > 1e-9:
                fast_ok = False
                break

        if not use_smt_fallback:
            if fast_ok and fast_has_constraint:
                return candidate
            continue  # Fast rejection — skip to next candidate.

        # ── SMT fallback (only when evaluate() returned None) ────
        try:
            with smt.create_context(timeout_ms=3000) as ctx:
                valid = True
                has_constraints = False

                for pre, post in unique_pairs:
                    if var not in post:
                        continue
                    if not isinstance(post[var], (int, float)):
                        valid = False
                        break

                    has_constraints = True

                    expr = candidate.to_smt(smt_vars, smt)
                    substitutions = [
                        (smt_vars[v], smt.RealVal(pre[v]))
                        for v in variables
                        if v in pre and isinstance(pre[v], (int, float))
                    ]
                    if substitutions:
                        expr = smt.substitute(expr, *substitutions)

                    expected = smt.RealVal(post[var])
                    ctx.add(smt.Eq(expr, expected))

                if not valid or not has_constraints:
                    continue

                if ctx.check() == SatResult.SAT:
                    return candidate
        except Exception:
            continue

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Per-transition update synthesis
# ═══════════════════════════════════════════════════════════════════════════

def _collect_pre_post_pairs(
    transition: Transition,
    efsm: EFSM,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Build (pre-state, post-state) pairs for *transition*.

    Each data sample on the transition represents a *post-state*
    observation.  The corresponding *pre-state* is reconstructed
    by looking at samples on the *incoming* transitions of the
    source state (or the initial payload if source is the initial
    state).

    When direct pre-state data is unavailable we fall back to
    using the sample itself as both pre and post, which still
    captures identity vs. increment patterns.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    # Gather incoming data to the source state as potential pre-states.
    incoming_samples: list[dict[str, Any]] = []
    for t in efsm.transitions:
        if t.target_id == transition.source_id:
            incoming_samples.extend(t.data_samples)

    for post_sample in transition.data_samples:
        if incoming_samples:
            # Pair each post sample with the chronologically closest
            # pre-sample (heuristic: closest by index order).
            pre = incoming_samples[min(len(pairs), len(incoming_samples) - 1)]
        else:
            # Source is initial state — use the post sample's data
            # as a proxy (captures the first observation).
            pre = post_sample
        pairs.append((pre, post_sample))

    return pairs


def synthesise_postconditions(
    efsm: EFSM,
    use_symbolic_pruning: bool = False,
) -> EFSM:
    """Annotate every transition in *efsm* with update rules.

    Strategy (per specification §4 Step 5, Reynolds et al. IJCAR 2020):
      For each variable on each transition, run GetAbductUCL
      (enumerative abduction) to find the simplest update
      expression consistent with all observed pre/post pairs.

    Variables that have no numeric post-state observations on a
    transition are skipped (the abduction problem would be vacuously
    satisfiable — nothing to abduce).

    Pre/post observation pairs are stored directly on each
    transition (populated during PTA construction and preserved
    through state merging).

    Parameters
    ----------
    efsm : EFSM
        The EFSM with guard formulas already populated (Step 4).
    use_symbolic_pruning : bool
        When ``True``, use the transition's guard formula to
        symbolically prune equivalent update-expression candidates
        before checking them against observations.  This can
        significantly reduce the search space when guards impose
        non-trivial constraints on the input variables.  Default
        is ``False`` (original behaviour).

    Returns a **new** EFSM with ``update_rule`` populated.
    """
    efsm = efsm.deep_copy()
    variables = sorted(efsm.variables)
    smt = get_solver()
    total_transitions = len(efsm.transitions)

    logger.info("Postcondition synthesis: %d transitions, %d variables",
                total_transitions, len(variables))

    for t_idx, transition in enumerate(efsm.transitions, 1):
        pre_post_pairs = transition.pre_post_pairs

        if not pre_post_pairs:
            continue

        logger.info("  [%d/%d] %s → %s (%s): %d observation pairs",
                    t_idx, total_transitions, transition.source_id,
                    transition.target_id, transition.activity,
                    len(pre_post_pairs))

        update_rule: dict[str, SMTExpr] = {}

        for var in variables:
            # Skip variables that have no numeric post-state value
            # in any observation.  The abduction problem is vacuously
            # satisfiable in that case — nothing to abduce.
            if not any(
                isinstance(post.get(var), (int, float))
                for _, post in pre_post_pairs
            ):
                continue

            # GetAbductUCL (Reynolds et al.)
            abduct = _get_abduct_ucl(
                var,
                pre_post_pairs,
                variables,
                guard_formula=transition.guard_formula,
                use_symbolic_pruning=use_symbolic_pruning,
            )
            if abduct is not None:
                smt_vars = {v: smt.Real(v) for v in variables}
                update_rule[var] = abduct.to_smt(smt_vars, smt)
            # else: no update discovered → variable unchanged (identity).

        if update_rule:
            transition.update_rule = update_rule

    return efsm
