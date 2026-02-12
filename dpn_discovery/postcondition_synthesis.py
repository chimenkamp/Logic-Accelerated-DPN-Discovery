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

from dataclasses import dataclass
from typing import Any

import z3

from dpn_discovery.models import EFSM, Transition


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

    def to_z3(self, z3_vars: dict[str, z3.ArithRef]) -> z3.ArithRef:
        """Compile to Z3 arithmetic expression."""
        match self.kind:
            case "const":
                return z3.RealVal(self.value)  # type: ignore[arg-type]
            case "var":
                return z3_vars[self.var_name]  # type: ignore[index]
            case "add":
                return self.children[0].to_z3(z3_vars) + self.children[1].to_z3(z3_vars)
            case "sub":
                return self.children[0].to_z3(z3_vars) - self.children[1].to_z3(z3_vars)
            case "mul":
                return self.children[0].to_z3(z3_vars) * self.children[1].to_z3(z3_vars)
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


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Expression grammar enumeration
# ═══════════════════════════════════════════════════════════════════════════

def _enumerate_candidates(
    variables: list[str],
    max_depth: int = 2,
) -> list[ExprNode]:
    """Enumerate candidate update expressions up to *max_depth*.

    Grammar (§4 Step 5 of spec):
        Expr → Const | Var | Expr + Expr | Const * Var + Const

    Enumeration order mirrors Reynolds et al.'s size-increasing
    strategy so the simplest (most general) hypothesis is tried
    first.

    Level 0: constants  0, 1, −1
    Level 1: variables  x, y, …
    Level 2: x+1, x−1, 2*x, x+y, …
    """
    constants = [
        ExprNode(kind="const", value=0),
        ExprNode(kind="const", value=1),
        ExprNode(kind="const", value=-1),
    ]

    var_nodes = [ExprNode(kind="var", var_name=v) for v in variables]

    # Depth-0 candidates.
    candidates: list[ExprNode] = list(constants)

    # Depth-1 candidates.
    candidates.extend(var_nodes)

    if max_depth < 2:
        return candidates

    # Depth-2 candidates: binary combinations.
    atoms = constants + var_nodes
    for lhs in atoms:
        for rhs in atoms:
            if lhs.kind == "const" and rhs.kind == "const":
                continue  # const ⊕ const is just another const — skip.
            candidates.append(ExprNode(kind="add", children=(lhs, rhs)))
            candidates.append(ExprNode(kind="sub", children=(lhs, rhs)))
            candidates.append(ExprNode(kind="mul", children=(lhs, rhs)))

    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Template-based synthesis with Z3  (spec §4 Step 5)
# ═══════════════════════════════════════════════════════════════════════════

def _synthesise_update_template(
    var: str,
    pre_post_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    variables: list[str],
) -> z3.ExprRef | None:
    """Try the explicit template  x' = c₁·x + c₂  first (spec §4 Step 5).

    This is faster than full enumeration for the common linear case
    and directly implements the spec's suggested template.
    """
    if not pre_post_pairs:
        return None

    c1 = z3.Real(f"_c1_{var}")
    c2 = z3.Real(f"_c2_{var}")
    solver = z3.Solver()
    solver.set("timeout", 5000)

    n_constraints = 0
    for pre, post in pre_post_pairs:
        if var not in pre or var not in post:
            continue
        if not isinstance(pre[var], (int, float)) or not isinstance(post[var], (int, float)):
            continue

        x_in = z3.RealVal(pre[var])
        x_out = z3.RealVal(post[var])
        solver.add(x_out == c1 * x_in + c2)
        n_constraints += 1

    if n_constraints == 0:
        return None

    if solver.check() == z3.sat:
        model = solver.model()
        c1_val = model.eval(c1, model_completion=True)
        c2_val = model.eval(c2, model_completion=True)
        x_var = z3.Real(var)
        return c1_val * x_var + c2_val  # type: ignore[return-value]

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 4.  GetAbductUCL — enumerative abduction with UNSAT-core learning
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


def _get_abduct_ucl(
    var: str,
    pre_post_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    variables: list[str],
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
    # De-duplicate observations to speed up Z3 calls.
    unique_pairs = _deduplicate_pairs(var, pre_post_pairs, variables)

    candidates = _enumerate_candidates(variables, max_depth=2)

    for candidate in candidates:
        # Build Z3 constraints.
        z3_vars = {v: z3.Real(v) for v in variables}
        solver = z3.Solver()
        solver.set("timeout", 3000)

        valid = True
        has_constraints = False

        for idx, (pre, post) in enumerate(unique_pairs):
            if var not in post:
                continue
            if not isinstance(post[var], (int, float)):
                valid = False
                break

            has_constraints = True

            # Substitute pre-state values into candidate expression.
            expr = candidate.to_z3(z3_vars)
            substitutions = [
                (z3_vars[v], z3.RealVal(pre[v]))
                for v in variables
                if v in pre and isinstance(pre[v], (int, float))
            ]
            if substitutions:
                expr = z3.substitute(expr, *substitutions)

            expected = z3.RealVal(post[var])
            solver.add(expr == expected)

        if not valid or not has_constraints:
            continue

        # Check satisfiability.
        if solver.check() == z3.sat:
            return candidate

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Per-transition update synthesis
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


def synthesise_postconditions(efsm: EFSM) -> EFSM:
    """Annotate every transition in *efsm* with update rules.

    Strategy (per specification §4 Step 5):
      1. Try the fast template  x' = c₁·x + c₂  via Z3.
      2. If that fails, fall back to GetAbductUCL (enumerative
         abduction with UNSAT-core learning).

    Pre/post observation pairs are stored directly on each
    transition (populated during PTA construction and preserved
    through state merging).

    Returns a **new** EFSM with ``update_rule`` populated.
    """
    efsm = efsm.deep_copy()
    variables = sorted(efsm.variables)

    for transition in efsm.transitions:
        pre_post_pairs = transition.pre_post_pairs

        if not pre_post_pairs:
            continue

        update_rule: dict[str, z3.ExprRef] = {}

        for var in variables:
            # Attempt 1: template synthesis  x' = c₁x + c₂.
            # result = _synthesise_update_template(var, pre_post_pairs, variables)
            # if result is not None:
            #     update_rule[var] = result
            #     continue

            # Attempt 2: GetAbductUCL (Reynolds et al.)
            abduct = _get_abduct_ucl(var, pre_post_pairs, variables)
            if abduct is not None:
                z3_vars = {v: z3.Real(v) for v in variables}
                update_rule[var] = abduct.to_z3(z3_vars)
            # else: no update discovered → variable unchanged (identity).

        if update_rule:
            transition.update_rule = update_rule

    return efsm
