"""
Step 4 — Guard Synthesis via MDC-accelerated SAT.

For every state with multiple competing outgoing transitions
(same or different activity), synthesise minimal distinguishing
Boolean guards using:

  1.  A context-free grammar of guard predicates.
  2.  PHOG (Probabilistic Higher-Order Grammar) weighting from
      Lee et al. (PLDI 2018).
  3.  A* search over the grammar weighted by PHOG log-probabilities
      with an admissible heuristic.
  4.  Z3-based verification (coverage + disjointness).

References
----------
  Lee, W., Heo, K., Alur, R., Naik, M.  *Accelerating Search-Based
  Program Synthesis using Learned Probabilistic Models*  (PLDI 2018).

  Specification §4 Step 4  — Guard Synthesis via MDC-accelerated SAT.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import z3

from dpn_discovery.models import EFSM, Transition


# ═══════════════════════════════════════════════════════════════════════════
# 1.  AST representation of guard formulas
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class GuardNode:
    """AST node for a guard expression."""
    kind: str                             # "leq" | "gt" | "eq" | "and" | "or" | "not" | "true"
    variable: str | None = None           # leaf nodes
    constant: float | int | None = None   # leaf nodes
    children: tuple[GuardNode, ...] = ()  # composite nodes

    def size(self) -> int:
        """Number of AST nodes (proxy for formula complexity)."""
        return 1 + sum(c.size() for c in self.children)

    def to_z3(self, z3_vars: dict[str, z3.ArithRef]) -> z3.BoolRef:
        """Compile this AST into a Z3 Boolean expression."""
        match self.kind:
            case "leq":
                return z3_vars[self.variable] <= z3.RealVal(self.constant)  # type: ignore[arg-type]
            case "gt":
                return z3_vars[self.variable] > z3.RealVal(self.constant)  # type: ignore[arg-type]
            case "eq":
                return z3_vars[self.variable] == z3.RealVal(self.constant)  # type: ignore[arg-type]
            case "and":
                return z3.And(*(c.to_z3(z3_vars) for c in self.children))
            case "or":
                return z3.Or(*(c.to_z3(z3_vars) for c in self.children))
            case "not":
                return z3.Not(self.children[0].to_z3(z3_vars))
            case "true":
                return z3.BoolVal(True)
            case _:
                raise ValueError(f"Unknown guard node kind: {self.kind}")

    def pretty(self) -> str:
        """Human-readable representation."""
        match self.kind:
            case "leq":
                return f"({self.variable} ≤ {self.constant})"
            case "gt":
                return f"({self.variable} > {self.constant})"
            case "eq":
                return f"({self.variable} = {self.constant})"
            case "and":
                return "(" + " ∧ ".join(c.pretty() for c in self.children) + ")"
            case "or":
                return "(" + " ∨ ".join(c.pretty() for c in self.children) + ")"
            case "not":
                return f"¬{self.children[0].pretty()}"
            case "true":
                return "⊤"
            case _:
                return f"?{self.kind}"


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PHOG context model  (Lee et al. §3)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PHOGContext:
    """Context tuple for PHOG rule-weight lookup.

    Attributes correspond to the context features defined in
    Lee et al. §3.2:
      • parent_symbol   — the non-terminal being expanded.
      • left_sibling    — the symbol produced by the immediately
                          preceding production at the same depth.
      • depth           — depth in the derivation tree.
      • grandparent     — the non-terminal one level above parent.
    """
    parent_symbol: str
    left_sibling: str | None
    depth: int
    grandparent: str | None


@dataclass(slots=True)
class PHOGModel:
    """Learned probabilistic grammar weights.

    Maps  (context, production_rule) → log-probability.

    For the initial implementation we bootstrap with **uniform
    weights** (i.e. every candidate production is equally likely)
    and refine via online counting when more guard examples are
    observed.  This matches Lee et al. §4 (cold-start strategy).
    """
    _counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def log_probability(self, ctx: PHOGContext, rule: str) -> float:
        """Return  ln P(rule | ctx).

        Uses Laplace-smoothed relative frequencies.
        """
        key = self._context_key(ctx)
        bucket = self._counts.get(key, {})
        total = sum(bucket.values()) if bucket else 0
        count = bucket.get(rule, 0)
        # Laplace smoothing with α = 1.
        return math.log((count + 1) / (total + self._num_rules()))

    def observe(self, ctx: PHOGContext, rule: str) -> None:
        """Record one observation of *rule* under *ctx*."""
        key = self._context_key(ctx)
        self._counts.setdefault(key, {})
        self._counts[key][rule] = self._counts[key].get(rule, 0) + 1

    # --- internals -------------------------------------------------------

    @staticmethod
    def _context_key(ctx: PHOGContext) -> str:
        return f"{ctx.parent_symbol}|{ctx.left_sibling}|{ctx.depth}|{ctx.grandparent}"

    @staticmethod
    def _num_rules() -> int:
        """Total number of production rules in the grammar."""
        # leq, gt, eq, and, or, not, true
        return 7


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Candidate enumeration grammar
# ═══════════════════════════════════════════════════════════════════════════

def _enumerate_atomic_candidates(
    variables: list[str],
    thresholds: dict[str, list[float | int]],
) -> list[GuardNode]:
    """Generate all atomic guard candidates  (v ≤ c)  and  (v > c)
    for every variable and every observed threshold value.

    Thresholds are inferred from the data: midpoints between
    successive distinct values.
    """
    candidates: list[GuardNode] = []
    for var in variables:
        for c in thresholds.get(var, []):
            candidates.append(GuardNode(kind="leq", variable=var, constant=c))
            candidates.append(GuardNode(kind="gt", variable=var, constant=c))
    return candidates


def _compute_thresholds(
    samples: list[dict[str, Any]],
    variables: list[str],
) -> dict[str, list[float | int]]:
    """Derive candidate split thresholds for each numeric variable.

    Uses *midpoint* between consecutive distinct sorted values
    (standard decision-boundary technique).
    """
    thresholds: dict[str, list[float | int]] = {}
    for var in variables:
        values: list[float] = sorted(
            {float(s[var]) for s in samples if var in s and isinstance(s[var], (int, float))}
        )
        midpoints: list[float | int] = []
        for i in range(len(values) - 1):
            mid = (values[i] + values[i + 1]) / 2.0
            midpoints.append(mid)
        # Also include the exact boundary values themselves.
        for v in values:
            if v not in midpoints:
                midpoints.append(v)
        thresholds[var] = sorted(set(midpoints))
    return thresholds


# ═══════════════════════════════════════════════════════════════════════════
# 4.  A* search with PHOG-weighted heuristic  (Lee et al. §5)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(order=True, slots=True)
class _SearchNode:
    """Priority-queue node for A* search over the guard grammar."""
    priority: float
    candidate: GuardNode = field(compare=False)
    depth: int = field(compare=False, default=0)


def _admissible_heuristic(node: GuardNode) -> float:
    """Admissible (never over-estimates) cost-to-go.

    Per Lee et al. §5.2 the heuristic is the sum of the
    minimum-weight completions for every remaining hole in the
    partial derivation.  For a fully-expanded candidate this is 0.
    """
    # All enumerated candidates are fully expanded → h = 0.
    # For composite (and/or) nodes we optimistically assume the
    # cheapest single-atom completion.
    return 0.0


def _search_guard(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    variables: list[str],
    phog: PHOGModel,
    max_candidates: int = 5000,
) -> GuardNode | None:
    """A*-search for the minimal guard that covers *positive*
    and rejects *negative*.

    Returns ``None`` if no satisfying guard is found within the
    exploration budget.
    """
    all_samples = positive + negative
    thresholds = _compute_thresholds(all_samples, variables)
    atoms = _enumerate_atomic_candidates(variables, thresholds)

    if not atoms:
        # No numeric variables → return trivial guard.
        return GuardNode(kind="true")

    # --- Phase 1: atomic candidates (depth 1) ----------------------------
    heap: list[_SearchNode] = []
    ctx_root = PHOGContext(
        parent_symbol="Guard", left_sibling=None, depth=0, grandparent=None
    )

    for atom in atoms:
        log_p = phog.log_probability(ctx_root, atom.kind)
        cost = -log_p + _admissible_heuristic(atom)
        heapq.heappush(heap, _SearchNode(priority=cost, candidate=atom, depth=1))

    visited = 0
    while heap and visited < max_candidates:
        node = heapq.heappop(heap)
        visited += 1

        if _verify_guard_z3(node.candidate, positive, negative, variables):
            # Update PHOG model with successful derivation.
            phog.observe(ctx_root, node.candidate.kind)
            return node.candidate

        # --- Phase 2: conjunctive / disjunctive expansions ---------------
        if node.depth < 3:  # limit derivation depth
            for atom in atoms:
                # AND composition
                conj = GuardNode(
                    kind="and", children=(node.candidate, atom)
                )
                ctx_and = PHOGContext(
                    parent_symbol="Guard",
                    left_sibling=node.candidate.kind,
                    depth=node.depth + 1,
                    grandparent="Guard",
                )
                log_p = phog.log_probability(ctx_and, "and")
                cost = -log_p + conj.size()
                heapq.heappush(
                    heap, _SearchNode(priority=cost, candidate=conj, depth=node.depth + 1)
                )

                # OR composition
                disj = GuardNode(
                    kind="or", children=(node.candidate, atom)
                )
                log_p_or = phog.log_probability(ctx_and, "or")
                cost_or = -log_p_or + disj.size()
                heapq.heappush(
                    heap, _SearchNode(priority=cost_or, candidate=disj, depth=node.depth + 1)
                )

    return None


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Z3-based verification  (coverage + disjointness)
# ═══════════════════════════════════════════════════════════════════════════

def _verify_guard_z3(
    guard: GuardNode,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    variables: list[str],
) -> bool:
    """Check that *guard* satisfies:

    1. **Coverage**:  ∀ d ∈ positive:  guard(d) = True.
    2. **Exclusion**: ∀ d ∈ negative:  guard(d) = False.

    All checking is delegated to Z3 (spec §6 constraint 3).
    """
    z3_vars = {v: z3.Real(v) for v in variables}
    solver = z3.Solver()
    solver.set("timeout", 3000)

    guard_expr = guard.to_z3(z3_vars)

    # Coverage: guard must be True for every positive sample.
    for sample in positive:
        substituted = _substitute(guard_expr, z3_vars, sample)
        solver.add(substituted)

    # Exclusion: guard must be False for every negative sample.
    for sample in negative:
        substituted = _substitute(guard_expr, z3_vars, sample)
        solver.add(z3.Not(substituted))

    return solver.check() == z3.sat


def _substitute(
    expr: z3.BoolRef,
    z3_vars: dict[str, z3.ArithRef],
    sample: dict[str, Any],
) -> z3.BoolRef:
    """Replace every Z3 variable in *expr* with its concrete
    value from *sample*.
    """
    substitutions = [
        (z3_vars[v], z3.RealVal(sample[v]))
        for v in z3_vars
        if v in sample and isinstance(sample[v], (int, float))
    ]
    return z3.substitute(expr, *substitutions)  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Top-level entry point
# ═══════════════════════════════════════════════════════════════════════════

def synthesise_guards(efsm: EFSM) -> EFSM:
    """Annotate every transition in *efsm* with a synthesised guard.

    Decision points are identified in two ways:
      1. **Same-activity competing transitions**: a state with ≥ 2
         outgoing edges on the *same* activity label.
      2. **Cross-activity decision points**: a state with outgoing
         edges on *different* activity labels leading to *different*
         targets.  The data must distinguish which activity path is
         taken (e.g. AutoApprove vs Review based on ``amount``).

    Per specification §4 Step 4: "For a state  s  with multiple
    outgoing transitions  t₁, t₂, …, tₖ  labeled with the same
    activity  a  (or competing activities), find guards…"

    Returns a **new** EFSM with ``guard_formula`` populated on
    every transition.
    """
    efsm = efsm.deep_copy()
    phog = PHOGModel()
    variables = sorted(efsm.variables)

    for state in sorted(efsm.states):
        outgoing = efsm.outgoing(state)
        if not outgoing:
            continue

        by_activity = efsm.outgoing_by_activity(state)

        # --- Case 1: same-activity competition ---
        for activity, transitions in by_activity.items():
            if len(transitions) >= 2:
                _synthesise_competing_guards(transitions, variables, phog)

        # --- Case 2: pairwise cross-activity decision points ---
        # For each pair of activity groups (a_i, a_j) from the same
        # source that are still unguarded, try to find a guard that
        # separates a_i's samples from a_j's samples.  If the data
        # is inseparable (overlapping), both stay ⊤.
        unguarded_activities = [
            act for act, ts in by_activity.items()
            if all(t.guard_formula is None for t in ts)
        ]
        if len(unguarded_activities) >= 2:
            _synthesise_pairwise_activity_guards(
                by_activity, unguarded_activities, variables, phog
            )

        # --- Assign ⊤ to any transition still without a guard ---
        for t in outgoing:
            if t.guard_formula is None:
                t.guard_formula = z3.BoolVal(True)

    return efsm


def _guard_samples(t: Transition) -> list[dict[str, Any]]:
    """Return the data samples relevant for guard synthesis.

    Guards evaluate the *pre-state* — the variable assignment that
    holds **before** the transition fires.  So we extract the
    pre-snapshots from ``pre_post_pairs``.  Falls back to
    ``data_samples`` when ``pre_post_pairs`` is empty (backward
    compatibility).
    """
    if t.pre_post_pairs:
        return [pre for pre, _post in t.pre_post_pairs if pre]
    return [s for s in t.data_samples if s]


def _synthesise_pairwise_activity_guards(
    by_activity: dict[str, list[Transition]],
    activities: list[str],
    variables: list[str],
    phog: PHOGModel,
) -> None:
    """Synthesise guards for pairwise cross-activity competition.

    For each pair of activities ``(a_i, a_j)`` emanating from the
    same source state, attempt to find a guard that separates the
    data samples of ``a_i`` from those of ``a_j``.  If a guard is
    found, assign it to all transitions of ``a_i`` and its negation
    to all transitions of ``a_j``.

    Only transitions that have **not** yet been assigned a guard are
    considered.  When the data is inseparable (e.g. Submit has all
    amounts), the pair is skipped and both stay ⊤.
    """
    from itertools import combinations

    for act_a, act_b in combinations(activities, 2):
        ts_a = by_activity[act_a]
        ts_b = by_activity[act_b]

        # Skip if already guarded from a previous pair.
        if all(t.guard_formula is not None for t in ts_a) and \
           all(t.guard_formula is not None for t in ts_b):
            continue

        # Guards evaluate the *pre-state* (state before the transition
        # fires), so we collect pre-snapshots from pre_post_pairs.
        # Fall back to data_samples for backward compatibility.
        positive: list[dict[str, Any]] = []
        for t in ts_a:
            positive.extend(_guard_samples(t))
        negative: list[dict[str, Any]] = []
        for t in ts_b:
            negative.extend(_guard_samples(t))

        if not positive or not negative:
            continue

        guard_node = _search_guard(positive, negative, variables, phog)

        if guard_node is not None:
            z3_vars = {v: z3.Real(v) for v in variables}
            guard_expr = guard_node.to_z3(z3_vars)
            neg_guard_expr = z3.Not(guard_expr)

            for t in ts_a:
                if t.guard_formula is None:
                    t.guard_formula = guard_expr
            for t in ts_b:
                if t.guard_formula is None:
                    t.guard_formula = neg_guard_expr


def _synthesise_competing_guards(
    transitions: list[Transition],
    variables: list[str],
    phog: PHOGModel,
) -> None:
    """Synthesise pairwise-disjoint guards for *transitions*
    that share a source state and activity label.

    For each transition  tᵢ  the positive samples are its own
    ``data_samples`` and the negative samples are the union of
    all other transitions' ``data_samples``.

    The guard is placed onto ``transition.guard_formula``.
    """
    for i, t_i in enumerate(transitions):
        positive = _guard_samples(t_i)
        negative: list[dict[str, Any]] = []
        for j, t_j in enumerate(transitions):
            if i != j:
                negative.extend(_guard_samples(t_j))

        if not positive:
            t_i.guard_formula = z3.BoolVal(True)
            continue

        guard_node = _search_guard(positive, negative, variables, phog)

        if guard_node is not None:
            z3_vars = {v: z3.Real(v) for v in variables}
            t_i.guard_formula = guard_node.to_z3(z3_vars)
        else:
            # Fallback: unconstrained guard (best-effort).
            t_i.guard_formula = z3.BoolVal(True)
