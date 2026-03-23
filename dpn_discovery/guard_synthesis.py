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
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from dpn_discovery.smt import get_solver, SMTSolver, SMTBool, SMTArith, SatResult

logger = logging.getLogger(__name__)

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

    def to_smt(self, smt_vars: dict[str, SMTArith], solver: SMTSolver) -> SMTBool:
        """Compile this AST into an SMT Boolean expression."""
        match self.kind:
            case "leq":
                return solver.LE(smt_vars[self.variable], solver.RealVal(self.constant))
            case "gt":
                return solver.GT(smt_vars[self.variable], solver.RealVal(self.constant))
            case "eq":
                return solver.Eq(smt_vars[self.variable], solver.RealVal(self.constant))
            case "and":
                return solver.And(*(c.to_smt(smt_vars, solver) for c in self.children))
            case "or":
                return solver.Or(*(c.to_smt(smt_vars, solver) for c in self.children))
            case "not":
                return solver.Not(self.children[0].to_smt(smt_vars, solver))
            case "true":
                return solver.BoolVal(True)
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

    def evaluate(self, sample: dict[str, Any]) -> bool | None:
        """Evaluate the guard under a concrete variable assignment.

        Returns ``True`` / ``False`` when all referenced variables
        are present and numeric in *sample*, or ``None`` when a
        required variable is missing (caller should fall back to SMT).
        """
        match self.kind:
            case "leq":
                v = sample.get(self.variable)
                return v <= self.constant if isinstance(v, (int, float)) else None
            case "gt":
                v = sample.get(self.variable)
                return v > self.constant if isinstance(v, (int, float)) else None
            case "eq":
                v = sample.get(self.variable)
                return abs(v - self.constant) < 1e-9 if isinstance(v, (int, float)) else None
            case "and":
                for c in self.children:
                    r = c.evaluate(sample)
                    if r is None:
                        return None
                    if not r:
                        return False
                return True
            case "or":
                for c in self.children:
                    r = c.evaluate(sample)
                    if r is None:
                        return None
                    if r:
                        return True
                return False
            case "not":
                r = self.children[0].evaluate(sample)
                return (not r) if r is not None else None
            case "true":
                return True
            case _:
                return None


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


def _check_disjoint_from(
    candidate: GuardNode,
    prev_guards: list[GuardNode],
    variables: list[str],
) -> bool:
    """Check that *candidate* is symbolically disjoint from every
    guard in *prev_guards*.

    For each previous guard φ, checks that  φ ∧ candidate  is
    UNSAT — i.e. no variable assignment can satisfy both
    simultaneously.
    """
    if not prev_guards:
        return True

    smt = get_solver()
    smt_vars = {v: smt.Real(v) for v in variables}
    cand_expr = candidate.to_smt(smt_vars, smt)

    for prev in prev_guards:
        prev_expr = prev.to_smt(smt_vars, smt)
        with smt.create_context(timeout_ms=3000) as ctx:
            ctx.add(cand_expr)
            ctx.add(prev_expr)
            if ctx.check() != SatResult.UNSAT:
                return False
    return True


def _search_guard(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    variables: list[str],
    phog: PHOGModel,
    max_candidates: int = 5000,
    disjoint_from: list[GuardNode] | None = None,
) -> GuardNode | None:
    """A*-search for the minimal guard that covers *positive*
    and rejects *negative*, optionally disjoint from previously
    synthesised guards.

    Parameters
    ----------
    disjoint_from : list[GuardNode] | None
        When provided, the returned guard is guaranteed to be
        symbolically disjoint from every guard in this list
        (i.e. their conjunction is UNSAT for all variable
        assignments).

    Returns ``None`` if no satisfying guard is found within the
    exploration budget.
    """
    all_samples = positive + negative
    thresholds = _compute_thresholds(all_samples, variables)
    atoms = _enumerate_atomic_candidates(variables, thresholds)

    if not atoms:
        # No numeric variables → return trivial guard.
        return GuardNode(kind="true")

    ctx_root = PHOGContext(
        parent_symbol="Guard", left_sibling=None, depth=0, grandparent=None
    )

    # --- Quick feasibility pre-check ------------------------------------
    # If no single atom can separate *any* positive from *any* negative
    # sample, composites won't help either — bail out immediately.
    best_score = 0.0
    best_atom: GuardNode | None = None
    n_total = len(positive) + len(negative)
    for atom in atoms:
        tp = sum(1 for s in positive if atom.evaluate(s) is True)
        tn = sum(1 for s in negative if atom.evaluate(s) in (False, None))
        score = (tp + tn) / n_total if n_total else 0.0
        # Perfect atom — return immediately, skip A*.
        if tp == len(positive) and tn == len(negative):
            if disjoint_from and not _check_disjoint_from(atom, disjoint_from, variables):
                continue
            phog.observe(ctx_root, atom.kind)
            return atom
        if score > best_score:
            best_score = score
            best_atom = atom
    # If the best single atom can't beat random guessing, the search
    # over conjunctions / disjunctions won't find anything useful.
    if best_score <= 0.5:
        return None
    # Scale the exploration budget by how promising the best atom looks.
    # A best-score of 0.95 keeps the full budget; 0.6 limits to ~20%.
    if best_score < 0.95:
        scale = max(0.1, (best_score - 0.5) / (0.95 - 0.5))
        max_candidates = max(50, int(max_candidates * scale))

    # --- Phase 1: atomic candidates (depth 1) ----------------------------
    heap: list[_SearchNode] = []

    for atom in atoms:
        log_p = phog.log_probability(ctx_root, atom.kind)
        cost = -log_p + _admissible_heuristic(atom)
        heapq.heappush(heap, _SearchNode(priority=cost, candidate=atom, depth=1))

    visited = 0
    while heap and visited < max_candidates:
        node = heapq.heappop(heap)
        visited += 1

        if _verify_guard_fast(node.candidate, positive, negative, variables):
            # Check symbolic disjointness with previously synthesised
            # guards (sequential partition enforcement).
            if disjoint_from and not _check_disjoint_from(
                node.candidate, disjoint_from, variables
            ):
                # Overlaps with a previous guard — keep searching.
                continue
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

def _verify_guard_fast(
    guard: GuardNode,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    variables: list[str],
) -> bool:
    """Fast pure-Python guard verification.

    Uses ``GuardNode.evaluate()`` for concrete evaluation.
    Falls back to the SMT-based ``_verify_guard()`` only when
    ``evaluate`` returns ``None`` (i.e. a required variable is
    missing from a sample).
    """
    # Coverage: guard must be True for every positive sample.
    for sample in positive:
        result = guard.evaluate(sample)
        if result is None:
            # Fallback to SMT for this candidate.
            return _verify_guard(guard, positive, negative, variables)
        if not result:
            return False

    # Exclusion: guard must be False for every negative sample.
    for sample in negative:
        result = guard.evaluate(sample)
        if result is None:
            return _verify_guard(guard, positive, negative, variables)
        if result:
            return False

    return True


def _verify_guard(
    guard: GuardNode,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    variables: list[str],
) -> bool:
    """Check that *guard* satisfies:

    1. **Coverage**:  ∀ d ∈ positive:  guard(d) = True.
    2. **Exclusion**: ∀ d ∈ negative:  guard(d) = False.

    All checking is delegated to the SMT solver (spec §6 constraint 3).
    """
    smt = get_solver()
    smt_vars = {v: smt.Real(v) for v in variables}

    with smt.create_context(timeout_ms=3000) as ctx:
        guard_expr = guard.to_smt(smt_vars, smt)

        # Coverage: guard must be True for every positive sample.
        for sample in positive:
            substituted = _substitute(guard_expr, smt_vars, sample)
            ctx.add(substituted)

        # Exclusion: guard must be False for every negative sample.
        for sample in negative:
            substituted = _substitute(guard_expr, smt_vars, sample)
            ctx.add(smt.Not(substituted))

        return ctx.check() == SatResult.SAT


def _substitute(
    expr: SMTBool,
    smt_vars: dict[str, SMTArith],
    sample: dict[str, Any],
) -> SMTBool:
    """Replace every SMT variable in *expr* with its concrete
    value from *sample*.
    """
    smt = get_solver()
    substitutions = [
        (smt_vars[v], smt.RealVal(sample[v]))
        for v in smt_vars
        if v in sample and isinstance(sample[v], (int, float))
    ]
    return smt.substitute(expr, *substitutions)


def _verify_partition_smt(
    guard_nodes: list[GuardNode],
    variables: list[str],
    context: str,
) -> bool:
    """Verify that *guard_nodes* form a partition of the
    variable space — pairwise disjoint and jointly exhaustive.

    All guards are re-compiled from their AST representation
    using a **single** set of SMT variables, ensuring that
    solver backends with per-call variable scoping (e.g. Yices2)
    produce correct results.

    Returns ``True`` iff the partition property holds.  Logs
    warnings for every violation detected.
    """
    if len(guard_nodes) < 2:
        return True

    smt = get_solver()
    smt_vars = {v: smt.Real(v) for v in variables}
    guard_formulas = [gn.to_smt(smt_vars, smt) for gn in guard_nodes]
    all_ok = True

    # 1. Pairwise disjointness:  φᵢ ∧ φⱼ  must be UNSAT ∀ i ≠ j.
    for i in range(len(guard_formulas)):
        for j in range(i + 1, len(guard_formulas)):
            with smt.create_context(timeout_ms=3000) as ctx:
                ctx.add(guard_formulas[i])
                ctx.add(guard_formulas[j])
                if ctx.check() != SatResult.UNSAT:
                    logger.warning(
                        "  Partition violation (%s): guards %d and %d overlap",
                        context, i, j,
                    )
                    all_ok = False

    # 2. Exhaustive coverage:  ¬(φ₁ ∨ … ∨ φₖ)  must be UNSAT.
    with smt.create_context(timeout_ms=3000) as ctx:
        ctx.add(smt.Not(smt.Or(*guard_formulas)))
        if ctx.check() != SatResult.UNSAT:
            logger.warning(
                "  Partition violation (%s): guards are not exhaustive",
                context,
            )
            all_ok = False

    if all_ok:
        logger.info(
            "  Partition verified (%s): %d guards are pairwise "
            "disjoint and exhaustive",
            context, len(guard_formulas),
        )
    return all_ok


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
    states_sorted = sorted(efsm.states)
    total_states = len(states_sorted)

    logger.info("Guard synthesis: %d states, %d transitions, %d variables",
                total_states, len(efsm.transitions), len(variables))

    for state_idx, state in enumerate(states_sorted, 1):
        outgoing = efsm.outgoing(state)
        if not outgoing:
            continue

        by_activity = efsm.outgoing_by_activity(state)

        # --- Case 1: same-activity competition ---
        for activity, transitions in by_activity.items():
            if len(transitions) >= 2:
                logger.info("  [%d/%d] State %s: synthesising guards for %d competing '%s' transitions",
                            state_idx, total_states, state, len(transitions), activity)
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
            logger.info("  [%d/%d] State %s: pairwise cross-activity guards for %s",
                        state_idx, total_states, state, unguarded_activities)
            _synthesise_pairwise_activity_guards(
                by_activity, unguarded_activities, variables, phog
            )

        # --- Assign ⊤ to any transition still without a guard ---
        for t in outgoing:
            if t.guard_formula is None:
                t.guard_formula = get_solver().BoolVal(True)

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
    """Synthesise guards for cross-activity competition.

    Uses **sequential one-vs-rest** synthesis with disjointness
    constraints to guarantee a pairwise-disjoint and jointly-
    exhaustive guard partition:

      1. For branches 1 … k−1: find a guard that separates the
         branch's samples from all others AND is symbolically
         disjoint from every previously synthesised guard.
      2. For the last branch k: assign  ¬(φ₁ ∨ … ∨ φₖ₋₁),
         which is correct by construction.

    When the data is inseparable the transition stays ⊤.
    """
    # Identify activities that still need guards.
    unguarded = [
        act for act in activities
        if all(t.guard_formula is None for t in by_activity[act])
    ]

    if len(unguarded) < 2:
        return

    synthesized_nodes: list[GuardNode] = []
    all_guard_nodes: list[GuardNode] = []

    for idx, act in enumerate(unguarded):
        ts_act = by_activity[act]

        # Skip if already guarded (e.g. by same-activity handler).
        if all(t.guard_formula is not None for t in ts_act):
            continue

        is_last = (idx == len(unguarded) - 1)
        all_previous_ok = (len(synthesized_nodes) == idx)

        # ── Last branch: assign  ¬(φ₁ ∨ … ∨ φₖ₋₁)  ────────────────
        if is_last and all_previous_ok and synthesized_nodes:
            if len(synthesized_nodes) == 1:
                neg_node = GuardNode(
                    kind="not", children=(synthesized_nodes[0],),
                )
            else:
                or_node = GuardNode(
                    kind="or", children=tuple(synthesized_nodes),
                )
                neg_node = GuardNode(
                    kind="not", children=(or_node,),
                )

            smt = get_solver()
            smt_vars = {v: smt.Real(v) for v in variables}
            guard_expr = neg_node.to_smt(smt_vars, smt)
            for t in ts_act:
                if t.guard_formula is None:
                    t.guard_formula = guard_expr
            all_guard_nodes.append(neg_node)
            logger.debug(
                "  Last-branch negation guard for '%s': %s",
                act, neg_node.pretty(),
            )
            continue

        # ── One-vs-rest with disjointness constraints ───────────────
        positive: list[dict[str, Any]] = []
        for t in ts_act:
            positive.extend(_guard_samples(t))

        negative: list[dict[str, Any]] = []
        for other_act in activities:
            if other_act == act:
                continue
            for t in by_activity[other_act]:
                negative.extend(_guard_samples(t))

        if not positive or not negative:
            continue

        guard_node = _search_guard(
            positive, negative, variables, phog,
            disjoint_from=synthesized_nodes if synthesized_nodes else None,
        )

        if guard_node is not None:
            synthesized_nodes.append(guard_node)
            all_guard_nodes.append(guard_node)
            smt = get_solver()
            smt_vars = {v: smt.Real(v) for v in variables}
            guard_expr = guard_node.to_smt(smt_vars, smt)
            for t in ts_act:
                if t.guard_formula is None:
                    t.guard_formula = guard_expr

    # ── Post-synthesis partition verification ────────────────────────
    if len(all_guard_nodes) >= 2:
        _verify_partition_smt(all_guard_nodes, variables, "cross-activity")


def _synthesise_competing_guards(
    transitions: list[Transition],
    variables: list[str],
    phog: PHOGModel,
) -> None:
    """Synthesise pairwise-disjoint guards for *transitions*
    that share a source state and activity label.

    Uses sequential synthesis with disjointness constraints.
    The last transition receives  ¬(φ₁ ∨ … ∨ φₖ₋₁)  to
    guarantee an exhaustive and disjoint partition.
    """
    if len(transitions) < 2:
        # Single transition — trivially guarded.
        if transitions:
            transitions[0].guard_formula = get_solver().BoolVal(True)
        return

    synthesized_nodes: list[GuardNode] = []
    all_guard_nodes: list[GuardNode] = []

    for i, t_i in enumerate(transitions):
        is_last = (i == len(transitions) - 1)
        all_previous_ok = (len(synthesized_nodes) == i)

        # ── Last transition: negation ────────────────────────────────
        if is_last and all_previous_ok and synthesized_nodes:
            if len(synthesized_nodes) == 1:
                neg_node = GuardNode(
                    kind="not", children=(synthesized_nodes[0],),
                )
            else:
                or_node = GuardNode(
                    kind="or", children=tuple(synthesized_nodes),
                )
                neg_node = GuardNode(
                    kind="not", children=(or_node,),
                )

            smt = get_solver()
            smt_vars = {v: smt.Real(v) for v in variables}
            t_i.guard_formula = neg_node.to_smt(smt_vars, smt)
            all_guard_nodes.append(neg_node)
            continue

        # ── One-vs-rest with disjointness constraints ────────────────
        positive = _guard_samples(t_i)
        negative: list[dict[str, Any]] = []
        for j, t_j in enumerate(transitions):
            if i != j:
                negative.extend(_guard_samples(t_j))

        if not positive:
            t_i.guard_formula = get_solver().BoolVal(True)
            continue

        guard_node = _search_guard(
            positive, negative, variables, phog,
            disjoint_from=synthesized_nodes if synthesized_nodes else None,
        )

        if guard_node is not None:
            synthesized_nodes.append(guard_node)
            all_guard_nodes.append(guard_node)
            smt = get_solver()
            smt_vars = {v: smt.Real(v) for v in variables}
            t_i.guard_formula = guard_node.to_smt(smt_vars, smt)
        else:
            # Fallback: unconstrained guard (best-effort).
            t_i.guard_formula = get_solver().BoolVal(True)

    # ── Post-synthesis partition verification ────────────────────────
    if len(all_guard_nodes) >= 2:
        _verify_partition_smt(all_guard_nodes, variables, "same-activity")
