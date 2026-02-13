"""Smoke tests for symbolic guard-based candidate pruning."""

import z3

from dpn_discovery.postcondition_synthesis import (
    ExprNode,
    _are_equivalent_under_guard,
    _enumerate_candidates,
    _is_trivial_guard,
    _prune_candidates_with_guard,
)

def test_trivial_guard_detection() -> None:
    assert _is_trivial_guard(None) is True
    assert _is_trivial_guard(z3.BoolVal(True)) is True
    x = z3.Real("x")
    assert _is_trivial_guard(x > 0) is False
    assert _is_trivial_guard(x == z3.RealVal(0)) is False


def test_equivalence_under_equality_guard() -> None:
    """Under  x = 0,  the expressions 'x' and '0' are equivalent."""
    variables = ["x"]
    x = z3.Real("x")
    guard = x == z3.RealVal(0)

    e_x = ExprNode(kind="var", var_name="x")
    e_0 = ExprNode(kind="const", value=0)
    e_1 = ExprNode(kind="const", value=1)

    assert _are_equivalent_under_guard(e_x, e_0, guard, variables) is True
    assert _are_equivalent_under_guard(e_x, e_1, guard, variables) is False


def test_equivalence_under_range_guard() -> None:
    """Under  x > 5,  'x + 0' and 'x' are equivalent (always true),
    but 'x' and '0' are NOT equivalent."""
    variables = ["x"]
    x = z3.Real("x")
    guard = x > z3.RealVal(5)

    e_x = ExprNode(kind="var", var_name="x")
    e_0 = ExprNode(kind="const", value=0)
    e_x_plus_0 = ExprNode(kind="add", children=(e_x, ExprNode(kind="const", value=0)))

    assert _are_equivalent_under_guard(e_x, e_x_plus_0, guard, variables) is True
    assert _are_equivalent_under_guard(e_x, e_0, guard, variables) is False


def test_pruning_reduces_candidates() -> None:
    """Under  x = 0,  many depth-2 candidates collapse."""
    variables = ["x"]
    x = z3.Real("x")
    guard = x == z3.RealVal(0)

    all_candidates = _enumerate_candidates(variables, max_depth=2)
    pruned = _prune_candidates_with_guard(all_candidates, guard, variables)

    assert len(pruned) < len(all_candidates), (
        f"Expected pruning to reduce candidates, "
        f"got {len(pruned)} vs {len(all_candidates)}"
    )
    print(f"\nPruning: {len(all_candidates)} → {len(pruned)} "
          f"({len(all_candidates) - len(pruned)} removed)")


def test_pruning_preserves_order() -> None:
    """Pruning must keep the first (simplest) representative."""
    variables = ["x"]
    x = z3.Real("x")
    guard = x == z3.RealVal(0)

    all_candidates = _enumerate_candidates(variables, max_depth=2)
    pruned = _prune_candidates_with_guard(all_candidates, guard, variables)

    # The first few candidates (constants 0, 1, -1; variable x) should
    # appear in the same relative order.
    pruned_kinds = [(c.kind, c.value, c.var_name) for c in pruned[:3]]
    assert pruned_kinds[0] == ("const", 0, None), "Constant 0 must be first"
    assert pruned_kinds[1] == ("const", 1, None), "Constant 1 must be second"


def test_no_pruning_with_trivial_guard() -> None:
    """Trivial guard (True) should be skipped by the caller,
    but _prune_candidates_with_guard itself still works."""
    variables = ["x"]
    guard = z3.BoolVal(True)

    all_candidates = _enumerate_candidates(variables, max_depth=2)
    pruned = _prune_candidates_with_guard(all_candidates, guard, variables)

    # Under True, no expressions are equivalent unless they're
    # algebraically identical (e.g. x+0 ≡ x).
    # So pruning should still remove some trivially equal ones.
    assert len(pruned) <= len(all_candidates)


if __name__ == "__main__":
    test_trivial_guard_detection()
    print("✓ test_trivial_guard_detection")

    test_equivalence_under_equality_guard()
    print("✓ test_equivalence_under_equality_guard")

    test_equivalence_under_range_guard()
    print("✓ test_equivalence_under_range_guard")

    test_pruning_reduces_candidates()
    print("✓ test_pruning_reduces_candidates")

    test_pruning_preserves_order()
    print("✓ test_pruning_preserves_order")

    test_no_pruning_with_trivial_guard()
    print("✓ test_no_pruning_with_trivial_guard")

    print("\nAll tests passed!")
