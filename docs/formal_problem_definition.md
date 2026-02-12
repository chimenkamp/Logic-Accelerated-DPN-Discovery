# Formal Problem Definitions

This document provides the formal problem statements for the two
synthesis tasks that decorate the merged EFSM with semantic
information — **Guard Synthesis** and **Postcondition Synthesis** —
and clarifies the role of data in the preceding **State Merging**
step.

Each section states (i) the original problem as formulated in the
reference paper, (ii) how we adapt it to the domain of Data-Aware
Petri Net discovery, and (iii) the concrete grammar, context model,
and solver encoding used in the implementation.

---

## 1  State Merging: Control-Flow *and* Data Perspective

### 1.1  What the PTA Captures

The Prefix Tree Acceptor (PTA) is a tree-shaped EFSM built directly
from the event log.  Every edge stores the **data payload**
$\vec{d}_i$ of the event that fired the transition, plus the
**(pre, post) observation pair** derived from the consecutive events
in the originating trace.  Hence the PTA already embeds both
the *control-flow* skeleton (activity sequences) and the *data*
observations.

### 1.2  What the Merging Step Considers

The Blue-Fringe / EDSM algorithm (Walkinshaw et al., 2013) merges
pairs of states $(q, q')$ subject to a **fold-closure compatibility
check** that is *not* purely structural.  The compatibility predicate
evaluates three rules:

| # | Rule | Perspective |
|---|------|-------------|
| 1 | An accepting state and a non-accepting state must never be merged. | Control-flow |
| 2 | For every shared activity $a$ whose outgoing edges point to *different* effective targets, the data samples on $q$'s edges and $q'$'s edges must be **linearly separable**. | **Data** |
| 3 | Fold-closure: the respective successor states must themselves be fold-compatible (recursively). | Control-flow + Data (recursive) |

#### Rule 2 — Data Separability via Z3

The data check encodes a **linear separability** problem over Z3
rationals.  Given sample sets $A$ (from $q$) and $B$ (from $q'$)
with numeric variables $V_{\text{num}} \subseteq V$, we ask:

$$
\exists\, \vec{c}, c_0 \;\;\text{s.t.}\;\;
\begin{cases}
\forall\, \vec{d} \in A: & \sum_{v \in V_{\text{num}}} c_v \cdot d_v + c_0 \;\ge\; \varepsilon \\[4pt]
\forall\, \vec{d} \in B: & \sum_{v \in V_{\text{num}}} c_v \cdot d_v + c_0 \;\le\; -\varepsilon
\end{cases}
\qquad (\varepsilon = 1)
$$

If the Z3 solver returns **SAT**, a linear guard *exists* that can
later separate the two groups, so the merge is permitted.  If
**UNSAT**, the data distributions are inseparable and the merge is
blocked.

### 1.3  Summary

> **The state merging step is *not* a pure control-flow operation.**
> It jointly considers control-flow structure (accept/non-accept,
> alphabet compatibility, fold-closure) *and* the data perspective
> (linear separability of observations).  A merge is only performed
> when a future guard can provably distinguish the merged data sets.

After merging, each transition carries the **union** of all data
samples and pre/post pairs from the original PTA edges that were
folded together.  These accumulated observations are the input to
the two synthesis steps below.

---

## 2  Guard Synthesis — PHOG-Accelerated SyGuS

### 2.1  Reference

> Lee, W., Heo, K., Alur, R., Naik, M.
> *Accelerating Search-Based Program Synthesis using Learned
> Probabilistic Models.*  PLDI 2018.

The paper proposes weighting the productions of a context-free
grammar with a **Probabilistic Higher-Order Grammar (PHOG)** model
and using **A\* search** to enumerate candidates in order of
decreasing likelihood.  The grammar generates candidate programs
(in our case: guard predicates); Z3 decides their correctness.

### 2.2  Adaptation to DPN Guard Synthesis

In the original paper the target language is a general SyGuS
specification.  We instantiate the framework with a domain-specific
grammar for **Boolean guards over real-valued process variables**
and redefine the verification oracle as a coverage + exclusion
check on finite data-sample sets.

#### 2.2.1  The Synthesis Problem

**Given.**

| Symbol | Meaning |
|--------|---------|
| $V = \{v_1, \dots, v_n\}$ | Set of numeric process variables (e.g. `amount`, `counter`). |
| $D^+_i = \{d \mid d \text{ is a data sample on transition } t_i\}$ | Positive examples for transition $t_i$. |
| $D^-_i = \bigcup_{j \neq i} D^+_j$ | Negative examples: all samples on competing transitions. |
| $\mathcal{G}$ | Context-free grammar of guard expressions (§2.3). |
| $\mathcal{P}$ | PHOG probability model (§2.4). |

**Find.**

A guard expression $g_i \in L(\mathcal{G})$ of **minimal size**
such that:

1. **Coverage** — the guard accepts every positive example:
$$
\forall\, \vec{d} \in D^+_i:\; g_i(\vec{d}) = \top
$$

2. **Exclusion** — the guard rejects every negative example:
$$
\forall\, \vec{d} \in D^-_i:\; g_i(\vec{d}) = \bot
$$

3. **Minimality (MDC)** — among all satisfying guards, prefer
the one with the smallest AST `size()`.

The problem is solved independently for every **decision point** —
a state with multiple outgoing transitions that must be
distinguished.  Two kinds of decision points exist:

* **Same-activity competition:** $\ge 2$ transitions labelled
  with the same activity $a$ leave the same state.
* **Cross-activity competition:** transitions labelled with
  *different* activities leave the same state but go to
  *different* targets.  Here guards are synthesised **pairwise
  per activity group**: the positive set is one activity's samples,
  the negative set is the other's.

#### 2.2.2  Threshold Derivation

Before search begins, the set of candidate **constants** $C$ is
derived from the data:

1. Collect all distinct numeric values $\{v_1, \dots, v_k\}$ per
   variable (pooling $D^+ \cup D^-$, sorted).
2. Compute **midpoints** between consecutive values:
   $m_i = (v_i + v_{i+1}) / 2$.
3. Add the exact boundary values themselves.

These thresholds form the terminals of the grammar.

#### 2.2.3  The Guard Grammar $\mathcal{G}$

The context-free grammar is:

$$
\boxed{
\begin{aligned}
\langle\textit{Guard}\rangle \;&::=\; \langle\textit{Atom}\rangle
                  \;\mid\; \langle\textit{Guard}\rangle \;\wedge\; \langle\textit{Atom}\rangle
                  \;\mid\; \langle\textit{Guard}\rangle \;\vee\; \langle\textit{Atom}\rangle
                  \;\mid\; \top \\[6pt]
\langle\textit{Atom}\rangle   \;&::=\; (v \le c)
                  \;\mid\; (v > c)
                  \;\mid\; (v = c) \\[6pt]
v \;&\in\; V, \quad c \;\in\; C
\end{aligned}
}
$$

where:

| Non-terminal | Productions | AST `kind` |
|---|---|---|
| $\langle\textit{Guard}\rangle$ | Atom, And, Or, True | `"leq"`, `"gt"`, `"eq"`, `"and"`, `"or"`, `"true"` |
| $\langle\textit{Atom}\rangle$ | $v \le c$, $v > c$, $v = c$ | `"leq"`, `"gt"`, `"eq"` |

**Derivation depth** is bounded at 3 to keep the search space
manageable.  Atomic candidates (depth 1) are tried first; if no
atom satisfies coverage + exclusion, conjunctive and disjunctive
compositions (depth 2–3) are explored.

**Difference from the original paper.**  Lee et al. use the grammar
of a general SyGuS benchmark (e.g. bitvector or LIA expressions).
We replace this with a purpose-built grammar of **comparison
predicates** over real-valued process variables — a much narrower
language tailored to process-mining guards.

#### 2.2.4  The PHOG Context Model $\mathcal{P}$

A PHOG assigns a probability to each production rule *conditioned*
on a context tuple that captures the local derivation history.
We use the same four-feature context from Lee et al. §3.2:

$$
\text{ctx} = \bigl(\,
  \underbrace{N_{\text{parent}}}_{\text{non-terminal being expanded}},\;
  \underbrace{r_{\text{left}}}_{\text{left-sibling production}},\;
  \underbrace{d}_{\text{depth in derivation tree}},\;
  \underbrace{N_{\text{grand}}}_{\text{grandparent non-terminal}}
\,\bigr)
$$

The model maintains a count table
$\text{counts}[\text{ctx}][r]$ and returns **Laplace-smoothed
log-probabilities**:

$$
\ln P(r \mid \text{ctx}) \;=\;
\ln \frac{\text{count}(\text{ctx}, r) + 1}
         {\sum_{r'} \text{count}(\text{ctx}, r') + |\mathcal{R}|}
$$

where $|\mathcal{R}| = 7$ (the number of production rule types).

**Cold start.**  Initially all counts are zero, so every production
has equal probability $\frac{1}{|\mathcal{R}|}$.  As guards are
successfully synthesised the model is updated
(`PHOGModel.observe`), biasing future searches towards patterns
that have worked before.

**Difference from the original paper.**  Lee et al. train PHOG
offline on a corpus of existing solutions.  We bootstrap with
uniform weights and learn **online** within a single pipeline run.

#### 2.2.5  A\* Search

Candidates are expanded in best-first order using cost function:

$$
f(n) \;=\; \underbrace{-\ln P(n \mid \text{ctx})}_{g(n)\;\text{(PHOG cost)}}
       \;+\; \underbrace{h(n)}_{\text{admissible heuristic}}
$$

The heuristic $h(n)$ is the AST `size()` of the candidate
(admissible: never overestimates the final formula size).  For
fully-expanded atomic candidates $h = 0$.

Each candidate popped from the priority queue is sent to the
**Z3 verification oracle** (§2.2.6).  If it passes, the search
terminates.  Otherwise, the candidate is expanded into conjunctive
and disjunctive compositions with all atoms and re-inserted into
the queue.

The search budget is capped at 5 000 candidate evaluations.

#### 2.2.6  Z3 Verification Oracle

A candidate guard $g$ is accepted iff the following conjunction is
**satisfiable** (all clauses are ground after substitution, so this
reduces to Boolean constant propagation):

$$
\bigwedge_{\vec{d} \in D^+} g[\vec{d}/V]
\;\;\wedge\;\;
\bigwedge_{\vec{d} \in D^-} \neg\, g[\vec{d}/V]
$$

where $g[\vec{d}/V]$ denotes the Z3 `substitute` operation that
replaces every variable $v_i$ with the concrete value $d_i$.

---

## 3  Postcondition Synthesis — Abduction

### 3.1  Reference

> Reynolds, A., Barbosa, H., Nötzli, A., Barrett, C., Tinelli, C.
> *Scalable Algorithms for Abduction via Enumerative Syntax-Guided
> Synthesis.*  IJCAR 2020.

The paper presents **GetAbductUCL** — an enumerative abduction
algorithm that, given background axioms $A$ and a goal $G$, finds
the simplest formula $S$ such that $A \wedge S \models G$ and
$A \wedge S$ is consistent.  UNSAT-core learning is used to prune
the enumeration space.

### 3.2  Adaptation to DPN Postcondition Synthesis

We recast the original abduction problem into the domain of
*variable-update discovery* for process transitions.  The question
becomes: given the data state *before* a transition fires (pre) and
the data state *after* (post), what is the simplest arithmetic
expression that explains the observed transformation?

#### 3.2.1  The Abduction Problem

For each transition $t$ and each variable $v \in V$, we solve:

**Given.**

| Symbol | Instantiation in our domain |
|--------|---------------------------|
| **Axioms** $A$ | The theory of linear real arithmetic (LRA), provided implicitly by the Z3 solver.  No additional domain axioms are needed because the background theory of arithmetic is the only reasoning framework. |
| **Goal** $G$ | The conjunction of all observed input–output constraints: $$G \;=\; \bigwedge_{k=1}^{m} \bigl(v' = e(\vec{d}_k^{\,\text{pre}})\bigr)$$ where $(\vec{d}_k^{\,\text{pre}}, \vec{d}_k^{\,\text{post}})$ are the $m$ pre/post observation pairs stored on transition $t$, and $v' = d_k^{\,\text{post}}[v]$ is the observed post-value of variable $v$. |
| **Hypothesis** $S$ | An expression $e \in L(\mathcal{G}_{\text{upd}})$ from the update grammar (§3.3) such that: $$A \;\wedge\; \bigl(v' = e(\vec{d}^{\,\text{pre}})\bigr) \;\models\; G$$ i.e. $e$ explains *every* observed (pre, post) pair simultaneously. |

**Consistency.**  We require $A \wedge S \not\models \bot$, i.e.
the axioms of arithmetic together with the candidate expression
must not be contradictory.

**Simplicity.**  Among all satisfying hypotheses, prefer the one
with the smallest AST `size()` (size-increasing enumeration).

**Difference from the original paper.**  Reynolds et al. solve
general first-order abduction where $A$ can be an arbitrary
theory and $G$ an arbitrary formula.  We specialise to:

* $A$ = LRA (linear real arithmetic) — no user-supplied axioms.
* $G$ = a conjunction of ground equalities (variable = constant).
* $S$ = a closed-form arithmetic expression from a typed grammar.

This specialisation allows us to substitute concrete data values
directly, turning each Z3 query into a ground satisfiability check.

#### 3.2.2  Two-Phase Strategy

The implementation uses a **two-phase** approach:

1. **Template synthesis (fast path).**
   Fix the template $v' = c_1 \cdot v + c_2$ and solve for the
   two unknown coefficients.  This is a system of linear equations
   over Z3 reals — typically solved in microseconds.

2. **GetAbductUCL (fallback).**
   If the template is unsatisfiable (the update is non-linear,
   involves multiple variables, or is not of the form $c_1 v + c_2$),
   fall back to full enumerative abduction with UNSAT-core pruning.

### 3.3  The Update Grammar $\mathcal{G}_{\text{upd}}$

$$
\boxed{
\begin{aligned}
\langle\textit{Expr}\rangle \;&::=\; \langle\textit{Const}\rangle
            \;\mid\; \langle\textit{Var}\rangle
            \;\mid\; \langle\textit{Expr}\rangle + \langle\textit{Expr}\rangle
            \;\mid\; \langle\textit{Expr}\rangle - \langle\textit{Expr}\rangle
            \;\mid\; \langle\textit{Expr}\rangle \times \langle\textit{Expr}\rangle \\[6pt]
\langle\textit{Const}\rangle \;&::=\; 0 \;\mid\; 1 \;\mid\; {-1} \\[4pt]
\langle\textit{Var}\rangle   \;&::=\; v_1 \;\mid\; v_2 \;\mid\; \cdots \;\mid\; v_n
\end{aligned}
}
$$

**Enumeration depth** is bounded at 2, producing three levels of
candidates in size-increasing order:

| Depth | Candidates | Example |
|-------|-----------|---------|
| 0 | Constants | $0,\; 1,\; -1$ |
| 1 | Variables | $\texttt{amount},\; \texttt{counter}$ |
| 2 | Binary compositions (excluding const $\oplus$ const) | $\texttt{counter} + 1$, $2 \times \texttt{counter}$, $\texttt{amount} - \texttt{counter}$ |

**Difference from the original paper.**  Reynolds et al. enumerate
over a full SyGuS grammar (potentially infinite).  We restrict to
a small finite grammar (depth ≤ 2 with three constants and $|V|$
variables) because process updates are overwhelmingly linear
($v' = v + 1$, $v' = v$, $v' = 0$).  If needed, the depth bound
can be raised.

### 3.4  UNSAT-Core Learning (GetAbductUCL)

When a candidate expression $e$ fails — i.e. the conjunction

$$
\bigwedge_{k=1}^{m} \bigl(d_k^{\,\text{post}}[v] = e(\vec{d}_k^{\,\text{pre}})\bigr)
$$

is **UNSAT** — the Z3 solver returns an **unsatisfiable core**: a
minimal subset of observation indices $I \subseteq \{1, \dots, m\}$
that is already contradictory.

The learner records $I$.  Any future candidate $e'$ that also fails
on a *superset* $I' \supseteq I$ of the same observations can be
**pruned without querying Z3**, because the same structural conflict
will arise.

Formally, assertion tracking is used:

```
solver.assert_and_track(  e(d_k^pre) == d_k^post[v],  indicator_k  )
```

After UNSAT, `solver.unsat_core()` yields the failing indicator
variables, which are mapped back to observation indices.

**Difference from the original paper.**  Reynolds et al. use
UNSAT cores over *quantified* formulas with symbolic variables.
We operate on *ground* instances (all data values are concrete),
so core extraction is straightforward — each tracked assertion
corresponds to exactly one observation pair.

### 3.5  Pre/Post Observation Pairs

A critical implementation detail: the (pre, post) pairs must
reflect the **true temporal ordering** within each trace.

* **Pre-state**: the data payload of the *preceding* event in
  the same trace (empty dict `{}` for the first event).
* **Post-state**: the data payload of the *current* event.

These pairs are recorded during **PTA construction** (where the
trace order is known) and carried through state merging unchanged.
This avoids the need to reconstruct temporal relationships after
states have been folded.

---

## 4  Summary of Formal Objects

| Step | Paper | Key formal objects |
|------|-------|--------------------|
| State Merging | Walkinshaw et al. 2013 | Fold-closure predicate; linear-separator existence query over Z3 reals |
| Guard Synthesis | Lee et al. PLDI 2018 | Grammar $\mathcal{G}$ of Boolean predicates; PHOG context model $(\text{parent}, \text{left-sib}, \text{depth}, \text{grandparent})$; A\* cost $f = -\ln P + h$; Z3 coverage + exclusion oracle |
| Postcondition Synthesis | Reynolds et al. IJCAR 2020 | Axioms $A$ = LRA; Goal $G$ = ground equalities from observations; Hypothesis $S$ = expression from $\mathcal{G}_{\text{upd}}$; UNSAT-core pruning set |
