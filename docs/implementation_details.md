# Implementation Detail: Data-Aware Petri Net Discovery Pipeline

## 1. Introduction
This document defines the architecture, formalisms, and algorithms for a software pipeline that discovers a Data-Aware Petri Net (DPN) from an event log. The pipeline first constructs a guarded Extended Finite State Machine (EFSM) via state merging and SAT-based synthesis, and finally transforms it into a DPN.

**Target Environment:**
* **Language:** Python 3.14 (Strict Type Hinting, `match` statements)
* **Solver:** Z3 Theorem Prover (`z3-solver` package)
* **Goal:** Fully automated discovery without heuristic shortcuts.

---

## 2. Formal Definitions & Data Structures

### 2.1. Event Log & Traces
An **Event Log** $L$ is a multiset of traces.
* **Event** $e = (a, \vec{d})$: A tuple containing an activity label $a \in \Sigma$ and a data vector $\vec{d} \in D$ (where $D$ is the domain of variables $V$).
* **Trace** $\sigma = \langle e_1, e_2, \dots, e_n \rangle$: A sequence of events.

### 2.2. Extended Finite State Machine (EFSM)
An EFSM is a tuple $M = (S, s_0, \Sigma, V, T)$, where:
* $S$: Finite set of states.
* $s_0 \in S$: Initial state.
* $\Sigma$: Set of event labels (activities).
* $V$: Set of data variables.
* $T$: Set of transitions. Each transition $t \in T$ is a tuple $(s_{src}, s_{tgt}, a, g, u)$:
    * $s_{src}, s_{tgt} \in S$: Source and target states.
    * $a \in \Sigma$: Activity label.
    * $g(V) \to \{0, 1\}$: **Guard predicate** (Boolean formula over $V$).
    * $u(V) \to V$: **Update function** (maps input values to output values).

### 2.3. Data Petri Net (DPN)
A DPN is a tuple $N = (P, T, F, V, G, U)$, where:
* $P$: Set of places.
* $T$: Set of transitions.
* $F \subseteq (P \times T) \cup (T \times P)$: Flow relation.
* $G$: Function mapping transitions to guards.
* $U$: Function mapping transitions to variable update logic.

---

## 3. Pipeline Architecture

The software must implement the following 6-step sequential pipeline:

1.  **Log Preprocessing**: Parse log, identify decision variables $V$.
2.  **PTA Construction**: Build the Prefix Tree Acceptor from traces.
3.  **State Merging**: Generalize the PTA into an FSM using Data-Driven State Merging.
4.  **Guard Synthesis**: Decorate FSM transitions with guards using MDC-accelerated SAT.
5.  **Postcondition Synthesis**: Infer data update rules using Abduction.
6.  **DPN Transformation**: Convert the guarded EFSM into a Data Petri Net.

---

## 4. Detailed Algorithms & Z3 Implementation

### Step 1 & 2: Preprocessing and PTA Construction
**Goal**: Convert raw logs into a tree where edges represent activities and nodes represent states.

* **Input**: Event Log $L$.
* **Output**: A PTA (a tree-shaped FSM) where every unique prefix in $L$ is a state.
* **Implementation**:
    * Root state $s_0$.
    * For each trace $\sigma = \langle e_1, \dots, e_n \rangle$:
        * Traverse from $s_0$. If edge labeled $e_i.a$ exists, follow it.
        * If not, create new state $s_{new}$ and edge $(s_{curr}, s_{new}, e_i.a)$.
        * **Crucial**: Store the data payload $e_i.\vec{d}$ on the edge. This data collection is required for Steps 3 and 4.

### Step 3: Data-Driven State Merging (Walkinshaw et al.)
**Goal**: Fold the PTA into a graph by merging equivalent states.
**Algorithm**: `Blue-Fringe` or `EDSM` (Evidence-Driven State Merging) with a modified compatibility check.

* **Logic**: Two states $q$ and $q'$ are **compatible** if, for all shared future paths, the data values can be distinguished or unified.
* **Formal Check**:
    * Let $T_{out}(q)$ and $T_{out}(q')$ be the outgoing transitions of $q$ and $q'$.
    * If there exists activity $a$ such that $q \xrightarrow{a} \dots$ and $q' \xrightarrow{a} \dots$:
        * Collect data samples $D_q$ (from transitions exiting $q$ with label $a$).
        * Collect data samples $D_{q'}$ (from transitions exiting $q'$ with label $a$).
        * **Compatibility Condition**: A merge is valid if we can find a guard $g$ that covers both $D_q$ and $D_{q'}$ consistent with the global process behavior. (Simplification per Walkinshaw: Check if the data distributions are non-contradictory, often using a classifier test).
* **Agent Task**: Implement a `merge_states(pta, q, q')` function that merges nodes and unions their data samples.

### Step 4: Guard Synthesis via MDC-accelerated SAT
**Goal**: Assign a Boolean guard $g$ to each transition $t$ to distinguish which path is taken based on data $V$.



[Image of logic gate circuit diagram]


* **Problem Statement**: For a state $s$ with multiple outgoing transitions $t_1, t_2, \dots t_k$ labeled with the same activity $a$ (or competing activities), find guards $g_1, \dots, g_k$ such that:
    1.  **Coverage**: For every data sample $d \in Data(t_i)$, $g_i(d)$ is True.
    2.  **Disjointness**: For $i \neq j$, $g_i \land g_j$ is False (optional, but good for determinism).
    3.  **Minimality (MDC)**: The formula size is minimized (Minimal Distinguishing Constraints).

* **Z3 Implementation Strategy**:
    * **Variables**: Create Z3 variables for the coefficients of the guard (e.g., if linear: $c_1 \cdot v_1 + c_0 > 0$).
    * **Constraints**:
        ```python
        solver = z3.Solver()
        # For every data point d in Transition 1's observations:
        for d in data_t1:
            solver.add( Guard_T1(d) == True )
            solver.add( Guard_T2(d) == False ) # Negative example

        # Objective: Minimal structure (Use z3.Optimize() or iterative deepening on formula size)
        ```
    * **MDC Algorithm**: Start with simple predicates (e.g., atomic comparisons $v > c$). If UNSAT, increase complexity (Disjunctive Normal Form).

### Step 5: Postcondition Synthesis via Abduction (Pelyukh et al.)
**Goal**: Determine how variables are updated (e.g., `x = x + 1`).
**Formalism (Abduction)**:
* **Given**:
    * $Pre(\sigma)$: The state of data before the transition.
    * $Post(\sigma')$: The state of data observed after the transition.
    * $Background$: The axioms of arithmetic.
* **Find**: An update function $Op$ (the "Hypothesis") such that:
    $$ \forall \sigma . Pre(\sigma) \land Op(\sigma, \sigma') \implies Post(\sigma') $$
* **Z3 Implementation**:
    * Treat this as a **Syntax-Guided Synthesis (SyGuS)** problem or template-based synthesis.
    * **Template**: Assume $x' = c_1 \cdot x + c_2$.
    * **Constraint**: For all observed pairs $(d_{in}, d_{out})$ on this transition:
        `solver.add( d_out == c1 * d_in + c2 )`
    * Solve for $c_1, c_2$.

### Step 6: EFSM to Data Petri Net Mapping
**Goal**: Convert the discovered EFSM to a standard DPN format.

* **Algorithm**:
    1.  **Places**: Create a place $p_s$ for every state $s$ in the EFSM.
    2.  **Transitions**: For every transition $t = (s_{src}, s_{tgt}, a, g, u)$ in EFSM:
        * Create a DPN transition $t_{dpn}$.
        * Add arc $p_{s_{src}} \to t_{dpn}$.
        * Add arc $t_{dpn} \to p_{s_{tgt}}$.
        * Annotate $t_{dpn}$ with guard $g$ and variable update $u$.
    3.  **Output**: Serialize to PNML (Petri Net Markup Language) or a specific Python object structure.

---

## 5. Python 3.14 Implementation Specs

The implementation must use modern Python features.

### 5.1. Class Structure

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import z3

@dataclass
class Event:
    activity: str
    payload: Dict[str, float | int | str]

@dataclass
class Transition:
    source_id: str
    target_id: str
    activity: str
    data_samples: List[Dict[str, Any]] = field(default_factory=list)
    guard_formula: Optional[z3.ExprRef] = None
    update_rule: Optional[z3.ExprRef] = None

class DataPetriNet:
    def __init__(self):
        self.places: List[str] = []
        self.transitions: List[Transition] = []
        # Z3 Context
        self.ctx = z3.Context()

    def add_place(self, name: str):
        ...
```

### 5.2. Type Hinting

Use strict typing. For Z3 objects, use z3.ExprRef, z3.BoolRef, z3.ArithRef. Example:

```python
def synthesize_guard(positive_samples: list[dict], negative_samples: list[dict]) -> z3.BoolRef:
    ...
```

## 6. Assumptions and Constraints

    Completeness: The input log is assumed to be "complete enough" (covering all valid paths) for the PTA construction.

    Deterministic Data: The pipeline assumes the underlying process is deterministic with respect to data (i.e., same data state + same activity = same outcome).

    Z3 Usage: All logical synthesis must be delegated to Z3. Do not write manual heuristic parsers for logic.

    No "Magic" Strings: All variable names and activity labels must be dynamically extracted from the log.