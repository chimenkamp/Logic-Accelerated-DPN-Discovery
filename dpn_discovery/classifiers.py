"""
Data Classifiers for MINT-style EFSM Inference.

Implements the classifier layer from Walkinshaw et al. (2013),
Algorithm 3.  For each event label *l* in the traces a classifier
C_l is trained that maps a data vector to the predicted *next*
event label.  These classifiers are used during:

  1. PTA construction  – prefix sharing requires identical predictions.
  2. Transition equivalence – two same-label transitions are equivalent
     iff the classifier predicts the same next-event for all their data.
  3. Post-merge consistency – every transition's data must agree with
     the structural successors of its target state.

Reference
---------
  Walkinshaw, N., Taylor, R., Derrick, J.  *Inferring Extended
  Finite State Machine Models from Software Executions* (2013),
  §4 Algorithm 3 and §6.1.7.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Protocol

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost

from dpn_discovery.models import EventLog

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Classifier algorithm selection
# ═══════════════════════════════════════════════════════════════════════════

class ClassifierAlgorithm(Enum):
    """Supported data-classifier algorithms (cf. Walkinshaw §6.1.7)."""

    DECISION_TREE = auto()   # J48 / C4.5 equivalent
    NAIVE_BAYES = auto()     # GaussianNB
    ADABOOST = auto()        # AdaBoost with decision stumps


# ═══════════════════════════════════════════════════════════════════════════
# Classifier protocol & implementations
# ═══════════════════════════════════════════════════════════════════════════

class DataClassifier(Protocol):
    """Abstract interface for a per-label data classifier."""

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the classifier on feature matrix *X* and labels *y*."""
        ...

    def predict(self, x: np.ndarray) -> str:
        """Predict the next-event label for a single data vector *x*."""
        ...


class _DecisionTreeClassifier:
    """Wrapper around scikit-learn DecisionTreeClassifier (≈ J48/C4.5)."""

    def __init__(self) -> None:
        self._clf = SkDecisionTree()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._clf.fit(X, y)
        self._trained = True

    def predict(self, x: np.ndarray) -> str:
        return str(self._clf.predict(x.reshape(1, -1))[0])


class _NaiveBayesClassifier:
    """Wrapper around scikit-learn GaussianNB."""

    def __init__(self) -> None:
        self._clf = SkGaussianNB()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._clf.fit(X, y)
        self._trained = True

    def predict(self, x: np.ndarray) -> str:
        return str(self._clf.predict(x.reshape(1, -1))[0])


class _AdaBoostClassifier:
    """Wrapper around scikit-learn AdaBoostClassifier."""

    def __init__(self) -> None:
        self._clf = SkAdaBoost(algorithm="SAMME")
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._clf.fit(X, y)
        self._trained = True

    def predict(self, x: np.ndarray) -> str:
        return str(self._clf.predict(x.reshape(1, -1))[0])


def _make_classifier(algorithm: ClassifierAlgorithm) -> DataClassifier:
    """Instantiate a fresh (untrained) classifier of the given type."""
    match algorithm:
        case ClassifierAlgorithm.DECISION_TREE:
            return _DecisionTreeClassifier()
        case ClassifierAlgorithm.NAIVE_BAYES:
            return _NaiveBayesClassifier()
        case ClassifierAlgorithm.ADABOOST:
            return _AdaBoostClassifier()
        case _:
            raise ValueError(f"Unknown classifier algorithm: {algorithm}")


# ═══════════════════════════════════════════════════════════════════════════
# Training set preparation  (Algorithm 3, line 2)
# ═══════════════════════════════════════════════════════════════════════════

TrainingSet = dict[str, tuple[list[dict[str, Any]], list[str]]]
"""Per-label training data:  label → (list-of-data-vectors, list-of-next-labels)."""


def prepare_data_traces(
    log: EventLog,
    variables: set[str],
) -> TrainingSet:
    """Build per-label training sets from the event log.

    For each event at position *i* with label *l*, the training
    example is:  X = data-vector-at-event-i,  y = label-at-event-(i+1).

    The last event of every trace has no successor and is skipped.

    Parameters
    ----------
    log : EventLog
        The parsed event log.
    variables : set[str]
        The set of variable names to include in the feature vectors.

    Returns
    -------
    TrainingSet
        ``{label: (data_dicts, next_labels)}`` where *data_dicts*
        is a list of payload dictionaries and *next_labels* the
        corresponding list of successor event labels.
    """
    result: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}

    for trace in log.traces:
        events = trace.events
        for i in range(len(events) - 1):
            label = events[i].activity
            data = {v: events[i].payload.get(v, 0) for v in variables}
            next_label = events[i + 1].activity

            if label not in result:
                result[label] = ([], [])
            result[label][0].append(data)
            result[label][1].append(next_label)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Classifier training  (Algorithm 3, line 3)
# ═══════════════════════════════════════════════════════════════════════════

Classifiers = dict[str, DataClassifier]
"""Mapping  event-label → trained DataClassifier."""


def infer_classifiers(
    data_traces: TrainingSet,
    variables: list[str],
    algorithm: ClassifierAlgorithm = ClassifierAlgorithm.DECISION_TREE,
) -> Classifiers:
    """Train one classifier per event label.

    Parameters
    ----------
    data_traces : TrainingSet
        Output of :func:`prepare_data_traces`.
    variables : list[str]
        Ordered list of variable names (determines feature columns).
    algorithm : ClassifierAlgorithm
        Which scikit-learn algorithm to use.

    Returns
    -------
    Classifiers
        ``{label: trained_classifier}``.
    """
    classifiers: Classifiers = {}

    for label, (data_dicts, next_labels) in data_traces.items():
        if not data_dicts:
            continue

        # Build feature matrix.
        X = np.array(
            [[_to_float(d.get(v, 0)) for v in variables] for d in data_dicts],
            dtype=np.float64,
        )
        y = np.array(next_labels)

        # If all training examples have the same next-label the
        # classifier is trivial — but we still create one so the
        # API is uniform.
        clf = _make_classifier(algorithm)
        clf.train(X, y)
        classifiers[label] = clf

        logger.debug(
            "Classifier for '%s': %d samples, %d classes",
            label,
            len(data_dicts),
            len(set(next_labels)),
        )

    return classifiers


# ═══════════════════════════════════════════════════════════════════════════
# Prediction helper
# ═══════════════════════════════════════════════════════════════════════════

def predict_next_label(
    classifiers: Classifiers,
    label: str,
    data: dict[str, Any],
    variables: list[str],
) -> str | None:
    """Use classifier C_{label} to predict the next event.

    Returns ``None`` if no classifier exists for *label* (e.g. the
    label only ever appeared as the last event in every trace).
    """
    clf = classifiers.get(label)
    if clf is None:
        return None
    x = np.array([_to_float(data.get(v, 0)) for v in variables], dtype=np.float64)
    return clf.predict(x)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_float(value: Any) -> float:
    """Coerce a payload value to float for the feature matrix."""
    if isinstance(value, (int, float)):
        return float(value)
    # Categorical / string → hash-based numeric encoding.
    # Crude but functional; refinements can be added later.
    if isinstance(value, str):
        return float(hash(value) % 10_000)
    return 0.0
