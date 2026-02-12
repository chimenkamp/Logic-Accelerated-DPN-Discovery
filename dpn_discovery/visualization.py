"""
DPN & EFSM Visualizer — Graphviz-based rendering.

Provides ``DPNVisualizer`` to render:
  • A ``DataPetriNet`` as a bipartite Petri-net graph with
    guard and update-function annotations on transitions.
  • An ``EFSM`` (PTA or merged automaton) as a state-machine
    graph with labelled edges.

Uses the ``graphviz`` Python package (already a pm4py dependency)
for layout and rendering.

Usage::

    from dpn_discovery.visualization import DPNVisualizer

    viz = DPNVisualizer()
    viz.view_dpn(dpn, title="Discovered DPN")
    viz.view_efsm(pta, title="Prefix Tree Acceptor")
    viz.view_comparison(pta, dpn)
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import z3

try:
    import graphviz  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "graphviz Python package is required for visualization. "
        "Install with: pip install graphviz"
    ) from exc

from dpn_discovery.models import (
    EFSM,
    DataPetriNet,
    DPNTransition,
    Place,
    Transition,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class VisualizerSettings:
    """Settings controlling the visual appearance of diagrams.

    Attributes:
        output_format:  File format for rendered output ('png', 'pdf', 'svg').
        rankdir:        Graph layout direction ('LR' left-to-right, 'TB' top-bottom).
        font_name:      Font family for labels.
        font_size:      Base font size in points.
        place_color:    Fill colour for DPN places.
        trans_color:    Fill colour for DPN transitions (activity boxes).
        guard_color:    Font colour for guard annotations.
        update_color:   Font colour for update annotations.
        initial_color:  Border colour for the initial place / state.
        accept_color:   Fill colour for accepting states.
        state_color:    Fill colour for EFSM states.
        edge_color:     Edge colour.
        dpi:            Dots-per-inch for raster outputs.
        show_guards:    Whether to annotate guards on transitions.
        show_updates:   Whether to annotate update functions on transitions.
    """

    output_format: Literal["png", "pdf", "svg"] = "png"
    rankdir: Literal["LR", "TB"] = "LR"
    font_name: str = "Helvetica"
    font_size: int = 11
    place_color: str = "#E8F5E9"
    trans_color: str = "#E3F2FD"
    guard_color: str = "#C62828"
    update_color: str = "#1565C0"
    initial_color: str = "#2E7D32"
    accept_color: str = "#FFF9C4"
    state_color: str = "#F3E5F5"
    edge_color: str = "#424242"
    dpi: int = 150
    show_guards: bool = True
    show_updates: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _guard_str(formula: Optional[z3.BoolRef]) -> str:
    """Return a human-readable string for a Z3 guard formula."""
    if formula is None:
        return "True"
    s = str(formula)
    # Z3 uses 'And(…)' / 'Or(…)'; simplify common cases.
    return s


def _update_str(update_rule: Optional[dict[str, z3.ExprRef]]) -> str:
    """Return a compact string for an update rule."""
    if not update_rule:
        return ""
    parts: list[str] = []
    for var, expr in sorted(update_rule.items()):
        parts.append(f"{var}' := {expr}")
    return "\n".join(parts)


def _html_escape(text: str) -> str:
    """Escape text for use inside Graphviz HTML labels."""
    return html.escape(text, quote=True)


# ═══════════════════════════════════════════════════════════════════════════
# DPNVisualizer
# ═══════════════════════════════════════════════════════════════════════════

class DPNVisualizer:
    """Renders ``DataPetriNet`` and ``EFSM`` objects as Graphviz diagrams.

    Parameters
    ----------
    settings : VisualizerSettings | None
        Visual appearance settings.  Uses defaults when ``None``.
    """

    def __init__(self, settings: Optional[VisualizerSettings] = None) -> None:
        self.settings = settings or VisualizerSettings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_dpn(
        self,
        dpn: DataPetriNet,
        title: str = "Data Petri Net",
    ) -> graphviz.Digraph:
        """Build a ``graphviz.Digraph`` for *dpn*.

        Places are drawn as circles, transitions as rectangles.
        Guards and update functions are shown beneath each transition
        in coloured text (when enabled in settings).
        """
        s = self.settings
        dot = graphviz.Digraph(
            name="dpn",
            comment=title,
            format=s.output_format,
            graph_attr={
                "rankdir": s.rankdir,
                "label": title,
                "labelloc": "t",
                "fontname": s.font_name,
                "fontsize": str(s.font_size + 2),
                "dpi": str(s.dpi),
                "bgcolor": "white",
            },
            node_attr={
                "fontname": s.font_name,
                "fontsize": str(s.font_size),
            },
            edge_attr={
                "fontname": s.font_name,
                "fontsize": str(s.font_size - 1),
                "color": s.edge_color,
            },
        )

        # --- Places (circles) -----------------------------------------------
        for place in dpn.places:
            is_initial = place.name in dpn.initial_marking
            dot.node(
                place.name,
                label=place.name,
                shape="circle",
                style="filled",
                fillcolor=s.place_color,
                color=s.initial_color if is_initial else "#616161",
                penwidth="2.5" if is_initial else "1.0",
                width="0.5",
                fixedsize="false",
            )

        # --- Transitions (rectangles with annotations) -----------------------
        for trans in dpn.transitions:
            label = self._dpn_transition_label(trans)
            dot.node(
                trans.name,
                label=f"<{label}>",
                shape="box",
                style="filled,rounded",
                fillcolor=s.trans_color,
                color="#1565C0",
                penwidth="1.2",
            )

            # Arcs: input places → transition.
            for ip in trans.input_places:
                dot.edge(ip, trans.name)

            # Arcs: transition → output places.
            for op in trans.output_places:
                dot.edge(trans.name, op)

        return dot

    def render_efsm(
        self,
        efsm: EFSM,
        title: str = "EFSM",
    ) -> graphviz.Digraph:
        """Build a ``graphviz.Digraph`` for *efsm*.

        States are circles; transitions are labelled directed edges.
        Guard and update annotations are shown on edges when enabled.
        """
        s = self.settings
        dot = graphviz.Digraph(
            name="efsm",
            comment=title,
            format=s.output_format,
            graph_attr={
                "rankdir": s.rankdir,
                "label": title,
                "labelloc": "t",
                "fontname": s.font_name,
                "fontsize": str(s.font_size + 2),
                "dpi": str(s.dpi),
                "bgcolor": "white",
            },
            node_attr={
                "fontname": s.font_name,
                "fontsize": str(s.font_size),
            },
            edge_attr={
                "fontname": s.font_name,
                "fontsize": str(s.font_size - 1),
                "color": s.edge_color,
            },
        )

        # --- States (circles) ------------------------------------------------
        for state_id in sorted(efsm.states):
            is_initial = state_id == efsm.initial_state
            is_accept = state_id in efsm.accepting_states

            dot.node(
                state_id,
                label=state_id,
                shape="doublecircle" if is_accept else "circle",
                style="filled",
                fillcolor=s.accept_color if is_accept else s.state_color,
                color=s.initial_color if is_initial else "#616161",
                penwidth="2.5" if is_initial else "1.0",
            )

        # Invisible start arrow for the initial state.
        dot.node("__start__", label="", shape="none", width="0", height="0")
        dot.edge("__start__", efsm.initial_state, arrowsize="0.8")

        # --- Transitions (labelled edges) ------------------------------------
        for trans in efsm.transitions:
            label = self._efsm_edge_label(trans)
            dot.edge(
                trans.source_id,
                trans.target_id,
                label=f"<{label}>",
            )

        return dot

    def view_dpn(
        self,
        dpn: DataPetriNet,
        title: str = "Data Petri Net",
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Render and display *dpn*. Returns the path to the rendered file.

        If *output_path* is ``None``, a temp file is created.
        """
        dot = self.render_dpn(dpn, title=title)
        return self._render_and_view(dot, output_path)

    def view_efsm(
        self,
        efsm: EFSM,
        title: str = "EFSM",
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Render and display *efsm*. Returns the path to the rendered file."""
        dot = self.render_efsm(efsm, title=title)
        return self._render_and_view(dot, output_path)

    def view_comparison(
        self,
        efsm: EFSM,
        dpn: DataPetriNet,
        efsm_title: str = "EFSM (PTA / Merged)",
        dpn_title: str = "Data Petri Net",
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Render both the EFSM and DPN side-by-side in one diagram.

        Uses Graphviz subgraphs to place them next to each other.
        Returns the path to the rendered file.
        """
        s = self.settings
        combined = graphviz.Digraph(
            name="comparison",
            comment="EFSM ↔ DPN Comparison",
            format=s.output_format,
            graph_attr={
                "rankdir": s.rankdir,
                "label": f"{efsm_title}  vs  {dpn_title}",
                "labelloc": "t",
                "fontname": s.font_name,
                "fontsize": str(s.font_size + 4),
                "dpi": str(s.dpi),
                "bgcolor": "white",
                "compound": "true",
            },
        )

        # Left subgraph: EFSM.
        efsm_sub = self.render_efsm(efsm, title=efsm_title)
        efsm_sub.name = "cluster_efsm"
        efsm_sub.graph_attr["label"] = efsm_title
        efsm_sub.graph_attr["style"] = "dashed"
        efsm_sub.graph_attr["color"] = "#9E9E9E"
        combined.subgraph(efsm_sub)

        # Right subgraph: DPN.
        dpn_sub = self.render_dpn(dpn, title=dpn_title)
        dpn_sub.name = "cluster_dpn"
        dpn_sub.graph_attr["label"] = dpn_title
        dpn_sub.graph_attr["style"] = "dashed"
        dpn_sub.graph_attr["color"] = "#9E9E9E"
        combined.subgraph(dpn_sub)

        return self._render_and_view(combined, output_path)

    def save_dpn(
        self,
        dpn: DataPetriNet,
        output_path: str | Path,
        title: str = "Data Petri Net",
    ) -> Path:
        """Render *dpn* and save to *output_path* (no viewer launched)."""
        dot = self.render_dpn(dpn, title=title)
        return self._save(dot, output_path)

    def save_efsm(
        self,
        efsm: EFSM,
        output_path: str | Path,
        title: str = "EFSM",
    ) -> Path:
        """Render *efsm* and save to *output_path* (no viewer launched)."""
        dot = self.render_efsm(efsm, title=title)
        return self._save(dot, output_path)

    def save_comparison(
        self,
        efsm: EFSM,
        dpn: DataPetriNet,
        output_path: str | Path,
        efsm_title: str = "EFSM (PTA / Merged)",
        dpn_title: str = "Data Petri Net",
    ) -> Path:
        """Render the comparison and save to *output_path*."""
        # Reuse view_comparison logic but without launching viewer.
        s = self.settings
        combined = graphviz.Digraph(
            name="comparison",
            format=s.output_format,
            graph_attr={
                "rankdir": s.rankdir,
                "label": f"{efsm_title}  vs  {dpn_title}",
                "labelloc": "t",
                "fontname": s.font_name,
                "fontsize": str(s.font_size + 4),
                "dpi": str(s.dpi),
                "bgcolor": "white",
                "compound": "true",
            },
        )
        efsm_sub = self.render_efsm(efsm, title=efsm_title)
        efsm_sub.name = "cluster_efsm"
        efsm_sub.graph_attr["style"] = "dashed"
        efsm_sub.graph_attr["color"] = "#9E9E9E"
        combined.subgraph(efsm_sub)

        dpn_sub = self.render_dpn(dpn, title=dpn_title)
        dpn_sub.name = "cluster_dpn"
        dpn_sub.graph_attr["style"] = "dashed"
        dpn_sub.graph_attr["color"] = "#9E9E9E"
        combined.subgraph(dpn_sub)

        return self._save(combined, output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dpn_transition_label(self, trans: DPNTransition) -> str:
        """Build an HTML-like label for a DPN transition box.

        Uses a Graphviz HTML <TABLE> for proper layout with
        horizontal rules between sections.
        """
        s = self.settings

        # Activity name (strip t_ prefix and _N suffix).
        activity = trans.name
        if activity.startswith("t_"):
            parts = activity[2:].rsplit("_", 1)
            activity = parts[0] if len(parts) == 2 and parts[1].isdigit() else activity[2:]

        rows: list[str] = [
            f'<TR><TD><B>{_html_escape(activity)}</B></TD></TR>',
        ]

        # Guard.
        if s.show_guards and trans.guard is not None:
            guard = _guard_str(trans.guard)
            rows.append(f'<HR/>')
            rows.append(
                f'<TR><TD><FONT COLOR="{s.guard_color}" POINT-SIZE="{s.font_size - 1}">'
                f"[{_html_escape(guard)}]</FONT></TD></TR>"
            )

        # Update.
        if s.show_updates and trans.update_rule:
            upd = _update_str(trans.update_rule)
            if not (s.show_guards and trans.guard is not None):
                rows.append(f'<HR/>')
            escaped = _html_escape(upd).replace(chr(10), "<BR/>")
            rows.append(
                f'<TR><TD><FONT COLOR="{s.update_color}" POINT-SIZE="{s.font_size - 1}">'
                f'{escaped}</FONT></TD></TR>'
            )

        return f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">{"".join(rows)}</TABLE>'

    def _efsm_edge_label(self, trans: Transition) -> str:
        """Build an HTML-like edge label for an EFSM transition."""
        s = self.settings

        parts: list[str] = [f'<B>{_html_escape(trans.activity)}</B>']

        if s.show_guards and trans.guard_formula is not None:
            guard = _guard_str(trans.guard_formula)
            parts.append(
                f'<FONT COLOR="{s.guard_color}" POINT-SIZE="{s.font_size - 2}">'
                f"[{_html_escape(guard)}]</FONT>"
            )

        if s.show_updates and trans.update_rule:
            upd = _update_str(trans.update_rule)
            parts.append(
                f'<FONT COLOR="{s.update_color}" POINT-SIZE="{s.font_size - 2}">'
                f"{_html_escape(upd)}</FONT>"
            )

        return "<BR/>".join(parts)

    def _render_and_view(
        self,
        dot: graphviz.Digraph,
        output_path: Optional[str | Path],
    ) -> Path:
        """Render the graph and open the viewer."""
        if output_path:
            path = Path(output_path).with_suffix("")
            rendered = dot.render(
                filename=str(path),
                cleanup=True,
                view=True,
            )
        else:
            rendered = dot.render(
                cleanup=True,
                view=True,
            )
        return Path(rendered)

    def _save(
        self,
        dot: graphviz.Digraph,
        output_path: str | Path,
    ) -> Path:
        """Render the graph to a file without opening a viewer."""
        path = Path(output_path).with_suffix("")
        rendered = dot.render(
            filename=str(path),
            cleanup=True,
            view=False,
        )
        return Path(rendered)
