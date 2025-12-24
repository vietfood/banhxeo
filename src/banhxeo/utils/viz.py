"""
Thanks Gemini 3.0 Pro
"""

from typing import Set

import graphviz

from banhxeo.core.buffer import (
    BinaryOp,
    LazyBuffer,
    LoadOp,
    MovementOp,
    ReduceOp,
    UnaryOp,
)


def _format_op_label(lb: LazyBuffer) -> str:
    """
    Creates an HTML-like label for Graphviz.
    Highlights Shape/Strides because that's what debugging memory layout requires.
    """
    # Shorten the Enum class name (e.g. BinaryOp.ADD -> ADD)
    op_name = str(lb.op).split(".")[-1]
    op_category = str(lb.op).split(".")[0]

    # Visual cues for specific data
    shape_str = f"Shape: {lb.view.shape}"
    stride_str = f"Strides: {lb.view.strides}"
    offset_str = f"Offset: {lb.view.offset}"

    # Check if realized
    realized_mark = (
        "<b><font color='darkgreen'>[REALIZED]</font></b><br/>" if lb.realized else ""
    )

    # HTML Label construction
    return (
        f"<{realized_mark}"
        f"<b>{op_name}</b> <font point-size='10' color='gray'>({op_category})</font><br/>"
        f"<font point-size='10'>{shape_str}</font><br/>"
        f"<font point-size='10'>{stride_str}</font><br/>"
        f"<font point-size='10'>{offset_str}</font>>"
    )


def _get_node_style(lb: LazyBuffer) -> dict:
    """
    Color-coding based on operation type to separate Memory IO, Compute, and Metadata.
    """
    style = {"style": "filled", "shape": "box", "fontname": "Helvetica"}

    if isinstance(lb.op, LoadOp):
        # Input / Memory sources
        style["fillcolor"] = "#ffebcd"  # BlanchedAlmond
        style["color"] = "orange"
    elif isinstance(lb.op, LoadOp):
        style["fillcolor"] = "#e0f7fa"  # Cyan tint
        style["color"] = "#1E8F04"
        style["style"] = "filled,rounded"  # Round corners for metadata ops
    elif isinstance(lb.op, MovementOp):
        # Metadata / View movements (cheap)
        style["fillcolor"] = "#e0f7fa"  # Cyan tint
        style["color"] = "#006064"
        style["style"] = "filled,rounded"  # Round corners for metadata ops
    elif isinstance(lb.op, (BinaryOp, UnaryOp, ReduceOp)):
        # Compute (The heavy lifting)
        style["fillcolor"] = "#eeeeee"
        style["color"] = "black"

    # Dashed border if lazy (not realized)
    if not lb.realized:
        style["style"] = f"{style.get('style', '')},dashed"

    return style


def visualize_graph(root: LazyBuffer, filename: str = "lazy_graph", view: bool = True):
    """
    Walks the LazyBuffer graph from the root up to the sources (parents)
    and renders it using Graphviz.
    """
    dot = graphviz.Digraph(comment="LazyBuffer Graph", format="png")
    dot.attr(rankdir="BT")  # Bottom-to-Top: Inputs at bottom, Output at top

    # Track visited nodes by object ID to handle DAG structure (diamond dependencies)
    visited: Set[int] = set()

    def walk(lb: "LazyBuffer"):
        node_id = str(id(lb))

        if id(lb) in visited:
            return
        visited.add(id(lb))

        # Add Node
        dot.node(node_id, label=_format_op_label(lb), **_get_node_style(lb))

        # Add Edges (from src to current)
        for parent in lb.src:
            parent_id = str(id(parent))
            walk(parent)
            dot.edge(parent_id, node_id)

    walk(root)

    try:
        output_path = dot.render(filename, view=view)
        print(f"Graph rendered to {output_path}")
    except Exception as e:
        print(
            f"Failed to render graph. Ensure Graphviz is installed on your system.\nError: {e}"
        )
