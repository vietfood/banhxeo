"""
Thanks Gemini 3.0 Pro and Claude Sonnet 4.5
"""

from typing import List, Set, Tuple

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


def _get_op_symbol(lb: LazyBuffer) -> str:
    """Returns a visual symbol for each operation type"""
    if isinstance(lb.op, LoadOp):
        return "📥"
    elif isinstance(lb.op, MovementOp):
        return "↔️"
    elif isinstance(lb.op, BinaryOp):
        return "⊕"
    elif isinstance(lb.op, UnaryOp):
        return "◯"
    elif isinstance(lb.op, ReduceOp):
        return "Σ"
    return "?"


def _get_color_code(lb: LazyBuffer) -> str:
    """Returns ANSI color code for terminal output"""
    if isinstance(lb.op, LoadOp):
        return "\033[93m"  # Yellow
    elif isinstance(lb.op, MovementOp):
        return "\033[96m"  # Cyan
    elif isinstance(lb.op, (BinaryOp, UnaryOp, ReduceOp)):
        return "\033[92m"  # Green
    return "\033[0m"  # Reset


def visualize_schedule_cli(schedule: List[LazyBuffer], compact: bool = False):
    """
    Visualizes a linearized schedule in execution order.

    Args:
        schedule: List of LazyBuffer nodes in topological order
        compact: If True, shows minimal info per node
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    print("\n" + "=" * 100)
    print(f"{BOLD}LazyBuffer Schedule Visualization{RESET}")
    print("=" * 100)
    print(f"\nExecution order: {len(schedule)} operations")
    print(
        "\nLegend: 📥 LoadOp  ↔️ MovementOp  ⊕ BinaryOp  ◯ UnaryOp  Σ ReduceOp  |  ✓ Realized  ○ Lazy"
    )
    print("=" * 100 + "\n")

    for idx, lb in enumerate(schedule):
        op_name = str(lb.op).split(".")[-1]
        symbol = _get_op_symbol(lb)
        color = _get_color_code(lb)
        realized = "✓" if lb.realized else "○"

        if compact:
            print(f"{idx:3d}. {color}{symbol} {op_name:<15}{RESET} {realized}")
        else:
            shape = str(lb.view.shape)
            strides = str(lb.view.strides)
            offset = lb.view.offset

            # Truncate if too long
            if len(shape) > 20:
                shape = shape[:17] + "..."
            if len(strides) > 20:
                strides = strides[:17] + "..."

            print(
                f"{idx:3d}. {color}{symbol} {op_name:<15}{RESET} {realized}  "
                f"shape={shape:<20} strides={strides:<20} offset={offset}"
            )

    print("\n" + "=" * 100 + "\n")


def visualize_schedule_flow(schedule: List[LazyBuffer]):
    """
    Shows the schedule as a data flow diagram with arrows.
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print("\n" + "=" * 100)
    print(f"{BOLD}Schedule Data Flow{RESET}")
    print("=" * 100 + "\n")

    for idx, lb in enumerate(schedule):
        op_name = str(lb.op).split(".")[-1]
        symbol = _get_op_symbol(lb)
        color = _get_color_code(lb)
        realized = "✓" if lb.realized else "○"

        # Show dependencies
        if lb.src:
            deps = ", ".join(
                [str(schedule.index(src)) for src in lb.src if src in schedule]
            )
            deps_str = f"  ← depends on: [{deps}]"
        else:
            deps_str = "  (source)"

        print(f"{idx:3d}. {color}{symbol} {op_name:<15}{RESET} {realized} {deps_str}")

        # Draw arrow to next step
        if idx < len(schedule) - 1:
            print("      ↓")

    print("\n" + "=" * 100 + "\n")


def visualize_schedule_stats(schedule: List[LazyBuffer]):
    """
    Shows statistics about the schedule.
    """
    from collections import Counter

    RESET = "\033[0m"
    BOLD = "\033[1m"

    op_counts = Counter()
    realized_count = 0

    for lb in schedule:
        op_type = type(lb.op).__name__
        op_counts[op_type] += 1
        if lb.realized:
            realized_count += 1

    print("\n" + "=" * 60)
    print(f"{BOLD}Schedule Statistics{RESET}")
    print("=" * 60)
    print(f"\nTotal operations: {len(schedule)}")
    print(f"Realized nodes:   {realized_count}")
    print(f"Lazy nodes:       {len(schedule) - realized_count}")
    print("\nOperation breakdown:")
    for op_type, count in op_counts.most_common():
        print(f"  {op_type:<20} {count:>4}")
    print("=" * 60 + "\n")


def visualize_buffers_list(
    buffers: List[LazyBuffer], title: str = "LazyBuffer List", compact: bool = False
):
    """
    Visualizes a simple list of LazyBuffers (no dependency tracking).

    Args:
        buffers: List of LazyBuffer nodes
        title: Custom title for the visualization
        compact: If True, shows minimal info per node
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print("\n" + "=" * 100)
    print(f"{BOLD}{title}{RESET}")
    print("=" * 100)
    print(f"\nTotal buffers: {len(buffers)}")
    print(
        "\nLegend: 📥 LoadOp  ↔️ MovementOp  ⊕ BinaryOp  ◯ UnaryOp  Σ ReduceOp  |  ✓ Realized  ○ Lazy"
    )
    print("=" * 100 + "\n")

    for idx, lb in enumerate(buffers):
        op_name = str(lb.op).split(".")[-1]
        symbol = _get_op_symbol(lb)
        color = _get_color_code(lb)
        realized = "✓" if lb.realized else "○"

        if compact:
            print(f"{idx:3d}. {color}{symbol} {op_name:<15}{RESET} {realized}")
        else:
            shape = str(lb.view.shape)
            strides = str(lb.view.strides)
            offset = lb.view.offset

            # Truncate if too long
            if len(shape) > 20:
                shape = shape[:17] + "..."
            if len(strides) > 20:
                strides = strides[:17] + "..."

            print(
                f"{idx:3d}. {color}{symbol} {op_name:<15}{RESET} {realized}  "
                f"shape={shape:<20} strides={strides:<20} offset={offset}"
            )

    print("\n" + "=" * 100 + "\n")
