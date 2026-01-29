"""Utilities for computing Plotly-compatible colorscales from simple color lists."""

from typing import List, Literal, Optional, Tuple, Union

import numpy as np


def compute_colorscale(
    colors: List[str],
    scale_type: Literal["linear", "diverging", "quantile", "quantize"] = "linear",
    data: Optional[np.ndarray] = None,
    data_min: Optional[float] = None,
    data_max: Optional[float] = None,
    pivot: Optional[float] = None,
    symmetric: bool = False,
    zero_color: Optional[str] = None,
    nan_color: Optional[str] = None,
) -> Union[List[List[Union[float, str]]], tuple[List[List[Union[float, str]]], float]]:
    """Compute a Plotly colorscale from a list of colors and a scale type.

    Parameters
    ----------
    colors : list of str
        Colors as hex codes or CSS named colors (minimum 2).
    scale_type : str
        ``"linear"``, ``"diverging"``, ``"quantile"``, or ``"quantize"``.
    data : np.ndarray, optional
        Raw data values.  Required for ``"quantile"``; used to derive
        *data_min*/*data_max* when those are not supplied.
    data_min, data_max : float, optional
        Explicit data range.  Computed from *data* when omitted.
    pivot : float, optional
        Center value for ``"diverging"`` (default ``0``).
    symmetric : bool
        For ``"diverging"``: extend range so *pivot* is exactly centered.
    zero_color : str, optional
        Dedicated color for value ``0``.  A sharp breakpoint is inserted
        so that ``0`` maps to this color while values > 0 use the
        regular gradient.
    nan_color : str, optional
        Dedicated color for missing/NaN values.  When set, NaN cells are
        replaced with a sentinel value and a sharp breakpoint is inserted
        at the bottom of the colorscale.  Returns a tuple of
        ``(colorscale, sentinel)`` instead of just the colorscale.

    Returns
    -------
    list of [float, str] or tuple of (list, float)
        Plotly-compatible colorscale. When *nan_color* is set, returns
        ``(colorscale, sentinel)`` so the caller can replace NaN values.
    """
    if len(colors) < 2:
        raise ValueError("At least 2 colors are required.")

    if data is not None:
        clean = data[~np.isnan(data)]
        if data_min is None:
            data_min = float(clean.min()) if len(clean) else 0.0
        if data_max is None:
            data_max = float(clean.max()) if len(clean) else 1.0
    else:
        if data_min is None:
            data_min = 0.0
        if data_max is None:
            data_max = 1.0

    if data_min == data_max:
        data_max = data_min + 1.0

    builders = {
        "linear": _compute_linear_scale,
        "diverging": _compute_diverging_scale,
        "quantile": _compute_quantile_scale,
        "quantize": _compute_quantize_scale,
    }
    if scale_type not in builders:
        raise ValueError(
            f"Unknown scale_type {scale_type!r}. "
            f"Choose from: {', '.join(builders)}."
        )

    if scale_type == "quantile":
        if data is None:
            raise ValueError("scale_type='quantile' requires the data parameter.")
        scale = builders[scale_type](colors, data)
    elif scale_type == "diverging":
        scale = builders[scale_type](
            colors, data_min, data_max, pivot=pivot, symmetric=symmetric,
        )
    else:
        scale = builders[scale_type](colors, data_min, data_max)

    if zero_color is not None:
        scale = _apply_zero_color(scale, zero_color, data_min, data_max)

    if nan_color is not None:
        scale, sentinel = _apply_nan_color(scale, nan_color, data_min, data_max)
        return scale, sentinel

    return scale


# ---------------------------------------------------------------------------
# Scale builders
# ---------------------------------------------------------------------------

def _compute_linear_scale(
    colors: List[str], data_min: float, data_max: float,
) -> List[List[Union[float, str]]]:
    positions = np.linspace(0.0, 1.0, len(colors))
    return [[float(p), c] for p, c in zip(positions, colors)]


def _compute_diverging_scale(
    colors: List[str],
    data_min: float,
    data_max: float,
    pivot: Optional[float] = None,
    symmetric: bool = False,
) -> List[List[Union[float, str]]]:
    if pivot is None:
        pivot = 0.0

    if symmetric:
        max_dist = max(abs(data_max - pivot), abs(data_min - pivot))
        data_min = pivot - max_dist
        data_max = pivot + max_dist

    data_range = data_max - data_min
    if data_range == 0:
        data_range = 1.0

    pivot_pos = (pivot - data_min) / data_range
    pivot_pos = max(0.0, min(1.0, pivot_pos))

    n = len(colors)
    mid = n // 2

    # Build positions: first half maps to [0, pivot_pos],
    # second half maps to [pivot_pos, 1]
    positions = []
    for i in range(n):
        if i <= mid:
            positions.append(pivot_pos * i / mid if mid > 0 else 0.0)
        else:
            positions.append(
                pivot_pos + (1.0 - pivot_pos) * (i - mid) / (n - 1 - mid)
            )

    return [[float(p), c] for p, c in zip(positions, colors)]


def _compute_quantile_scale(
    colors: List[str], data: np.ndarray,
) -> List[List[Union[float, str]]]:
    clean = data[~np.isnan(data)]
    if len(clean) == 0:
        return _compute_linear_scale(colors, 0.0, 1.0)

    n = len(colors)
    # Compute quantile boundaries (n bins → n+1 edges)
    edges = np.quantile(clean, np.linspace(0.0, 1.0, n + 1))
    dmin, dmax = float(edges[0]), float(edges[-1])
    data_range = dmax - dmin if dmax != dmin else 1.0

    # Build piecewise-constant colorscale
    scale: List[List[Union[float, str]]] = []
    for i in range(n):
        lo = (float(edges[i]) - dmin) / data_range
        hi = (float(edges[i + 1]) - dmin) / data_range
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if i == 0:
            scale.append([lo, colors[i]])
        else:
            # sharp transition
            scale.append([lo, colors[i - 1]])
            scale.append([lo, colors[i]])
    scale.append([1.0, colors[-1]])
    return scale


def _compute_quantize_scale(
    colors: List[str], data_min: float, data_max: float,
) -> List[List[Union[float, str]]]:
    n = len(colors)
    scale: List[List[Union[float, str]]] = []
    for i in range(n):
        lo = i / n
        hi = (i + 1) / n
        if i == 0:
            scale.append([lo, colors[i]])
        else:
            scale.append([lo, colors[i - 1]])
            scale.append([lo, colors[i]])
    scale.append([1.0, colors[-1]])
    return scale


# ---------------------------------------------------------------------------
# Zero-color helper
# ---------------------------------------------------------------------------

def _apply_zero_color(
    scale: List[List[Union[float, str]]],
    zero_color: str,
    data_min: float,
    data_max: float,
) -> List[List[Union[float, str]]]:
    """Insert a sharp breakpoint so that value 0 maps to *zero_color*."""
    if data_min > 0:
        # No zeros possible in range — nothing to do
        return scale

    data_range = data_max - data_min
    if data_range <= 0:
        return [[0.0, zero_color], [1.0, zero_color]]

    zero_pos = (0.0 - data_min) / data_range
    # Small epsilon for the sharp transition
    eps = min(0.001, zero_pos * 0.1) if zero_pos > 0 else 0.001

    breakpoint = zero_pos + eps
    if breakpoint >= 1.0:
        # All values ≤ 0
        return [[0.0, zero_color], [1.0, zero_color]]

    new_scale: List[List[Union[float, str]]] = [
        [0.0, zero_color],
        [breakpoint, zero_color],
    ]

    # Re-map all original positions into [breakpoint, 1.0]
    old_lo = scale[0][0]
    old_hi = scale[-1][0]
    old_range = old_hi - old_lo if old_hi != old_lo else 1.0
    for p, c in scale:
        new_p = breakpoint + (1.0 - breakpoint) * (p - old_lo) / old_range
        new_scale.append([new_p, c])

    # Ensure it ends at 1.0
    new_scale[-1][0] = 1.0

    return new_scale


def _apply_nan_color(
    scale: List[List[Union[float, str]]],
    nan_color: str,
    data_min: float,
    data_max: float,
) -> tuple[List[List[Union[float, str]]], float]:
    """Insert a sharp breakpoint at the bottom so that a sentinel value maps to *nan_color*.

    Returns the modified scale and the sentinel value to use for NaN cells.
    """
    data_range = data_max - data_min
    if data_range <= 0:
        sentinel = data_min - 1.0
        return [[0.0, nan_color], [1.0, nan_color]], sentinel

    # Sentinel sits just below the real data range
    sentinel = data_min - data_range * 0.01
    new_range = data_max - sentinel

    # Breakpoint where sentinel region ends and real data begins
    breakpoint = (data_min - sentinel) / new_range
    eps = min(0.001, breakpoint * 0.1)
    breakpoint_end = breakpoint + eps
    if breakpoint_end >= 1.0:
        return [[0.0, nan_color], [1.0, nan_color]], sentinel

    new_scale: List[List[Union[float, str]]] = [
        [0.0, nan_color],
        [breakpoint_end, nan_color],
    ]

    # Re-map original positions into [breakpoint_end, 1.0]
    old_lo = scale[0][0]
    old_hi = scale[-1][0]
    old_range = old_hi - old_lo if old_hi != old_lo else 1.0
    for p, c in scale:
        new_p = breakpoint_end + (1.0 - breakpoint_end) * (p - old_lo) / old_range
        new_scale.append([new_p, c])

    new_scale[-1][0] = 1.0

    return new_scale, sentinel
