"""Utilities for computing Plotly-compatible colorscales from simple color lists."""

from typing import List, Literal, Optional, Tuple, Union

import numpy as np


def compute_colorscale(
    colors: Optional[List[str]] = None,
    scale_type: Literal["linear", "diverging", "quantile", "quantize", "categorical"] = "linear",
    data: Optional[np.ndarray] = None,
    data_min: Optional[float] = None,
    data_max: Optional[float] = None,
    pivot: Optional[float] = None,
    symmetric: bool = False,
    zero_color: Optional[str] = None,
    nan_color: Optional[str] = None,
    bins: Optional[List[Tuple[float, float, str]]] = None,
) -> Union[List[List[Union[float, str]]], tuple[List[List[Union[float, str]]], float]]:
    """Compute a Plotly colorscale from a list of colors and a scale type.

    Parameters
    ----------
    colors : list of str, optional
        Colors as hex codes or CSS named colors (minimum 2).
        Not required when *scale_type* is ``"categorical"`` (bins carry
        their own colors).
    scale_type : str
        ``"linear"``, ``"diverging"``, ``"quantile"``, ``"quantize"``,
        or ``"categorical"``.
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
    bins : list of (float, float, str), optional
        Categorical bin definitions.  Each tuple is ``(min_val, max_val,
        color)``.  Required when *scale_type* is ``"categorical"``.

    Returns
    -------
    list of [float, str] or tuple of (list, float)
        Plotly-compatible colorscale. When *nan_color* is set, returns
        ``(colorscale, sentinel)`` so the caller can replace NaN values.
    """
    if scale_type == "categorical":
        if not bins:
            raise ValueError("scale_type='categorical' requires the bins parameter.")
    else:
        if colors is None or len(colors) < 2:
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

    if scale_type == "categorical":
        scale = _compute_categorical_scale(bins, data_min, data_max)
    else:
        builders = {
            "linear": _compute_linear_scale,
            "diverging": _compute_diverging_scale,
            "quantile": _compute_quantile_scale,
            "quantize": _compute_quantize_scale,
        }
        if scale_type not in builders:
            raise ValueError(
                f"Unknown scale_type {scale_type!r}. "
                f"Choose from: {', '.join(builders)}, categorical."
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


def _compute_categorical_scale(
    bins: List[Tuple[float, float, str]],
    data_min: float,
    data_max: float,
) -> List[List[Union[float, str]]]:
    """Build a piecewise-constant colorscale from user-defined bins.

    Each bin is a ``(lo, hi, color)`` tuple.  Values in ``[lo, hi]`` map
    to the given color.  Bins must be sorted by *lo* and should cover
    the full data range.
    """
    bins = sorted(bins, key=lambda b: b[0])
    data_range = data_max - data_min
    if data_range <= 0:
        return [[0.0, bins[0][2]], [1.0, bins[-1][2]]]

    scale: List[List[Union[float, str]]] = []
    for i, (lo, hi, color) in enumerate(bins):
        # Clamp to data range and normalize to [0, 1]
        norm_lo = max(0.0, min(1.0, (lo - data_min) / data_range))
        norm_hi = max(0.0, min(1.0, (min(hi, data_max) - data_min) / data_range))

        if i == 0:
            scale.append([0.0, color])
        else:
            # Sharp transition from previous color to this one
            scale.append([norm_lo, bins[i - 1][2]])
            scale.append([norm_lo, color])

    scale.append([1.0, bins[-1][2]])
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


def extract_legend_bins(
    scale_type: str,
    colors: Optional[List[str]] = None,
    data: Optional[np.ndarray] = None,
    data_min: float = 0.0,
    data_max: float = 1.0,
    bins: Optional[List[Tuple[float, float, str]]] = None,
) -> List[Tuple[float, float, str, str]]:
    """Extract discrete bin definitions for use in a clickable legend.

    Parameters
    ----------
    scale_type : str
        One of ``"quantile"``, ``"quantize"``, or ``"categorical"``.
    colors : list of str, optional
        Colors used by quantile/quantize scales.
    data : np.ndarray, optional
        Raw data values (required for ``"quantile"``).
    data_min, data_max : float
        Data range boundaries.
    bins : list of (float, float, str), optional
        Categorical bin definitions (required for ``"categorical"``).

    Returns
    -------
    list of (float, float, str, str)
        Each tuple is ``(min_val, max_val, color, label)``.
    """
    if scale_type == "categorical":
        if not bins:
            raise ValueError("scale_type='categorical' requires the bins parameter.")
        sorted_bins = sorted(bins, key=lambda b: b[0])
        return [
            (lo, hi, color, f"{lo:g} – {hi:g}")
            for lo, hi, color in sorted_bins
        ]

    if colors is None or len(colors) < 2:
        raise ValueError("At least 2 colors are required for legend bins.")

    n = len(colors)

    if scale_type == "quantile":
        if data is None:
            raise ValueError("scale_type='quantile' requires data.")
        clean = data[~np.isnan(data)]
        if len(clean) == 0:
            edges = np.linspace(data_min, data_max, n + 1)
        else:
            edges = np.quantile(clean, np.linspace(0.0, 1.0, n + 1))
        result = []
        for i in range(n):
            lo, hi = float(edges[i]), float(edges[i + 1])
            result.append((lo, hi, colors[i], f"{lo:g} – {hi:g}"))
        return result

    if scale_type == "quantize":
        data_range = data_max - data_min
        result = []
        for i in range(n):
            lo = data_min + data_range * i / n
            hi = data_min + data_range * (i + 1) / n
            result.append((lo, hi, colors[i], f"{lo:g} – {hi:g}"))
        return result

    raise ValueError(
        f"Cannot extract legend bins for scale_type={scale_type!r}. "
        "Use 'quantile', 'quantize', or 'categorical'."
    )


def _resolve_plotly_colorscale(colorscale):
    """Resolve a colorscale name or list into a list of [[pos, color], ...].

    Accepts Plotly named colorscales (e.g. ``"blues"``) or already-expanded
    lists like ``[[0, "#fff"], [1, "#000"]]``.
    """
    if isinstance(colorscale, str):
        import plotly.colors as pc

        name = colorscale.capitalize()
        # Try sequential, diverging, then qualitative
        for attr in ("sequential", "diverging", "qualitative"):
            module = getattr(pc, attr, None)
            if module and hasattr(module, name):
                raw = getattr(module, name)
                # Plotly stores them as list of rgb/hex strings
                if raw and isinstance(raw[0], str):
                    n = len(raw)
                    return [[i / (n - 1), c] for i, c in enumerate(raw)]
                return [list(pair) for pair in raw]
        # Fallback: use plotly's built-in get_colorscale
        return [list(pair) for pair in pc.get_colorscale(colorscale)]
    # Already a list
    return [list(pair) for pair in colorscale]


def build_composite_colorscale(layer_colorscales, overlap_colorscale, n_stops=6):
    """Build a single colorscale with distinct bands for each layer + overlap.

    Parameters
    ----------
    layer_colorscales : list of (str or list)
        One colorscale per layer (e.g. ``["blues", "reds"]``).
    overlap_colorscale : str or list
        Colorscale for days where multiple layers overlap.
    n_stops : int
        Number of color stops to sample from each sub-colorscale.

    Returns
    -------
    list of [float, str]
        Unified Plotly colorscale where each band occupies an equal
        fraction of [0, 1].
    """
    n_bands = len(layer_colorscales) + 1  # layers + overlap
    band_width = 1.0 / n_bands

    all_scales = [_resolve_plotly_colorscale(cs) for cs in layer_colorscales]
    all_scales.append(_resolve_plotly_colorscale(overlap_colorscale))

    composite = []
    for band_idx, scale in enumerate(all_scales):
        band_start = band_idx * band_width
        band_end = (band_idx + 1) * band_width

        # Sample n_stops evenly from the source scale
        for i in range(n_stops):
            t = i / (n_stops - 1)  # 0..1 within source scale
            # Interpolate color from source scale
            color = _interpolate_color(scale, t)
            pos = band_start + t * (band_end - band_start)
            # Clamp to avoid floating point issues
            pos = max(0.0, min(1.0, pos))
            composite.append([pos, color])

        # Add sharp boundary at band end (duplicate position with next band's
        # first color) to prevent gradient bleeding between bands.
        # Not needed for the last band.
        if band_idx < len(all_scales) - 1:
            next_scale = all_scales[band_idx + 1]
            next_color = _interpolate_color(next_scale, 0.0)
            composite.append([band_end, next_color])

    return composite


def _interpolate_color(scale, t):
    """Linearly interpolate a color at position *t* (0–1) from a colorscale.

    Parameters
    ----------
    scale : list of [float, str]
        Plotly colorscale.
    t : float
        Position in [0, 1].

    Returns
    -------
    str
        Hex color string.
    """
    import plotly.colors as pc

    if t <= 0:
        return _to_hex(scale[0][1])
    if t >= 1:
        return _to_hex(scale[-1][1])

    # Find the two surrounding stops
    for i in range(len(scale) - 1):
        p0, c0 = scale[i]
        p1, c1 = scale[i + 1]
        if p0 <= t <= p1:
            # Local interpolation factor
            f = (t - p0) / (p1 - p0) if p1 != p0 else 0.0
            r0, g0, b0 = _parse_color(c0)
            r1, g1, b1 = _parse_color(c1)
            r = int(r0 + f * (r1 - r0))
            g = int(g0 + f * (g1 - g0))
            b = int(b0 + f * (b1 - b0))
            return f"#{r:02x}{g:02x}{b:02x}"

    return _to_hex(scale[-1][1])


def _parse_color(color_str):
    """Parse a color string (hex or rgb()) into (r, g, b) ints."""
    import plotly.colors as pc

    c = str(color_str).strip()
    if c.startswith("#"):
        c = c.lstrip("#")
        if len(c) == 3:
            c = c[0] * 2 + c[1] * 2 + c[2] * 2
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    if c.startswith("rgb"):
        nums = c.replace("rgba", "").replace("rgb", "").strip("() ")
        parts = [int(float(x.strip())) for x in nums.split(",")[:3]]
        return parts[0], parts[1], parts[2]
    # Try plotly's conversion
    try:
        converted = pc.unconvert_from_RGB_255(pc.convert_to_RGB_255(pc.label_rgb(pc.unlabel_rgb(c))))
    except Exception:
        return 128, 128, 128  # fallback grey
    return int(converted[0] * 255), int(converted[1] * 255), int(converted[2] * 255)


def _to_hex(color_str):
    """Convert any color string to hex."""
    r, g, b = _parse_color(color_str)
    return f"#{r:02x}{g:02x}{b:02x}"
