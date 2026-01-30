"""Tests for colorscale_utils, focusing on the categorical scale."""

import numpy as np
import pytest

from plotly_calheatmap.colorscale_utils import (
    _compute_categorical_scale,
    compute_colorscale,
)


class TestComputeCategoricalScale:
    def test_basic_bins(self):
        bins = [(0, 0, "gray"), (1, 3, "lightgreen"), (4, 10, "darkgreen")]
        scale = _compute_categorical_scale(bins, data_min=0, data_max=10)
        # Should start at 0 and end at 1
        assert scale[0][0] == 0.0
        assert scale[-1][0] == 1.0
        # First color should be gray
        assert scale[0][1] == "gray"
        # Last color should be darkgreen
        assert scale[-1][1] == "darkgreen"

    def test_sharp_transitions(self):
        bins = [(0, 0, "gray"), (1, 5, "green")]
        scale = _compute_categorical_scale(bins, data_min=0, data_max=5)
        # At the boundary (1/5 = 0.2), there should be two entries
        boundary_entries = [e for e in scale if abs(e[0] - 0.2) < 1e-9]
        assert len(boundary_entries) == 2
        assert boundary_entries[0][1] == "gray"
        assert boundary_entries[1][1] == "green"

    def test_single_bin(self):
        bins = [(0, 10, "blue")]
        scale = _compute_categorical_scale(bins, data_min=0, data_max=10)
        assert scale[0][1] == "blue"
        assert scale[-1][1] == "blue"

    def test_unsorted_bins_get_sorted(self):
        bins = [(4, 10, "darkgreen"), (0, 0, "gray"), (1, 3, "lightgreen")]
        scale = _compute_categorical_scale(bins, data_min=0, data_max=10)
        assert scale[0][1] == "gray"
        assert scale[-1][1] == "darkgreen"


class TestComputeColorscaleCategorical:
    def test_categorical_requires_bins(self):
        with pytest.raises(ValueError, match="bins"):
            compute_colorscale(scale_type="categorical")

    def test_categorical_with_bins(self):
        bins = [(0, 0, "gray"), (1, 3, "lightgreen"), (4, 10, "darkgreen")]
        scale = compute_colorscale(
            scale_type="categorical",
            bins=bins,
            data_min=0,
            data_max=10,
        )
        assert isinstance(scale, list)
        assert scale[0][0] == 0.0
        assert scale[-1][0] == 1.0

    def test_categorical_with_nan_color(self):
        bins = [(0, 5, "green"), (6, 10, "red")]
        result = compute_colorscale(
            scale_type="categorical",
            bins=bins,
            data_min=0,
            data_max=10,
            nan_color="#ccc",
        )
        assert isinstance(result, tuple)
        scale, sentinel = result
        assert sentinel < 0

    def test_categorical_with_data_array(self):
        bins = [(0, 2, "gray"), (3, 5, "green")]
        data = np.array([0, 1, 2, 3, 4, 5])
        scale = compute_colorscale(
            scale_type="categorical",
            bins=bins,
            data=data,
        )
        assert isinstance(scale, list)

    def test_non_categorical_still_requires_colors(self):
        with pytest.raises(ValueError, match="colors"):
            compute_colorscale(scale_type="linear")
