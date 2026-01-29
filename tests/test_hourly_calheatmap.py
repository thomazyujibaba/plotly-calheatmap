from datetime import datetime
from unittest import TestCase

import pandas as pd
from plotly import graph_objects as go

from plotly_calheatmap.hourly_calheatmap import hourly_calheatmap


class TestHourlyCalplot(TestCase):
    def setUp(self) -> None:
        # One month of hourly data
        dates = pd.date_range("2023-01-01", "2023-01-31 23:00:00", freq="h")
        self.one_month_df = pd.DataFrame(
            {"ds": dates, "value": range(len(dates))}
        )

        # Two years, sparse data
        dates_multi = pd.date_range("2022-06-01", "2023-06-30 23:00:00", freq="3h")
        self.multi_year_df = pd.DataFrame(
            {"ds": dates_multi, "value": range(len(dates_multi))}
        )

    def test_basic_one_month(self) -> None:
        fig = hourly_calheatmap(self.one_month_df, "ds", "value")
        self.assertIsInstance(fig, go.Figure)
        # Should have 1 heatmap trace (1 year × 1 month)
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        self.assertEqual(len(heatmaps), 1)

    def test_multi_year(self) -> None:
        fig = hourly_calheatmap(self.multi_year_df, "ds", "value")
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        # 2 years × multiple months present
        self.assertGreater(len(heatmaps), 2)

    def test_aggregation(self) -> None:
        # Duplicate rows to test aggregation
        df = pd.DataFrame({
            "ds": [datetime(2023, 1, 1, 10)] * 3,
            "value": [10, 20, 30],
        })
        fig = hourly_calheatmap(df, "ds", "value", agg="sum")
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        self.assertEqual(len(heatmaps), 1)
        # z value at day=1, hour=10 should be 60
        z = heatmaps[0].z
        self.assertEqual(z[10][0], 60.0)

    def test_dark_theme(self) -> None:
        fig = hourly_calheatmap(self.one_month_df, "ds", "value", dark_theme=True)
        self.assertEqual(fig.layout.paper_bgcolor, "#333")

    def test_colorscale(self) -> None:
        fig = hourly_calheatmap(
            self.one_month_df, "ds", "value", colorscale="blues"
        )
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        # Plotly resolves named colorscales to tuples; just check it's not the default
        self.assertIsNotNone(heatmaps[0].colorscale)

    def test_showscale(self) -> None:
        fig = hourly_calheatmap(
            self.one_month_df, "ds", "value", showscale=True
        )
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        has_showscale = any(t.showscale for t in heatmaps)
        self.assertTrue(has_showscale)

    def test_locale(self) -> None:
        fig = hourly_calheatmap(
            self.one_month_df, "ds", "value", locale="pt_BR"
        )
        # Subplot title should use Portuguese month name
        annotations = [a.text for a in fig.layout.annotations]
        self.assertTrue(any("janeiro" in a.lower() for a in annotations))

    def test_navigation(self) -> None:
        fig = hourly_calheatmap(
            self.multi_year_df, "ds", "value", navigation=True
        )
        # Should have navigation buttons
        self.assertIsNotNone(fig.layout.updatemenus)
        self.assertEqual(len(fig.layout.updatemenus), 1)
        buttons = fig.layout.updatemenus[0].buttons
        self.assertEqual(len(buttons), 2)  # 2 years
        # Only first year traces should be visible
        visible_count = sum(1 for t in fig.data if t.visible is not False)
        self.assertEqual(visible_count, 12)  # 12 months for first year

    def test_colorbar_nticks(self) -> None:
        fig = hourly_calheatmap(
            self.one_month_df, "ds", "value", showscale=True
        )
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        colorbar = heatmaps[0].colorbar
        self.assertEqual(colorbar.nticks, 5)
