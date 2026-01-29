import unittest

from plotly_calheatmap.i18n import get_localized_day_abbrs, get_localized_month_names


class TestI18n(unittest.TestCase):
    def test_english_month_names_default(self) -> None:
        names = get_localized_month_names()
        self.assertEqual(len(names), 12)
        self.assertEqual(names[0], "January")
        self.assertEqual(names[11], "December")

    def test_english_day_abbrs_default(self) -> None:
        abbrs = get_localized_day_abbrs()
        self.assertEqual(len(abbrs), 7)
        self.assertEqual(abbrs[0], "Mon")
        self.assertEqual(abbrs[6], "Sun")

    def test_portuguese_month_names(self) -> None:
        names = get_localized_month_names("pt_BR")
        self.assertEqual(len(names), 12)
        self.assertEqual(names[0], "janeiro")
        self.assertEqual(names[11], "dezembro")

    def test_portuguese_day_abbrs(self) -> None:
        abbrs = get_localized_day_abbrs("pt_BR")
        self.assertEqual(len(abbrs), 7)

    def test_spanish_month_names(self) -> None:
        names = get_localized_month_names("es")
        self.assertEqual(len(names), 12)
        self.assertEqual(names[0], "enero")

    def test_explicit_english_locale(self) -> None:
        names = get_localized_month_names("en")
        self.assertEqual(names[0], "January")
