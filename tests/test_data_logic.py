import unittest

import pandas as pd

from data_logic import (
    BulkFillConfig,
    build_column_description,
    build_dataset_overview_text,
    bulk_fill_missing,
    compute_outlier_records,
    convert_series_type,
    fill_missing_values,
)


class DataLogicTests(unittest.TestCase):
    def test_compute_outlier_records_detects_extreme_value(self):
        series = pd.Series([10, 11, 12, 13, 99], index=[0, 1, 2, 3, 4])

        records, bounds = compute_outlier_records(series)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["row_index"], 4)
        self.assertEqual(records[0]["side"], "above")
        self.assertLess(bounds["upper"], 99)

    def test_convert_series_type_to_integer_marks_invalid_values(self):
        series = pd.Series(["1", "2", "3.5", "x", None])

        result = convert_series_type(series, "integer")

        self.assertEqual(result.invalid_count, 2)
        self.assertEqual(str(result.converted.dtype), "Int64")
        self.assertEqual(result.converted.iloc[0], 1)
        self.assertTrue(pd.isna(result.converted.iloc[2]))
        self.assertTrue(pd.isna(result.converted.iloc[3]))

    def test_convert_series_type_to_boolean_supports_russian_values(self):
        series = pd.Series(["да", "нет", "true", "off", "непонятно"])

        result = convert_series_type(series, "boolean")

        self.assertEqual(result.invalid_count, 1)
        self.assertEqual(result.converted.tolist()[:4], [True, False, True, False])
        self.assertTrue(pd.isna(result.converted.iloc[4]))

    def test_fill_missing_values_uses_group_and_global_fallback(self):
        df = pd.DataFrame(
            {
                "city": ["A", "A", "B", "B", "C"],
                "segment": ["x", "x", "x", "y", "z"],
                "income": [10.0, None, 30.0, None, None],
            }
        )

        result = fill_missing_values(
            df,
            "income",
            "mean",
            group_columns=("city",),
            fallback_to_global=True,
        )

        self.assertEqual(result.filled_count, 3)
        self.assertEqual(result.remaining_missing, 0)
        self.assertEqual(result.series.tolist(), [10.0, 10.0, 30.0, 30.0, 20.0])

    def test_bulk_fill_missing_respects_config(self):
        df = pd.DataFrame(
            {
                "age": [10.0, None, 30.0],
                "city": ["A", None, "A"],
                "notes": [None, None, None],
            }
        )
        config = BulkFillConfig(
            numeric_method="mean",
            categorical_method="constant",
            categorical_constant="unknown",
            fallback_to_global=True,
        )

        result_df, report = bulk_fill_missing(df, config)

        self.assertEqual(result_df["age"].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(result_df["city"].tolist(), ["A", "unknown", "A"])
        self.assertIn("notes", report["skipped_columns"])

    def test_dataset_and_column_descriptions_include_key_info(self):
        df = pd.DataFrame({"age": [10, 20, None], "city": ["A", "B", "A"]})

        overview = build_dataset_overview_text(df)
        column_text = build_column_description(df, "age")

        self.assertIn("Строк: 3", overview)
        self.assertIn("Пропусков: 1", overview)
        self.assertIn("Столбец: age", column_text)
        self.assertIn("Пропусков: 1", column_text)


if __name__ == "__main__":
    unittest.main()
