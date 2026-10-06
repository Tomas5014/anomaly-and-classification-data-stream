import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_classification_labeling_results import (
    CUMULATIVE_REQUIRED_COLUMNS,
    aggregate_results,
    load_results,
    parse_prequential_configs,
    plot_heatmap_c,
    slugify,
)


def cumulative_row(**overrides):
    row = {
        "Exec_ID": "20260101_000000",
        "Dataset": "Cenário_25",
        "Experiment": "C",
        "Model": "HoeffdingTree",
        "Scenario": "Default_FullFeatures",
        "Delay_Percentage": 0.0,
        "Label_Budget_Percentage": 100.0,
        "Effective_Query_Percentage_avg": 100.0,
        "F1_avg": 80.0,
        "F1_std": 2.0,
        "Prec_avg": 85.0,
        "Prec_std": 1.0,
        "Rec_avg": 75.0,
        "Rec_std": 3.0,
        "FP_avg": 4.0,
        "FP_std": 1.0,
        "FN_avg": 5.0,
        "FN_std": 1.0,
    }
    row.update(overrides)
    return row


class ClassificationLabelingPlotsTest(unittest.TestCase):
    def test_parse_prequential_configs(self):
        self.assertEqual(
            parse_prequential_configs(["0:100", "5:10", "5:10"]),
            ((0.0, 100.0), (5.0, 10.0)),
        )
        with self.assertRaisesRegex(ValueError, "ATRASO:ORCAMENTO"):
            parse_prequential_configs(["invalid"])

    def test_slugify_removes_accents_and_spaces(self):
        self.assertEqual(slugify("Adaptação 25"), "Adaptacao_25")

    def test_load_results_selects_latest_execution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            old = pd.DataFrame(
                [cumulative_row(Exec_ID="20260101_000000", F1_avg=10.0)]
            )
            new = pd.DataFrame(
                [cumulative_row(Exec_ID="20260102_000000", F1_avg=90.0)]
            )
            old.to_csv(directory / "old_cumulative.csv", sep=";", index=False)
            new.to_csv(directory / "new_cumulative.csv", sep=";", index=False)

            result = load_results(directory, "cumulative", "latest")

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["Exec_ID"], "20260102_000000")
            self.assertEqual(result.iloc[0]["F1_avg"], 90.0)

    def test_load_results_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pd.DataFrame([{"Exec_ID": "1"}]).to_csv(
                directory / "invalid_cumulative.csv", sep=";", index=False
            )
            with self.assertRaisesRegex(ValueError, "colunas obrigatórias"):
                load_results(directory, "cumulative")

    def test_required_columns_match_fixture(self):
        self.assertFalse(CUMULATIVE_REQUIRED_COLUMNS - set(cumulative_row()))

    def test_aggregate_results_gives_equal_weight_to_each_dataset(self):
        data = pd.DataFrame(
            [
                cumulative_row(Dataset="D1", Exec_ID="1", F1_avg=0.0),
                cumulative_row(Dataset="D1", Exec_ID="2", F1_avg=100.0),
                cumulative_row(Dataset="D2", Exec_ID="1", F1_avg=100.0),
            ]
        )

        aggregate = aggregate_results(data)

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate.iloc[0]["Dataset_Count"], 2)
        self.assertAlmostEqual(aggregate.iloc[0]["F1_avg"], 75.0)
        self.assertAlmostEqual(aggregate.iloc[0]["F1_std"], 25.0)

    def test_heatmap_is_written(self):
        rows = []
        for delay in (0.0, 1.0, 5.0, 10.0):
            for budget in (1.0, 3.0, 5.0, 10.0, 30.0, 100.0):
                rows.append(
                    cumulative_row(
                        Delay_Percentage=delay,
                        Label_Budget_Percentage=budget,
                        F1_avg=max(0.0, 100.0 - delay * 5.0 - (100.0 - budget) / 2.0),
                    )
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_base = Path(temporary_directory) / "heatmap"
            paths = plot_heatmap_c(
                pd.DataFrame(rows),
                "Teste",
                output_base,
                ("png",),
                72,
            )

            self.assertEqual(paths, [output_base.with_suffix(".png")])
            self.assertTrue(paths[0].is_file())
            self.assertGreater(paths[0].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
