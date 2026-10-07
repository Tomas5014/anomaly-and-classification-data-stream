import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_classification_labeling_results import (
    CUMULATIVE_REQUIRED_COLUMNS,
    aggregate_results,
    filter_results,
    generate_individual_plots,
    generate_aggregate_plots,
    generate_prequential_plots,
    load_results,
    parse_prequential_configs,
    plot_heatmap_c,
    slugify,
)
from src.Results.Plots import Plots


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
    def test_results_plots_selects_non_interactive_backend(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import src.Results.Plots; "
                    "import matplotlib; "
                    "print(matplotlib.get_backend())"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip().lower(), "agg")

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

    def test_latest_execution_is_selected_separately_by_training_mode(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            legacy = pd.DataFrame([cumulative_row(F1_avg=10.0)])
            binary = pd.DataFrame([cumulative_row(
                Exec_ID="20260102_000000", Training_Label_Mode="binary",
                Evaluation_Label_Mode="binary", F1_avg=90.0,
            )])
            legacy.to_csv(directory / "legacy_cumulative.csv", sep=";", index=False)
            binary.to_csv(directory / "binary_cumulative.csv", sep=";", index=False)
            result = load_results(directory, "cumulative", "latest")
            self.assertEqual(len(result), 2)
            self.assertEqual(set(result.Training_Label_Mode), {"binary", "multiclass"})
            selected = filter_results(result, training_label_mode="binary")
            self.assertEqual(selected.iloc[0].F1_avg, 90.0)

    def test_aggregation_never_mixes_training_modes(self):
        data = pd.DataFrame([
            cumulative_row(Dataset="D1", F1_avg=10.0),
            cumulative_row(Dataset="D2", F1_avg=30.0),
            cumulative_row(Dataset="D1", Training_Label_Mode="binary", F1_avg=90.0),
            cumulative_row(Dataset="D2", Training_Label_Mode="binary", F1_avg=100.0),
        ])
        result = aggregate_results(data).set_index("Training_Label_Mode")
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc["multiclass", "F1_avg"], 20.0)
        self.assertEqual(result.loc["binary", "F1_avg"], 95.0)
        self.assertEqual(result.loc["binary", "Dataset_Count"], 2)

    def test_protocol_plots_use_separate_paths_and_manifest_rows(self):
        data = pd.DataFrame([
            cumulative_row(Training_Label_Mode="binary"),
            cumulative_row(Training_Label_Mode="multiclass"),
        ])
        prequential = data.assign(Window_Index=1, Window_Instances=100, Instance=100)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            records = generate_individual_plots(data, output_dir, ("png",), 36)
            records += generate_aggregate_plots(data, output_dir, ("png",), 36)
            records += generate_prequential_plots(
                prequential, output_dir, ((0.0, 100.0),), 1, ("png",), 36,
            )
            self.assertEqual({r["Training_Label_Mode"] for r in records}, {"binary", "multiclass"})
            self.assertEqual(len({r["File"] for r in records}), len(records))
            for record in records:
                self.assertIn(record["Training_Label_Mode"], Path(record["File"]).parts)
                self.assertTrue(Path(record["File"]).is_file())

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

    def test_detailed_fp_fn_and_label_flow_plot_is_written(self):
        data = {
            "instances": [3, 5, 7],
            "fp_mean": [0.0, 1.0, 0.0],
            "fp_std": [0.0, 0.2, 0.0],
            "fn_mean": [1.0, 0.0, 2.0],
            "fn_std": [0.1, 0.0, 0.3],
            "queried_window_mean": [2.0, 2.0, 2.0],
            "queried_window_std": [0.0, 0.0, 0.0],
            "delivered_window_mean": [0.0, 1.0, 2.0],
            "delivered_window_std": [0.0, 0.1, 0.0],
            "pending_window_mean": [2.0, 3.0, 3.0],
            "pending_window_std": [0.0, 0.2, 0.2],
            "initial_training_instances": 2,
            "initial_training_end_index": 1,
            "delay_instances": 2,
            "total_instances": 8,
            "run_count": 5,
            "labeling": {
                "effective_query_percentage": (100.0, 0.0),
                "queried_instances": (6.0, 0.0),
                "delivered_during_stream": (3.0, 0.2),
                "flushed_after_stream": (3.0, 0.2),
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Plots(["BENIGN", "DNS"]).plot_labeling_fp_fn(
                model_name="HoeffdingTree",
                data=data,
                attack_regions=[(2, 3, 1), (6, 7, 1)],
                title="Adaptação_25",
                window_size=2,
                scenario_name="Default_FullFeatures",
                configuration_slug="experiment_C_delay_25pct_labels_100pct",
                experiment="C",
                delay_percentage=25.0,
                label_budget_percentage=100.0,
                exec_id="20261006_120000",
                output_dir=temporary_directory,
            )

            output_path = Path(path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
