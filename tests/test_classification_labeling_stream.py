import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_classification_labeling_results import (
    generate_stream_plots, load_results, main, stream_history,
)
from run_classification_labeling import build_model_factories, build_stream
from src.Classification.Labeling import ClassificationLabelingExperimentRunner
from src.Results.Plots import Plots
from tests.test_classification_labeling import FakeLearner, FakeStream


class ClassificationLabelingStreamTest(unittest.TestCase):
    labels = [0, 0, 1, 1, 0, 0, 1, 1]

    def suite(self, directory=None, experiments=("A", "B", "C")):
        stream = FakeStream(self.labels)
        stream.training_label_mode = "binary"
        runner = ClassificationLabelingExperimentRunner(
            ["BENIGN", "ATTACK"], n_runs=2, attack_gap_tolerance=1,
        )
        return runner.run_suite(
            stream, {"HoeffdingTree": lambda run_seed=None: FakeLearner()},
            experiments=experiments, window_evaluation=3, experiment_name="Tiny_25",
            exec_id="20260101_000000", save_csv=directory is not None,
            output_dir=str(directory) if directory else "unused",
        )

    def test_warmup_predicts_before_training_and_does_not_enter_evaluation(self):
        class DiagnosticLearner(FakeLearner):
            def predict(self, instance):
                self.assert_unseen = instance.index not in self.trained_indices
                if not self.assert_unseen:
                    raise AssertionError("Training leaked into the current prediction")
                return 1 - instance.y_index if instance.index < 4 else instance.y_index

        runner = ClassificationLabelingExperimentRunner(["BENIGN", "ATTACK"], attack_gap_tolerance=1)
        learner = DiagnosticLearner()
        result = runner.prequential_test(FakeStream(self.labels), learner, 0.25, 1.0, 3)
        self.assertEqual(result["warmup"]["instances"], [2, 3])
        self.assertEqual(result["warmup"]["window_sizes"], [3, 1])
        self.assertEqual(result["warmup"]["fp"], [2, 0])
        self.assertEqual(result["warmup"]["fn"], [1, 1])
        self.assertEqual(result["instances"], [6, 7])
        self.assertEqual(result["window_sizes"], [3, 1])
        self.assertEqual(result["y_true"], result["y_pred"])
        self.assertEqual(result["queried_instances"], 4)
        self.assertEqual(learner.trained_indices, list(range(8)))

    def test_export_keeps_warmup_out_of_evaluation_and_cumulative_counts(self):
        suite = self.suite(experiments=("A",))
        for _, row in suite["cumulative"].iterrows():
            windows = suite["prequential"][suite["prequential"].Delay_Percentage == row.Delay_Percentage]
            self.assertEqual(windows.Window_Instances.sum(), 4)
            self.assertEqual(windows.FN_avg.sum(), row.FN_avg)
            self.assertEqual(windows.FP_avg.sum(), row.FP_avg)
            self.assertEqual(row.Queried_avg, 4)
        self.assertTrue((suite["warmup"].Instance < 4).all())
        self.assertTrue((suite["prequential"].Window_Start >= 4).all())

    def test_lines_use_window_end_indices_and_leave_bottom_padding(self):
        suite = self.suite(experiments=("A",))
        row = suite["cumulative"].iloc[0]
        windows = suite["prequential"][suite["prequential"].Delay_Percentage == row.Delay_Percentage]
        history = stream_history(row, windows, suite["warmup"])
        fig, axis = plt.subplots()
        try:
            Plots._draw_count_windows(axis, history, "fn", "gray", "warmup", warmup=True)
            Plots._draw_count_windows(axis, history, "fn", "blue", "evaluation")
            self.assertEqual(len(axis.lines), 2)  # never connect across phases
            warmup_line, evaluation_line = axis.lines
            np.testing.assert_array_equal(warmup_line.get_xdata(), [2, 3])
            np.testing.assert_array_equal(evaluation_line.get_xdata(), [6, 7])
            np.testing.assert_array_equal(warmup_line.get_ydata(), history["warmup"]["fn_mean"])
            np.testing.assert_array_equal(evaluation_line.get_ydata(), history["fn_mean"])
            for line in axis.lines:
                self.assertEqual(line.get_marker(), "o")
                self.assertEqual(line.get_drawstyle(), "default")
            self.assertEqual(warmup_line.get_linestyle(), "--")
            self.assertEqual(evaluation_line.get_linestyle(), "-")
            self.assertFalse(axis.patches)  # no stairs/step patches
            Plots._pad_count_axis(axis)
            self.assertLess(axis.get_ylim()[0], 0)
        finally:
            plt.close(fig)

    def test_partial_single_window_still_has_a_visible_marker(self):
        data = {"initial_training_instances": 2,
                "warmup": {"instances": [1], "window_sizes": [2],
                           "fp_mean": [0], "fp_std": [0]}}
        fig, axis = plt.subplots()
        try:
            Plots._draw_count_windows(axis, data, "fp", "gray", "warmup", warmup=True)
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1])
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0])
            self.assertEqual(axis.lines[0].get_marker(), "o")
        finally:
            plt.close(fig)

    def test_line_plot_keeps_window_alignment_validation(self):
        data = {"initial_training_instances": 4, "instances": [6, 7],
                "window_sizes": [2, 1], "fn_mean": [1, 0], "fn_std": [0, 0]}
        fig, axis = plt.subplots()
        try:
            with self.assertRaisesRegex(ValueError, "nonconsecutive"):
                Plots._draw_count_windows(axis, data, "fn", "blue", "evaluation")
        finally:
            plt.close(fig)

    def test_comparison_groups_have_four_delays_six_budgets_and_twelve_c_panels(self):
        suite = self.suite()
        figures = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"Label": ["BENIGN", "BENIGN", "Z_First", "Z_First",
                                    "BENIGN", "BENIGN", "A_Second", "A_Second"]}).to_csv(
                root / "Tiny_25.csv", index=False,
            )
            with patch("plot_classification_labeling_results.Plots", wraps=Plots) as plots_type, \
                 patch("matplotlib.figure.Figure.savefig", autospec=True,
                       side_effect=lambda fig, *args, **kwargs: figures.append(fig)) as save:
                records = generate_stream_plots(
                    suite["cumulative"], suite["prequential"], suite["warmup"], root, root,
                )
            # Header and external legend must survive bbox_inches='tight'.
            for call in save.call_args_list:
                self.assertEqual(len(call.kwargs["bbox_extra_artists"]), 2)
            self.assertEqual(plots_type.call_args.args[0], ["BENIGN", "Z_First", "A_Second"])
        self.assertEqual(len(records), 3)
        self.assertEqual([len(fig.axes) for fig in figures], [2, 2, 12])
        self.assertEqual(len(figures[0].axes[0].get_legend_handles_labels()[1]), 5)  # warmup + 4 delays
        self.assertEqual(len(figures[1].axes[0].get_legend_handles_labels()[1]), 7)  # warmup + 6 budgets
        self.assertTrue(all(len(axis.get_legend_handles_labels()[1]) == 5 for axis in figures[2].axes))
        self.assertTrue(all(axis.get_ylim()[0] < 0 for fig in figures for axis in fig.axes))

    def test_comparison_header_legend_and_panel_titles_do_not_overlap(self):
        suite = self.suite()
        with tempfile.TemporaryDirectory() as directory:
            for experiment in ("A", "B", "C"):
                rows = suite["cumulative"][suite["cumulative"].Experiment == experiment]
                for warmup_recorded in (True, False):
                    configurations = []
                    for _, row in rows.iterrows():
                        windows = suite["prequential"][
                            (suite["prequential"].Experiment == experiment)
                            & (suite["prequential"].Delay_Percentage == row.Delay_Percentage)
                            & (suite["prequential"].Label_Budget_Percentage == row.Label_Budget_Percentage)
                        ]
                        configurations.append({
                            "delay_percentage": row.Delay_Percentage,
                            "label_budget_percentage": row.Label_Budget_Percentage,
                            "data": stream_history(
                                row, windows, suite["warmup"] if warmup_recorded else pd.DataFrame(),
                            ),
                        })
                    for dpi in (72, 150):
                        with self.subTest(experiment=experiment, warmup=warmup_recorded, dpi=dpi):
                            figures = []
                            with patch("matplotlib.figure.Figure.savefig", autospec=True,
                                       side_effect=lambda fig, *args, **kwargs: figures.append(fig)):
                                Plots(["BENIGN", "ATTACK"]).plot_labeling_comparison(
                                    "AdaptiveRandomForest", configurations,
                                    title="Consistência_25", experiment=experiment,
                                    output_dir=directory, dpi=dpi,
                                )
                            self.assertEqual(len(figures), 1)
                            fig = figures[0]
                            # Match the export renderer even though savefig itself is mocked.
                            fig.set_dpi(dpi)
                            fig.canvas.draw()
                            renderer = fig.canvas.get_renderer()
                            heading = fig._suptitle.get_window_extent(renderer)
                            legend = fig.legends[0].get_window_extent(renderer)
                            heading_gap = heading.y0 - legend.y1
                            self.assertGreater(
                                heading_gap, 0,
                                f"Header overlaps legend by {-heading_gap:.2f} pixels",
                            )
                            for axis in fig.axes[:2]:
                                panel_title = axis.title.get_window_extent(renderer)
                                panel_gap = legend.y0 - panel_title.y1
                                self.assertGreater(
                                    panel_gap, 0,
                                    f"Legend overlaps first-row panel title by {-panel_gap:.2f} pixels",
                                )

    def test_historical_warmup_is_not_invented_or_borrowed_from_another_execution(self):
        suite = self.suite(experiments=("A",))
        row = suite["cumulative"].iloc[0]
        windows = suite["prequential"][suite["prequential"].Delay_Percentage == row.Delay_Percentage]
        wrong_warmup = suite["warmup"].assign(Exec_ID="another_run")
        self.assertNotIn("warmup", stream_history(row, windows, wrong_warmup))
        self.assertNotIn("warmup", stream_history(row, windows, pd.DataFrame()))
        legacy = windows.drop(columns=[c for c in windows if "_Window_" in c or c.startswith("Pending_At_")])
        self.assertFalse(stream_history(row, legacy, pd.DataFrame())["label_flow_recorded"])

    def test_cli_reads_warmup_and_regenerates_stream_pngs_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(load_results(root, "warmup").empty)
            self.suite(root, experiments=("A",))
            pd.DataFrame({"Label": ["BENIGN" if y == 0 else "DNS" for y in self.labels]}).to_csv(
                root / "Tiny_25.csv", index=False,
            )
            self.assertFalse(load_results(root, "warmup").empty)
            result = main([
                "--input-dir", str(root), "--data-root", str(root),
                "--output-dir", str(root / "figures"), "--stream-only",
                "--stream-plots", "all", "--experiments", "A", "--dpi", "36",
            ])
            self.assertEqual(len(result["paths"]), 5)
            self.assertTrue(all(Path(path).is_file() for path in result["paths"]))

    def test_warmup_predictions_work_with_all_four_actual_capy_moa_classifiers(self):
        frame = pd.DataFrame({"feature": list(range(8)),
                              "Label": ["BENIGN" if y == 0 else "DNS" for y in self.labels]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Tiny_25.csv"
            frame.to_csv(path, index=False)
            stream, targets, _ = build_stream(path)
            runner = ClassificationLabelingExperimentRunner(targets, attack_gap_tolerance=1)
            factories = build_model_factories(stream.get_schema(), ["ARF", "LB", "HAT", "HT"])
            for name, factory in factories.items():
                with self.subTest(model=name):
                    result = runner.prequential_test(stream, factory(run_seed=42), 0.25, 0.30, 3)
                    self.assertEqual(result["warmup"]["instances"], [2, 3])
                    self.assertEqual(len(result["y_true"]), 4)
                    self.assertEqual(sum(result["warmup"]["window_sizes"]), 4)
                    class SkipWarmupPrediction:
                        def __init__(self, learner):
                            self.learner = learner
                            self.calls = 0

                        def predict(self, instance):
                            self.calls += 1
                            return 0 if self.calls <= 4 else self.learner.predict(instance)

                        def train(self, instance):
                            self.learner.train(instance)

                    baseline = runner.prequential_test(
                        stream, SkipWarmupPrediction(factory(run_seed=42)), 0.25, 0.30, 3,
                    )
                    self.assertEqual(result["y_pred"], baseline["y_pred"])
                    self.assertEqual(result["queried_indices"], baseline["queried_indices"])


if __name__ == "__main__":
    unittest.main()
