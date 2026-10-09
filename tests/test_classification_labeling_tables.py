import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

from tabulate_classification_labeling_results import (
    GRIDS,
    KEY,
    METRICS,
    MODEL_NAMES,
    aggregate_scenarios,
    compare_modes,
    dataset_order,
    experiment_membership,
    format_number,
    generate_tables,
    load_batch,
    shared_labeling,
    validate_batch,
)


def batch_fixture(exec_id, mode, datasets=("Consistência_25",)):
    rows = []
    for dataset in datasets:
        initial = 20
        evaluation = 80 if dataset.endswith("_25") else 180
        total = initial + evaluation
        for model in MODEL_NAMES:
            for experiment, grid in GRIDS.items():
                for delay, budget in sorted(grid):
                    queried = evaluation * budget / 100
                    tail = queried * delay / 100
                    row = dict(
                        Exec_ID=exec_id, Dataset=dataset, Scenario="Default_FullFeatures",
                        Training_Label_Mode=mode, Evaluation_Label_Mode="binary",
                        Experiment=experiment, Model=model, Delay_Percentage=delay,
                        Label_Budget_Percentage=budget, Delay_Instances=round(total * delay / 100),
                        Initial_Training_Instances=initial, Evaluation_Instances=evaluation,
                        Window_Evaluation=100, Runs=5, Queried_avg=queried,
                        Queried_std=0.0 if budget == 100 else 0.2,
                        Effective_Query_Percentage_avg=budget,
                        Effective_Query_Percentage_std=0.0 if budget == 100 else 20 / evaluation,
                        Delivered_During_Stream_avg=queried-tail, Delivered_During_Stream_std=0.0,
                        Flushed_After_Stream_avg=tail, Flushed_After_Stream_std=0.0,
                    )
                    for metric in METRICS:
                        row[f"{metric}_avg"] = 70.0
                        row[f"{metric}_std"] = 2.34
                    row.update(
                        F1_avg=72.3456789 + (10 if dataset.endswith("_200") else 0)
                        + (2 if mode == "binary" else 0),
                        MCC_avg=0.7654321, MCC_std=0.012345,
                        FP_avg=3.4, FP_std=0.49, FN_avg=5.6, FN_std=0.8,
                        Time_avg=1.123456789,
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def write_batch(directory, exec_id, mode, datasets=("Consistência_25",), legacy=False):
    frame = batch_fixture(exec_id, mode, datasets)
    paths = []
    for dataset, group in frame.groupby("Dataset"):
        infix = "_binaryTraining" if mode == "binary" else ""
        path = directory / f"{dataset}_Default_FullFeatures{infix}_{exec_id}_cumulative.csv"
        if legacy:
            group = group.drop(columns=["Training_Label_Mode", "Evaluation_Label_Mode"])
        group.to_csv(path, sep=";", index=False)
        prequential = path.with_name(path.name.replace("_cumulative.csv", "_prequential.csv"))
        group.to_csv(prequential, sep=";", index=False)
        paths.extend((path, prequential))
    return paths


class ClassificationLabelingTablesTest(unittest.TestCase):
    def test_portuguese_numbers_preserve_fractional_errors(self):
        self.assertEqual(format_number(3.4), "3,40")
        self.assertEqual(format_number(12345.678), "12.345,68")
        self.assertEqual(format_number(-0.00001, signed=True), "+0,00")
        self.assertEqual(format_number(0.7654321, 4), "0,7654")

    def test_natural_dataset_order(self):
        self.assertEqual(sorted(["Recorrência_25", "Consistência_1000", "Consistência_25",
                                 "Consistência_200"], key=dataset_order),
                         ["Consistência_25", "Consistência_200", "Consistência_1000", "Recorrência_25"])

    def test_experiment_equivalences(self):
        self.assertEqual(experiment_membership(0, 100), "A,B,C")
        self.assertEqual(experiment_membership(5, 100), "A,C")
        self.assertEqual(experiment_membership(0, 3), "B,C")
        self.assertEqual(experiment_membership(5, 3), "C")

    def test_complete_grid_passes_and_duplicate_rejected(self):
        frame = batch_fixture("b", "binary")
        validate_batch(frame)
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            validate_batch(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))

    def test_incomplete_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "grade incompleta"):
            validate_batch(batch_fixture("b", "binary").iloc[1:])

    def test_equivalent_a_and_c_must_have_same_metrics(self):
        frame = batch_fixture("b", "binary")
        frame.loc[frame.Experiment.eq("A"), "F1_avg"] = 1.0
        with self.assertRaisesRegex(ValueError, "equivalentes em C divergem"):
            validate_batch(frame)

    def test_consultation_conservation_checked(self):
        frame = batch_fixture("b", "binary")
        frame.loc[0, "Flushed_After_Stream_avg"] = 500.0
        with self.assertRaisesRegex(ValueError, "entregas"):
            validate_batch(frame)

    def test_compare_uses_unique_c_and_unrounded_means(self):
        multi = batch_fixture("m", "multiclass")
        binary = batch_fixture("b", "binary")
        binary.loc[:, "F1_avg"] = multi.F1_avg + 0.004
        pair = compare_modes(pd.concat([multi, binary], ignore_index=True))
        self.assertEqual(len(pair), 96)
        self.assertFalse(pair.duplicated(KEY).any())
        np.testing.assert_allclose(pair.Delta_F1, 0.004, atol=1e-12)

    def test_compare_rejects_unmatched_dataset_and_query_cost(self):
        frame = pd.concat([batch_fixture("m", "multiclass"),
                           batch_fixture("b", "binary", ("Consistência_200",))], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "mesmas condições"):
            compare_modes(frame)
        frame = pd.concat([batch_fixture("m", "multiclass"), batch_fixture("b", "binary")], ignore_index=True)
        frame.loc[frame.Training_Label_Mode.eq("binary"), "Queried_avg"] += 1
        with self.assertRaisesRegex(ValueError, "custos"):
            compare_modes(frame)

    def test_summary_std_is_between_scenarios_not_run_std(self):
        frame = batch_fixture("b", "binary", ("Consistência_25", "Consistência_200"))
        frame.loc[:, "Total_Label_Cost_avg"] = (
            100 * (frame.Initial_Training_Instances + frame.Queried_avg)
            / (frame.Initial_Training_Instances + frame.Evaluation_Instances)
        )
        frame.loc[:, "Delivered_Percentage_avg"] = (
            100 * frame.Delivered_During_Stream_avg / frame.Evaluation_Instances
        )
        summary = aggregate_scenarios(frame)
        self.assertEqual(len(summary), 136)
        np.testing.assert_allclose(summary.F1_mean, 79.3456789)
        np.testing.assert_allclose(summary.F1_std_between_scenarios, 5.0)
        self.assertTrue(summary.Dataset_Count.eq(2).all())
        self.assertNotIn("F1_std", summary)
        with self.assertRaisesRegex(ValueError, "múltiplas observações"):
            aggregate_scenarios(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))

    def test_explicit_batch_selection_legacy_annotation_and_total_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_batch(root, "m", "multiclass", legacy=True)
            write_batch(root, "extra", "multiclass", legacy=True)
            paths = write_batch(root, "b", "binary")
            multi = load_batch(root, "m", "multiclass", "Default_FullFeatures")
            binary = load_batch(root, "b", "binary", "Default_FullFeatures")
            self.assertTrue(multi.Exec_ID.eq("m").all())
            self.assertTrue(multi.Training_Label_Mode.eq("multiclass").all())
            row = binary.loc[binary.Label_Budget_Percentage.eq(1)].iloc[0]
            self.assertAlmostEqual(row.Total_Label_Cost_avg, 20.8)
            self.assertAlmostEqual(row.Total_Label_Cost_std, 0.2)
            tables = shared_labeling(pd.concat([multi, binary], ignore_index=True))
            self.assertEqual(len(tables), 24)
            bad = pd.read_csv(paths[0], sep=";").drop(columns=["Training_Label_Mode"])
            bad.to_csv(paths[0], sep=";", index=False)
            with self.assertRaisesRegex(ValueError, "falta Training_Label_Mode"):
                load_batch(root, "b", "binary", "Default_FullFeatures")

    def test_end_to_end_tables_links_counts_precision_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = ("Consistência_25", "Consistência_200")
            paths = write_batch(root, "m", "multiclass", datasets, legacy=True)
            paths += write_batch(root, "b", "binary", datasets)
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            data = pd.concat([
                load_batch(root, "m", "multiclass", "Default_FullFeatures"),
                load_batch(root, "b", "binary", "Default_FullFeatures"),
            ], ignore_index=True)
            output = root / "tables"
            index = root / "index.md"
            manifest = generate_tables(data, output, index)
            self.assertEqual(len(manifest), 20)
            expected_counts = {"acumulados_completos.csv": 544, "condicoes_unicas.csv": 384,
                               "comparacao_binario_multiclasse.csv": 192,
                               "resumos_por_experimento.csv": 272, "custos_rotulagem.csv": 48}
            for filename, count in expected_counts.items():
                exported = pd.read_csv(output / filename, sep=";", float_precision="round_trip")
                self.assertEqual(len(exported), count)
                if filename == "acumulados_completos.csv":
                    self.assertAlmostEqual(exported.F1_avg.iloc[0], data.F1_avg.iloc[0], places=12)
                    self.assertEqual(exported.FP_avg.iloc[0], 3.4)
            exported = pd.read_csv(output / "acumulados_completos.csv", sep=";")
            for column in ("Source_Cumulative", "Source_Prequential"):
                for value in exported[column].unique():
                    self.assertFalse(Path(value).is_absolute())
                    self.assertTrue((output / value).is_file())
            for document in [index] + list(output.rglob("*.md")):
                text = document.read_text(encoding="utf-8")
                for target in re.findall(r"\]\(([^)]+)\)", text):
                    self.assertTrue((document.parent / unquote(target)).is_file(), (document, target))
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if line.startswith("| ") and (i == 0 or not lines[i-1].startswith("| ")):
                        self.assertTrue(i == 0 or lines[i-1] == "", (document, i))
            individual = (output / "binary" / "Consistencia_25.md").read_text()
            self.assertEqual(individual.count("### Experimento"), 12)
            self.assertIn("3,40 ± 0,49", individual)
            manifest_frame = pd.read_csv(output / "manifest.csv", sep=";")
            self.assertEqual(len(manifest_frame), 19)  # Não lista o próprio manifesto.
            self.assertTrue(all((output / p).is_file() for p in manifest_frame.File))
            for path, before in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
