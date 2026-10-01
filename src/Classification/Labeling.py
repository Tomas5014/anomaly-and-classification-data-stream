from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import re
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.Results.Metrics import Metrics
from src.Results.Plots import Plots


DEFAULT_DELAY_FRACTIONS = (0.0, 0.01, 0.05, 0.10)
DEFAULT_LABEL_PROBABILITIES = (0.01, 0.03, 0.05, 0.10, 0.30, 1.0)


@dataclass(frozen=True)
class LabelingExperimentConfig:
    """Configuration for delayed labels and random active learning."""

    experiment: str
    delay_fraction: float
    label_probability: float

    def __post_init__(self):
        experiment = self.experiment.upper()
        if experiment not in {"A", "B", "C"}:
            raise ValueError("experiment must be A, B, or C")
        if not 0.0 <= self.delay_fraction <= 1.0:
            raise ValueError("delay_fraction must be between 0 and 1")
        if not 0.0 <= self.label_probability <= 1.0:
            raise ValueError("label_probability must be between 0 and 1")
        object.__setattr__(self, "experiment", experiment)

    @property
    def delay_percentage(self) -> float:
        return self.delay_fraction * 100.0

    @property
    def label_budget_percentage(self) -> float:
        return self.label_probability * 100.0

    @property
    def slug(self) -> str:
        delay = _format_percentage(self.delay_percentage)
        budget = _format_percentage(self.label_budget_percentage)
        return f"experiment_{self.experiment}_delay_{delay}pct_labels_{budget}pct"


def build_experiment_configs(
    experiments: Sequence[str] = ("A", "B", "C"),
    delay_fractions: Sequence[float] = DEFAULT_DELAY_FRACTIONS,
    label_probabilities: Sequence[float] = DEFAULT_LABEL_PROBABILITIES,
) -> list[LabelingExperimentConfig]:
    """Build the three experiment grids requested by the project.

    Experiment A varies label delay with full supervision. Experiment B varies
    random label availability without delay. Experiment C uses the Cartesian
    product of both sets.
    """

    normalized_experiments = [str(value).upper() for value in experiments]
    unknown = sorted(set(normalized_experiments) - {"A", "B", "C"})
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")

    configs = []
    for experiment in normalized_experiments:
        if experiment == "A":
            configs.extend(
                LabelingExperimentConfig("A", delay, 1.0)
                for delay in delay_fractions
            )
        elif experiment == "B":
            configs.extend(
                LabelingExperimentConfig("B", 0.0, probability)
                for probability in label_probabilities
            )
        else:
            configs.extend(
                LabelingExperimentConfig("C", delay, probability)
                for delay in delay_fractions
                for probability in label_probabilities
            )
    return configs


def calculate_delay_instances(total_instances: int, delay_fraction: float) -> int:
    """Convert a scenario-relative delay to the nearest number of instances."""

    if total_instances <= 0:
        raise ValueError("total_instances must be positive")
    if not 0.0 <= delay_fraction <= 1.0:
        raise ValueError("delay_fraction must be between 0 and 1")
    return int(round(total_instances * delay_fraction))


class ClassificationLabelingExperimentRunner:
    """Run classification with random active learning and delayed labels.

    The first benign block and the immediately following attack region are
    always used as fully supervised initial training. Short benign gaps inside
    an attack region do not split it. Metrics start after that region.
    """

    def __init__(
        self,
        target_names=None,
        n_runs: int = 1,
        random_seed: int = 42,
        attack_gap_tolerance: int = 1000,
    ):
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if attack_gap_tolerance < 0:
            raise ValueError("attack_gap_tolerance must be non-negative")

        self.target_names = target_names if target_names is not None else ["Normal", "Attack"]
        self.n_runs = int(n_runs)
        self.random_seed = int(random_seed)
        self.attack_gap_tolerance = int(attack_gap_tolerance)
        self.normal_class_idx = 0

        for index, name in enumerate(self.target_names):
            if str(name).strip().upper() in {"BENIGN", "NORMAL", "0"}:
                self.normal_class_idx = index
                break

        self.metrics = Metrics()
        self.plots = Plots(self.target_names)

    def prequential_test(
        self,
        stream,
        learner,
        delay_fraction: float,
        label_probability: float,
        window_evaluation: int | None = 100,
        sampling_seed: int | None = None,
        flush_pending_labels: bool = True,
    ) -> dict:
        """Evaluate one learner run using causal label delivery.

        Labels delayed by ``d`` become available before prediction at time
        ``t + d``. A zero-delay label is applied after prediction of its own
        instance, preserving test-then-train order.
        """

        if not 0.0 <= label_probability <= 1.0:
            raise ValueError("label_probability must be between 0 and 1")
        if window_evaluation is not None and window_evaluation <= 0:
            raise ValueError("window_evaluation must be positive or None")

        total_instances = self._stream_size(stream)
        delay_instances = calculate_delay_instances(total_instances, delay_fraction)
        initial_training_end = self._find_initial_training_end(stream)
        rng = np.random.default_rng(self.random_seed if sampling_seed is None else sampling_seed)

        stream.restart()
        pending_labels = deque()
        y_true = []
        y_pred = []
        true_labels_multi = []
        queried_indices = []
        instances = []
        window_sizes = []
        f1_values = []
        precision_values = []
        recall_values = []
        fp_values = []
        fn_values = []

        initial_training_instances = 0
        evaluation_instances = 0
        delivered_during_stream = 0
        last_window_size = 0

        start_time = time.time()

        for stream_index in range(total_instances):
            if not stream.has_more_instances():
                raise ValueError("Stream ended before its reported length")

            instance = stream.next_instance()
            true_label_multiclass = int(instance.y_index)
            true_labels_multi.append(true_label_multiclass)
            is_normal = true_label_multiclass == self.normal_class_idx

            if stream_index < initial_training_end:
                learner.train(instance)
                initial_training_instances += 1
                continue

            while pending_labels and pending_labels[0][0] <= stream_index:
                _, delayed_instance = pending_labels.popleft()
                learner.train(delayed_instance)
                delivered_during_stream += 1

            prediction = learner.predict(instance)
            prediction = self.normal_class_idx if prediction is None else int(prediction)

            y_true.append(0 if is_normal else 1)
            y_pred.append(0 if prediction == self.normal_class_idx else 1)
            evaluation_instances += 1

            selected_for_labeling = (
                label_probability >= 1.0 or rng.random() < label_probability
            )
            if selected_for_labeling:
                queried_indices.append(stream_index)
                if delay_instances == 0:
                    learner.train(instance)
                    delivered_during_stream += 1
                else:
                    pending_labels.append((stream_index + delay_instances, instance))

            if (
                window_evaluation is not None
                and evaluation_instances % window_evaluation == 0
            ):
                self._append_window_metrics(
                    y_true[-window_evaluation:],
                    y_pred[-window_evaluation:],
                    stream_index,
                    instances,
                    f1_values,
                    precision_values,
                    recall_values,
                    fp_values,
                    fn_values,
                )
                last_window_size = window_evaluation
                window_sizes.append(window_evaluation)

        if window_evaluation is not None:
            remainder = evaluation_instances % window_evaluation
            if remainder:
                self._append_window_metrics(
                    y_true[-remainder:],
                    y_pred[-remainder:],
                    total_instances - 1,
                    instances,
                    f1_values,
                    precision_values,
                    recall_values,
                    fp_values,
                    fn_values,
                )
                last_window_size = remainder
                window_sizes.append(remainder)

        pending_at_stream_end = len(pending_labels)
        if flush_pending_labels:
            while pending_labels:
                _, delayed_instance = pending_labels.popleft()
                learner.train(delayed_instance)

        execution_time = time.time() - start_time
        effective_query_percentage = (
            len(queried_indices) / evaluation_instances * 100.0
            if evaluation_instances
            else 0.0
        )

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "true_labels_multi": true_labels_multi,
            "instances": instances,
            "window_sizes": window_sizes,
            "f1": f1_values,
            "precision": precision_values,
            "recall": recall_values,
            "fp": fp_values,
            "fn": fn_values,
            "exec_time": execution_time,
            "total_instances": total_instances,
            "initial_training_instances": initial_training_instances,
            "initial_training_end_index": initial_training_end - 1,
            "evaluation_instances": evaluation_instances,
            "delay_instances": delay_instances,
            "queried_instances": len(queried_indices),
            "queried_indices": queried_indices,
            "delivered_during_stream": delivered_during_stream,
            "flushed_after_stream": pending_at_stream_end if flush_pending_labels else 0,
            "pending_instances": len(pending_labels),
            "effective_query_percentage": effective_query_percentage,
            "last_window_size": last_window_size,
        }

    def run_configuration(
        self,
        stream,
        algorithms: Mapping[str, Callable],
        config: LabelingExperimentConfig,
        window_evaluation: int | None = 100,
    ) -> dict:
        """Run one labeling configuration for all supplied model factories."""

        predictions_history = {}

        for algorithm_name, learner_factory in algorithms.items():
            if not callable(learner_factory):
                raise TypeError(
                    f"Algorithm '{algorithm_name}' must be a factory accepting run_seed"
                )

            runs = []
            for run_index in range(self.n_runs):
                run_seed = self.random_seed + run_index
                learner = learner_factory(run_seed=run_seed)
                runs.append(
                    self.prequential_test(
                        stream=stream,
                        learner=learner,
                        delay_fraction=config.delay_fraction,
                        label_probability=config.label_probability,
                        window_evaluation=window_evaluation,
                        sampling_seed=run_seed,
                    )
                )

            predictions_history[algorithm_name] = self._aggregate_runs(runs)

        return predictions_history

    def run_suite(
        self,
        stream,
        algorithms: Mapping[str, Callable],
        experiments: Sequence[str] = ("A", "B", "C"),
        window_evaluation: int | None = 100,
        experiment_name: str = "General",
        scenario_name: str = "Default_FullFeatures",
        exec_id: str = "N/A",
        output_dir: str = "output/ClassificationLabeling",
        save_csv: bool = True,
        generate_plots: bool = False,
        reuse_equivalent_configurations: bool = True,
    ) -> dict:
        """Run experiments A, B, and C and optionally save consolidated CSVs."""

        configs = build_experiment_configs(experiments=experiments)
        results_by_config = {}
        result_cache = {}
        cumulative_rows = []
        prequential_rows = []

        for config in configs:
            cache_key = (config.delay_fraction, config.label_probability)
            if reuse_equivalent_configurations and cache_key in result_cache:
                predictions_history = result_cache[cache_key]
            else:
                predictions_history = self.run_configuration(
                    stream=stream,
                    algorithms=algorithms,
                    config=config,
                    window_evaluation=window_evaluation,
                )
                result_cache[cache_key] = predictions_history

            results_by_config[config.slug] = predictions_history
            cumulative_rows.extend(
                self._cumulative_rows(
                    predictions_history,
                    config,
                    experiment_name,
                    scenario_name,
                    exec_id,
                    window_evaluation,
                )
            )
            prequential_rows.extend(
                self._prequential_rows(
                    predictions_history,
                    config,
                    experiment_name,
                    scenario_name,
                    exec_id,
                    window_evaluation,
                )
            )

            if generate_plots and window_evaluation is not None:
                self._generate_plots(
                    predictions_history,
                    config,
                    experiment_name,
                    scenario_name,
                    window_evaluation,
                )

        cumulative_df = pd.DataFrame(cumulative_rows)
        prequential_df = pd.DataFrame(prequential_rows)
        paths = {}

        if save_csv:
            os.makedirs(output_dir, exist_ok=True)
            prefix = _clean_filename(f"{experiment_name}_{scenario_name}_{exec_id}")
            cumulative_path = os.path.join(output_dir, f"{prefix}_cumulative.csv")
            prequential_path = os.path.join(output_dir, f"{prefix}_prequential.csv")
            cumulative_df.to_csv(cumulative_path, sep=";", index=False)
            prequential_df.to_csv(prequential_path, sep=";", index=False)
            paths = {
                "cumulative": cumulative_path,
                "prequential": prequential_path,
            }

        return {
            "results": results_by_config,
            "cumulative": cumulative_df,
            "prequential": prequential_df,
            "paths": paths,
        }

    def _stream_size(self, stream) -> int:
        try:
            total_instances = len(stream)
        except (TypeError, AttributeError):
            stream.restart()
            total_instances = 0
            while stream.has_more_instances():
                stream.next_instance()
                total_instances += 1
            stream.restart()

        if total_instances <= 0:
            raise ValueError("Stream must contain at least one instance")
        return int(total_instances)

    def _find_initial_training_end(self, stream) -> int:
        """Return exclusive end of first benign block plus first attack region."""

        stream.restart()
        first_label = None
        first_attack_index = None
        last_attack_index = None

        index = 0
        while stream.has_more_instances():
            instance = stream.next_instance()
            label = int(instance.y_index)
            is_attack = label != self.normal_class_idx

            if first_label is None:
                first_label = label
                if is_attack:
                    stream.restart()
                    raise ValueError("Scenario must start with a benign block")

            if is_attack:
                if first_attack_index is None:
                    first_attack_index = index
                elif index - last_attack_index > self.attack_gap_tolerance:
                    stream.restart()
                    return last_attack_index + 1
                last_attack_index = index

            index += 1

        stream.restart()

        if first_attack_index is None:
            raise ValueError(
                "Scenario does not contain an attack block after the first benign block"
            )
        if last_attack_index >= index - 1:
            raise ValueError(
                "Scenario ends inside the initial attack block; no evaluation region remains"
            )
        return last_attack_index + 1

    def _append_window_metrics(
        self,
        y_true,
        y_pred,
        stream_index,
        instances,
        f1_values,
        precision_values,
        recall_values,
        fp_values,
        fn_values,
    ):
        f1, precision, recall, _, fp, fn = self.metrics.calc_sklearn_metrics(
            y_true, y_pred
        )
        instances.append(stream_index)
        f1_values.append(f1)
        precision_values.append(precision)
        recall_values.append(recall)
        fp_values.append(fp)
        fn_values.append(fn)

    def _aggregate_runs(self, runs: Sequence[dict]) -> dict:
        if not runs:
            raise ValueError("At least one run is required")

        cumulative_metrics = np.array(
            [self.metrics.calc_sklearn_metrics(run["y_true"], run["y_pred"]) for run in runs],
            dtype=float,
        )

        result = {
            "instances": runs[0]["instances"],
            "window_sizes": runs[0]["window_sizes"],
            "true_labels_multi": runs[0]["true_labels_multi"],
            "total_instances": runs[0]["total_instances"],
            "initial_training_instances": runs[0]["initial_training_instances"],
            "evaluation_instances": runs[0]["evaluation_instances"],
            "delay_instances": runs[0]["delay_instances"],
            "run_count": len(runs),
            "cumulative": {
                "f1": _mean_std(cumulative_metrics[:, 0]),
                "prec": _mean_std(cumulative_metrics[:, 1]),
                "rec": _mean_std(cumulative_metrics[:, 2]),
                "mcc": _mean_std(cumulative_metrics[:, 3]),
                "fp": _mean_std(cumulative_metrics[:, 4]),
                "fn": _mean_std(cumulative_metrics[:, 5]),
            },
            "exec_time_mean": float(np.mean([run["exec_time"] for run in runs])),
            "exec_time_std": float(np.std([run["exec_time"] for run in runs])),
            "labeling": {
                "queried_instances": _mean_std(
                    [run["queried_instances"] for run in runs]
                ),
                "delivered_during_stream": _mean_std(
                    [run["delivered_during_stream"] for run in runs]
                ),
                "flushed_after_stream": _mean_std(
                    [run["flushed_after_stream"] for run in runs]
                ),
                "effective_query_percentage": _mean_std(
                    [run["effective_query_percentage"] for run in runs]
                ),
            },
        }

        for key, output_key in (
            ("f1", "f1"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("fp", "fp"),
            ("fn", "fn"),
        ):
            matrix = np.asarray([run[key] for run in runs], dtype=float)
            result[f"{output_key}_mean"] = np.mean(matrix, axis=0)
            result[f"{output_key}_std"] = np.std(matrix, axis=0)

        return result

    def _cumulative_rows(
        self,
        predictions_history,
        config,
        experiment_name,
        scenario_name,
        exec_id,
        window_evaluation,
    ):
        rows = []
        for model_name, data in predictions_history.items():
            cumulative = data["cumulative"]
            labeling = data["labeling"]
            rows.append(
                {
                    "Exec_ID": exec_id,
                    "Dataset": experiment_name,
                    "Experiment": config.experiment,
                    "Model": model_name,
                    "Scenario": scenario_name,
                    "Delay_Percentage": config.delay_percentage,
                    "Delay_Instances": data["delay_instances"],
                    "Label_Budget_Percentage": config.label_budget_percentage,
                    "Initial_Training_Instances": data["initial_training_instances"],
                    "Evaluation_Instances": data["evaluation_instances"],
                    "Window_Evaluation": window_evaluation,
                    "Runs": data["run_count"],
                    "Queried_avg": labeling["queried_instances"][0],
                    "Queried_std": labeling["queried_instances"][1],
                    "Effective_Query_Percentage_avg": labeling["effective_query_percentage"][0],
                    "Effective_Query_Percentage_std": labeling["effective_query_percentage"][1],
                    "Delivered_During_Stream_avg": labeling["delivered_during_stream"][0],
                    "Delivered_During_Stream_std": labeling["delivered_during_stream"][1],
                    "Flushed_After_Stream_avg": labeling["flushed_after_stream"][0],
                    "Flushed_After_Stream_std": labeling["flushed_after_stream"][1],
                    "F1_avg": cumulative["f1"][0],
                    "F1_std": cumulative["f1"][1],
                    "Prec_avg": cumulative["prec"][0],
                    "Prec_std": cumulative["prec"][1],
                    "Rec_avg": cumulative["rec"][0],
                    "Rec_std": cumulative["rec"][1],
                    "MCC_avg": cumulative["mcc"][0],
                    "MCC_std": cumulative["mcc"][1],
                    "FP_avg": cumulative["fp"][0],
                    "FP_std": cumulative["fp"][1],
                    "FN_avg": cumulative["fn"][0],
                    "FN_std": cumulative["fn"][1],
                    "Time_avg": data["exec_time_mean"],
                    "Time_std": data["exec_time_std"],
                }
            )
        return rows

    def _prequential_rows(
        self,
        predictions_history,
        config,
        experiment_name,
        scenario_name,
        exec_id,
        window_evaluation,
    ):
        rows = []
        for model_name, data in predictions_history.items():
            for index, instance_index in enumerate(data["instances"]):
                rows.append(
                    {
                        "Exec_ID": exec_id,
                        "Dataset": experiment_name,
                        "Experiment": config.experiment,
                        "Model": model_name,
                        "Scenario": scenario_name,
                        "Delay_Percentage": config.delay_percentage,
                        "Delay_Instances": data["delay_instances"],
                        "Label_Budget_Percentage": config.label_budget_percentage,
                        "Initial_Training_Instances": data["initial_training_instances"],
                        "Window_Evaluation": window_evaluation,
                        "Window_Index": index + 1,
                        "Window_Instances": data["window_sizes"][index],
                        "Instance": instance_index,
                        "F1_avg": data["f1_mean"][index],
                        "F1_std": data["f1_std"][index],
                        "Prec_avg": data["precision_mean"][index],
                        "Prec_std": data["precision_std"][index],
                        "Rec_avg": data["recall_mean"][index],
                        "Rec_std": data["recall_std"][index],
                        "FP_avg": data["fp_mean"][index],
                        "FP_std": data["fp_std"][index],
                        "FN_avg": data["fn_mean"][index],
                        "FN_std": data["fn_std"][index],
                    }
                )
        return rows

    def _generate_plots(
        self,
        predictions_history,
        config,
        experiment_name,
        scenario_name,
        window_evaluation,
    ):
        for model_name, data in predictions_history.items():
            single_model_result = {model_name: data}
            attack_regions = self.metrics.extract_attack_regions(
                data["true_labels_multi"], normal_class_idx=self.normal_class_idx
            )
            plot_scenario = f"{scenario_name}_{config.slug}"
            plot_strategy = os.path.join("classification_labeling", config.experiment)
            self.plots.plot_metrics(
                results=single_model_result,
                attack_regions=attack_regions,
                title=experiment_name,
                window_size=window_evaluation,
                scenario_name=plot_scenario,
                discretization_strategy=plot_strategy,
            )
            self.plots.plot_fp_fn(
                results=single_model_result,
                attack_regions=attack_regions,
                title=experiment_name,
                window_size=window_evaluation,
                scenario_name=plot_scenario,
                discretization_strategy=plot_strategy,
            )


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)), float(np.std(array))


def _format_percentage(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _clean_filename(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", str(value)).strip("_") or "results"
