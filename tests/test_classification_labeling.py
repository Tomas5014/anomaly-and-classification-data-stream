import unittest

from src.Classification.Labeling import (
    ClassificationLabelingExperimentRunner,
    build_experiment_configs,
    calculate_delay_instances,
)


class FakeInstance:
    def __init__(self, index, y_index):
        self.index = index
        self.y_index = y_index


class FakeStream:
    def __init__(self, labels):
        self.instances = [FakeInstance(index, label) for index, label in enumerate(labels)]
        self.position = 0

    def __len__(self):
        return len(self.instances)

    def restart(self):
        self.position = 0

    def has_more_instances(self):
        return self.position < len(self.instances)

    def next_instance(self):
        instance = self.instances[self.position]
        self.position += 1
        return instance


class FakeLearner:
    def __init__(self):
        self.trained_indices = []
        self.predicted_indices = []

    def predict(self, instance):
        self.predicted_indices.append(instance.index)
        return 0

    def train(self, instance):
        self.trained_indices.append(instance.index)


class ClassificationLabelingTest(unittest.TestCase):
    def setUp(self):
        self.runner = ClassificationLabelingExperimentRunner(
            target_names=["BENIGN", "ATTACK"],
            random_seed=7,
            attack_gap_tolerance=1,
        )
        self.labels = [0, 0, 1, 1, 0, 0, 1, 1]

    def test_experiment_grid_sizes(self):
        self.assertEqual(len(build_experiment_configs(("A",))), 4)
        self.assertEqual(len(build_experiment_configs(("B",))), 6)
        self.assertEqual(len(build_experiment_configs(("C",))), 24)
        self.assertEqual(len(build_experiment_configs()), 34)

    def test_delay_uses_total_scenario_size(self):
        self.assertEqual(calculate_delay_instances(15_000, 0.01), 150)
        self.assertEqual(calculate_delay_instances(15_000, 0.05), 750)
        self.assertEqual(calculate_delay_instances(15_000, 0.10), 1_500)

    def test_initial_blocks_are_fully_supervised_and_excluded_from_evaluation(self):
        learner = FakeLearner()
        result = self.runner.prequential_test(
            stream=FakeStream(self.labels),
            learner=learner,
            delay_fraction=0.0,
            label_probability=0.0,
            window_evaluation=4,
        )

        self.assertEqual(result["initial_training_instances"], 4)
        self.assertEqual(result["evaluation_instances"], 4)
        self.assertEqual(learner.trained_indices, [0, 1, 2, 3])
        self.assertEqual(learner.predicted_indices, [4, 5, 6, 7])
        self.assertEqual(result["queried_instances"], 0)

    def test_zero_delay_and_full_budget_match_test_then_train(self):
        learner = FakeLearner()
        result = self.runner.prequential_test(
            stream=FakeStream(self.labels),
            learner=learner,
            delay_fraction=0.0,
            label_probability=1.0,
            window_evaluation=3,
        )

        self.assertEqual(learner.trained_indices, list(range(8)))
        self.assertEqual(result["queried_instances"], 4)
        self.assertEqual(result["delivered_during_stream"], 4)
        self.assertEqual(result["flushed_after_stream"], 0)
        self.assertEqual(result["window_sizes"], [3, 1])
        self.assertEqual(len(result["instances"]), 2)
        self.assertEqual(result["queried_window"], [3, 1])
        self.assertEqual(result["delivered_window"], [3, 1])
        self.assertEqual(result["pending_window"], [0, 0])

    def test_delayed_label_flow_is_recorded_per_window(self):
        result = self.runner.prequential_test(
            stream=FakeStream(self.labels),
            learner=FakeLearner(),
            delay_fraction=0.25,
            label_probability=1.0,
            window_evaluation=2,
        )

        self.assertEqual(result["queried_window"], [2, 2])
        self.assertEqual(result["delivered_window"], [0, 2])
        self.assertEqual(result["pending_window"], [2, 2])
        self.assertEqual(result["flushed_after_stream"], 2)

    def test_delayed_labels_are_causal_and_flushed(self):
        learner = FakeLearner()
        result = self.runner.prequential_test(
            stream=FakeStream(self.labels),
            learner=learner,
            delay_fraction=0.25,
            label_probability=1.0,
            window_evaluation=None,
        )

        self.assertEqual(result["delay_instances"], 2)
        self.assertEqual(result["delivered_during_stream"], 2)
        self.assertEqual(result["flushed_after_stream"], 2)
        self.assertEqual(learner.trained_indices, list(range(8)))

    def test_prequential_table_contains_label_flow_by_window(self):
        suite = self.runner.run_suite(
            stream=FakeStream(self.labels),
            algorithms={"FakeModel": lambda run_seed=None: FakeLearner()},
            experiments=("A",),
            window_evaluation=2,
            save_csv=False,
        )

        prequential = suite["prequential"]
        expected_columns = {
            "Queried_Window_avg",
            "Queried_Window_std",
            "Delivered_Window_avg",
            "Delivered_Window_std",
            "Pending_At_Window_End_avg",
            "Pending_At_Window_End_std",
        }
        self.assertTrue(expected_columns.issubset(prequential.columns))
        first_window = prequential.iloc[0]
        self.assertEqual(first_window["Queried_Window_avg"], 2)
        self.assertEqual(first_window["Delivered_Window_avg"], 2)
        self.assertEqual(first_window["Pending_At_Window_End_avg"], 0)

    def test_random_sampling_is_reproducible(self):
        labels = [0, 0, 1, 1] + [0, 1] * 50
        first = self.runner.prequential_test(
            FakeStream(labels),
            FakeLearner(),
            delay_fraction=0.0,
            label_probability=0.30,
            sampling_seed=123,
            window_evaluation=10,
        )
        second = self.runner.prequential_test(
            FakeStream(labels),
            FakeLearner(),
            delay_fraction=0.0,
            label_probability=0.30,
            sampling_seed=123,
            window_evaluation=10,
        )

        self.assertEqual(first["queried_indices"], second["queried_indices"])

    def test_fragmented_attack_region_remains_in_initial_training(self):
        runner = ClassificationLabelingExperimentRunner(
            target_names=["BENIGN", "ATTACK"],
            attack_gap_tolerance=2,
        )
        learner = FakeLearner()
        result = runner.prequential_test(
            FakeStream([0, 0, 1, 0, 1, 0, 0, 0, 1]),
            learner,
            delay_fraction=0.0,
            label_probability=0.0,
            window_evaluation=None,
        )

        self.assertEqual(result["initial_training_instances"], 5)
        self.assertEqual(learner.trained_indices, [0, 1, 2, 3, 4])
        self.assertEqual(learner.predicted_indices[0], 5)

    def test_invalid_initial_scenario_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "start with a benign block"):
            self.runner.prequential_test(
                FakeStream([1, 1, 0]),
                FakeLearner(),
                delay_fraction=0.0,
                label_probability=1.0,
            )


if __name__ == "__main__":
    unittest.main()
