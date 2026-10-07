from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.Classification.Labeling import (
    ClassificationLabelingExperimentRunner,
    build_experiment_configs,
)
from src.Classification.Models import get_classification_models
from src.Data.Processor import DataStreamProcessor


CATEGORIES = ("Consistência", "Generalização", "Adaptação", "Recorrência")
SCENARIO_SIZES = ("25", "200", "1000")
MODEL_NAMES = ("LB", "HAT", "ARF", "HT")
MODEL_OUTPUT_NAMES = {
    "LB": "LeveragingBagging",
    "HAT": "HoeffdingAdaptiveTree",
    "ARF": "AdaptiveRandomForest",
    "HT": "HoeffdingTree",
}

SELECTED_FEATURES = [
    "Max Packet Length",
    "Average Packet Size",
    "Fwd Packet Length Min",
    "Min Packet Length",
    "Fwd Packet Length Max",
    "Packet Length Mean",
    "Fwd Packet Length Mean",
    "Avg Fwd Segment Size",
    "min_seg_size_forward",
    "ACK Flag Count",
    "Flow Duration",
    "Fwd IAT Total",
    "Flow IAT Max",
    "Fwd IAT Max",
    "Flow IAT Std",
    "Fwd IAT Std",
    "Fwd IAT Mean",
    "Flow IAT Mean",
    "Total Length of Fwd Packets",
    "Subflow Fwd Bytes",
    "act_data_pkt_fwd",
    "Subflow Fwd Packets",
    "Total Fwd Packets",
    "Down/Up Ratio",
    "Init_Win_bytes_backward",
    "Total Length of Bwd Packets",
    "Subflow Bwd Bytes",
    "Flow IAT Min",
    "Bwd Packet Length Max",
    "URG Flag Count",
    "Bwd IAT Total",
    "Bwd Packets/s",
    "Init_Win_bytes_forward",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run classification experiments with active learning and delayed labels."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/15k"),
        help="Root containing one folder per scenario category.",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        nargs="*",
        help="Explicit CSV paths. Overrides categories and sizes.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORIES,
        default=list(CATEGORIES),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=SCENARIO_SIZES,
        default=list(SCENARIO_SIZES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("A", "B", "C"),
        default=["A", "B", "C"],
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=("full", "selected"),
        default=["full"],
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument(
        "--training-label-mode",
        choices=("binary", "multiclass"),
        default="binary",
        help="Training target. Original attack families are preserved for plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--attack-gap-tolerance", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ClassificationLabeling"),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Gera, durante o experimento, gráficos detalhados de FP/FN, "
            "regiões de ataque e dinâmica dos rótulos por janela."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_datasets(args) -> list[Path]:
    if args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = [
            args.data_root / category / f"{category}_{size}.csv"
            for category in args.categories
            for size in args.sizes
        ]

    missing = [str(path) for path in datasets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing datasets: {missing}")
    return datasets


def build_stream(dataset_path: Path, selected_features=None, binary_label=True):
    dataframe = pd.read_csv(dataset_path)
    processor = DataStreamProcessor(
        logging=False,
        selected_features=selected_features,
    )
    return processor.create_stream(
        df=dataframe,
        target_label_col="Label",
        binary_label=binary_label,
        preserve_label_metadata=True,
        normalize_method="MinMaxScaler",
        threshold_var=None,
        threshold_corr=None,
        top_n_features=None,
        return_stream=True,
        extra_ignore_cols=[
            "Source IP",
            "Source Port",
            "Destination IP",
            "Destination Port",
            "Protocol",
            "Inbound",
        ],
        imputation_method="mediana",
    )


def build_model_factories(schema, model_names):
    factories = {}
    for model_name in model_names:
        output_name = MODEL_OUTPUT_NAMES[model_name]

        def make_model(run_seed=None, selected=model_name, output=output_name):
            return get_classification_models(
                schema=schema,
                selected_models=[selected],
                run_seed=run_seed,
            )[output]

        factories[output_name] = make_model
    return factories


def feature_configurations(names):
    configs = []
    if "full" in names:
        configs.append(("Default_FullFeatures", None))
    if "selected" in names:
        configs.append(("Default_33Features", SELECTED_FEATURES))
    return configs


def print_plan(args, datasets, feature_configs):
    configs = build_experiment_configs(tuple(args.experiments))
    unique_configs = {
        (config.delay_fraction, config.label_probability) for config in configs
    }
    model_runs = (
        len(datasets)
        * len(feature_configs)
        * len(unique_configs)
        * len(args.models)
        * args.n_runs
    )
    print(f"Datasets: {len(datasets)}")
    print(f"Feature sets: {len(feature_configs)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Training labels: {args.training_label_mode}; evaluation labels: binary")
    print(f"Logical configurations: {len(configs)}")
    print(f"Unique configurations: {len(unique_configs)}")
    print(f"Model runs: {model_runs}")
    for dataset in datasets:
        print(f" - {dataset}")


def main():
    args = parse_args()
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive")
    if args.window <= 0:
        raise ValueError("--window must be positive")

    datasets = resolve_datasets(args)
    feature_configs = feature_configurations(args.feature_sets)
    print_plan(args, datasets, feature_configs)
    if args.dry_run:
        return

    exec_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_jobs = len(datasets) * len(feature_configs)
    job_index = 0

    for dataset_path in datasets:
        experiment_name = dataset_path.stem

        for scenario_name, selected_features in feature_configs:
            job_index += 1
            print(
                f"[{job_index}/{total_jobs}] {experiment_name} | {scenario_name}"
            )
            stream, targets, features = build_stream(
                dataset_path,
                selected_features=selected_features,
                binary_label=args.training_label_mode == "binary",
            )
            algorithms = build_model_factories(stream.get_schema(), args.models)
            runner = ClassificationLabelingExperimentRunner(
                target_names=targets,
                n_runs=args.n_runs,
                random_seed=args.seed,
                attack_gap_tolerance=args.attack_gap_tolerance,
            )
            suite = runner.run_suite(
                stream=stream,
                algorithms=algorithms,
                experiments=tuple(args.experiments),
                window_evaluation=args.window,
                experiment_name=experiment_name,
                scenario_name=scenario_name,
                exec_id=exec_id,
                output_dir=str(args.output_dir),
                save_csv=True,
                generate_plots=args.plots,
            )
            print(f"  Features: {len(features)}")
            print(f"  Cumulative: {suite['paths']['cumulative']}")
            print(f"  Prequential: {suite['paths']['prequential']}")
            if suite["paths"]["plots"]:
                print(f"  Detailed plots: {len(suite['paths']['plots'])}")
                print(f"  First plot: {suite['paths']['plots'][0]}")


if __name__ == "__main__":
    main()
