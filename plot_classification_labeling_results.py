from __future__ import annotations

import argparse
import hashlib
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_ALIASES = {
    "LB": "LeveragingBagging",
    "HAT": "HoeffdingAdaptiveTree",
    "ARF": "AdaptiveRandomForest",
    "HT": "HoeffdingTree",
}

MODEL_SHORT_NAMES = {value: key for key, value in MODEL_ALIASES.items()}
MODEL_ORDER = list(MODEL_ALIASES.values())

METRICS = (
    ("F1_avg", "F1_std", "F1-score (%)"),
    ("Prec_avg", "Prec_std", "Precisão (%)"),
    ("Rec_avg", "Rec_std", "Recall (%)"),
)

ERROR_METRICS = (
    ("FP_avg", "FP_std", "Falsos positivos"),
    ("FN_avg", "FN_std", "Falsos negativos"),
)

CUMULATIVE_REQUIRED_COLUMNS = {
    "Exec_ID",
    "Dataset",
    "Experiment",
    "Model",
    "Scenario",
    "Delay_Percentage",
    "Label_Budget_Percentage",
    "Effective_Query_Percentage_avg",
    "F1_avg",
    "F1_std",
    "Prec_avg",
    "Prec_std",
    "Rec_avg",
    "Rec_std",
    "FP_avg",
    "FP_std",
    "FN_avg",
    "FN_std",
}

PREQUENTIAL_REQUIRED_COLUMNS = {
    "Exec_ID",
    "Dataset",
    "Experiment",
    "Model",
    "Scenario",
    "Delay_Percentage",
    "Label_Budget_Percentage",
    "Window_Index",
    "Window_Instances",
    "Instance",
    "F1_avg",
    "F1_std",
    "Prec_avg",
    "Prec_std",
    "Rec_avg",
    "Rec_std",
}

CUMULATIVE_KEY = [
    "Exec_ID",
    "Dataset",
    "Experiment",
    "Model",
    "Scenario",
    "Delay_Percentage",
    "Label_Budget_Percentage",
]

PREQUENTIAL_KEY = CUMULATIVE_KEY + ["Window_Index"]

DEFAULT_PREQUENTIAL_CONFIGS = ((0.0, 100.0), (1.0, 30.0), (5.0, 10.0), (10.0, 1.0))


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Gera gráficos dos experimentos de active learning e atraso de "
            "rótulos usando os CSVs existentes, sem retreinar os modelos."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/ClassificationLabeling"),
        help="Diretório que contém os CSVs cumulativos e prequenciais.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Diretório dos gráficos. Padrão: <input-dir>/plots.",
    )
    parser.add_argument(
        "--exec-id",
        default="latest",
        help=(
            "Execução usada: 'latest' escolhe a mais recente por dataset e "
            "cenário, 'all' mantém todas, ou informe um Exec_ID específico."
        ),
    )
    parser.add_argument("--datasets", nargs="+", help="Filtra nomes da coluna Dataset.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Filtra modelos. Aceita LB, HAT, ARF, HT ou nomes completos.",
    )
    parser.add_argument("--scenarios", nargs="+", help="Filtra a coluna Scenario.")
    parser.add_argument(
        "--scope",
        choices=("all", "individual", "aggregate"),
        default="all",
        help="Gera figuras individuais, agregadas ou ambas.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png"],
        help="Formatos de saída.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Resolução de saída.")
    parser.add_argument(
        "--include-prequential",
        action="store_true",
        help="Gera curvas prequenciais para configurações representativas.",
    )
    parser.add_argument(
        "--prequential-configs",
        nargs="+",
        default=["0:100", "1:30", "5:10", "10:1"],
        metavar="ATRASO:ORCAMENTO",
        help="Pares do Experimento C mostrados nas curvas prequenciais.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Número de janelas da média móvel prequencial.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida entradas e mostra o plano sem criar imagens.",
    )
    return parser.parse_args(argv)


def parse_prequential_configs(values: Sequence[str]) -> tuple[tuple[float, float], ...]:
    configs = []
    for value in values:
        parts = str(value).split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Configuração prequencial inválida '{value}'. Use ATRASO:ORCAMENTO."
            )
        try:
            delay, budget = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError(
                f"Configuração prequencial inválida '{value}'. Use números."
            ) from exc
        if not 0.0 <= delay <= 100.0 or not 0.0 <= budget <= 100.0:
            raise ValueError(
                f"Configuração prequencial fora da faixa 0–100: '{value}'."
            )
        configs.append((delay, budget))
    return tuple(dict.fromkeys(configs))


def load_results(input_dir: Path, kind: str, exec_selector: str = "latest") -> pd.DataFrame:
    if kind not in {"cumulative", "prequential"}:
        raise ValueError("kind must be 'cumulative' or 'prequential'")

    paths = sorted(input_dir.glob(f"*_{kind}.csv"))
    if not paths:
        raise FileNotFoundError(
            f"Nenhum arquivo '*_{kind}.csv' encontrado em {input_dir}"
        )

    required = (
        CUMULATIVE_REQUIRED_COLUMNS
        if kind == "cumulative"
        else PREQUENTIAL_REQUIRED_COLUMNS
    )
    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep=";")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} não possui as colunas obrigatórias: {missing}")
        frame = frame.assign(
            Exec_ID=frame["Exec_ID"].astype(str),
            _Source_File=str(path),
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = _select_executions(data, exec_selector)
    key = CUMULATIVE_KEY if kind == "cumulative" else PREQUENTIAL_KEY
    data = data.sort_values(key + ["_Source_File"]).drop_duplicates(key, keep="last")
    return data.reset_index(drop=True)


def _select_executions(data: pd.DataFrame, selector: str) -> pd.DataFrame:
    selector = str(selector)
    if selector.lower() == "all":
        return data.copy()
    if selector.lower() == "latest":
        latest = data.groupby(["Dataset", "Scenario"])["Exec_ID"].transform("max")
        return data[data["Exec_ID"] == latest].copy()

    selected = data[data["Exec_ID"] == selector].copy()
    if selected.empty:
        available = ", ".join(sorted(data["Exec_ID"].unique()))
        raise ValueError(
            f"Exec_ID '{selector}' não encontrado. Disponíveis: {available}"
        )
    return selected


def filter_results(
    data: pd.DataFrame,
    datasets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
) -> pd.DataFrame:
    result = data.copy()
    filters = {
        "Dataset": list(datasets) if datasets else None,
        "Model": normalize_models(models, result["Model"].unique()) if models else None,
        "Scenario": list(scenarios) if scenarios else None,
    }
    for column, values in filters.items():
        if not values:
            continue
        missing = sorted(set(values) - set(result[column].unique()))
        if missing:
            available = ", ".join(sorted(map(str, result[column].unique())))
            raise ValueError(
                f"Valores de {column} não encontrados: {missing}. Disponíveis: {available}"
            )
        result = result[result[column].isin(values)]

    if result.empty:
        raise ValueError("Os filtros removeram todos os resultados")
    return result.reset_index(drop=True)


def normalize_models(
    values: Sequence[str] | None, available_models: Iterable[str]
) -> list[str]:
    if not values:
        return []
    available = set(map(str, available_models))
    normalized = []
    for value in values:
        text = str(value)
        model = MODEL_ALIASES.get(text.upper(), text)
        if model not in available:
            raise ValueError(
                f"Modelo '{value}' não encontrado. Disponíveis: {sorted(available)}"
            )
        normalized.append(model)
    return list(dict.fromkeys(normalized))


def aggregate_results(data: pd.DataFrame) -> pd.DataFrame:
    """Create an unweighted macro-average without over-weighting repeated runs."""

    config_columns = [
        "Dataset",
        "Scenario",
        "Experiment",
        "Model",
        "Delay_Percentage",
        "Label_Budget_Percentage",
    ]
    value_columns = [
        "F1_avg",
        "Prec_avg",
        "Rec_avg",
        "FP_avg",
        "FN_avg",
        "Effective_Query_Percentage_avg",
    ]

    # If --exec-id all is used, first collapse repeated executions so that each
    # dataset has the same weight in the macro-average.
    per_dataset = data.groupby(config_columns, as_index=False)[value_columns].mean()
    aggregate_columns = [
        "Scenario",
        "Experiment",
        "Model",
        "Delay_Percentage",
        "Label_Budget_Percentage",
    ]

    rows = []
    for keys, group in per_dataset.groupby(aggregate_columns, sort=True):
        row = dict(zip(aggregate_columns, keys))
        row["Dataset"] = f"Média de {group['Dataset'].nunique()} datasets"
        row["Dataset_Count"] = int(group["Dataset"].nunique())
        for column in value_columns:
            row[column] = float(group[column].mean())
            if column in {"F1_avg", "Prec_avg", "Rec_avg", "FP_avg", "FN_avg"}:
                std_column = column.replace("_avg", "_std")
                row[std_column] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def slugify(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ascii_value).strip("_") or "result"


def selection_slug(data: pd.DataFrame, include_datasets: bool = True) -> str:
    datasets = sorted(map(str, data["Dataset"].unique()))
    models = _ordered_models(data)
    executions = sorted(map(str, data["Exec_ID"].unique())) if "Exec_ID" in data else []
    payload = "|".join(datasets + models + executions)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    model_text = "-".join(_model_label(model) for model in models)
    if include_datasets:
        return f"datasets_{len(datasets)}_models_{model_text}_{digest}"
    return f"models_{model_text}_{digest}"


def _ordered_models(data: pd.DataFrame) -> list[str]:
    present = list(map(str, data["Model"].unique()))
    known = [model for model in MODEL_ORDER if model in present]
    return known + sorted(set(present) - set(known))


def _model_label(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, model)


def _model_colors(models: Sequence[str]) -> dict[str, tuple]:
    colors = sns.color_palette("colorblind", n_colors=max(len(models), 1))
    return {model: colors[index] for index, model in enumerate(models)}


def _format_percent(value: float) -> str:
    return f"{value:g}%"


def _finish_figure(fig, legend_axes, title: str):
    handles, labels = legend_axes.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=min(len(labels), 6),
            frameon=False,
        )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))


def save_figure(
    fig,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for output_format in dict.fromkeys(formats):
        path = output_base.with_suffix(f".{output_format}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_performance(
    data: pd.DataFrame,
    experiment: str,
    title_context: str,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    subset = data[data["Experiment"] == experiment]
    if subset.empty:
        return []

    x_column = "Delay_Percentage" if experiment == "A" else "Label_Budget_Percentage"
    x_label = "Atraso do rótulo (%)" if experiment == "A" else "Orçamento de rótulos (%)"
    x_values = sorted(subset[x_column].unique())
    positions = np.arange(len(x_values))
    models = _ordered_models(subset)
    colors = _model_colors(models)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    for axis, (mean_column, std_column, y_label) in zip(axes, METRICS):
        for model in models:
            model_data = (
                subset[subset["Model"] == model]
                .set_index(x_column)
                .reindex(x_values)
            )
            means = model_data[mean_column].to_numpy(dtype=float)
            stds = model_data[std_column].fillna(0).to_numpy(dtype=float)
            axis.plot(
                positions,
                means,
                marker="o",
                linewidth=2.1,
                color=colors[model],
                label=_model_label(model),
            )
            axis.fill_between(
                positions,
                np.clip(means - stds, 0, 100),
                np.clip(means + stds, 0, 100),
                color=colors[model],
                alpha=0.13,
            )
        axis.set_xticks(positions, [_format_percent(value) for value in x_values])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_ylim(0, 105)
        axis.grid(True, alpha=0.25, linestyle=":")

    _finish_figure(
        fig,
        axes[0],
        f"Experimento {experiment} — desempenho — {title_context}",
    )
    return save_figure(fig, output_base, formats, dpi)


def plot_errors(
    data: pd.DataFrame,
    experiment: str,
    title_context: str,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    subset = data[data["Experiment"] == experiment]
    if subset.empty:
        return []

    x_column = "Delay_Percentage" if experiment == "A" else "Label_Budget_Percentage"
    x_label = "Atraso do rótulo (%)" if experiment == "A" else "Orçamento de rótulos (%)"
    x_values = sorted(subset[x_column].unique())
    positions = np.arange(len(x_values))
    models = _ordered_models(subset)
    colors = _model_colors(models)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for axis, (mean_column, std_column, y_label) in zip(axes, ERROR_METRICS):
        for model in models:
            model_data = (
                subset[subset["Model"] == model]
                .set_index(x_column)
                .reindex(x_values)
            )
            means = model_data[mean_column].to_numpy(dtype=float)
            stds = model_data[std_column].fillna(0).to_numpy(dtype=float)
            axis.errorbar(
                positions,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=colors[model],
                label=_model_label(model),
            )
        axis.set_xticks(positions, [_format_percent(value) for value in x_values])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_ylim(bottom=0)
        axis.grid(True, alpha=0.25, linestyle=":")

    _finish_figure(
        fig,
        axes[0],
        f"Experimento {experiment} — FP e FN — {title_context}",
    )
    return save_figure(fig, output_base, formats, dpi)


def plot_heatmap_c(
    data: pd.DataFrame,
    title_context: str,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    subset = data[data["Experiment"] == "C"]
    if subset.empty:
        return []

    models = _ordered_models(subset)
    column_count = min(2, len(models))
    row_count = math.ceil(len(models) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7 * column_count, 5 * row_count),
        squeeze=False,
    )
    delays = sorted(subset["Delay_Percentage"].unique())
    budgets = sorted(subset["Label_Budget_Percentage"].unique())

    for index, model in enumerate(models):
        axis = axes.flat[index]
        matrix = subset[subset["Model"] == model].pivot_table(
            index="Delay_Percentage",
            columns="Label_Budget_Percentage",
            values="F1_avg",
            aggfunc="mean",
        )
        matrix = matrix.reindex(index=delays, columns=budgets)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            linewidths=0.4,
            cbar=index == len(models) - 1,
            cbar_kws={"label": "F1-score (%)"},
            ax=axis,
        )
        axis.set_title(_model_label(model), fontweight="bold")
        axis.set_xlabel("Orçamento de rótulos (%)")
        axis.set_ylabel("Atraso do rótulo (%)")
        axis.set_xticklabels([_format_percent(value) for value in budgets], rotation=0)
        axis.set_yticklabels([_format_percent(value) for value in delays], rotation=0)

    for index in range(len(models), len(axes.flat)):
        axes.flat[index].set_visible(False)

    fig.suptitle(
        f"Experimento C — interação entre atraso e orçamento — {title_context}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, output_base, formats, dpi)


def plot_cost_performance(
    data: pd.DataFrame,
    title_context: str,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    subset = data[data["Experiment"] == "C"]
    if subset.empty:
        return []

    models = _ordered_models(subset)
    column_count = min(2, len(models))
    row_count = math.ceil(len(models) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7 * column_count, 5 * row_count),
        squeeze=False,
    )
    delays = sorted(subset["Delay_Percentage"].unique())
    delay_colors = dict(
        zip(delays, sns.color_palette("viridis", n_colors=max(len(delays), 1)))
    )

    for index, model in enumerate(models):
        axis = axes.flat[index]
        model_data = subset[subset["Model"] == model]
        for delay in delays:
            group = model_data[model_data["Delay_Percentage"] == delay].sort_values(
                "Effective_Query_Percentage_avg"
            )
            if group.empty:
                continue
            axis.errorbar(
                group["Effective_Query_Percentage_avg"],
                group["F1_avg"],
                yerr=group["F1_std"].fillna(0),
                marker="o",
                capsize=2,
                linewidth=1.8,
                color=delay_colors[delay],
                label=f"Atraso {_format_percent(delay)}",
            )
        axis.set_title(_model_label(model), fontweight="bold")
        axis.set_xlabel("Rótulos efetivamente consultados (%)")
        axis.set_ylabel("F1-score (%)")
        axis.set_xlim(0, 105)
        axis.set_ylim(0, 105)
        axis.grid(True, alpha=0.25, linestyle=":")

    for index in range(len(models), len(axes.flat)):
        axes.flat[index].set_visible(False)

    _finish_figure(
        fig,
        axes.flat[0],
        f"Experimento C — custo de rotulagem × desempenho — {title_context}",
    )
    return save_figure(fig, output_base, formats, dpi)


def plot_prequential(
    data: pd.DataFrame,
    model: str,
    configs: Sequence[tuple[float, float]],
    smooth: int,
    title_context: str,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    subset = data[(data["Experiment"] == "C") & (data["Model"] == model)]
    if subset.empty:
        return []

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    colors = sns.color_palette("tab10", n_colors=max(len(configs), 1))
    plotted = False

    for color, (delay, budget) in zip(colors, configs):
        group = subset[
            np.isclose(subset["Delay_Percentage"], delay)
            & np.isclose(subset["Label_Budget_Percentage"], budget)
        ].sort_values("Instance")
        if group.empty:
            continue
        plotted = True
        label = f"Atraso {_format_percent(delay)} | rótulos {_format_percent(budget)}"
        for axis, (mean_column, std_column, y_label) in zip(axes, METRICS):
            means = group[mean_column].rolling(smooth, min_periods=1).mean()
            stds = group[std_column].rolling(smooth, min_periods=1).mean().fillna(0)
            x_values = group["Instance"].to_numpy(dtype=float)
            mean_values = means.to_numpy(dtype=float)
            std_values = stds.to_numpy(dtype=float)
            axis.plot(x_values, mean_values, color=color, linewidth=1.8, label=label)
            axis.fill_between(
                x_values,
                np.clip(mean_values - std_values, 0, 100),
                np.clip(mean_values + std_values, 0, 100),
                color=color,
                alpha=0.10,
            )
            axis.set_ylabel(y_label)
            axis.set_ylim(0, 105)
            axis.grid(True, alpha=0.25, linestyle=":")

    if not plotted:
        plt.close(fig)
        return []

    axes[-1].set_xlabel("Índice da instância no cenário")
    _finish_figure(
        fig,
        axes[0],
        f"Experimento C — evolução prequencial — {_model_label(model)} — {title_context}",
    )
    return save_figure(fig, output_base, formats, dpi)


def generate_individual_plots(
    data: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[dict]:
    records = []
    group_columns = ["Exec_ID", "Dataset", "Scenario"]
    for (exec_id, dataset, scenario), group in data.groupby(group_columns, sort=True):
        directory = (
            output_dir
            / "individual"
            / slugify(scenario)
            / slugify(dataset)
            / slugify(exec_id)
            / selection_slug(group, include_datasets=False)
        )
        title = f"{dataset} | {scenario}"
        specifications = [
            ("performance_A", plot_performance, (group, "A", title, directory / "performance_experiment_A", formats, dpi)),
            ("performance_B", plot_performance, (group, "B", title, directory / "performance_experiment_B", formats, dpi)),
            ("heatmap_C", plot_heatmap_c, (group, title, directory / "heatmap_experiment_C_f1", formats, dpi)),
            ("cost_C", plot_cost_performance, (group, title, directory / "cost_performance_experiment_C", formats, dpi)),
            ("errors_A", plot_errors, (group, "A", title, directory / "errors_experiment_A", formats, dpi)),
            ("errors_B", plot_errors, (group, "B", title, directory / "errors_experiment_B", formats, dpi)),
        ]
        for plot_type, function, arguments in specifications:
            for path in function(*arguments):
                records.append(
                    _manifest_record(path, plot_type, dataset, scenario, str(exec_id))
                )
    return records


def generate_aggregate_plots(
    data: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[dict]:
    records = []
    aggregate = aggregate_results(data)
    for scenario, group in aggregate.groupby("Scenario", sort=True):
        dataset_count = int(group["Dataset_Count"].max())
        title = f"média macro de {dataset_count} datasets | {scenario}"
        source_group = data[data["Scenario"] == scenario]
        directory = (
            output_dir
            / "aggregated"
            / slugify(scenario)
            / selection_slug(source_group)
        )
        specifications = [
            ("aggregate_performance_A", plot_performance, (group, "A", title, directory / "performance_experiment_A", formats, dpi)),
            ("aggregate_performance_B", plot_performance, (group, "B", title, directory / "performance_experiment_B", formats, dpi)),
            ("aggregate_heatmap_C", plot_heatmap_c, (group, title, directory / "heatmap_experiment_C_f1", formats, dpi)),
            ("aggregate_cost_C", plot_cost_performance, (group, title, directory / "cost_performance_experiment_C", formats, dpi)),
            ("aggregate_errors_A", plot_errors, (group, "A", title, directory / "errors_experiment_A", formats, dpi)),
            ("aggregate_errors_B", plot_errors, (group, "B", title, directory / "errors_experiment_B", formats, dpi)),
        ]
        for plot_type, function, arguments in specifications:
            for path in function(*arguments):
                records.append(
                    _manifest_record(path, plot_type, "AGGREGATED", scenario, "macro")
                )
    return records


def generate_prequential_plots(
    data: pd.DataFrame,
    output_dir: Path,
    configs: Sequence[tuple[float, float]],
    smooth: int,
    formats: Sequence[str],
    dpi: int,
) -> list[dict]:
    records = []
    group_columns = ["Exec_ID", "Dataset", "Scenario", "Model"]
    config_slug = "-".join(
        f"d{value:g}_b{budget:g}" for value, budget in configs
    ).replace(".", "p")
    for (exec_id, dataset, scenario, model), group in data.groupby(
        group_columns, sort=True
    ):
        directory = (
            output_dir
            / "prequential"
            / slugify(scenario)
            / slugify(dataset)
            / slugify(exec_id)
        )
        title = f"{dataset} | {scenario}"
        paths = plot_prequential(
            group,
            model,
            configs,
            smooth,
            title,
            directory
            / (
                f"{slugify(_model_label(model))}_prequential_"
                f"{config_slug}_smooth{smooth}"
            ),
            formats,
            dpi,
        )
        for path in paths:
            records.append(
                _manifest_record(
                    path,
                    "prequential_C",
                    dataset,
                    scenario,
                    str(exec_id),
                    model,
                )
            )
    return records


def _manifest_record(
    path: Path,
    plot_type: str,
    dataset: str,
    scenario: str,
    exec_id: str,
    model: str = "ALL",
) -> dict:
    return {
        "File": str(path),
        "Plot_Type": plot_type,
        "Dataset": dataset,
        "Scenario": scenario,
        "Exec_ID": exec_id,
        "Model": model,
    }


def print_plan(
    cumulative: pd.DataFrame,
    scope: str,
    formats: Sequence[str],
    include_prequential: bool,
):
    individual_groups = cumulative.groupby(["Exec_ID", "Dataset", "Scenario"]).ngroups
    scenarios = cumulative["Scenario"].nunique()
    model_groups = cumulative.groupby(["Exec_ID", "Dataset", "Scenario", "Model"]).ngroups
    individual_figures = individual_groups * 6 if scope in {"all", "individual"} else 0
    aggregate_figures = scenarios * 6 if scope in {"all", "aggregate"} else 0
    prequential_figures = model_groups if include_prequential else 0
    total_figures = individual_figures + aggregate_figures + prequential_figures
    print(f"Datasets: {cumulative['Dataset'].nunique()}")
    print(f"Cenários de atributos: {scenarios}")
    print(f"Modelos: {', '.join(_model_label(m) for m in _ordered_models(cumulative))}")
    print(f"Exec_IDs selecionados: {', '.join(sorted(cumulative['Exec_ID'].unique()))}")
    print(f"Figuras planejadas: {total_figures}")
    print(f"Arquivos planejados: {total_figures * len(set(formats))}")


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.smooth <= 0:
        raise ValueError("--smooth must be positive")

    configs = parse_prequential_configs(args.prequential_configs)
    output_dir = args.output_dir or args.input_dir / "plots"
    sns.set_theme(style="whitegrid", context="notebook")

    cumulative = load_results(args.input_dir, "cumulative", args.exec_id)
    cumulative = filter_results(
        cumulative,
        datasets=args.datasets,
        models=args.models,
        scenarios=args.scenarios,
    )
    print_plan(
        cumulative,
        args.scope,
        args.formats,
        args.include_prequential,
    )
    if args.dry_run:
        return {"cumulative": cumulative, "prequential": None, "paths": []}

    records = []
    if args.scope in {"all", "individual"}:
        records.extend(
            generate_individual_plots(cumulative, output_dir, args.formats, args.dpi)
        )
    if args.scope in {"all", "aggregate"}:
        records.extend(
            generate_aggregate_plots(cumulative, output_dir, args.formats, args.dpi)
        )

    prequential = None
    if args.include_prequential:
        prequential = load_results(args.input_dir, "prequential", args.exec_id)
        prequential = filter_results(
            prequential,
            datasets=args.datasets,
            models=args.models,
            scenarios=args.scenarios,
        )
        records.extend(
            generate_prequential_plots(
                prequential,
                output_dir,
                configs,
                args.smooth,
                args.formats,
                args.dpi,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    current_manifest = pd.DataFrame(records)
    if manifest_path.is_file():
        previous_manifest = pd.read_csv(manifest_path, sep=";")
        current_manifest = pd.concat(
            [previous_manifest, current_manifest], ignore_index=True
        ).drop_duplicates("File", keep="last")
    current_manifest.to_csv(manifest_path, sep=";", index=False)
    print(f"Arquivos gerados: {len(records)}")
    print(f"Manifesto: {manifest_path}")
    return {
        "cumulative": cumulative,
        "prequential": prequential,
        "paths": [record["File"] for record in records],
        "manifest": str(manifest_path),
    }


if __name__ == "__main__":
    main()
