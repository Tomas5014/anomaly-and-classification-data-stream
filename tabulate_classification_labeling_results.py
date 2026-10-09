"""Organiza CSVs de classificação A/B/C em Markdown e CSV, sem retreinar."""

from __future__ import annotations

import argparse
import itertools
import os
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd


MODEL_NAMES = {
    "AdaptiveRandomForest": "ARF — Adaptive Random Forest",
    "LeveragingBagging": "LB — Leveraging Bagging",
    "HoeffdingAdaptiveTree": "HAT — Hoeffding Adaptive Tree",
    "HoeffdingTree": "HT — Hoeffding Tree",
}
MODE_NAMES = {"binary": "binário", "multiclass": "multiclasse"}
CATEGORY_ORDER = {name: i for i, name in enumerate(
    ("Consistência", "Generalização", "Adaptação", "Recorrência")
)}
DELAYS = (0.0, 1.0, 5.0, 10.0)
BUDGETS = (1.0, 3.0, 5.0, 10.0, 30.0, 100.0)
GRIDS = {
    "A": set(itertools.product(DELAYS, (100.0,))),
    "B": set(itertools.product((0.0,), BUDGETS)),
    "C": set(itertools.product(DELAYS, BUDGETS)),
}
METRICS = ("F1", "Prec", "Rec", "MCC", "FP", "FN", "Time")
LABEL_METRICS = (
    "Queried", "Effective_Query_Percentage", "Delivered_During_Stream",
    "Flushed_After_Stream",
)
KEY = ["Dataset", "Scenario", "Model", "Delay_Percentage", "Label_Budget_Percentage"]
PAIRS = [f"{metric}_{suffix}" for metric in METRICS + LABEL_METRICS
         for suffix in ("avg", "std")]
PROTOCOL = ["Delay_Instances", "Initial_Training_Instances", "Evaluation_Instances",
            "Window_Evaluation", "Runs"]
REQUIRED = set(KEY + PAIRS + PROTOCOL + ["Exec_ID", "Experiment"])
PERFORMANCE_HEADERS = (
    "Atraso (%)", "Orçamento (%)", "F1 (%)", "Precisão (%)", "Recall (%)",
    "MCC", "FP", "FN", "Tempo (s)",
)


def slug(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()


def dataset_order(dataset: str) -> tuple[int, str, int]:
    category, size = dataset.rsplit("_", 1)
    return CATEGORY_ORDER.get(category, len(CATEGORY_ORDER)), category, int(size)


def ordered(data: pd.DataFrame) -> pd.DataFrame:
    """Ordenação natural dos cenários e estável dos modelos/condições."""
    result = data.copy()
    result.loc[:, "_Dataset_Order"] = result.Dataset.map(dataset_order) if "Dataset" in result else 0
    result.loc[:, "_Model_Order"] = result.Model.map({m: i for i, m in enumerate(MODEL_NAMES)})
    columns = [c for c in ("Training_Label_Mode", "_Dataset_Order", "Experiment",
                          "Delay_Percentage", "Label_Budget_Percentage", "_Model_Order")
               if c in result]
    return result.sort_values(columns).drop(columns=["_Dataset_Order", "_Model_Order"])


def format_number(value: float, decimals: int = 2, signed: bool = False) -> str:
    if abs(value) < 0.5 * 10 ** -decimals:
        value = 0.0
    rendered = format(value, f"{ '+' if signed else ''},.{decimals}f")
    return rendered.replace(",", "_").replace(".", ",").replace("_", ".")


def mean_std(row, metric: str, aggregate: bool = False) -> str:
    decimals = 4 if metric == "MCC" else 2
    mean_key = f"{metric}_mean" if aggregate else f"{metric}_avg"
    std_key = f"{metric}_std_between_scenarios" if aggregate else f"{metric}_std"
    return f"{format_number(row[mean_key], decimals)} ± {format_number(row[std_key], decimals)}"


def markdown_table(headers, rows) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n\n"


def file_link(label: str, path: Path, parent: Path) -> str:
    relative = Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix()
    return f"[{label}]({quote(relative, safe='/._-')})"


def load_batch(input_dir: Path, exec_id: str, mode: str, scenario: str) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob(f"*_{exec_id}_cumulative.csv")):
        frame = pd.read_csv(path, sep=";", dtype={"Exec_ID": str}, float_precision="round_trip")
        missing = REQUIRED - set(frame)
        if missing:
            raise ValueError(f"{path}: colunas obrigatórias ausentes: {sorted(missing)}")
        if not frame.Exec_ID.eq(exec_id).all():
            raise ValueError(f"{path}: Exec_ID incompatível com o nome do arquivo")
        # Os CSVs históricos deste projeto não registravam o modo. Só a seleção
        # explícita da execução multiclasse autoriza interpretar esse legado.
        if "Training_Label_Mode" not in frame:
            if mode != "multiclass" or "_binaryTraining_" in path.name:
                raise ValueError(f"{path}: falta Training_Label_Mode para treinamento binário")
            frame.loc[:, "Training_Label_Mode"] = "multiclass"
        if "Evaluation_Label_Mode" not in frame:
            if mode != "multiclass":
                raise ValueError(f"{path}: falta Evaluation_Label_Mode")
            frame.loc[:, "Evaluation_Label_Mode"] = "binary"
        frame = frame.loc[frame.Scenario.eq(scenario)].copy()
        if frame.empty:
            continue
        if not frame.Training_Label_Mode.eq(mode).all():
            raise ValueError(f"{path}: modo de treinamento incompatível com a seleção")
        if not frame.Evaluation_Label_Mode.eq("binary").all():
            raise ValueError(f"{path}: esta comparação exige avaliação binária")
        frame.loc[:, "Source_Cumulative"] = str(path.resolve())
        frame.loc[:, "Source_Prequential"] = str(path.with_name(
            path.name.replace("_cumulative.csv", "_prequential.csv")
        ).resolve())
        if not Path(frame.Source_Prequential.iloc[0]).is_file():
            raise FileNotFoundError(frame.Source_Prequential.iloc[0])
        frames.append(frame)
    if not frames:
        raise ValueError(f"Execução {exec_id}/{mode}/{scenario} não encontrada em {input_dir}")
    result = pd.concat(frames, ignore_index=True)
    validate_batch(result)
    n = result.Initial_Training_Instances + result.Evaluation_Instances
    result.loc[:, "Total_Instances"] = n
    result.loc[:, "Total_Label_Cost_avg"] = 100 * (result.Initial_Training_Instances + result.Queried_avg) / n
    result.loc[:, "Total_Label_Cost_std"] = 100 * result.Queried_std / n
    result.loc[:, "Delivered_Percentage_avg"] = 100 * result.Delivered_During_Stream_avg / result.Evaluation_Instances
    result.loc[:, "Delivered_Percentage_std"] = 100 * result.Delivered_During_Stream_std / result.Evaluation_Instances
    return ordered(result).reset_index(drop=True)


def validate_batch(data: pd.DataFrame) -> None:
    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Resultados contêm valores ausentes")
    if data.duplicated(KEY + ["Experiment"]).any():
        raise ValueError("Chaves de resultados duplicadas; nenhuma linha será descartada silenciosamente")
    if not np.isfinite(data[PAIRS + PROTOCOL].to_numpy(dtype=float)).all():
        raise ValueError("Resultados numéricos não finitos")
    if (data[[c for c in PAIRS if c.endswith("_std")]] < 0).any().any():
        raise ValueError("Desvios padrão negativos")
    if not data.Experiment.isin(GRIDS).all():
        raise ValueError("Somente os experimentos A, B e C são aceitos")
    for (dataset, scenario), group in data.groupby(["Dataset", "Scenario"]):
        if set(group.Model) != set(MODEL_NAMES):
            raise ValueError(f"{dataset}/{scenario}: são necessários os quatro modelos")
        if (group.Initial_Training_Instances.nunique() != 1
                or group.Evaluation_Instances.nunique() != 1
                or group.Window_Evaluation.nunique() != 1
                or group.Runs.nunique() != 1):
            raise ValueError(f"{dataset}: protocolo inconsistente entre condições")
        for model in MODEL_NAMES:
            for experiment, grid in GRIDS.items():
                selected = group.loc[group.Model.eq(model) & group.Experiment.eq(experiment)]
                actual = set(zip(selected.Delay_Percentage, selected.Label_Budget_Percentage))
                if actual != grid:
                    raise ValueError(f"{dataset}/{model}/{experiment}: grade incompleta ou inesperada")
    if (data.Evaluation_Instances <= 0).any() or (data.Initial_Training_Instances < 0).any():
        raise ValueError("Quantidades inválidas de instâncias")
    if (data.Runs < 1).any() or (data.Window_Evaluation < 1).any():
        raise ValueError("Número de repetições/janela inválido")
    if not np.allclose(data.Queried_avg, data.Delivered_During_Stream_avg + data.Flushed_After_Stream_avg,
                       rtol=0, atol=1e-9):
        raise ValueError("Consultas não correspondem às entregas no fluxo mais entregas finais")
    if not np.allclose(data.Effective_Query_Percentage_avg,
                       100 * data.Queried_avg / data.Evaluation_Instances, rtol=0, atol=1e-9):
        raise ValueError("Percentuais efetivos de consulta inconsistentes")
    n = data.Initial_Training_Instances + data.Evaluation_Instances
    if not np.array_equal(data.Delay_Instances.to_numpy(), np.rint(n * data.Delay_Percentage / 100)):
        raise ValueError("Atraso não corresponde ao percentual do tamanho total do cenário")
    c = data.loc[data.Experiment.eq("C")]
    for experiment in ("A", "B"):
        aligned = data.loc[data.Experiment.eq(experiment)].merge(
            c, on=KEY, suffixes=("_ab", "_c"), validate="one_to_one"
        )
        if len(aligned) != len(data.loc[data.Experiment.eq(experiment)]):
            raise ValueError("Condição A/B ausente em C")
        columns = PAIRS + PROTOCOL
        if not np.allclose(aligned[[f"{x}_ab" for x in columns]],
                           aligned[[f"{x}_c" for x in columns]], rtol=0, atol=1e-9):
            raise ValueError("A/B e seus equivalentes em C divergem")


def compare_modes(data: pd.DataFrame) -> pd.DataFrame:
    unique = data.loc[data.Experiment.eq("C")]
    binary = unique.loc[unique.Training_Label_Mode.eq("binary")]
    multi = unique.loc[unique.Training_Label_Mode.eq("multiclass")]
    if (set(map(tuple, multi[KEY].to_numpy()))
            != set(map(tuple, binary[KEY].to_numpy()))):
        raise ValueError("Os dois modos não têm as mesmas condições/cenários")
    result = multi.merge(binary, on=KEY, suffixes=("_multiclass", "_binary"),
                         validate="one_to_one")
    shared = PROTOCOL + [f"{m}_{s}" for m in LABEL_METRICS for s in ("avg", "std")]
    if not np.allclose(result[[f"{x}_multiclass" for x in shared]],
                       result[[f"{x}_binary" for x in shared]], rtol=0, atol=1e-9):
        raise ValueError("Protocolos/custos de rotulagem incompatíveis entre modos")
    for metric in METRICS:
        result.loc[:, f"Delta_{metric}"] = result[f"{metric}_avg_binary"] - result[f"{metric}_avg_multiclass"]
    return ordered(result).reset_index(drop=True)


def aggregate_scenarios(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["Training_Label_Mode", "Scenario", "Experiment", "Model",
            "Delay_Percentage", "Label_Budget_Percentage"]
    rows = []
    for values, group in data.groupby(keys):
        if group.Dataset.duplicated().any():
            raise ValueError("Agregação não aceita múltiplas observações do mesmo cenário")
        row = dict(zip(keys, values))
        row["Dataset_Count"] = len(group)
        for metric in METRICS + LABEL_METRICS + ("Total_Label_Cost", "Delivered_Percentage"):
            row[f"{metric}_mean"] = group[f"{metric}_avg"].mean()
            row[f"{metric}_std_between_scenarios"] = group[f"{metric}_avg"].std(ddof=0)
        rows.append(row)
    return ordered(pd.DataFrame(rows)).reset_index(drop=True)


def shared_labeling(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["Dataset", "Scenario", "Delay_Percentage", "Label_Budget_Percentage"]
    columns = PROTOCOL + ["Total_Instances"] + [f"{m}_{s}" for m in
        LABEL_METRICS + ("Total_Label_Cost", "Delivered_Percentage") for s in ("avg", "std")]
    rows = []
    for values, group in data.loc[data.Experiment.eq("C")].groupby(keys):
        if not np.allclose(group[columns], group[columns].iloc[0].to_numpy(), rtol=0, atol=1e-9):
            raise ValueError("Custo de rotulagem difere entre modelos ou modos")
        rows.append(dict(zip(keys, values)) | group.iloc[0][columns].to_dict())
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def performance_rows(group: pd.DataFrame, aggregate: bool = False):
    for _, row in group.sort_values(["Delay_Percentage", "Label_Budget_Percentage"]).iterrows():
        yield [format_number(row.Delay_Percentage, 0), format_number(row.Label_Budget_Percentage, 0)] + [
            mean_std(row, metric, aggregate) for metric in METRICS
        ]


def experiment_membership(delay: float, budget: float) -> str:
    return ",".join(exp for exp, grid in GRIDS.items() if (delay, budget) in grid)


def generate_tables(data: pd.DataFrame, output_dir: Path, index_path: Path) -> list[dict]:
    """Escreve somente tabelas derivadas; os CSVs/plots originais são imutáveis."""
    comparisons = compare_modes(data)
    aggregates = aggregate_scenarios(data)
    labeling = shared_labeling(data)
    unique = data.loc[data.Experiment.eq("C")].copy()
    unique.loc[:, "Experiments_Applicable"] = [experiment_membership(d, b) for d, b in
                                       zip(unique.Delay_Percentage, unique.Label_Budget_Percentage)]
    comparisons.loc[:, "Experiments_Applicable"] = [experiment_membership(d, b) for d, b in
                                             zip(comparisons.Delay_Percentage, comparisons.Label_Budget_Percentage)]
    manifest = []

    def save_csv(name, frame):
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Caminhos portáveis relativos ao CSV de saída; precisão não arredondada.
        exported = frame.copy()
        for column in [c for c in exported if c.startswith("Source_")]:
            exported.loc[:, column] = exported[column].map(
                lambda value: Path(os.path.relpath(value, path.parent.resolve())).as_posix()
            )
        exported.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
        manifest.append({"File": str(path.resolve()), "Type": "csv", "Rows": len(frame)})
        return path

    def save_markdown(path, content, count):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        manifest.append({"File": str(path.resolve()), "Type": "markdown", "Rows": count})

    csvs = {
        "Acumulados completos (A/B/C)": save_csv("acumulados_completos.csv", data),
        "Condições únicas (sem repetir A/B/C)": save_csv("condicoes_unicas.csv", unique),
        "Comparação binário menos multiclasse": save_csv("comparacao_binario_multiclasse.csv", comparisons),
        "Resumos por modo e experimento": save_csv("resumos_por_experimento.csv", aggregates),
        "Custos de rotulagem": save_csv("custos_rotulagem.csv", labeling),
    }
    datasets = sorted(data.Dataset.unique(), key=dataset_order)
    mode_files, aggregate_files, comparison_files = {}, {}, {}
    labeling_path = output_dir / "rotulagem.md"
    common_legend = (
        "Valores: média ± DP das repetições; DP populacional (`ddof=0`). "
        "F1/precisão/recall em %, MCC entre −1 e 1, FP/FN em número de erros "
        "e tempo em segundos. FP/FN médios podem ser fracionários. "
        "Arredondamento apenas para exibição; os CSVs consolidados preservam a precisão.\n\n"
    )
    for mode in MODE_NAMES:
        selected = data.loc[data.Training_Label_Mode.eq(mode)]
        for dataset in datasets:
            group = selected.loc[selected.Dataset.eq(dataset)]
            first = group.iloc[0]
            path = output_dir / mode / f"{slug(dataset)}.md"
            mode_files[mode, dataset] = path
            content = f"# {dataset} — treinamento {MODE_NAMES[mode]}\n\n"
            content += file_link("Índice geral", index_path, path.parent) + "\n\n"
            content += (
                f"Execução: `{first.Exec_ID}`. Configuração: `{first.Scenario}`. "
                f"Avaliação binária. {int(first.Runs)} repetições; janela de "
                f"{int(first.Window_Evaluation)} instâncias. Treino inicial: "
                f"{format_number(first.Initial_Training_Instances, 0)}; avaliação: "
                f"{format_number(first.Evaluation_Instances, 0)}.\n\n"
            )
            content += common_legend
            content += (
                "A: atraso, orçamento 100%. B: Random Sampling, atraso zero. "
                "C: todas as combinações. A e B repetem condições presentes em C, "
                "não são novas repetições independentes.\n\n"
                + file_link("Consultas, entregas, pendências finais e custo total", labeling_path, path.parent)
                + "\n\n"
            )
            for model, name in MODEL_NAMES.items():
                content += f"## {name}\n\n"
                for experiment in GRIDS:
                    rows = group.loc[group.Model.eq(model) & group.Experiment.eq(experiment)]
                    content += f"### Experimento {experiment}\n\n"
                    content += markdown_table(PERFORMANCE_HEADERS, performance_rows(rows))
            content += "## Dados originais, incluindo todas as janelas\n\n"
            content += file_link("CSV acumulado", Path(first.Source_Cumulative), path.parent) + " · "
            content += file_link("CSV prequential (por janela)", Path(first.Source_Prequential), path.parent) + "\n\n"
            save_markdown(path, content, len(group))
        for experiment in GRIDS:
            group = aggregates.loc[aggregates.Training_Label_Mode.eq(mode) & aggregates.Experiment.eq(experiment)]
            path = output_dir / "resumos" / f"{mode}_experimento_{experiment}.md"
            aggregate_files[mode, experiment] = path
            content = f"# Experimento {experiment} — resumo do treinamento {MODE_NAMES[mode]}\n\n"
            content += file_link("Índice geral", index_path, path.parent) + "\n\n"
            content += (
                f"Média não ponderada dos {len(datasets)} cenários. Cada valor é "
                "média ± DP ENTRE CENÁRIOS, calculado sobre as médias das repetições "
                "de cada cenário (`ddof=0`). Este DP não é o DP das cinco repetições "
                "e não é intervalo de confiança. FP/FN são médias por cenário, não "
                "somas de erros. F1/precisão/recall em %, MCC em [−1, 1], tempo em segundos.\n\n"
            )
            for model, name in MODEL_NAMES.items():
                content += f"## {name}\n\n"
                content += markdown_table(PERFORMANCE_HEADERS, performance_rows(group.loc[group.Model.eq(model)], True))
            save_markdown(path, content, len(group))
    for dataset in datasets:
        group = comparisons.loc[comparisons.Dataset.eq(dataset)]
        path = output_dir / "comparacao" / f"{slug(dataset)}.md"
        comparison_files[dataset] = path
        content = f"# {dataset} — binário × multiclasse\n\n"
        content += file_link("Índice geral", index_path, path.parent) + "\n\n"
        content += (
            "As 24 condições únicas de C incluem todas as condições de A e B. "
            "A coluna Experimentos mostra onde cada condição também aparece. "
            "F1 é média ± DP das repetições. Δ = binário − multiclasse, calculado "
            "antes do arredondamento. ΔF1/ΔPrecisão/ΔRecall em p.p.; ΔMCC na "
            "escala do coeficiente; ΔFP/ΔFN em número médio de erros. "
            "Para desempenho, delta positivo indica melhora; para erros, negativo "
            "indica redução. Não são testes de significância.\n\n"
        )
        for model, name in MODEL_NAMES.items():
            content += f"## {name}\n\n"
            rows = []
            for _, row in group.loc[group.Model.eq(model)].sort_values(["Delay_Percentage", "Label_Budget_Percentage"]).iterrows():
                rows.append([
                    format_number(row.Delay_Percentage, 0), format_number(row.Label_Budget_Percentage, 0),
                    row.Experiments_Applicable,
                    f"{format_number(row.F1_avg_multiclass)} ± {format_number(row.F1_std_multiclass)}",
                    f"{format_number(row.F1_avg_binary)} ± {format_number(row.F1_std_binary)}",
                ] + [format_number(row[f"Delta_{m}"], 4 if m == "MCC" else 2, True)
                     for m in ("F1", "Prec", "Rec", "MCC", "FP", "FN")])
            content += markdown_table(("Atraso (%)", "Orçamento (%)", "Experimentos",
                "F1 multiclasse (%)", "F1 binário (%)", "ΔF1", "ΔPrecisão", "ΔRecall",
                "ΔMCC", "ΔFP", "ΔFN"), rows)
        content += "Os valores completos e diferenças de tempo também estão no " + file_link(
            "CSV de comparação", csvs["Comparação binário menos multiclasse"], path.parent
        ) + ". Tempos de execuções em dias distintos não isolam o efeito do alvo.\n"
        save_markdown(path, content, len(group))
    content = "# Rotulagem — consultas, entregas e custo total\n\n"
    content += file_link("Índice geral", index_path, labeling_path.parent) + "\n\n"
    content += (
        "Média ± DP das repetições. Custos e protocolos foram verificados como "
        "iguais entre os quatro modelos e os dois modos, por isso são apresentados "
        "uma única vez. Todas as condições de A e B estão neste grid de C.\n\n"
        "O orçamento é a probabilidade de consulta do Random Sampling, não uma "
        "cota exata. Consulta efetiva (%) = 100 × consultas / instâncias avaliadas. "
        "Custo total (%) = 100 × (treino inicial + consultas) / total de instâncias. "
        "O treino inicial tem todos os rótulos e está excluído da consulta efetiva, "
        "mas incluído no custo total.\n\n"
        "Entregues no fluxo são usados em atualizações durante a avaliação. "
        "Entregues após o fim esvaziam a fila, mas não melhoram as previsões já "
        "avaliadas. Consultas = entregas no fluxo + entregas após o fim, em média; "
        "os DP dessas parcelas não se somam. O atraso em instâncias usa o "
        "tamanho total, incluindo o treino inicial.\n\n"
    )
    for dataset in datasets:
        group = labeling.loc[labeling.Dataset.eq(dataset)]
        first = group.iloc[0]
        content += f"## {dataset}\n\n"
        content += (f"Total: {format_number(first.Total_Instances, 0)}; treino inicial: "
                    f"{format_number(first.Initial_Training_Instances, 0)}; "
                    f"avaliadas: {format_number(first.Evaluation_Instances, 0)}.\n\n")
        rows = ([format_number(row.Delay_Percentage, 0), format_number(row.Delay_Instances, 0),
                 format_number(row.Label_Budget_Percentage, 0)] + [mean_std(row, m) for m in
                 ("Queried", "Effective_Query_Percentage", "Total_Label_Cost",
                  "Delivered_During_Stream", "Flushed_After_Stream")]
                for _, row in group.iterrows())
        content += markdown_table(("Atraso (%)", "Atraso (instâncias)", "Orçamento (%)",
            "Consultados", "Consulta efetiva (%)", "Custo total (%)", "Entregues no fluxo",
            "Entregues após o fim"), rows)
    save_markdown(labeling_path, content, len(labeling))
    content = "# Tabelas dos resultados de classificação\n\n"
    content += f"Gerado em {date.today().isoformat()}, usando apenas resultados já existentes.\n\n"
    content += (
        "Este índice reúne todos os resultados acumulados dos experimentos A/B/C "
        "das duas execuções completas analisadas e dá acesso a todos os resultados "
        "por janela. Escopo: classificação, configuração `"
        + str(data.Scenario.iloc[0])
        + "`. Não inclui detecção de anomalias, experimentos padrões com outro "
        "protocolo, resultados otimizados ou seleção de features inexistentes "
        "nessas execuções. Os CSVs e gráficos originais não foram alterados.\n\n"
    )
    content += "## 1. Execuções e cobertura\n\n"
    coverage = []
    for mode in MODE_NAMES:
        group = data.loc[data.Training_Label_Mode.eq(mode)]
        coverage.append([MODE_NAMES[mode], group.Exec_ID.iloc[0], "binária", group.Dataset.nunique(),
                         group.Model.nunique(), int(group.Runs.iloc[0]), len(group), int(group.Experiment.eq("C").sum())])
    content += markdown_table(("Treinamento", "Exec_ID", "Avaliação", "Cenários", "Modelos",
                              "Repetições", "Linhas A/B/C", "Condições únicas"), coverage)
    content += (
        "A usa atrasos de 0%, 1%, 5% e 10%, com todos os rótulos. B usa orçamentos "
        "de 1%, 3%, 5%, 10%, 30% e 100%, sem atraso. C cruza essas opções. "
        "Cada cenário/modelo tem 34 linhas A/B/C, mas apenas 24 condições únicas. "
        "Não contar as linhas equivalentes como novos experimentos independentes. "
        "O resultado histórico adicional `20261001_091516` não é incluído.\n\n"
    )
    content += "## 2. Como ler as tabelas\n\n"
    content += common_legend
    content += (
        "Nas tabelas individuais, o DP descreve a variação entre as repetições. "
        "Nos resumos gerais, o DP descreve a variação ENTRE CENÁRIOS, e a média "
        "dá o mesmo peso a cada cenário; não é F1 após concatenar as previsões. "
        "DP não é margem de erro nem intervalo de confiança. Um DP exibido como "
        "0,00 pode ser um valor muito pequeno arredondado.\n\n"
        "Δ = binário − multiclasse. F1/precisão/recall maiores são melhores; "
        "FP/FN menores são melhores. O alvo positivo avaliado é ATTACK nos dois "
        "modos. Treinamento multiclasse não significa avaliação multiclasse.\n\n"
        "Atraso (%) é proporcional ao total do cenário, não apenas ao trecho "
        "avaliado. Orçamento (%) é probabilidade de consulta; não inclui o treino "
        "inicial totalmente rotulado. Não há condição com orçamento zero.\n\n"
    )
    content += "## 3. Referência: todos os rótulos, atraso zero\n\n"
    content += "Média não ponderada entre cenários; DP entre cenários. ΔF1 em pontos percentuais.\n\n"
    baseline = aggregates.loc[aggregates.Experiment.eq("C") & aggregates.Delay_Percentage.eq(0)
                              & aggregates.Label_Budget_Percentage.eq(100)]
    rows = []
    for model, name in MODEL_NAMES.items():
        b = baseline.loc[baseline.Model.eq(model) & baseline.Training_Label_Mode.eq("binary")].iloc[0]
        m = baseline.loc[baseline.Model.eq(model) & baseline.Training_Label_Mode.eq("multiclass")].iloc[0]
        rows.append([name.split(" — ")[0], mean_std(m, "F1", True), mean_std(b, "F1", True),
                     format_number(b.F1_mean - m.F1_mean, 2, True)])
    content += markdown_table(("Modelo", "F1 multiclasse (%)", "F1 binário (%)", "ΔF1 (p.p.)"), rows)
    content += "## 4. Todos os resumos por experimento\n\n"
    rows = [[exp, file_link("Treinamento binário", aggregate_files["binary", exp], index_path.parent),
             file_link("Treinamento multiclasse", aggregate_files["multiclass", exp], index_path.parent)] for exp in GRIDS]
    content += markdown_table(("Experimento", "Binário", "Multiclasse"), rows)
    content += "## 5. Todos os resultados por cenário e modelo\n\n"
    content += "Cada tabela individual contém os quatro modelos, os três experimentos e todas as métricas acumuladas.\n\n"
    rows = []
    for dataset in datasets:
        first = data.loc[data.Dataset.eq(dataset)].iloc[0]
        rows.append([dataset, format_number(first.Total_Instances, 0),
                     format_number(first.Initial_Training_Instances, 0), format_number(first.Evaluation_Instances, 0),
                     file_link("A/B/C binário", mode_files["binary", dataset], index_path.parent),
                     file_link("A/B/C multiclasse", mode_files["multiclass", dataset], index_path.parent),
                     file_link("Comparação completa", comparison_files[dataset], index_path.parent)])
    content += markdown_table(("Cenário", "Total", "Treino inicial", "Avaliação", "Binário", "Multiclasse", "Comparação"), rows)
    content += "## 6. Custos de rotulagem completos\n\n"
    content += file_link("Todas as consultas, entregas e custos, por cenário e condição", labeling_path, index_path.parent) + ".\n\n"
    content += "## 7. Resultados por janela e arquivos de origem\n\n"
    content += (
        "As tabelas completas por janela já estão nos CSVs prequenciais abaixo. "
        "Cada arquivo contém todos os modelos e A/B/C; não há seleção de melhores "
        "janelas. `Window_Index` identifica a janela; `Instance` é a posição final "
        "(base zero); `Window_Instances` é o tamanho efetivo. A última janela pode "
        "ser menor. Filtre `Experiment`, `Model`, `Delay_Percentage` e "
        "`Label_Budget_Percentage` para acompanhar uma condição.\n\n"
        "F1/precisão/recall/FP/FN por janela são média e DP das repetições, não "
        "métricas acumuladas. A versão binária inclui consultas, entregas e "
        "pendências por janela; esses campos não foram registrados no legado "
        "multiclasse e não foram preenchidos artificialmente. Janelas sem "
        "ataques podem ter F1/recall zero mesmo com previsões benignas corretas.\n\n"
    )
    rows = []
    for dataset in datasets:
        row = [dataset]
        for mode in MODE_NAMES:
            first = data.loc[data.Dataset.eq(dataset) & data.Training_Label_Mode.eq(mode)].iloc[0]
            row.extend([file_link("Acumulado", Path(first.Source_Cumulative), index_path.parent),
                        file_link("Por janela", Path(first.Source_Prequential), index_path.parent)])
        rows.append(row)
    content += markdown_table(("Cenário", "Binário: acumulado", "Binário: janelas",
                              "Multiclasse: acumulado", "Multiclasse: janelas"), rows)
    content += "## 8. Planilhas CSV para filtrar ou importar no TCC\n\n"
    content += (
        "Separador `;`, codificação UTF-8 com BOM, números com ponto decimal e "
        "sem arredondamento deliberado. Ao importar no Excel/LibreOffice, use "
        "essas opções e trate `Exec_ID` como texto. Os caminhos de origem são "
        "relativos à pasta do CSV consolidado.\n\n"
    )
    counts = (len(data), len(unique), len(comparisons), len(aggregates), len(labeling))
    content += markdown_table(("Planilha", "Linhas", "Conteúdo"), [
        [file_link(name, path, index_path.parent), count, description]
        for (name, path), count, description in zip(csvs.items(), counts, (
            "Todas as médias/DP e metadados originais de A/B/C, mais custos derivados.",
            "24 condições por cenário/modelo/modo; Experiments_Applicable identifica A/B/C equivalentes.",
            "Condições emparelhadas; todas as médias/DP dos dois modos e diferenças das médias.",
            "Média e DP entre cenários, em colunas explicitamente separadas das métricas individuais.",
            "Um registro por cenário/atraso/orçamento; custo comum aos modelos e modos.",
        ))
    ])
    content += "## 9. Limitações para o TCC\n\n"
    content += (
        "Esta organização não corrige o protocolo histórico: há pré-processamento "
        "com estatísticas globais do cenário, condições dependentes, poucos "
        "ataques em alguns cenários e execuções realizadas em dias distintos. "
        "Não foram recuperados resultados individuais das cinco repetições; os "
        "arquivos existentes registram apenas média e DP. Não inferir significância "
        "estatística apenas dos deltas ou DP destas tabelas.\n\n"
    )
    for name in ("ANALISE_RESULTADOS_BINARIOS_TCC.md", "ANALISE_RESULTADOS_TCC.md"):
        path = Path(__file__).resolve().parent / name
        if path.is_file():
            content += file_link(name, path, index_path.parent) + "\n\n"
    content += "## 10. Como regenerar\n\n"
    content += (
        "Execute na raiz do repositório. Apenas as tabelas derivadas e este índice "
        "serão reescritos; não há treinamento nem dependência do Java/CapyMOA.\n\n"
        "```bash\n.venv/bin/python tabulate_classification_labeling_results.py \\\n"
        f"  --binary-exec-id {data.loc[data.Training_Label_Mode.eq('binary'), 'Exec_ID'].iloc[0]} \\\n"
        f"  --multiclass-exec-id {data.loc[data.Training_Label_Mode.eq('multiclass'), 'Exec_ID'].iloc[0]}\n```\n\n"
        "O gerador confere grades completas, duplicações, equivalência A/B/C e "
        "compatibilidade entre os modos antes de escrever. Os IDs explícitos "
        "evitam misturar execuções parciais ou contar arquivos históricos adicionais. "
        "Use `--help` para mudar as pastas, o índice ou a configuração.\n"
    )
    save_markdown(index_path, content, len(data))
    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.loc[:, "File"] = manifest_frame.File.map(
        lambda p: Path(os.path.relpath(p, output_dir.resolve())).as_posix()
    )
    save_csv("manifest.csv", manifest_frame)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output/ClassificationLabeling"))
    parser.add_argument("--output-dir", type=Path, help="Padrão: <input-dir>/tables")
    parser.add_argument("--index-path", type=Path, default=Path("TABELAS_RESULTADOS_CLASSIFICACAO.md"))
    parser.add_argument("--binary-exec-id", default="20261007_130844")
    parser.add_argument("--multiclass-exec-id", default="20261001_101354")
    parser.add_argument("--scenario", default="Default_FullFeatures")
    args = parser.parse_args(argv)
    binary = load_batch(args.input_dir, args.binary_exec_id, "binary", args.scenario)
    multi = load_batch(args.input_dir, args.multiclass_exec_id, "multiclass", args.scenario)
    data = ordered(pd.concat([binary, multi], ignore_index=True)).reset_index(drop=True)
    if data.Runs.nunique() != 1:
        raise ValueError("Os modos têm quantidades diferentes de repetições")
    output_dir = args.output_dir or args.input_dir / "tables"
    manifest = generate_tables(data, output_dir, args.index_path)
    print(f"Índice: {args.index_path}")
    print(f"{len(data)} linhas A/B/C; {int(data.Experiment.eq('C').sum())} condições únicas nos dois modos.")
    print(f"{len(manifest)} arquivos gerados. Tabelas: {output_dir}")


if __name__ == "__main__":
    main()
