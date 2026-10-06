# Guia do Código dos Experimentos de Classificação

## 1. Arquitetura

```text
CSV do cenário
    ↓
DataStreamProcessor
    ↓
NumpyStream do CapyMOA
    ↓
Fábricas de classificadores
    ↓
ClassificationLabelingExperimentRunner
    ├── treinamento inicial
    ├── predição prequencial
    ├── Random Sampling
    ├── fila de rótulos atrasados
    └── agregação de métricas
    ↓
CSVs cumulativo e prequencial + gráficos opcionais
```

Arquivos envolvidos:

| Arquivo | Responsabilidade |
|---|---|
| `run_classification_labeling.py` | CLI e orquestração |
| `plot_classification_labeling_results.py` | Gráficos a partir dos CSVs existentes |
| `src/Classification/Labeling.py` | Regras dos experimentos A, B e C |
| `src/Classification/Models.py` | Classificadores CapyMOA |
| `src/Data/Processor.py` | Pré-processamento e criação do fluxo |
| `src/Results/Metrics.py` | Cálculo das métricas |
| `src/Results/Plots.py` | Gráficos opcionais |
| `tests/test_classification_labeling.py` | Testes da nova lógica |

## 2. `src/Classification/Labeling.py`

### 2.1. Constantes

```python
DEFAULT_DELAY_FRACTIONS = (0.0, 0.01, 0.05, 0.10)
DEFAULT_LABEL_PROBABILITIES = (0.01, 0.03, 0.05, 0.10, 0.30, 1.0)
```

São frações entre zero e um. A conversão para porcentagem ocorre na exportação.

### 2.2. `LabelingExperimentConfig`

`dataclass` imutável que representa uma configuração:

```python
LabelingExperimentConfig(
    experiment="C",
    delay_fraction=0.05,
    label_probability=0.10,
)
```

`__post_init__` valida os dados. `delay_percentage` e
`label_budget_percentage` convertem frações em porcentagem. `slug` cria um
identificador como:

```text
experiment_C_delay_5pct_labels_10pct
```

### 2.3. `build_experiment_configs`

Cria a grade solicitada:

- A: cada atraso com probabilidade 1,0;
- B: atraso zero com cada probabilidade;
- C: produto cartesiano de atrasos e probabilidades.

Retorna uma lista de `LabelingExperimentConfig`.

### 2.4. `calculate_delay_instances`

Converte a fração em número de instâncias:

```python
delay = calculate_delay_instances(15000, 0.05)
# 750
```

Valida o tamanho do fluxo e a faixa da fração. O arredondamento usa `round`.

## 3. `ClassificationLabelingExperimentRunner`

Classe principal dos experimentos.

### 3.1. Construtor

```python
runner = ClassificationLabelingExperimentRunner(
    target_names=targets,
    n_runs=5,
    random_seed=42,
    attack_gap_tolerance=1000,
)
```

- `target_names`: classes na ordem do fluxo;
- `n_runs`: repetições por modelo e configuração;
- `random_seed`: semente inicial;
- `attack_gap_tolerance`: distância para agrupar a primeira região de ataque.

O construtor procura `BENIGN`, `NORMAL` ou `0` para identificar a classe normal.
Também instancia as classes existentes `Metrics` e `Plots`.

### 3.2. `prequential_test`

Executa uma repetição:

```python
result = runner.prequential_test(
    stream=stream,
    learner=model,
    delay_fraction=0.05,
    label_probability=0.10,
    window_evaluation=100,
    sampling_seed=42,
)
```

Primeiro, o método descobre o tamanho do fluxo, calcula o atraso, encontra o fim
do treinamento inicial, reinicia o fluxo e cria o gerador aleatório.

A fila `pending_labels` é um `deque`. Cada item guarda:

```text
(índice_de_entrega, instância_original)
```

Durante o treinamento inicial, somente `learner.train(instance)` é chamado.
Depois dele, rótulos vencidos são entregues antes da predição atual:

```python
while pending_labels and pending_labels[0][0] <= stream_index:
    _, delayed_instance = pending_labels.popleft()
    learner.train(delayed_instance)
```

Depois ocorre a predição. Se o classificador retornar `None`, a classe normal é
usada como resposta segura. Para as métricas, a saída multiclasse vira binária:
normal é 0; qualquer ataque é 1.

O Random Sampling ocorre depois da predição:

```python
selected = label_probability >= 1.0 or rng.random() < label_probability
```

Se selecionada, a instância é treinada após a predição para atraso zero ou entra
na fila para entrega futura.

A cada `window_evaluation` instâncias, `_append_window_metrics` calcula as
métricas da janela. A janela parcial final também é calculada. Com
`window_evaluation=None`, não são produzidas séries por janela.

No final, rótulos pendentes são treinados por padrão. Isso não altera previsões
passadas; apenas permite contabilizar quantos ultrapassaram o fim do fluxo.

Principais campos retornados:

| Campo | Significado |
|---|---|
| `y_true`, `y_pred` | valores binários das métricas cumulativas |
| `true_labels_multi` | classes originais do fluxo |
| `instances` | índice final de cada janela |
| `window_sizes` | tamanho real de cada janela |
| `f1`, `precision`, `recall`, `fp`, `fn` | métricas por janela |
| `initial_training_instances` | tamanho do treinamento inicial |
| `evaluation_instances` | instâncias avaliadas |
| `delay_instances` | atraso em instâncias |
| `queried_instances` | instâncias escolhidas |
| `queried_indices` | posições escolhidas |
| `delivered_during_stream` | rótulos entregues durante o fluxo |
| `flushed_after_stream` | rótulos entregues depois do fim |
| `effective_query_percentage` | orçamento efetivamente usado |
| `exec_time` | tempo da repetição |

### 3.3. `_find_initial_training_end`

Percorre os rótulos antes da avaliação. Registra o último ataque da primeira
região. Quando surge outro ataque a uma distância maior que a tolerância, retorna
a posição imediatamente posterior ao ataque anterior. O retorno é exclusivo:

```python
if stream_index < initial_training_end:
    learner.train(instance)
```

O fluxo é reiniciado depois da busca.

### 3.4. `run_configuration`

Executa uma configuração para todas as fábricas de modelos. Uma fábrica é usada
porque cada repetição precisa de um modelo novo, sem estado anterior:

```python
def make_ht(run_seed=None):
    return get_classification_models(
        schema=stream.get_schema(),
        selected_models=["HT"],
        run_seed=run_seed,
    )["HoeffdingTree"]
```

Cada repetição usa `random_seed + run_index` e chama `prequential_test`.

### 3.5. `_aggregate_runs`

Calcula métricas cumulativas por repetição, média e desvio-padrão entre
repetições, estatísticas por janela, seleção de rótulos e tempo. `numpy.std` usa
o padrão `ddof=0`, portanto o desvio é populacional.

### 3.6. `run_suite`

Método de alto nível:

```python
suite = runner.run_suite(
    stream=stream,
    algorithms={"HoeffdingTree": make_ht},
    experiments=("A", "B", "C"),
    window_evaluation=100,
    experiment_name="Consistência_25",
    scenario_name="Default_FullFeatures",
    exec_id="20260930_120000",
)
```

Ele cria a grade, executa combinações únicas, agrega resultados, cria
`DataFrame`s, salva CSVs e gera gráficos opcionais.

O cache usa a chave:

```python
(delay_fraction, label_probability)
```

Assim, combinações iguais em A, B e C não treinam novamente. A identificação de
cada experimento continua presente nos CSVs.

O retorno contém:

```python
{
    "results": resultados_por_configuração,
    "cumulative": dataframe_cumulativo,
    "prequential": dataframe_prequencial,
    "paths": caminhos_dos_csvs,
}
```

### 3.7. Métodos auxiliares

- `_stream_size`: obtém o tamanho, percorrendo o fluxo se `len` não existir;
- `_append_window_metrics`: calcula métricas de uma janela;
- `_cumulative_rows`: cria linhas do CSV cumulativo;
- `_prequential_rows`: cria linhas do CSV prequencial;
- `_generate_plots`: chama os gráficos existentes;
- `_mean_std`: calcula média e desvio-padrão;
- `_format_percentage`: cria texto seguro para percentuais;
- `_clean_filename`: limpa nomes de arquivos.

## 4. `run_classification_labeling.py`

Permite executar tudo sem editar notebook.

### 4.1. Argumentos

| Argumento | Finalidade | Padrão |
|---|---|---|
| `--data-root` | raiz dos cenários | `data/15k` |
| `--datasets` | CSVs explícitos | não definido |
| `--categories` | categorias | todas |
| `--sizes` | 25, 200 e/ou 1000 | todos |
| `--models` | LB, HAT, ARF e/ou HT | todos |
| `--experiments` | A, B e/ou C | todos |
| `--feature-sets` | `full` e/ou `selected` | `full` |
| `--n-runs` | repetições | 5 |
| `--seed` | semente inicial | 42 |
| `--window` | tamanho da janela | 100 |
| `--attack-gap-tolerance` | tolerância da primeira região | 1000 |
| `--output-dir` | diretório de saída | `output/ClassificationLabeling` |
| `--plots` | habilita gráficos | não |
| `--dry-run` | mostra o plano sem treinar | não |

`--datasets` substitui a combinação de categorias e tamanhos.

### 4.2. Funções

`resolve_datasets` monta caminhos no padrão
`data/15k/<categoria>/<categoria>_<tamanho>.csv` e valida sua existência.

`build_stream` carrega o CSV e usa `DataStreamProcessor` com rótulos multiclasse,
Min-Max, imputação por mediana e remoção de identificadores de rede.

`build_model_factories` cria uma função por modelo. O argumento padrão
`selected=model_name` captura corretamente o valor de cada iteração e evita que
todas as fábricas apontem para o último modelo.

`feature_configurations` traduz `full` para `Default_FullFeatures` e `selected`
para `Default_33Features`.

`print_plan` informa datasets, atributos, modelos, configurações lógicas,
configurações únicas e total de execuções. `--dry-run` termina depois disso.

`main` valida argumentos, cria os fluxos e modelos, instancia o runner, executa a
suíte e imprime os caminhos dos resultados. Um `exec_id` de data e hora separa
execuções.

## 5. Módulos reutilizados

### `src/Classification/Models.py`

`get_classification_models` instancia LB, HAT, ARF e HT. O executor envia
`run_seed`, criando modelos independentes e reproduzíveis.

### `src/Data/Processor.py`

`DataStreamProcessor` limpa o `DataFrame`, trata ausentes, normaliza, codifica
rótulos e cria o `NumpyStream`.

### `src/Results/Metrics.py`

`calc_sklearn_metrics` retorna F1, precisão, recall, MCC, FP e FN. F1, precisão
e recall são multiplicados por 100.

### `src/Results/Plots.py`

`plot_metrics` e `plot_fp_fn` são chamados somente com `--plots`.

## 6. Testes

`tests/test_classification_labeling.py` usa `FakeStream` e `FakeLearner` para
verificar a ordem causal sem depender do Java ou de modelos reais.

```bash
python -m unittest discover -s tests -v
```

## 7. Exemplos

Teste rápido:

```bash
python run_classification_labeling.py \
  --categories Consistência \
  --sizes 25 \
  --models HT \
  --experiments A \
  --n-runs 1
```

Somente aprendizado ativo:

```bash
python run_classification_labeling.py \
  --experiments B \
  --models HT HAT \
  --n-runs 5
```

Experimento C com gráficos:

```bash
python run_classification_labeling.py \
  --experiments C \
  --models LB HAT ARF HT \
  --plots
```

## 8. Metodologia

Para protocolo, fórmulas, justificativas e interpretação das métricas, consulte
[METODOLOGIA_CLASSIFICACAO.md](METODOLOGIA_CLASSIFICACAO.md).

Para geração de figuras cumulativas, agregadas e prequenciais, consulte
[GRAFICOS_CLASSIFICACAO.md](GRAFICOS_CLASSIFICACAO.md).

Para uma explicação passo a passo da ausência de rótulos, Random Sampling, fila
de atrasos e integração com o CapyMOA, consulte
[IMPLEMENTACAO_ROTULAGEM_CLASSIFICACAO.md](IMPLEMENTACAO_ROTULAGEM_CLASSIFICACAO.md).
