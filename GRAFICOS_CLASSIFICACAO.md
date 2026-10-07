# Gráficos dos Experimentos de Classificação

## 1. Objetivo

O arquivo `plot_classification_labeling_results.py` transforma os CSVs já
gerados pelos experimentos de active learning e atraso de rótulos em figuras.
Ele não carrega os datasets originais, não instancia classificadores e não
repete treinamentos.

O processo é:

```text
CSVs cumulativos e prequenciais
              ↓
validação das colunas
              ↓
seleção da execução mais recente
              ↓
filtros opcionais
              ↓
gráficos individuais e agregados
              ↓
PNG, PDF ou SVG + manifest.csv
```

Isso separa duas responsabilidades:

- `run_classification_labeling.py`: executa os modelos, produz dados e, com
  `--plots`, gera diagnósticos temporais detalhados de cada configuração;
- `plot_classification_labeling_results.py`: lê os dados e produz figuras.

Assim, é possível alterar aparência, filtros e formatos sem gastar tempo
treinando os modelos novamente.

O protocolo principal agora usa treinamento e avaliação binários. O gerador
separa os resultados por `Training_Label_Mode`, nos títulos, diretórios e
agregações. Resultados multiclasse anteriores não são convertidos para binários.
É necessário executar novamente `run_classification_labeling.py` para gerar
as novas previsões.

## 2. Uso rápido

Gerar os gráficos cumulativos de todos os resultados mais recentes:

```bash
python plot_classification_labeling_results.py
```

Gerar apenas resultados do novo protocolo:

```bash
python plot_classification_labeling_results.py --training-label-mode binary
```

Se ainda não houver CSVs binários, esse filtro informa que não encontrou o modo
solicitado. Sem filtro, os modos disponíveis são processados separadamente.

As imagens são gravadas por padrão em:

```text
output/ClassificationLabeling/plots
```

Visualizar o plano sem criar arquivos:

```bash
python plot_classification_labeling_results.py --dry-run
```

Gerar também as curvas prequenciais:

```bash
python plot_classification_labeling_results.py --include-prequential
```

## 3. Entradas

O diretório padrão é `output/ClassificationLabeling`. O programa procura apenas
arquivos no primeiro nível desse diretório com estes padrões:

```text
*_cumulative.csv
*_prequential.csv
```

Os arquivos usam `;` como separador.

Os novos CSVs incluem `Training_Label_Mode` (`binary` ou `multiclass`) e
`Evaluation_Label_Mode` (`binary`). Na leitura dos arquivos históricos deste
projeto sem essas colunas, são assumidos treinamento multiclasse e avaliação
binária, sem alterar os arquivos. A deduplicação também distingue os modos.

### 3.1. CSV cumulativo

É usado nos gráficos dos Experimentos A, B e C, nos gráficos de erros e nas
figuras agregadas. Entre as colunas obrigatórias estão:

- identificação: `Exec_ID`, `Dataset`, `Experiment`, `Model`, `Scenario`;
- configuração: `Delay_Percentage`, `Label_Budget_Percentage`;
- custo real: `Effective_Query_Percentage_avg`;
- desempenho: `F1_avg`, `Prec_avg`, `Rec_avg`;
- dispersão entre repetições: `F1_std`, `Prec_std`, `Rec_std`;
- erros: `FP_avg`, `FP_std`, `FN_avg`, `FN_std`.

### 3.2. CSV prequencial

É carregado somente quando `--include-prequential` é informado. Suas colunas
obrigatórias incluem:

- identificação e configuração experimental;
- `Window_Index`, `Window_Instances` e `Instance`;
- média e desvio-padrão de F1, precisão e recall por janela.

Os arquivos produzidos pela versão atual também incluem FP/FN, rótulos
consultados e entregues e quantidade pendente ao final de cada janela. Essas
colunas adicionais alimentam o diagnóstico detalhado criado com `--plots`.

O programa interrompe a execução e informa o arquivo quando alguma coluna
obrigatória não existe.

## 4. Seleção da execução

O diretório pode conter resultados repetidos, como duas execuções do mesmo
dataset. O argumento `--exec-id` controla esse comportamento.

### `--exec-id latest`

É o padrão. Para cada combinação de `Dataset`, `Scenario` e
`Training_Label_Mode`, mantém somente o maior `Exec_ID`. Como o executor usa o
formato `AAAAMMDD_HHMMSS`, a ordenação
textual também representa a ordem cronológica.

Exemplo: se existem as execuções `20261001_091516` e `20261001_101354` para o
mesmo dataset, somente `20261001_101354` é usada.

### Execução específica

```bash
python plot_classification_labeling_results.py \
  --exec-id 20261001_101354
```

### Todas as execuções

```bash
python plot_classification_labeling_results.py --exec-id all
```

No modo `all`, as figuras individuais mantêm cada `Exec_ID` em um diretório
separado. Para a média agregada, execuções repetidas do mesmo dataset são
primeiro promediadas. Assim, um dataset executado duas vezes não recebe peso
duplo.

## 5. Gráficos gerados

Por padrão, são produzidas seis figuras para cada combinação de dataset,
cenário de atributos, modo de treinamento e execução. Também são produzidas
seis figuras agregadas para cada cenário de atributos e modo de treinamento.

### 5.1. Desempenho do Experimento A

Arquivo:

```text
performance_experiment_A.png
```

Possui três painéis:

- F1-score × atraso;
- precisão × atraso;
- recall × atraso.

Cada linha representa um classificador. O eixo horizontal contém 0%, 1%, 5% e
10% de atraso. A região transparente ao redor de cada linha representa o
desvio-padrão.

Uso principal: observar quanto cada modelo perde quando os rótulos demoram para
chegar.

### 5.2. Desempenho do Experimento B

Arquivo:

```text
performance_experiment_B.png
```

Possui F1, precisão e recall em função do orçamento nominal de rótulos:

```text
1%, 3%, 5%, 10%, 30% e 100%
```

Uso principal: identificar quanto desempenho é obtido com cada custo de
rotulagem e comparar a configuração totalmente supervisionada com orçamentos
menores.

### 5.3. Heatmap do Experimento C

Arquivo:

```text
heatmap_experiment_C_f1.png
```

O heatmap usa:

- linhas: atraso do rótulo;
- colunas: orçamento de rótulos;
- cor e anotação: F1-score médio.

Cada classificador possui um painel próprio. A escala é fixa entre 0 e 100 para
permitir comparação visual direta entre modelos e datasets.

Uso principal: encontrar regiões em que atraso e falta de rótulos interagem e
identificar configurações robustas.

### 5.4. Custo de rotulagem × desempenho

Arquivo:

```text
cost_performance_experiment_C.png
```

O eixo horizontal não usa o orçamento nominal. Ele usa
`Effective_Query_Percentage_avg`, isto é, o percentual realmente selecionado
pelo Random Sampling. O eixo vertical mostra F1.

Cada linha representa um atraso. Os painéis separam os modelos.

Uso principal: analisar o compromisso entre custo real de rotulagem e
desempenho, além de procurar configurações de Pareto: mais F1 com menos rótulos.

### 5.5. Falsos positivos e falsos negativos

Arquivos:

```text
errors_experiment_A.png
errors_experiment_B.png
```

Cada figura possui dois painéis:

- quantidade média de falsos positivos;
- quantidade média de falsos negativos.

As barras verticais representam o desvio-padrão. Esses gráficos são importantes
porque dois modelos com F1 semelhante podem ter comportamentos operacionais
muito diferentes.

### 5.6. Evolução prequencial

Arquivo por modelo:

```text
<MODELO>_prequential_selected.png
```

É gerado apenas com `--include-prequential`. Possui três painéis temporais:

- F1 por janela;
- precisão por janela;
- recall por janela.

Por padrão são mostradas quatro configurações representativas do Experimento C:

| Atraso | Orçamento |
|---:|---:|
| 0% | 100% |
| 1% | 30% |
| 5% | 10% |
| 10% | 1% |

Essas combinações podem ser substituídas:

```bash
python plot_classification_labeling_results.py \
  --include-prequential \
  --prequential-configs 0:100 0:10 5:30 10:100
```

O formato de cada valor é:

```text
ATRASO:ORCAMENTO
```

### 5.7. FP/FN, ataques e dinâmica dos rótulos

Este diagnóstico é gerado durante o experimento com:

```bash
python run_classification_labeling.py \
  --categories Adaptação \
  --sizes 25 \
  --models ARF \
  --experiments A B C \
  --plots
```

Cada configuração produz uma imagem em:

```text
output/ClassificationLabeling/plots/stream/
└── <binary|multiclass>/<cenário de features>/<dataset>/<execução>/<modelo>/
    └── experiment_<A|B|C>_delay_<D>pct_labels_<B>pct_FP_FN_Labeling.png
```

A figura possui três painéis temporais:

1. FP médio por janela e seu desvio-padrão;
2. FN médio por janela e seu desvio-padrão;
3. rótulos consultados, entregues e pendentes por janela.

As famílias de ataque são obtidas dos metadados originais preservados pelo
processador. O classificador aprende `BENIGN/ATTACK`, mas as faixas continuam
identificando DNS, Syn e demais tipos reais. O título informa o modo de treino.

As faixas coloridas mostram as regiões e os tipos de ataque. A área cinza
hachurada identifica o treinamento inicial. Quando existe atraso, uma linha
pontilhada marca o primeiro instante em que um rótulo consultado imediatamente
após o treinamento poderia retornar.

O título registra experimento, atraso percentual e em instâncias, orçamento
nominal, consulta efetiva, tamanho do treinamento inicial, janela, número de
execuções, total consultado, total entregue durante o fluxo e total liberado
depois do fim. Diferentemente dos gráficos acumulados `errors_experiment_A/B`,
esta figura mostra em qual trecho do fluxo cada erro ocorreu.

## 6. Suavização prequencial

As métricas prequenciais podem variar muito entre janelas, principalmente em
cenários raros. O programa aplica média móvel causal de cinco janelas:

```text
valor_suavizado[t] = média das últimas 5 janelas até t
```

Não são usadas janelas futuras. O tamanho pode ser alterado:

```bash
python plot_classification_labeling_results.py \
  --include-prequential \
  --smooth 10
```

Para desabilitar a suavização:

```bash
python plot_classification_labeling_results.py \
  --include-prequential \
  --smooth 1
```

## 7. Figuras individuais e agregadas

O argumento `--scope` possui três valores.

### Tudo

```bash
--scope all
```

Gera figuras individuais e agregadas. É o padrão.

### Somente datasets individuais

```bash
--scope individual
```

Preserva os valores de cada cenário separadamente.

### Somente agregados

```bash
--scope aggregate
```

Gera uma síntese entre datasets, útil para uma visão global e mais rápida.

### 7.1. Como a agregação é calculada

A agregação é uma média macro não ponderada:

```text
métrica agregada = soma da métrica de cada dataset / número de datasets
```

Cada dataset tem o mesmo peso, independentemente de seu número de instâncias.

Quando existem várias execuções do mesmo dataset, elas são promediadas antes da
média global. O desvio-padrão das figuras agregadas representa dispersão entre
datasets, não dispersão entre repetições do modelo.

Essa diferença é importante:

- figura individual: faixa ou barra = desvio entre as repetições daquele
  experimento;
- figura agregada: faixa ou barra = desvio entre datasets.

As médias são calculadas separadamente para cada modo de treinamento. Mesmo
com `--exec-id all`, resultados binários e multiclasse não entram na mesma média.

## 8. Estrutura de saída

Exemplo:

```text
output/ClassificationLabeling/plots/
├── manifest.csv
├── individual/
│   └── <binary|multiclass>/Default_FullFeatures/
│       └── Adaptacao_25/
│           └── 20261001_101354/
│               └── models_LB-HAT-ARF-HT_<hash>/
│                   ├── performance_experiment_A.png
│                   ├── performance_experiment_B.png
│                   ├── heatmap_experiment_C_f1.png
│                   ├── cost_performance_experiment_C.png
│                   ├── errors_experiment_A.png
│                   └── errors_experiment_B.png
├── aggregated/
│   └── <binary|multiclass>/Default_FullFeatures/
│       └── datasets_12_models_LB-HAT-ARF-HT_<hash>/
│           ├── performance_experiment_A.png
│           ├── performance_experiment_B.png
│           ├── heatmap_experiment_C_f1.png
│           ├── cost_performance_experiment_C.png
│           ├── errors_experiment_A.png
│           └── errors_experiment_B.png
└── prequential/
    └── <binary|multiclass>/Default_FullFeatures/
        └── Adaptacao_25/
            └── 20261001_101354/
                ├── LB_prequential_<configurações>_smooth5.png
                ├── HAT_prequential_<configurações>_smooth5.png
                ├── ARF_prequential_<configurações>_smooth5.png
                └── HT_prequential_<configurações>_smooth5.png
```

Os acentos são removidos somente dos nomes de diretórios e arquivos. Títulos e
rótulos internos permanecem em português.

O sufixo hexadecimal `<hash>` identifica o conjunto selecionado de datasets,
modelos e execuções. Dessa forma, uma execução posterior com filtros não
sobrescreve os gráficos completos.

## 9. Manifesto

Cada execução cria `manifest.csv`. Ele usa `;` como separador e registra:

Além da identificação dos arquivos, os novos registros incluem
`Training_Label_Mode` e `Evaluation_Label_Mode`. Registros antigos já existentes
no manifesto permanecem intactos.

- `File`: caminho da imagem;
- `Plot_Type`: tipo do gráfico;
- `Dataset`: dataset ou `AGGREGATED`;
- `Scenario`: conjunto de atributos;
- `Exec_ID`: execução ou `macro`;
- `Model`: modelo específico nas curvas prequenciais ou `ALL`.

O manifesto permite conferir quantas figuras foram geradas e automatizar sua
inclusão em relatórios. Quando já existe, novas entradas são combinadas com as
anteriores e caminhos repetidos são atualizados, não duplicados.

## 10. Filtros

### Dataset

```bash
python plot_classification_labeling_results.py \
  --datasets Adaptação_25 Consistência_25
```

Os nomes precisam corresponder à coluna `Dataset`.

### Modelo

Abreviações e nomes completos são aceitos:

```bash
python plot_classification_labeling_results.py --models HT HAT
```

Equivalências:

| Abreviação | Nome no CSV |
|---|---|
| LB | `LeveragingBagging` |
| HAT | `HoeffdingAdaptiveTree` |
| ARF | `AdaptiveRandomForest` |
| HT | `HoeffdingTree` |

### Cenário de atributos

```bash
python plot_classification_labeling_results.py \
  --scenarios Default_FullFeatures
```

Os filtros são aplicados aos CSVs cumulativos e prequenciais.

## 11. Formatos e resolução

PNG em 300 DPI é o padrão:

```bash
python plot_classification_labeling_results.py --formats png --dpi 300
```

Gerar simultaneamente PNG, PDF e SVG:

```bash
python plot_classification_labeling_results.py \
  --formats png pdf svg \
  --dpi 300
```

Recomendações:

- PNG: visualização rápida, apresentação e README;
- PDF: artigos e documentos LaTeX;
- SVG: edição vetorial em Inkscape ou Illustrator.

## 12. Diretórios personalizados

```bash
python plot_classification_labeling_results.py \
  --input-dir output/ClassificationLabeling \
  --output-dir output/FigurasArtigo
```

O diretório de saída é criado automaticamente.

## 13. Todos os argumentos

| Argumento | Padrão | Função |
|---|---|---|
| `--input-dir` | `output/ClassificationLabeling` | diretório dos CSVs |
| `--output-dir` | `<input-dir>/plots` | diretório das figuras |
| `--exec-id` | `latest` | seleciona execução |
| `--datasets` | todos | filtra datasets |
| `--models` | todos | filtra classificadores |
| `--scenarios` | todos | filtra conjuntos de atributos |
| `--training-label-mode` | todos, separados | filtra `binary` ou `multiclass` |
| `--scope` | `all` | individual, aggregate ou ambos |
| `--formats` | `png` | formatos das imagens |
| `--dpi` | 300 | resolução raster |
| `--include-prequential` | desabilitado | gera séries temporais |
| `--prequential-configs` | quatro pares | escolhe curvas temporais |
| `--smooth` | 5 | média móvel em janelas |
| `--dry-run` | desabilitado | mostra plano sem escrever |

Ajuda da CLI:

```bash
python plot_classification_labeling_results.py --help
```

## 14. Funcionamento interno do código

### `load_results`

Descobre arquivos, lê com pandas, valida as colunas, converte `Exec_ID` para
texto, registra o arquivo de origem, seleciona execuções e remove linhas
duplicadas pela chave experimental.

As chaves cumulativas são:

```text
Exec_ID + Dataset + Experiment + Model + Scenario
+ Training_Label_Mode + Delay_Percentage + Label_Budget_Percentage
```

No prequencial, `Window_Index` também faz parte da chave.

### `_select_executions`

Implementa `latest`, `all` e a seleção explícita de um `Exec_ID`.

### `filter_results`

Aplica os filtros. Se um valor não existe, informa os valores disponíveis em vez
de produzir silenciosamente um conjunto vazio.

### `aggregate_results`

Realiza duas etapas:

1. combina execuções repetidas dentro de cada dataset;
2. calcula média e desvio-padrão entre datasets.

Esse procedimento impede sobrepeso causado por arquivos duplicados.

### `plot_performance`

É reutilizada pelos Experimentos A e B. A diferença é somente o eixo horizontal:
atraso em A e orçamento em B.

### `plot_heatmap_c`

Constrói a matriz com `pivot_table`, ordena atrasos e orçamentos e usa a mesma
escala de cores em todos os modelos.

### `plot_cost_performance`

Usa o orçamento efetivamente observado no eixo horizontal e cria uma curva para
cada atraso.

### `plot_errors`

Produz os painéis de FP e FN com barras de erro.

### `plot_prequential`

Filtra o Experimento C, localiza os pares atraso/orçamento solicitados, ordena as
janelas por `Instance` e aplica média móvel causal.

### `save_figure`

Cria diretórios, grava todos os formatos pedidos e fecha a figura para liberar
memória. Isso é importante ao gerar dezenas de imagens.

### `generate_individual_plots`

Agrupa por `Exec_ID`, `Dataset`, `Scenario` e `Training_Label_Mode`, isolando
cada execução e protocolo.

### `generate_aggregate_plots`

Calcula a média macro e gera uma síntese por `Scenario` e `Training_Label_Mode`.

### `generate_prequential_plots`

Agrupa por execução, dataset, cenário, modelo e modo de treinamento. Cada figura compara somente as
configurações prequenciais escolhidas.

## 15. Cuidados de interpretação

### F1 zero em janela sem ataque

Uma janela com apenas tráfego benigno pode possuir:

```text
F1 = 0, FP = 0 e FN = 0
```

Isso acontece porque o F1 da classe positiva é definido como zero quando não há
ataques reais nem previstos. Não representa necessariamente falha do modelo.

### Orçamento nominal e efetivo

Random Sampling usa probabilidade, não uma quantidade fixa. Por isso, o gráfico
de custo utiliza o percentual efetivo, enquanto os gráficos B e C usam o
percentual nominal da configuração.

### Não somar A, B e C como observações independentes

Algumas configurações são equivalentes entre experimentos. O gerador mantém
cada experimento separado e nunca concatena A, B e C para calcular uma única
média de desempenho.

### Média agregada não é ponderada pelo tamanho

Todos os datasets possuem o mesmo peso. Para uma média por instância seria
necessário recalcular métricas a partir das predições individuais, que não estão
nos CSVs consolidados.

### Prequencial não mostra automaticamente regiões de ataque

Os CSVs prequenciais não armazenam a classe de cada instância. Por isso, o novo
gerador não sombreia ondas de ataque. Essa funcionalidade exigiria carregar os
datasets originais ou exportar as regiões junto aos resultados.

## 16. Testes

Os testes verificam:

- interpretação dos pares atraso/orçamento;
- normalização de nomes com acentos;
- escolha da execução mais recente;
- erro para CSV sem colunas obrigatórias;
- média macro com peso igual por dataset;
- criação real de uma imagem de heatmap.

Executar:

```bash
python -m unittest discover -s tests -v
```

Para tratar avisos futuros do pandas como erros:

```bash
python -W error::FutureWarning -m unittest discover -s tests -v
```

## 17. Exemplos completos

### Apenas o dataset Adaptação 25

```bash
python plot_classification_labeling_results.py \
  --datasets Adaptação_25
```

### Apenas HT e HAT, com prequencial

```bash
python plot_classification_labeling_results.py \
  --models HT HAT \
  --include-prequential
```

### Figuras agregadas para artigo

```bash
python plot_classification_labeling_results.py \
  --scope aggregate \
  --formats png pdf \
  --dpi 300
```

### Execução completa

```bash
python plot_classification_labeling_results.py \
  --exec-id latest \
  --scope all \
  --formats png \
  --dpi 300 \
  --include-prequential \
  --smooth 5
```
