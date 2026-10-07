# Metodologia: Active Learning e Atraso de Rótulos na Classificação

## 1. Objetivo

Esta extensão avalia classificadores incrementais em fluxos de dados quando os
rótulos não estão sempre disponíveis imediatamente. O trabalho considera apenas
os métodos de classificação do projeto:

- Leveraging Bagging (LB);
- Hoeffding Adaptive Tree (HAT);
- Adaptive Random Forest (ARF);
- Hoeffding Tree (HT).

Foram implementados dois problemas comuns em aplicações reais:

1. **Atraso de rótulo (*delayed labeling*):** o rótulo verdadeiro existe, mas só
   fica disponível depois de certo número de instâncias.
2. **Aprendizado ativo (*active learning*):** somente uma parte das instâncias é
   escolhida para receber rótulo e atualizar o modelo.

Os mecanismos podem ser avaliados separadamente ou em conjunto.

## 2. Protocolo geral

Cada arquivo CSV representa um cenário ordenado temporalmente. O cenário é
convertido em um fluxo e processado uma única vez, do começo ao fim.

O protocolo possui duas fases:

1. **Treinamento inicial supervisionado:** usa o primeiro bloco benigno e a
   primeira região de ataque. Todas essas instâncias possuem rótulo imediato.
   Não há atraso nem seleção por orçamento nessa fase.
2. **Avaliação prequencial:** para cada instância restante, o modelo primeiro
   realiza a predição e depois pode ser atualizado quando o rótulo estiver
   disponível.

As instâncias do treinamento inicial não participam das métricas. Isso evita que
amostras já vistas no treinamento sejam contabilizadas como previsões do modelo.

### 2.1. Identificação do treinamento inicial

Os cenários reais possuem pequenas interrupções benignas dentro de uma mesma
onda de ataque. Por isso, a primeira região não é interpretada apenas como um
bloco estritamente contínuo de rótulos de ataque.

O algoritmo:

1. exige que o cenário comece com tráfego benigno;
2. encontra a primeira instância de ataque;
3. mantém ataques sucessivos na mesma região enquanto a distância entre seus
   índices não ultrapassa `attack_gap_tolerance`;
4. encerra o treinamento inicial na última instância de ataque dessa região.

O valor padrão é `attack_gap_tolerance=1000`, igual à convenção usada para as
regiões de ataque nos gráficos existentes. O limite é medido pela diferença
entre os índices de duas instâncias de ataque consecutivas.

O código interrompe a execução quando o cenário:

- não começa com um bloco benigno;
- não possui ataque depois do bloco benigno;
- termina dentro da primeira região de ataque e não deixa dados para avaliação.

## 3. Experimentos

### 3.1. Experimento A — somente atraso de rótulo

Todas as instâncias da fase de avaliação são selecionadas para rotulagem. O que
muda é o momento em que o rótulo chega.

| Atraso proporcional | Probabilidade de rotulagem |
|---:|---:|
| 0% | 100% |
| 1% | 100% |
| 5% | 100% |
| 10% | 100% |

Esse experimento possui 4 configurações.

### 3.2. Experimento B — somente aprendizado ativo

Não existe atraso. Cada instância possui a mesma probabilidade de ser escolhida
para rotulagem por **Random Sampling**.

| Atraso proporcional | Probabilidade de rotulagem |
|---:|---:|
| 0% | 1% |
| 0% | 3% |
| 0% | 5% |
| 0% | 10% |
| 0% | 30% |
| 0% | 100% |

A configuração de 100% representa o fluxo totalmente supervisionado, seguindo
a ordem teste-depois-treino. Esse experimento possui 6 configurações.

### 3.3. Experimento C — atraso e aprendizado ativo

Combina os quatro atrasos do Experimento A com as seis probabilidades do
Experimento B por produto cartesiano:

| Parâmetro | Valores |
|---|---|
| Atraso | 0%, 1%, 5% e 10% |
| Probabilidade de rotulagem | 1%, 3%, 5%, 10%, 30% e 100% |

Esse experimento possui 24 configurações.

No total existem 34 configurações lógicas: 4 de A, 6 de B e 24 de C. Algumas
são equivalentes entre experimentos. Quando A, B e C são executados juntos, o
código calcula somente as 24 combinações únicas e reutiliza os resultados. Os
CSVs continuam contendo as 34 identificações lógicas para permitir a análise
separada de cada experimento.

## 4. Cálculo do atraso

O atraso é proporcional ao tamanho total do cenário, incluindo o treinamento
inicial:

```text
D = round(N × percentual_de_atraso)
```

Onde `N` é o número total de instâncias e `D` é o atraso inteiro.

Exemplo para um cenário com 15.000 instâncias:

| Percentual | Atraso resultante |
|---:|---:|
| 0% | 0 instâncias |
| 1% | 150 instâncias |
| 5% | 750 instâncias |
| 10% | 1.500 instâncias |

Se uma instância selecionada aparece na posição `t`, seu rótulo com atraso `D`
fica disponível antes da predição da posição `t + D`. Para atraso zero, a
predição é feita primeiro e o treinamento ocorre logo depois.

Rótulos cujo prazo ultrapassa o fim do fluxo são aplicados somente depois da
avaliação. Eles são registrados em `Flushed_After_Stream`, mas não alteram as
métricas já calculadas.

## 5. Random Sampling

O aprendizado ativo usa amostragem de Bernoulli. Depois da predição de cada
instância, é gerado um número pseudoaleatório. A instância é solicitada quando:

```text
número_aleatório < probabilidade_de_rotulagem
```

Assim, 10% representa uma probabilidade de 0,10 para cada instância, não uma
quantidade fixa de exatamente 10% do fluxo. O percentual efetivamente
selecionado é salvo nos resultados.

Cada repetição utiliza uma semente determinística:

```text
semente_da_repetição = semente_base + índice_da_repetição
```

A mesma semente alimenta o modelo e a seleção aleatória. Isso torna os resultados
reproduzíveis. Para uma mesma repetição, orçamentos menores formam subconjuntos
dos maiores porque usam a mesma sequência aleatória.

## 6. Ordem causal por instância

Depois do treinamento inicial, cada posição `t` segue esta ordem:

1. entregar e treinar rótulos antigos cujo prazo chegou em `t`;
2. prever a classe da instância atual;
3. registrar rótulo real e predição para as métricas;
4. aplicar Random Sampling;
5. se a instância foi selecionada:
   - com atraso zero, treinar imediatamente depois da predição;
   - com atraso maior que zero, guardar a instância para `t + D`.

Essa ordem impede que o rótulo atual seja usado antes de sua própria predição.

## 7. Dados e pré-processamento

O executor lê os CSVs em `data/15k` por padrão. São esperadas as categorias
Adaptação, Consistência, Generalização e Recorrência, com cenários de 25, 200 e
1000 ataques.

O `DataStreamProcessor`:

1. remove identificadores e colunas ignoradas;
2. mantém somente atributos numéricos;
3. substitui valores infinitos por limites numéricos;
4. preenche valores ausentes com a mediana;
5. normaliza os atributos com `MinMaxScaler`;
6. codifica `BENIGN` como classe normal de índice zero;
7. cria um `NumpyStream` do CapyMOA.

O protocolo principal usa treinamento e avaliação binários: `0 = BENIGN` e
`1 = ATTACK`, agrupando todas as famílias de ataque no rótulo positivo. Isso
alinha a tarefa aprendida com as métricas de benigno contra ataque.

O executor usa `binary_label=True` e `preserve_label_metadata=True`. Os nomes e
índices das famílias originais ficam em `stream.original_target_names` e
`stream.original_label_indices`, apenas para anotar os gráficos. Não são
features nem rótulos adicionais entregues ao classificador. A fila de atraso
guarda instâncias com o rótulo binário, inclusive para o treinamento inicial.

Para estudar a formulação anterior, `--training-label-mode multiclass` mantém
o treinamento multiclasse com avaliação binária. Esse modo não é misturado nas
médias com o binário. Mudar o alvo pode alterar as previsões, mas não garante
melhora de F1; é necessário executar novamente a grade experimental.

Há duas opções de atributos:

- `full`: todos os atributos numéricos disponíveis;
- `selected`: conjunto fixo de 33 atributos definido no executor.

> **Observação metodológica:** imputação e normalização são ajustadas sobre o
> DataFrame completo antes da criação do fluxo. Logo, o `MinMaxScaler` conhece os
> limites globais do cenário. Uma avaliação com pré-processamento estritamente
> causal deverá usar um transformador incremental ajustado apenas ao passado.

## 8. Métricas

### 8.1. Cumulativas

Calculadas sobre toda a fase de avaliação:

- F1-score (%);
- precisão (%);
- recall (%);
- MCC;
- falsos positivos (FP);
- falsos negativos (FN);
- tempo de execução.

Com várias repetições, são armazenados média e desvio-padrão populacional.

### 8.2. Prequenciais

O fluxo é dividido em janelas consecutivas, com tamanho padrão de 100. Para cada
janela são calculados F1, precisão, recall, FP e FN. A última janela é mantida
mesmo quando é menor; `Window_Instances` informa seu tamanho real.

Também são auditados por janela:

- rótulos consultados pelo Random Sampling;
- rótulos entregues ao modelo naquela janela;
- rótulos que continuam pendentes ao final da janela.

## 9. Arquivos gerados

Por padrão, os resultados ficam em `output/ClassificationLabeling`:

- `*_cumulative.csv`: uma linha por experimento, configuração e modelo;
- `*_prequential.csv`: uma linha por janela, configuração e modelo.

Os arquivos usam `;` como separador. A opção `--plots` também gera gráficos de
F1, precisão e recall e uma figura detalhada com FP/FN, regiões e nomes dos
ataques e dinâmica da rotulagem. Esses gráficos ficam em
`output/ClassificationLabeling/plots/stream`.

Os dois CSVs incluem `Training_Label_Mode` (`binary` ou `multiclass`) e
`Evaluation_Label_Mode` (`binary`). Os nomes novos contêm `binaryTraining` ou
`multiclassTraining`, além do identificador da execução. Os arquivos históricos
permanecem intactos e continuam representando o treinamento multiclasse; não é
possível obter resultados binários apenas recalculando suas métricas.

## 10. Execução

Visualizar o plano sem treinar:

```bash
python run_classification_labeling.py --dry-run
```

Executar os três experimentos para um cenário:

```bash
python run_classification_labeling.py \
  --categories Consistência \
  --sizes 25 \
  --models LB HAT ARF HT \
  --experiments A B C \
  --feature-sets full \
  --n-runs 5
```

Executar todos os cenários com os padrões:

```bash
python run_classification_labeling.py
```

Listar opções:

```bash
python run_classification_labeling.py --help
```

## 11. Reprodutibilidade e testes

Os testes verificam grade experimental, atraso proporcional, treinamento
inicial, ordem teste-depois-treino, causalidade, reprodutibilidade do Random
Sampling, ataques fragmentados e cenários inválidos.

```bash
python -m unittest discover -s tests -v
```

## 12. Guia técnico

A explicação de classes, funções e estruturas internas está em
[GUIA_CODIGO_CLASSIFICACAO.md](GUIA_CODIGO_CLASSIFICACAO.md).

A mecânica detalhada da ausência e do atraso de rótulos está em
[IMPLEMENTACAO_ROTULAGEM_CLASSIFICACAO.md](IMPLEMENTACAO_ROTULAGEM_CLASSIFICACAO.md).
