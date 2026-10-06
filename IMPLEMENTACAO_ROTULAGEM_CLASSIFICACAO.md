# Implementação de Active Learning e Atraso de Rótulos

## 1. Objetivo deste documento

Este documento explica como foram implementados:

- o treinamento inicial totalmente supervisionado;
- a ausência de rótulos durante a avaliação;
- o Random Sampling;
- o atraso na chegada dos rótulos;
- a combinação das duas estratégias;
- a avaliação prequencial;
- a integração com os classificadores do CapyMOA.

A implementação principal está em
[`src/Classification/Labeling.py`](src/Classification/Labeling.py). O executor
de linha de comando está em
[`run_classification_labeling.py`](run_classification_labeling.py).

O trabalho descrito aqui considera somente classificação. O pipeline de
detecção de anomalias não participa desses experimentos.

## 2. Resposta direta: isso é nativo do CapyMOA?

Não. O programa não utiliza uma implementação nativa do CapyMOA para active
learning ou delayed labeling.

O CapyMOA fornece:

- `NumpyStream`, usado para representar o fluxo;
- as instâncias do fluxo;
- Leveraging Bagging, Hoeffding Adaptive Tree, Adaptive Random Forest e
  Hoeffding Tree;
- `learner.predict(instance)`, usado para prever;
- `learner.train(instance)`, usado para atualizar o classificador.

O código deste projeto implementa:

- a grade dos Experimentos A, B e C;
- a identificação do treinamento inicial;
- o sorteio das instâncias que recebem rótulo;
- a ausência de atualização para instâncias não selecionadas;
- a fila de rótulos atrasados;
- a entrega causal desses rótulos;
- a contabilidade dos rótulos entregues e pendentes;
- as repetições, sementes, métricas e exportação.

Em termos simples, o CapyMOA implementa **como o modelo aprende**. O runner do
projeto implementa **quando o modelo pode aprender**.

## 3. Divisão de responsabilidades

| Responsabilidade | Responsável |
|---|---|
| Ler uma instância do fluxo | CapyMOA |
| Representar features e classe | CapyMOA |
| Realizar a predição | Classificador CapyMOA |
| Atualizar árvore ou ensemble | Classificador CapyMOA |
| Decidir se o rótulo será solicitado | Código do projeto |
| Impedir atualização sem rótulo | Código do projeto |
| Calcular o momento de chegada | Código do projeto |
| Guardar rótulos futuros | Código do projeto |
| Entregar rótulos atrasados | Código do projeto |
| Calcular métricas | Código do projeto e scikit-learn |
| Exportar resultados | Código do projeto e pandas |

Os classificadores são criados em
[`src/Classification/Models.py`](src/Classification/Models.py).

## 4. Visão geral do fluxo

```text
CSV do cenário
      │
      ▼
DataStreamProcessor
      │
      ├── limpeza
      ├── imputação
      ├── normalização
      └── codificação da classe
      │
      ▼
NumpyStream do CapyMOA
      │
      ▼
Treinamento inicial supervisionado
      │
      ▼
Avaliação prequencial
      │
      ├── entregar rótulos antigos
      ├── prever instância atual
      ├── registrar resultado
      ├── executar Random Sampling
      └── treinar agora ou agendar rótulo
      │
      ▼
Métricas cumulativas e por janela
      │
      ▼
CSVs e gráficos
```

## 5. Criação do fluxo CapyMOA

O executor lê o CSV:

```python
dataframe = pd.read_csv(dataset_path)
```

Depois utiliza `DataStreamProcessor.create_stream`:

```python
stream, targets, features = processor.create_stream(
    df=dataframe,
    target_label_col="Label",
    binary_label=False,
    normalize_method="MinMaxScaler",
    return_stream=True,
    imputation_method="mediana",
)
```

No final do processamento, é criado um `NumpyStream`:

```python
stream_obj = NumpyStream(
    final_x_array,
    y,
    target_name="Class",
    feature_names=feature_names,
    target_type="categorical",
)
```

Cada chamada a:

```python
instance = stream.next_instance()
```

retorna uma instância com features e classe verdadeira. O índice da classe pode
ser acessado por:

```python
instance.y_index
```

## 6. O significado de “ausência de rótulo”

Os experimentos são realizados offline sobre datasets completamente rotulados.
Logo, o avaliador precisa conhecer a classe verdadeira para calcular F1,
precisão, recall, FP e FN.

A ausência de rótulo é simulada do ponto de vista do classificador:

- o avaliador conhece a classe verdadeira;
- o modelo realiza a predição sem usar essa classe;
- somente uma instância selecionada pode ser enviada a `learner.train`;
- uma instância não selecionada nunca atualiza o modelo.

Portanto, o rótulo não é apagado fisicamente do objeto CapyMOA. Seu acesso para
treinamento é controlado pelo runner.

Essa separação é comum em avaliações offline:

```text
Verdade conhecida pelo avaliador ≠ verdade disponível ao modelo
```

O método `predict` do classificador utiliza as features para realizar a
predição. O rótulo da instância somente é usado pelo modelo quando o runner chama
`train`.

## 7. Configuração experimental

Cada combinação é representada por:

```python
@dataclass(frozen=True)
class LabelingExperimentConfig:
    experiment: str
    delay_fraction: float
    label_probability: float
```

### `delay_fraction`

Fração do tamanho total do cenário usada para calcular o atraso:

```text
0,00 → 0%
0,01 → 1%
0,05 → 5%
0,10 → 10%
```

### `label_probability`

Probabilidade de cada instância de avaliação ser selecionada:

```text
0,01 → 1%
0,03 → 3%
0,05 → 5%
0,10 → 10%
0,30 → 30%
1,00 → 100%
```

As validações impedem experimentos desconhecidos e probabilidades fora do
intervalo de zero a um.

## 8. Treinamento inicial supervisionado

Os três experimentos começam da mesma forma. O primeiro bloco benigno e a
primeira região de ataque são usados como treinamento inicial.

Durante essa fase:

- todas as classes estão disponíveis;
- não existe Random Sampling;
- não existe atraso;
- todas as instâncias atualizam o modelo imediatamente;
- essas instâncias não participam das métricas.

O laço executa:

```python
if stream_index < initial_training_end:
    learner.train(instance)
    initial_training_instances += 1
    continue
```

O `continue` impede que a instância de treinamento inicial seja prevista ou
submetida às regras de rotulagem.

### 8.1. Identificação da primeira região de ataque

O método `_find_initial_training_end` faz uma leitura preliminar do fluxo. Ele:

1. exige que a primeira instância seja benigna;
2. encontra a primeira instância de ataque;
3. registra a última instância de ataque da região;
4. considera pequenos intervalos benignos como parte da mesma região;
5. encerra a região quando a distância entre ataques supera a tolerância.

A condição de encerramento é:

```python
index - last_attack_index > attack_gap_tolerance
```

O padrão é:

```python
attack_gap_tolerance = 1000
```

Isso foi necessário porque a ordenação temporal dos cenários pode inserir
instâncias benignas dentro de uma mesma onda conceitual de ataque.

Depois de localizar o limite, o fluxo é reiniciado para a execução real.

## 9. Avaliação prequencial

Após o treinamento inicial, é utilizado o protocolo teste-depois-treino:

```text
1. prever a instância;
2. registrar a predição;
3. obter ou não o rótulo;
4. atualizar o modelo quando permitido.
```

O modelo nunca é treinado com a instância atual antes de tentar classificá-la.

Isso preserva a causalidade e impede que uma instância seja avaliada depois de
já ter sido vista pelo classificador.

## 10. Random Sampling

O active learning implementado é Random Sampling de Bernoulli. Para cada
instância da região de avaliação, é realizado um sorteio independente:

```python
selected_for_labeling = (
    label_probability >= 1.0
    or rng.random() < label_probability
)
```

### Exemplo com orçamento de 10%

```text
Probabilidade configurada = 0,10

Instância 1: número 0,72 → não recebe rótulo
Instância 2: número 0,04 → recebe rótulo
Instância 3: número 0,31 → não recebe rótulo
Instância 4: número 0,09 → recebe rótulo
```

A regra é:

```text
número aleatório < probabilidade configurada
```

### 10.1. O orçamento é esperado, não exato

Uma probabilidade de 10% não seleciona obrigatoriamente dez de cada cem
instâncias. Ela produz aproximadamente 10% em fluxos grandes.

Por isso o resultado armazena:

- `Label_Budget_Percentage`: orçamento nominal;
- `Effective_Query_Percentage_avg`: percentual realmente selecionado.

### 10.2. Instância não selecionada

Quando o sorteio falha:

- a predição continua nas métricas;
- a instância não é treinada imediatamente;
- a instância não entra na fila de atraso;
- o modelo nunca recebe seu rótulo.

Ausência de rótulo não é implementada como “atraso infinito”. A instância é
simplesmente descartada para fins de treinamento.

## 11. Cálculo do atraso

O atraso proporcional é convertido para número de instâncias:

```python
delay_instances = round(total_instances * delay_fraction)
```

Formalmente:

```text
D = round(N × d)
```

Onde:

- `N` é o número total de instâncias do cenário;
- `d` é a fração de atraso;
- `D` é o atraso em instâncias.

Exemplo:

```text
N = 15.000
d = 0,05

D = round(15.000 × 0,05)
D = 750 instâncias
```

O cálculo usa o cenário completo, incluindo a região de treinamento inicial.
Isso segue o protocolo definido para os experimentos.

## 12. Fila de rótulos atrasados

A estrutura usada é `collections.deque`:

```python
pending_labels = deque()
```

Cada entrada possui:

```text
(índice de entrega, instância original)
```

Quando uma instância selecionada aparece na posição `t`, é adicionada assim:

```python
pending_labels.append(
    (stream_index + delay_instances, instance)
)
```

Exemplo:

```text
posição atual = 5.000
atraso = 750
posição de entrega = 5.750
```

A fila recebe:

```text
(5750, instância_da_posição_5000)
```

O objeto guardado contém as features e o rótulo verdadeiro originais.

### 12.1. Por que `deque`?

Como todas as instâncias usam o mesmo atraso dentro de uma configuração, as
entregas são inseridas em ordem cronológica. `deque` permite retirar o primeiro
elemento em tempo constante, sem deslocar toda a lista.

## 13. Entrega causal dos rótulos

Antes da predição atual, o runner verifica quais rótulos ficaram disponíveis:

```python
while pending_labels and pending_labels[0][0] <= stream_index:
    _, delayed_instance = pending_labels.popleft()
    learner.train(delayed_instance)
    delivered_during_stream += 1
```

Um rótulo da posição `t` com atraso `D` fica disponível antes da predição da
posição `t + D`.

### Exemplo com atraso de três instâncias

```text
Posição 100
    entrega rótulos antigos
    prevê a instância 100
    registra a predição
    seleciona a instância 100
    agenda seu rótulo para 103

Posição 101
    entrega rótulos agendados para 101
    prevê a instância 101

Posição 102
    entrega rótulos agendados para 102
    prevê a instância 102

Posição 103
    entrega o rótulo da instância 100
    treina com a instância 100
    prevê a instância 103
```

A predição da posição 100 não utiliza o próprio rótulo.

## 14. Caso especial: atraso zero

Com atraso zero, a fila não é necessária:

```python
if delay_instances == 0:
    learner.train(instance)
```

Ainda assim, `train` é chamado somente depois de `predict`:

```text
prever → registrar → selecionar → treinar
```

Portanto, atraso zero e orçamento de 100% representam o protocolo prequencial
totalmente supervisionado.

## 15. Atraso e ausência combinados

No Experimento C, o fluxo decisório é:

```text
Instância atual
      │
      ▼
Realizar predição
      │
      ▼
Executar Random Sampling
      │
      ├── não selecionada
      │       └── nunca atualiza o modelo
      │
      └── selecionada
              │
              ├── atraso = 0
              │       └── treina depois da predição
              │
              └── atraso > 0
                      └── entra na fila para t + D
```

Somente instâncias selecionadas entram na fila de atraso.

## 16. Ordem completa por instância

Depois do treinamento inicial, cada instância segue exatamente esta ordem:

```text
┌──────────────────────────────────────────┐
│ 1. Ler a próxima instância               │
├──────────────────────────────────────────┤
│ 2. Entregar rótulos antigos vencidos     │
├──────────────────────────────────────────┤
│ 3. Prever a instância atual               │
├──────────────────────────────────────────┤
│ 4. Registrar y_true e y_pred              │
├──────────────────────────────────────────┤
│ 5. Executar Random Sampling               │
├──────────────────────────────────────────┤
│ 6. Se selecionada:                        │
│    - treinar agora se D = 0               │
│    - agendar para t + D se D > 0          │
├──────────────────────────────────────────┤
│ 7. Fechar a janela quando necessário      │
└──────────────────────────────────────────┘
```

Essa lógica está no método `prequential_test`.

## 17. Pseudocódigo completo

```python
calcular tamanho do cenário
calcular atraso em instâncias
encontrar fim do treinamento inicial
reiniciar o fluxo
criar fila vazia

para cada índice t:
    instância = próxima instância

    se t pertence ao treinamento inicial:
        treinar imediatamente
        continuar

    enquanto existir rótulo com entrega <= t:
        retirar rótulo da fila
        treinar com a instância antiga

    predição = prever instância atual
    registrar predição e classe verdadeira

    selecionada = sorteio < probabilidade

    se selecionada:
        se atraso == 0:
            treinar com a instância atual
        senão:
            adicionar (t + atraso, instância) à fila

    se uma janela terminou:
        calcular métricas da janela

registrar quantidade restante na fila
entregar rótulos restantes após o fluxo
calcular métricas cumulativas
```

## 18. Rótulos pendentes no final

Algumas entregas podem estar programadas para depois da última instância.

Exemplo:

```text
último índice = 15.000
instância selecionada = 14.900
atraso = 750
entrega prevista = 15.650
```

O programa registra quantos rótulos ainda estavam pendentes:

```python
pending_at_stream_end = len(pending_labels)
```

Depois, por padrão, entrega todos:

```python
while pending_labels:
    _, delayed_instance = pending_labels.popleft()
    learner.train(delayed_instance)
```

Esses rótulos aparecem em:

```text
Flushed_After_Stream_avg
```

Eles não alteram as métricas, porque não existem previsões posteriores. O
treinamento final serve para completar a contabilidade e deixar o modelo em um
estado no qual todos os rótulos selecionados foram consumidos.

A relação esperada é:

```text
Queried = Delivered_During_Stream + Flushed_After_Stream
```

## 19. Experimentos A, B e C

### Experimento A — somente atraso

```text
probabilidade de rótulo = 100%
atrasos = 0%, 1%, 5% e 10%
```

Todas as instâncias de avaliação são rotuladas. Apenas o momento da atualização
muda.

### Experimento B — somente ausência

```text
atraso = 0%
probabilidades = 1%, 3%, 5%, 10%, 30% e 100%
```

Uma instância selecionada atualiza o modelo imediatamente depois da própria
predição. Uma instância não selecionada nunca o atualiza.

### Experimento C — combinação

```text
4 atrasos × 6 probabilidades = 24 configurações
```

Cada instância precisa primeiro ser selecionada. Se for selecionada, seu rótulo
é entregue imediatamente ou no futuro, conforme o atraso.

## 20. Multiclasse no treinamento e binário na avaliação

O fluxo é criado com:

```python
binary_label=False
```

Os classificadores aprendem as classes originais:

```text
0 = BENIGN
1 = tipo de ataque A
2 = tipo de ataque B
3 = tipo de ataque C
...
```

Para as métricas, o resultado é convertido para normal contra ataque:

```python
y_true.append(0 if is_normal else 1)
y_pred.append(0 if prediction == normal_class_idx else 1)
```

Consequências:

- o aprendizado do modelo é multiclasse;
- F1, precisão, recall, FP e FN são binários;
- prever o tipo errado de ataque ainda conta como detecção de ataque;
- as métricas atuais não avaliam a identificação exata da família do ataque.

## 21. Métricas por janela

O tamanho padrão da janela é 100. A cada 100 instâncias avaliadas, são
calculados:

- F1;
- precisão;
- recall;
- FP;
- FN.

Se o número de instâncias não for múltiplo de 100, uma janela parcial é gerada.

Exemplo:

```text
1.050 instâncias avaliadas

10 janelas de 100
1 janela de 50
```

`Window_Instances` registra o tamanho real da janela.

As métricas cumulativas usam todas as previsões da região de avaliação.

## 22. Repetições e sementes

Cada configuração pode ser executada várias vezes. A semente da repetição é:

```python
run_seed = random_seed + run_index
```

Com semente inicial 42 e cinco repetições:

```text
repetição 1 → 42
repetição 2 → 43
repetição 3 → 44
repetição 4 → 45
repetição 5 → 46
```

A semente é usada tanto no modelo quanto no Random Sampling. Isso permite
reproduzir a seleção das instâncias e a inicialização de modelos aleatórios.

Para orçamentos menores que 100%, a mesma sequência pseudoaleatória é usada em
uma determinada repetição. Consequentemente, as seleções de orçamentos menores
tendem a ser subconjuntos das seleções maiores.

## 23. Agregação das repetições

Depois das repetições, `_aggregate_runs` calcula:

- média e desvio-padrão das métricas cumulativas;
- média e desvio-padrão de cada janela;
- média e desvio-padrão do número de consultas;
- percentual efetivo de consultas;
- rótulos entregues durante o fluxo;
- rótulos entregues depois do fluxo;
- tempo de execução.

O desvio-padrão usa `numpy.std` com `ddof=0`, ou seja, desvio populacional.

## 24. Cache de configurações equivalentes

Existem 34 configurações lógicas:

```text
Experimento A = 4
Experimento B = 6
Experimento C = 24
Total lógico = 34
```

Porém, A e B contêm combinações que também existem em C. Existem apenas 24
pares únicos de atraso e orçamento.

O cache usa:

```python
cache_key = (
    config.delay_fraction,
    config.label_probability,
)
```

Quando uma combinação já foi executada, seu resultado é reutilizado. Isso evita
retreinar o mesmo modelo, mas mantém linhas separadas nos CSVs para A, B e C.

## 25. Campos de auditoria nos resultados

Além das métricas, os CSVs armazenam campos que permitem verificar a lógica de
rotulagem:

| Campo | Significado |
|---|---|
| `Delay_Percentage` | atraso nominal |
| `Delay_Instances` | atraso convertido em instâncias |
| `Label_Budget_Percentage` | orçamento nominal |
| `Initial_Training_Instances` | tamanho do treinamento inicial |
| `Evaluation_Instances` | quantidade avaliada |
| `Queried_avg` | quantidade média selecionada |
| `Effective_Query_Percentage_avg` | percentual realmente selecionado |
| `Delivered_During_Stream_avg` | rótulos entregues durante a avaliação |
| `Flushed_After_Stream_avg` | rótulos entregues depois do final |
| `Runs` | número de repetições |

Esses campos foram usados para validar:

```text
Queried ≈ Delivered_During_Stream + Flushed_After_Stream
```

## 26. Exemplo completo

Considere:

```text
cenário = 15.000 instâncias
fim do treinamento inicial = 3.775
atraso = 5% = 750 instâncias
orçamento = 10%
```

Na posição 4.000:

1. são entregues rótulos antigos agendados para 4.000;
2. o modelo prevê a classe da instância 4.000;
3. a previsão é registrada;
4. o Random Sampling gera 0,07;
5. `0,07 < 0,10`, logo a instância é selecionada;
6. o rótulo é agendado para 4.750;
7. o modelo ainda não é treinado com a instância 4.000.

Na posição 4.750:

1. a instância original de 4.000 é retirada da fila;
2. `learner.train(instancia_4000)` é executado;
3. o modelo prevê a instância 4.750.

Se o sorteio da posição 4.000 tivesse produzido 0,45, a instância nunca seria
treinada.

## 27. Testes automatizados

Os testes estão em
[`tests/test_classification_labeling.py`](tests/test_classification_labeling.py).

### Grade experimental

Confirma:

```text
A = 4 configurações
B = 6 configurações
C = 24 configurações
Total = 34 configurações lógicas
```

### Cálculo do atraso

Confirma:

```text
15.000 × 1% = 150
15.000 × 5% = 750
15.000 × 10% = 1.500
```

### Treinamento inicial

Em um fluxo artificial, verifica que os índices iniciais são treinados e não
previstos.

### Teste-depois-treino

Com atraso zero e orçamento de 100%, verifica que todas as instâncias de
avaliação são previstas antes de serem treinadas.

### Causalidade do atraso

Com atraso de duas instâncias, verifica quantos rótulos chegam durante o fluxo e
quantos ficam para o final.

### Reprodutibilidade

Executa Random Sampling duas vezes com a mesma semente e exige índices
selecionados idênticos.

### Região inicial fragmentada

Verifica que pequenos espaços benignos não dividem incorretamente a primeira
onda de ataque.

Executar os testes:

```bash
python -m unittest discover -s tests -v
```

## 28. Limitações metodológicas

### 28.1. O avaliador possui todos os rótulos

Isso é necessário para calcular métricas offline. A indisponibilidade é aplicada
somente ao modelo.

### 28.2. Random Sampling não garante quantidade exata

As taxas representam probabilidades. O percentual realizado varia entre
repetições.

Se fosse necessário selecionar exatamente 10% do fluxo, seria preciso usar um
orçamento fixo e conhecer antecipadamente o tamanho da região de avaliação.

### 28.3. O atraso é medido em instâncias

Ele não representa segundos, minutos ou horas. Um atraso de 750 significa 750
novas instâncias processadas antes da entrega.

### 28.4. O treinamento inicial é identificado offline

O programa usa os rótulos para localizar a primeira região de ataque. Isso é
uma definição experimental, não um detector online do início da onda.

### 28.5. A normalização atual conhece o cenário completo

O `MinMaxScaler` é ajustado antes da criação do fluxo. Portanto, conhece os
limites globais dos atributos.

Isso não invalida a fila de rótulos, mas impede afirmar que todo o pipeline de
pré-processamento é estritamente causal. Uma versão totalmente online precisaria
de normalização incremental.

### 28.6. Métricas binárias não avaliam o tipo exato de ataque

O modelo é multiclasse, mas as métricas atuais medem normal contra ataque.

### 28.7. Rótulos entregues depois do final não afetam as métricas

Eles atualizam apenas o estado terminal do modelo. Não existem previsões futuras
para demonstrar o efeito desse treinamento.

## 29. Possíveis extensões

### Seleção por incerteza

Substituir o sorteio aleatório por uma regra baseada na confiança do modelo.
Isso exigiria probabilidades ou votos das classes.

### Orçamento fixo

Selecionar exatamente uma quantidade predefinida de instâncias em vez de uma
probabilidade de Bernoulli.

### Atraso variável

Sortear um atraso diferente para cada rótulo, simulando tempos de resposta de
analistas.

### Atraso em tempo real

Usar timestamps do dataset em vez de quantidade de instâncias.

### Fila com prioridade

Para atrasos variáveis, substituir `deque` por uma fila de prioridade ordenada
pelo instante de entrega.

### Métricas multiclasse

Adicionar macro-F1, weighted-F1 e matriz de confusão das famílias de ataque.

### Pré-processamento incremental

Substituir o ajuste global do `MinMaxScaler` por estatísticas atualizadas apenas
com dados já observados.

## 30. Resumo da implementação

A lógica central pode ser representada por:

```python
# Rótulos antigos ficam disponíveis antes da predição atual.
deliver_due_labels()

# Teste.
prediction = learner.predict(instance)
record_metrics(prediction, true_label)

# Active learning.
if rng.random() < label_probability:

    # Rótulo imediato.
    if delay_instances == 0:
        learner.train(instance)

    # Rótulo futuro.
    else:
        pending_labels.append(
            (current_index + delay_instances, instance)
        )
```

CapyMOA fornece os classificadores incrementais, o fluxo e as operações de
predição e treinamento. O runner deste projeto controla quando cada rótulo fica
disponível e quando o modelo pode ser atualizado.

## 31. Documentos relacionados

- [`METODOLOGIA_CLASSIFICACAO.md`](METODOLOGIA_CLASSIFICACAO.md): protocolo e
  justificativa experimental;
- [`GUIA_CODIGO_CLASSIFICACAO.md`](GUIA_CODIGO_CLASSIFICACAO.md): organização
  geral das classes e funções;
- [`GRAFICOS_CLASSIFICACAO.md`](GRAFICOS_CLASSIFICACAO.md): geração e
  interpretação dos gráficos;
- [`CLASSIFICATION_LABELING.md`](CLASSIFICATION_LABELING.md): resumo original da
  API de rotulagem.
