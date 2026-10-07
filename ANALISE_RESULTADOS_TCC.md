# Análise dos resultados de classificação para o TCC

Data da análise: 7 de outubro de 2026.

Escopo: resultados históricos dos experimentos A, B e C, anteriores à execução do novo treinamento binário. A detecção de anomalias não faz parte desta análise. Nenhum novo experimento foi executado e nenhum resultado original foi alterado.

## 1. Parecer geral

Os arquivos analisados são internamente consistentes com o protocolo implementado: o atraso tem o tamanho esperado, a seleção aleatória de rótulos coincide com as sementes utilizadas, as contagens de rótulos fecham e os erros por janela somam os erros acumulados. Não encontrei, nessas verificações, evidência de falha na implementação de atraso ou Random Sampling.

Entretanto, consistência dos CSVs não equivale a validade metodológica completa. Há uma limitação importante: normalização e imputação utilizam estatísticas do cenário inteiro, incluindo instâncias futuras. Portanto, os resultados são úteis como análise exploratória e comparação sob um pré-processamento offline comum, mas não devem ser apresentados como uma avaliação ponta a ponta estritamente causal.

Os principais achados são:

1. A disponibilidade imediata de rótulos favorece fortemente o desempenho, sobretudo nos ataques curtos.
2. A proporção de atraso em relação ao tamanho total do cenário esconde diferenças grandes em relação à duração dos ataques.
3. Random Sampling com orçamento muito baixo frequentemente não consulta nenhum ataque nos cenários mais desbalanceados.
4. O efeito conjunto de atraso e orçamento não é simplesmente aditivo: mais atualizações atrasadas nem sempre melhoram o resultado.
5. Leveraging Bagging e Adaptive Random Forest apresentam os maiores F1 médios na referência totalmente supervisionada, mas nenhum vence em todos os cenários.
6. Os resultados existentes usam treinamento multiclasse e avaliação binária. Não são resultados do novo treinamento binário.

## 2. Material analisado e critérios de inclusão

Foram utilizados os 12 pares de arquivos `*_cumulative.csv` e `*_prequential.csv` da execução `20261001_101354`, localizados em:

`/home/tomas/repos/anomaly-and-classification-data-stream/output/ClassificationLabeling/`

Cada par corresponde a uma das quatro categorias — Consistência, Generalização, Adaptação e Recorrência — combinada com os tamanhos nominais 25, 200 e 1000.

Também foram consultados os rótulos dos cenários originais em `data/15k/`, o código da implementação e o histórico do Git. O código histórico em `e51942c` confirma o uso de treinamento multiclasse, MinMaxScaler, imputação por mediana, cinco repetições e sementes 42 a 46.

O conjunto principal contém:

| Item | Quantidade |
|---|---:|
| Cenários | 12 |
| Modelos | 4 |
| Condições lógicas A + B + C por cenário/modelo | 34 |
| Combinações únicas de atraso e orçamento | 24 |
| Linhas acumuladas | 1.632 |
| Linhas acumuladas sem duplicar condições equivalentes | 1.152 |
| Repetições por condição única | 5 |
| Execuções de modelo representadas pelas condições únicas | 5.760 |
| Linhas de métricas por janela, incluindo as duplicações A/B/C | 204.952 |

As 34 condições lógicas são quatro de A, seis de B e 24 de C. Como A e B são subconjuntos de C, o código reutiliza as condições equivalentes. Isso evita trabalho repetido, mas significa que os resultados compartilhados entre A, B e C não constituem evidências independentes.

O par adicional de `Consistência_25`, identificado por `20261001_091516`, foi excluído da agregação para não dar peso duplo ao cenário. Suas métricas e contagens coincidem com a execução mais recente; os tempos de execução podem diferir.

Todos os resultados principais são `Default_FullFeatures`, com 77 features. Aqui, Default significa a configuração fixa definida pelo projeto, não necessariamente todos os parâmetros padrão da biblioteca. Não há, nesta execução, comparação entre configuração otimizada e Default, nem entre todas as features e features selecionadas.

## 3. O que exatamente foi avaliado

### 3.1 Treinamento e avaliação

O histórico utiliza treinamento multiclasse: o modelo recebe BENIGN e as famílias de ataque presentes no cenário. Para calcular as métricas, tanto o rótulo verdadeiro quanto a previsão são convertidos em:

- 0: benigno;
- 1: qualquer ataque.

Consequentemente, um ataque Syn previsto como DrDoS_DNS conta como detecção correta de ataque. Essas métricas medem detecção binária por um classificador treinado de forma multiclasse; não medem a identificação correta da família.

O treino inicial contém o primeiro trecho benigno e a primeira região de ataque, integralmente rotulados e sem atraso. As métricas começam somente depois desse prefixo.

Após o treino inicial, o modelo prevê antes de receber o rótulo da própria instância. Com atraso zero, a atualização ocorre depois dessa previsão. Com atraso positivo, a instância original e seu rótulo são guardados na fila e utilizados na atualização quando o atraso vence.

O avaliador conhece todos os rótulos para medir os erros, mesmo quando o modelo não os recebe. Isso é uma simulação de disponibilidade de rótulos; não significa que o modelo tenha sido treinado com todos eles.

### 3.2 Significado do atraso e do orçamento

O atraso é calculado por:

`d = round(N × proporção_de_atraso)`

N inclui o treino inicial. O atraso é medido em posições do fluxo, não em segundos, minutos ou horas. A aplicação desse atraso restringe somente as atualizações posteriores ao treino inicial.

Random Sampling seleciona cada instância avaliada com probabilidade p. É uma seleção Bernoulli, não uma cota que obriga a consultar exatamente p% das instâncias. Por isso, o percentual efetivo fica próximo do nominal, mas não necessariamente igual.

As mesmas sementes são utilizadas entre modelos e condições. Para p menor que 100%, os mesmos números aleatórios geram conjuntos de consultas aninhados à medida que p aumenta. Isso melhora a comparabilidade, mas também cria dependência entre condições.

Random Sampling é uma referência não informada de consulta de rótulos: não prioriza incerteza, diversidade ou relevância. A literatura de active learning para fluxos também investiga essas escolhas informadas; este estudo ainda não compara Random Sampling com elas. [Ienco, Žliobaitė e Pfahringer, 2014](https://proceedings.mlr.press/v36/ienco14.html).

### 3.3 Agregação adotada nesta análise

Cada `F1_avg`, `Prec_avg` e `Rec_avg` já representa a média das métricas de cinco execuções novas do modelo, sobre o mesmo cenário. Os respectivos `*_std` usam desvio padrão populacional das cinco métricas, com `ddof=0`.

Nas tabelas globais deste relatório, foi calculada a média aritmética das médias dos 12 cenários: cada cenário tem o mesmo peso. Não se trata de F1 calculado após concatenar todas as previsões, nem de uma média ponderada pela quantidade de ataques.

F1, precisão e recall estão em porcentagem. Diferenças entre dois valores percentuais são apresentadas em pontos percentuais, abreviados como p.p. MCC está na escala de -1 a 1.

Precisão média e recall médio não devem ser inseridos na fórmula de F1 para reconstruir `F1_avg`: a média de uma função não linear geralmente difere da função aplicada às médias. O mesmo cuidado vale para F1 reconstruído a partir das contagens médias de FP/FN.

## 4. Auditoria de integridade

Foram realizados 398 checks automáticos sobre os resultados e os rótulos originais. Todos passaram. As verificações incluíram:

- quantidade total de instâncias e tamanho do prefixo de treino;
- cinco repetições por resultado;
- ausência de campos ausentes nas colunas originais dos resultados acumulados;
- limites válidos das métricas e das contagens de FP/FN;
- atraso compatível com o tamanho real de cada cenário;
- igualdade exata, dentro da tolerância numérica, entre A e C com orçamento de 100%, e entre B e C com atraso zero;
- consultas iguais à soma dos rótulos entregues durante o fluxo e dos entregues depois de seu término;
- percentual efetivo compatível com as consultas;
- consultas aleatórias reproduzidas pelas sementes 42 a 46, incluindo média e desvio padrão;
- quantidade de entregas durante o fluxo compatível com a seleção e o atraso;
- recall acumulado compatível com os positivos originais e os FN;
- soma dos FP/FN das janelas igual ao total acumulado;
- cobertura completa da avaliação pelas janelas, inclusive a última janela parcial;
- limites dos erros de cada janela compatíveis com os positivos e negativos daquela janela.

Essa auditoria indica coerência do processamento e dos arquivos. Ela não prova ausência de qualquer defeito, não recupera todas as previsões originais e não substitui uma avaliação da metodologia.

## 5. Caracterização real dos cenários

| Cenário | Total N | Treino inicial | Ataques no treino | Instâncias avaliadas | Ataques avaliados | Ataques na avaliação (%) |
|---|---:|---:|---:|---:|---:|---:|
| Adaptação_25 | 15.075 | 3.775 | 25 | 11.300 | 46 | 0,41 |
| Adaptação_200 | 15.600 | 3.950 | 177 | 11.650 | 393 | 3,37 |
| Adaptação_1000 | 18.000 | 4.750 | 634 | 13.250 | 1.933 | 14,59 |
| Consistência_25 | 15.075 | 3.775 | 25 | 11.300 | 47 | 0,42 |
| Consistência_200 | 15.600 | 3.950 | 177 | 11.650 | 327 | 2,81 |
| Consistência_1000 | 18.000 | 4.750 | 634 | 13.250 | 1.859 | 14,03 |
| Generalização_25 | 15.075 | 3.775 | 25 | 11.300 | 48 | 0,42 |
| Generalização_200 | 15.600 | 3.950 | 177 | 11.650 | 395 | 3,39 |
| Generalização_1000 | 18.000 | 4.750 | 634 | 13.250 | 1.935 | 14,60 |
| Recorrência_25 | 15.072 | 2.154 | 12 | 12.918 | 58 | 0,45 |
| Recorrência_200 | 15.600 | 2.242 | 92 | 13.358 | 483 | 3,62 |
| Recorrência_1000 | 18.000 | 2.642 | 470 | 15.358 | 2.162 | 14,08 |

Somando os cenários, há 194.697 posições de fluxo, das quais 44.463 são destinadas ao treino inicial e 150.234 à avaliação. A avaliação contém 9.686 ataques. Essa soma contabiliza posições nos diferentes cenários, não necessariamente registros de origem únicos.

### 5.1 O tamanho nominal não é a quantidade exata de ataques

Os blocos extraídos dos arquivos de ataque também podem conter registros BENIGN. O gerador permite coletar linhas sem filtrar previamente a classe.

Por exemplo, a primeira região de tamanho nominal 1000, nos cenários Adaptação, Consistência e Generalização, contém 634 ataques DrDoS_DNS e 366 benignos. No primeiro bloco de tamanho 200, há 177 ataques. Em `Adaptação_25`, o segundo bloco contém 23 ataques Syn e dois benignos.

No TCC, é importante distinguir tamanho nominal do bloco, duração da região e quantidade efetiva de instâncias de ataque.

### 5.2 Recorrência utiliza blocos menores

Adaptação, Consistência e Generalização apresentam três regiões de ataque. Recorrência apresenta seis regiões. Os spans observados das regiões de Recorrência são aproximadamente 12, 100 e 500 instâncias para os tamanhos nominais 25, 200 e 1000, respectivamente; uma região de Recorrência_200 tem span de 97.

Assim, comparar categorias com o mesmo sufixo não significa comparar blocos individuais de mesma duração. Recorrência também tem um prefixo inicial proporcionalmente menor: aproximadamente 14% a 15% do cenário, contra aproximadamente 25% a 26% nas outras categorias.

### 5.3 Famílias e ordem temporal

As sequências observadas são:

- Consistência: DNS → DNS → DNS;
- Generalização: DNS → LDAP → DNS;
- Adaptação: DNS → Syn → DNS;
- Recorrência: DNS → Syn → LDAP → DNS → Syn → LDAP.

O treino inicial inclui a primeira região DNS. Syn e/ou LDAP surgem somente depois, conforme a categoria.

Essas sequências descrevem a construção dos cenários. Os nomes das categorias, isoladamente, não demonstram uma mudança matemática específica em P(X), P(Y) ou P(Y|X). A chegada de uma família nova e a mudança de prevalência são fatores observáveis; mecanismos específicos de concept drift exigem caracterização adicional.

## 6. Referência: 100% de rótulos, sem atraso

### 6.1 Desempenho por cenário

F1 médio (%):

| Cenário | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| Adaptação_25 | 78,55 | 85,62 | 75,95 | 76,92 |
| Adaptação_200 | 95,49 | 86,33 | 94,67 | 86,12 |
| Adaptação_1000 | 98,91 | 96,33 | 97,42 | 85,05 |
| Consistência_25 | 90,69 | 95,14 | 88,64 | 88,64 |
| Consistência_200 | 93,19 | 93,62 | 92,21 | 91,59 |
| Consistência_1000 | 98,91 | 98,77 | 97,84 | 94,96 |
| Generalização_25 | 88,90 | 95,46 | 89,13 | 89,13 |
| Generalização_200 | 95,49 | 97,15 | 96,39 | 95,05 |
| Generalização_1000 | 98,98 | 98,69 | 96,89 | 95,43 |
| Recorrência_25 | 76,77 | 80,41 | 77,89 | 66,02 |
| Recorrência_200 | 93,88 | 92,36 | 87,47 | 87,74 |
| Recorrência_1000 | 98,44 | 97,14 | 90,52 | 92,52 |

ARF tem o maior F1 em seis cenários, e LB nos outros seis. HAT e HT não lideram nenhum dos 12 nesta referência. Isso não implica que os ensembles sejam os melhores sob qualquer combinação de atraso e orçamento.

### 6.2 Médias globais e custo computacional

| Modelo | F1 (%) | Precisão (%) | Recall (%) | MCC | Taxa de FP (%) | Tempo médio por execução (s) |
|---|---:|---:|---:|---:|---:|---:|
| ARF | 92,35 | 92,71 | 92,22 | 0,9226 | 0,1346 | 21,92 |
| LB | 93,09 | 97,41 | 89,66 | 0,9310 | 0,0806 | 21,28 |
| HAT | 90,42 | 96,57 | 85,67 | 0,9045 | 0,1250 | 2,08 |
| HT | 87,43 | 94,57 | 81,68 | 0,8726 | 0,2048 | 1,88 |

A taxa de FP é calculada em cada cenário por `FP / quantidade de benignos avaliados`, e depois promediada entre cenários. Não deve ser confundida com `1 − precisão`.

LB tem o maior F1 médio e menor taxa média de FP. ARF tem o maior recall médio. Portanto, há um compromisso entre deixar passar ataques e produzir alarmes falsos; a escolha depende da prioridade operacional.

A vantagem média de LB sobre ARF em F1 é de aproximadamente 0,74 p.p. Esta análise é descritiva e não estabelece significância estatística dessa diferença.

Os ensembles levam aproximadamente dez vezes o tempo de HAT e HT nesta referência. HAT oferece F1 relativamente próximo dos ensembles com menor tempo, embora com menor recall. Esses tempos descrevem esta execução e este ambiente, não um benchmark universal de hardware ou de latência de predição.

## 7. Experimento A: somente atraso

### 7.1 Efeito global

F1 médio nos 12 cenários (%), com 100% de consultas:

| Atraso nominal | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| 0% | 92,35 | 93,09 | 90,42 | 87,43 |
| 1% | 61,35 | 70,94 | 55,67 | 54,72 |
| 5% | 37,82 | 40,15 | 41,74 | 38,03 |
| 10% | 28,89 | 32,90 | 43,64 | 35,90 |

Mesmo 1% gera perdas médias grandes. ARF perde aproximadamente 31,00 p.p.; LB, 22,14 p.p.; HAT, 34,75 p.p.; HT, 32,71 p.p.

Com atraso de 10%, ARF cai de 92,35% para 28,89%, uma perda de 63,46 p.p. Seu recall médio cai de 92,22% para 51,07%, e sua precisão de 92,71% para 27,58%. A taxa média de FP aumenta de 0,1346% para 7,2613%.

Portanto, a perda não é apenas consequência de deixar passar ataques: também há produção de alarmes falsos durante períodos benignos.

HAT apresenta F1 médio maior em 10% do que em 5% de atraso, 43,64% contra 41,74%. Isso mostra que a relação não é perfeitamente monotônica para cada modelo. Não é evidência de benefício geral do atraso.

### 7.2 Dependência do tamanho dos blocos

Média de F1 entre quatro categorias e quatro modelos (%):

| Tamanho nominal | Atraso 0% | Atraso 1% | Atraso 5% | Atraso 10% |
|---|---:|---:|---:|---:|
| 25 | 83,99 | 35,81 | 18,49 | 14,43 |
| 200 | 92,42 | 59,94 | 45,58 | 39,34 |
| 1000 | 96,05 | 86,27 | 54,24 | 52,22 |

Os atrasos efetivos são:

| Tamanho nominal | Total típico | Atraso 1% | Atraso 5% | Atraso 10% |
|---|---:|---:|---:|---:|
| 25, exceto Recorrência | 15.075 | 151 | 754 | 1.508 |
| 200 | 15.600 | 156 | 780 | 1.560 |
| 1000 | 18.000 | 180 | 900 | 1.800 |

Em Recorrência_25, N é 15.072 e o atraso de 10% é 1.507.

Para uma região de 25 instâncias, 151 posições de atraso equivalem a aproximadamente seis vezes sua duração. O primeiro rótulo daquela região só chega depois de ela ter terminado. Já em uma região de 1000 posições, um atraso de 180 ainda permite atualizações durante parte do ataque. Em Recorrência, as regiões são menores e essa comparação deve usar aproximadamente 12, 100 ou 500 posições.

Assim, o percentual sobre N precisa ser discutido juntamente com a razão `atraso / duração da região`. Chamar 1% de atraso de pequeno, sem esse contexto, seria enganoso.

### 7.3 Diferenças entre categorias

Média de F1 entre os três tamanhos e os quatro modelos (%):

| Categoria | Atraso 0% | Atraso 1% | Atraso 5% | Atraso 10% |
|---|---:|---:|---:|---:|
| Consistência | 93,68 | 80,30 | 57,63 | 51,10 |
| Generalização | 94,72 | 61,40 | 37,76 | 33,42 |
| Adaptação | 88,11 | 53,12 | 31,73 | 27,44 |
| Recorrência | 86,76 | 47,87 | 30,61 | 29,38 |

Consistência é a mais resistente ao atraso nesta comparação. A reutilização da família DNS é uma explicação plausível, pois exige menos aprendizado de famílias novas. Entretanto, as categorias também diferem em composição, duração de blocos e prefixo de treino; os números não isolam somente a presença de uma família nova.

### 7.4 Estudo temporal: Adaptação_25

Existem 46 ataques avaliados: 23 Syn e 23 DrDoS_DNS. O primeiro bloco DNS de 25 ataques está inteiramente no treino.

Resultados acumulados:

| Modelo | Atraso | F1 médio ± DP (%) | FP médio | FN médio |
|---|---:|---:|---:|---:|
| ARF | 0% | 78,55 ± 1,28 | 10,6 | 9,4 |
| ARF | 1% | 36,99 ± 0,70 | 51,0 | 24,0 |
| ARF | 5% | 13,58 ± 0,45 | 253,4 | 24,2 |
| ARF | 10% | 8,40 ± 0,19 | 456,2 | 24,0 |
| LB | 0% | 85,62 ± 2,07 | 2,6 | 9,6 |
| LB | 1% | 59,95 ± 0,40 | 5,4 | 24,0 |
| LB | 5% | 22,85 ± 0,10 | 124,6 | 24,0 |
| LB | 10% | 15,56 ± 0,17 | 214,8 | 24,0 |

Nos atrasos de 1%, 5% e 10%, todos os quatro modelos produzem 23 FN na janela que contém os 23 ataques Syn: nenhum desses ataques é detectado. Sem atraso, ARF tem 8,4 FN nessa janela; LB, 8,6; HAT e HT, 11.

ARF e LB continuam detectando quase todos os ataques DNS da região posterior, com aproximadamente um FN. HAT e HT, por outro lado, deixam passar todos os 46 ataques avaliados nas três condições com atraso, resultando em F1 zero. F1 zero não significa ausência de alarmes: esses modelos também produzem FP.

Nas condições com atraso, os FP de ARF e LB estão em janelas sem ataques. Para ARF com atraso de 10%, uma janela de 100 instâncias que termina no índice 4.574 contém, em média, 55,4 FP e nenhum ataque real. Essa janela ocorre antes do primeiro ataque avaliado, cujo início é o índice 7.525.

Esse detalhe impede atribuir todos os FP à chegada tardia dos rótulos de ataques avaliados: parte importante do problema já aparece no período benigno imediatamente após o treino. O atraso também retarda o feedback corretivo de exemplos benignos. É plausível que o modelo mantenha por mais tempo um estado influenciado pelo fim do prefixo de treino, mas o mecanismo interno exato não está registrado nos CSVs.

Os índices são baseados em zero. A janela Syn vai de 7.475 a 7.574, e a janela DNS posterior de 11.275 a 11.374. Cada uma contém uma única família de ataque, permitindo associar seus FN à família correspondente. Em janelas com mais de uma família, essa atribuição exigiria previsões por instância.

## 8. Experimento B: somente Random Sampling

### 8.1 Efeito global do orçamento

F1 médio nos 12 cenários (%), sem atraso:

| Orçamento nominal | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| 1% | 47,41 | 48,29 | 56,25 | 49,21 |
| 3% | 57,42 | 61,51 | 58,22 | 51,37 |
| 5% | 65,20 | 67,06 | 61,70 | 57,77 |
| 10% | 72,47 | 74,58 | 65,83 | 65,24 |
| 30% | 82,72 | 83,40 | 80,10 | 79,20 |
| 100% | 92,35 | 93,09 | 90,42 | 87,43 |

Na média global, os quatro modelos melhoram quando o orçamento aumenta. Com 30%, LB mantém aproximadamente 89,6% do F1 da referência completa, mas ainda perde 9,68 p.p. ARF perde 9,63 p.p.

Esses resultados indicam um compromisso entre custo e desempenho, não um orçamento universalmente ideal. O orçamento de 30% ainda é insuficiente em alguns cenários curtos.

O desempenho com poucos rótulos também pode ser prejudicado por perda de precisão. Por exemplo, ARF com orçamento de 1% tem recall médio de 73,69%, mas precisão média de 43,60%, contra 92,22% e 92,71% na referência. Consultar menos benignos reduz oportunidades de corrigir alarmes falsos, além de reduzir oportunidades de aprender ataques.

### 8.2 O efeito do desbalanceamento

Média de F1 entre quatro categorias e quatro modelos (%):

| Tamanho nominal | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 25 | 11,79 | 19,22 | 24,56 | 35,26 | 62,45 | 83,99 |
| 200 | 59,43 | 66,84 | 75,38 | 80,49 | 86,53 | 92,42 |
| 1000 | 79,64 | 85,34 | 88,85 | 92,84 | 95,09 | 96,05 |

A quantidade de ataques muda muito entre esses grupos, e Random Sampling não garante a consulta de ataques.

Para k ataques e probabilidade de consulta p, a probabilidade de não consultar nenhum deles é:

`P(nenhum ataque consultado) = (1 − p)^k`

Em Adaptação_25, k = 46. Com p = 1%, a probabilidade é aproximadamente 62,98%. Reproduzindo a seleção das sementes 42 a 46, três das cinco execuções não consultam nenhum ataque avaliado. A média é de apenas 0,6 ataque consultado por execução.

Isso não significa treinamento sem rótulos: todos os modelos já receberam os 25 ataques e os benignos do prefixo inicial. Significa ausência de novos rótulos de ataque nessas três repetições.

| Cenário | Ataques avaliados | Ataques consultados com p = 1%, média | Repetições sem consultar ataques |
|---|---:|---:|---:|
| Adaptação_25 | 46 | 0,6 | 3 de 5 |
| Consistência_25 | 47 | 0,6 | 3 de 5 |
| Generalização_25 | 48 | 0,6 | 3 de 5 |
| Recorrência_25 | 58 | 0,8 | 1 de 5 |
| Adaptação_200 | 393 | 3,8 | 0 de 5 |
| Adaptação_1000 | 1.933 | 17,2 | 0 de 5 |

Essas consultas de ataques foram reconstruídas a partir das sementes; os CSVs históricos registram consultas totais, não sua divisão por classe. A reconstrução reproduz exatamente a média e o desvio padrão das consultas totais observadas nos arquivos.

Para um bloco com 23 ataques e p = 1%, a chance de nenhuma consulta é aproximadamente 79,36%. Portanto, a chance de aprender online a partir de um ataque curto é pequena, mesmo sem atraso.

### 8.3 Exemplo de aumento de alarmes falsos

Em Adaptação_25, ARF apresenta:

- 100%: F1 = 78,55%, FP = 10,6 e FN = 9,4;
- 1%: F1 = 7,30%, FP = 569,6 e FN = 23,8.

A condição de 1% consulta, em média, 115,6 instâncias da avaliação, das quais apenas 0,6 são ataques. O aumento de FP ajuda a explicar por que F1 se torna muito baixo, apesar de o modelo ainda detectar parte dos ataques conhecidos.

### 8.4 Não monotonicidade em casos individuais

A tendência global de melhora não vale para toda configuração individual. Em Adaptação_1000, HT tem F1 de 85,05% com 100% dos rótulos, 93,01% com 10% e 93,23% com 30%.

Esse resultado mostra que a trajetória de atualização influencia o modelo. Menos rótulos pode alterar divisões, previsões nas folhas e adaptação ao fluxo. Entretanto, sem registrar os estados internos, não é possível determinar qual mecanismo explica esse caso. Também não se deve escolher o melhor orçamento observado e anunciá-lo como ótimo geral sem validação adicional.

## 9. Custo de rotulagem: o prefixo inicial precisa entrar na conta

`Effective_Query_Percentage_avg` utiliza somente as instâncias avaliadas como denominador. Ele não inclui o custo do treino inicial totalmente rotulado.

Para medir a proporção de instâncias rotuladas no cenário inteiro, a conta apropriada é:

`custo_total_percentual = 100 × (treino_inicial + consultas_na_avaliação) / N`

Média dessa proporção entre os 12 cenários:

| Orçamento nominal na avaliação | Consultas efetivas na avaliação (%) | Instâncias rotuladas no cenário inteiro (%) |
|---|---:|---:|
| 1% | 1,01 | 23,58 |
| 3% | 2,91 | 25,04 |
| 5% | 4,97 | 26,64 |
| 10% | 10,05 | 30,55 |
| 30% | 30,28 | 46,18 |
| 100% | 100,00 | 100,00 |

Assim, não é correto afirmar que o orçamento de 1% representa economia de 99% de rótulos considerando o cenário completo. Nesta média, o custo é de aproximadamente 23,58%, ou economia de aproximadamente 76,42% em relação à rotulagem de todas as instâncias.

Em Adaptação_25 especificamente, o custo com p = 1% é `(3.775 + 115,6) / 15.075`, aproximadamente 25,81% do cenário.

Essas são medidas de quantidade de rótulos por execução, não valores monetários nem medições de tempo de um anotador humano. Repetições do experimento também não implicam que um humano precise pagar novamente pela anotação dos mesmos registros.

O atraso, isoladamente, não economiza consultas. No experimento A, 100% das instâncias avaliadas são consultadas em todas as condições; o que muda é quando o rótulo pode influenciar previsões.

## 10. Experimento C: atraso e Random Sampling

As tabelas seguintes mostram o F1 médio nos 12 cenários (%). As linhas são o atraso nominal; as colunas, o orçamento de consultas.

### 10.1 ARF

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 47,41 | 57,42 | 65,20 | 72,47 | 82,72 | 92,35 |
| 1% | 44,39 | 51,34 | 57,57 | 59,45 | 59,40 | 61,35 |
| 5% | 38,07 | 42,06 | 42,24 | 41,00 | 35,73 | 37,82 |
| 10% | 35,08 | 37,65 | 37,47 | 36,64 | 31,43 | 28,89 |

### 10.2 LB

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 48,29 | 61,51 | 67,06 | 74,58 | 83,40 | 93,09 |
| 1% | 45,20 | 55,78 | 59,14 | 63,33 | 66,71 | 70,94 |
| 5% | 39,74 | 44,75 | 45,81 | 46,85 | 44,21 | 40,15 |
| 10% | 36,61 | 39,85 | 39,95 | 40,37 | 38,58 | 32,90 |

### 10.3 HAT

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 56,25 | 58,22 | 61,70 | 65,83 | 80,10 | 90,42 |
| 1% | 55,36 | 55,17 | 56,66 | 56,21 | 54,74 | 55,67 |
| 5% | 55,02 | 52,83 | 53,43 | 52,30 | 45,67 | 41,74 |
| 10% | 56,10 | 52,74 | 53,25 | 52,63 | 48,43 | 43,64 |

### 10.4 HT

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 49,21 | 51,37 | 57,77 | 65,24 | 79,20 | 87,43 |
| 1% | 46,36 | 47,07 | 50,08 | 52,19 | 56,28 | 54,72 |
| 5% | 40,69 | 38,99 | 39,92 | 41,83 | 41,87 | 38,03 |
| 10% | 41,37 | 38,02 | 37,93 | 40,56 | 40,41 | 35,90 |

### 10.5 Interpretação da interação

Sem atraso, a disponibilidade de mais rótulos favorece os resultados médios. Com atraso de 5% ou 10%, essa relação se enfraquece ou se inverte em algumas configurações.

Por exemplo, HAT com atraso de 10% apresenta F1 médio de 56,10% com orçamento de 1%, contra 43,64% com orçamento de 100%. Isso representa uma melhora relativa à condição de supervisão completa atrasada, não à referência sem atraso de 90,42%.

Um caso individual particularmente expressivo é Generalização_1000 com HAT:

| Condição | F1 médio ± DP (%) | Recall médio (%) | FP médio | FN médio |
|---|---:|---:|---:|---:|
| Sem atraso, 100% | 96,89 ± 0,00 | 94,88 | 19,0 | 99,0 |
| Atraso 10%, orçamento 100% | 72,39 ± 0,00 | 58,60 | 64,0 | 801,0 |
| Atraso 10%, orçamento 1% | 97,47 ± 0,07 | 97,83 | 56,2 | 42,0 |

O caso sugere que determinadas trajetórias de atualização atrasada podem prejudicar o modelo, enquanto uma menor taxa de atualização preserva um estado de detecção eficaz. Mas a explicação é uma hipótese: não foram persistidos logs de divisões, previsões internas das folhas, substituições de árvores ou alarmes de drift.

Além disso, uma previsão de outra família de ataque conta como detecção correta. Portanto, o alto desempenho nesse caso não demonstra que o modelo aprendeu a identificar LDAP com pouquíssimos rótulos.

A relevância de considerar orçamento e latência conjuntamente é discutida na literatura: em fluxos não estacionários, o rótulo consultado pode chegar quando a distribuição já mudou. Isso oferece contexto para a investigação, mas não prova o mecanismo deste caso específico. [Castellani, Schmitt e Hammer, 2022](https://arxiv.org/abs/2204.06822).

Não é seguro concluir que active learning reduz sempre o prejuízo do atraso, nem que rótulos atrasados são sempre prejudiciais. Os efeitos dependem do modelo, do cenário e da sequência de atualizações.

## 11. Rótulos consultados versus efetivamente utilizados durante a avaliação

O código entrega os rótulos pendentes após o fim do fluxo. Essas atualizações finais não alteram as previsões e métricas já calculadas.

Em Adaptação_25, com 100% de consultas:

| Atraso em instâncias | Consultados | Entregues durante o fluxo | Entregues após o fluxo |
|---|---:|---:|---:|
| 0 | 11.300 | 11.300 | 0 |
| 151 | 11.300 | 11.149 | 151 |
| 754 | 11.300 | 10.546 | 754 |
| 1.508 | 11.300 | 9.792 | 1.508 |

Embora todas as consultas sejam feitas, os últimos 1.508 rótulos da condição de 10% não podem beneficiar nenhuma previsão avaliada.

Na média dos cenários, com orçamento de 100%, as proporções de rótulos da avaliação entregues durante o fluxo são 100%, 98,70%, 93,50% e 87,00%, para atrasos de 0%, 1%, 5% e 10%.

Para orçamento nominal de 1% e atraso de 10%, aproximadamente 1,01% das instâncias avaliadas são consultadas, mas apenas 0,88% têm seus rótulos entregues antes do fim do fluxo.

Portanto, um gráfico de desempenho versus consultas descreve o custo de solicitação. Para discutir a quantidade de supervisão capaz de influenciar a avaliação, também é necessário considerar as entregas durante o fluxo.

## 12. Variabilidade, desvios padrão e janelas

### 12.1 O que o DP individual representa

Nos resultados individuais, o desvio padrão mede a variação da métrica entre cinco repetições do mesmo fluxo. Quando há Random Sampling, mudam as consultas; nos modelos com aleatoriedade relevante, também muda a construção do modelo.

Essas cinco repetições não são cinco capturas independentes de tráfego. O cenário e sua ordem permanecem iguais.

Na referência sem atraso e com 100% dos rótulos, HAT e HT apresentam DP de F1 igual a zero em todos os cenários. Isso significa que as repetições produziram a mesma métrica nessas condições; não garante estabilidade em novos dados nem ausência de incerteza científica.

Em contraste, há configurações instáveis. Adaptação_200 com HAT, atraso de 1% e orçamento de 30%, apresenta F1 de 45,54% ± 23,77 p.p. Recorrência_1000 com HAT, atraso de 5% e orçamento de 30%, apresenta 14,18% ± 21,84 p.p.

Médias assim devem ser acompanhadas da dispersão. Os CSVs não contêm as cinco métricas individuais, portanto não permitem reconstruir fielmente sua distribuição ou identificar quantas repetições tiveram resultados extremos.

### 12.2 O DP agregado tem outro significado

Nos gráficos agregados, o código calcula o desvio padrão entre as médias dos cenários, com `ddof=0`. Essa banda descreve heterogeneidade entre cenários, e não variação das cinco repetições de uma única condição.

Assim, uma banda grande no gráfico agregado pode indicar que uma configuração funciona bem em cenários de 1000 e mal nos de 25, mesmo que cada cenário individual tenha pouca variação entre sementes.

Não se deve apresentar média ± DP como um intervalo de confiança de 95%, nem interpretar o DP como a probabilidade de um erro. Também não foi realizado teste de significância nesta análise.

### 12.3 Limitações do F1 por janela

As janelas têm 100 instâncias, exceto a última quando necessário. Em uma janela sem ataques, o código registra F1 e recall zero por convenção de `zero_division=0`, inclusive quando todos os benignos são classificados corretamente.

Por isso, longos trechos de F1 zero no gráfico não significam, por si só, falha do modelo. Nesses trechos, é preciso olhar FP e a presença ou ausência de ataques.

FP/FN podem ser somados entre janelas; F1, precisão e recall não. A média de F1 das janelas também não representa, em geral, o F1 acumulado.

Ataques com 12 ou 25 posições são menores que uma janela. O gráfico agrega sua ocorrência com instâncias benignas próximas. Isso permite localizar a região, mas não medir exatamente o tempo até a primeira detecção a partir do CSV janelado.

### 12.4 Acurácia isolada seria enganosa

Um classificador que previsse sempre BENIGN teria acurácia de aproximadamente 85,40% a 99,59%, dependendo do cenário, mas recall de ataque zero. Em Adaptação_25, teria aproximadamente 99,59% de acurácia.

Portanto, é adequado priorizar F1, recall, precisão, MCC e taxas de FP/FN, em vez de utilizar apenas acurácia.

## 13. Custo computacional

No experimento B, ARF cai de aproximadamente 21,92 s por execução com 100% de consultas para 9,52 s com 1%; LB cai de 21,28 s para 9,29 s. A redução é de aproximadamente 56% em ambos os casos.

O tempo não diminui na mesma proporção que as consultas, porque todas as instâncias continuam sendo previstas e avaliadas, e o treino inicial permanece totalmente supervisionado.

HAT e HT apresentam reduções menores: aproximadamente 2,08 s para 1,78 s, e 1,88 s para 1,71 s, respectivamente.

No experimento A, os tempos ficam próximos entre os atrasos. Isso faz sentido: não há redução na quantidade de consultas e as atualizações pendentes são realizadas depois do fluxo.

O cronômetro do runner inclui o treino inicial, a passagem de avaliação, atualizações, cálculos de métricas por janela e o esvaziamento final da fila. Não representa o tempo humano de obtenção dos rótulos nem apenas o custo de predição. A criação do stream e seu pré-processamento ocorrem antes desse cronômetro.

Somando `Time_avg × Runs` somente nas condições únicas de C, obtém-se aproximadamente 11,77 horas registradas. Somar todas as linhas A/B/C produziria aproximadamente 17,85 horas, com dupla contagem de condições reaproveitadas. Nenhuma dessas somas representa necessariamente o tempo total de parede da execução completa, pois há trabalho fora dos cronômetros.

## 14. Comparação com os experimentos padrões antigos

Há duas situações diferentes no histórico, que não devem ser confundidas.

### 14.1 Tabela antiga com reaproveitamento de modelo

O notebook histórico em `850debe`, nas células de ARF, LB e HAT, cria os modelos uma vez e passa seus objetos para cinco rodadas. A implementação antiga reiniciava o stream, mas não garantia um modelo novo entre as rodadas.

A auditoria anterior identificou que os classificadores utilizados não ofereciam o `reset()` esperado naquele caminho. Assim, o modelo podia iniciar uma nova passagem já treinado com todo o cenário anterior. As rodadas posteriores deixavam de representar cinco execuções independentes de avaliação prequential.

O valor de ARF em Adaptação_25 de 92,5829% ± 6,6397 p.p., apresentado naquela tabela antiga, foi reproduzido pela sequência de passagens sobre o mesmo modelo. Ele não deve ser utilizado como referência de cinco repetições novas para comparar com A/B/C.

Essa conclusão se aplica àquela versão e àquela tabela. Não significa que todos os resultados históricos do repositório tenham o mesmo problema.

### 14.2 Resultados posteriores com protocolo diferente

Há resultados padrões posteriores com modelos novos e `Warmup = 2000`, por exemplo o CSV de ARF identificado por `20260614_0639`. Em Adaptação_25, ele registra F1 de aproximadamente 82,7650% ± 0,8845 p.p.

O experimento A/B/C, por sua vez, começa a avaliação depois de 3.775 instâncias nesse cenário. Portanto, as métricas são calculadas sobre populações diferentes: a referência padrão inclui o primeiro ataque, enquanto A/B/C o utiliza somente para treino.

A auditoria controlada anterior encontrou previsões coincidentes ao alinhar o trecho avaliado e as configurações. A diferença de F1 acumulado, por si só, não indica defeito no atraso zero.

Para uma comparação publicável, o trecho de treino, o trecho de avaliação, o pré-processamento, os parâmetros, o alvo e as sementes devem ser alinhados. Sem isso, uma diferença pode vir do protocolo, não do mecanismo estudado.

## 15. Limitações metodológicas que precisam constar no TCC

### 15.1 Estatísticas futuras no pré-processamento

Em `src/Data/Processor.py`, `_normalize_data()` usa `scaler.fit_transform(X)` sobre o cenário inteiro. A imputação por mediana também calcula a estatística sobre todas as linhas antes de o runner separar o treino inicial.

A documentação do scikit-learn recomenda aprender as transformações somente com o conjunto de treino e aplicar a transformação aprendida aos dados posteriores; ajustar o pré-processamento com dados de avaliação constitui vazamento de informação. [Documentação oficial sobre data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

A inspeção dos dados confirma que essa exposição não é apenas uma possibilidade abstrata:

| Grupo | Quantidade de features com mínimo ou máximo global diferente do treino inicial |
|---|---:|
| Adaptação | 37 a 38 |
| Consistência | 37 a 38 |
| Generalização | 38 |
| Recorrência | 43 |

Todos os cenários apresentam uma feature com valores ausentes cuja mediana global difere da mediana do prefixo inicial. Existem também valores infinitos nas entradas, substituídos por extremos de float32 antes da normalização.

Esses números demonstram uso efetivo de estatísticas futuras. Não quantificam quanto o vazamento mudou cada resultado: esse efeito ainda não foi medido e pode variar entre modelos. Não é correto afirmar, sem uma comparação controlada, que todo F1 necessariamente diminuiu ou aumentou por causa disso.

O pré-processamento comum mantém alguma utilidade comparativa dos resultados históricos, mas não elimina o problema de validade externa ou garante que o ranking se preserve após uma correção causal. A migração para alvo binário, sozinha, não resolve essa limitação.

### 15.2 Generalização limitada pelo benchmark construído

São cenários construídos com fontes e procedimentos compartilhados, não 12 observações independentes de sistemas de produção. A construção do tráfego benigno usa amostragem com semente fixa, e as famílias de ataque vêm de arquivos de origem comuns.

Não é seguro extrapolar as médias diretamente para qualquer rede, considerar os cenários independentes para testes estatísticos ou concluir resistência geral a todas as formas de drift.

### 15.3 Fatores de confusão entre tamanhos e categorias

Ao mudar o tamanho nominal, mudam simultaneamente a duração das regiões, a prevalência de ataque, a quantidade de positivos no treino e o tamanho total. Recorrência também muda o número de regiões e a proporção do prefixo inicial.

Logo, a melhoria nos cenários maiores não pode ser atribuída exclusivamente à quantidade de instâncias, e a diferença entre categorias não isola apenas a família de ataque.

### 15.4 Cinco sementes não são cinco cenários independentes

As repetições medem variabilidade de execução e consulta sobre um fluxo fixo. Para estudar variabilidade da construção do cenário, seriam necessárias outras construções ou capturas, com metodologia explicitamente definida.

### 15.5 Ausência de métricas individuais e logs internos persistidos

Os CSVs têm médias e DP, mas não as métricas completas de cada semente nem as previsões de cada instância. Não é possível realizar um teste pareado fiel, reconstruir a distribuição das métricas, medir atraso até primeira detecção ou provar o mecanismo interno de uma melhora inesperada apenas com esses arquivos.

Também não há logs suficientes para afirmar quantas substituições de árvores ou detecções de drift explicaram uma mudança de desempenho.

### 15.6 Falta de controle sem atualização após o treino

O grid atual não inclui orçamento de 0%. Portanto, não se pode medir diretamente quanto uma condição de orçamento muito baixo melhora em relação a um modelo congelado após o treino inicial. Adicionar esse controle seria uma expansão do experimento, não uma correção silenciosa dos resultados existentes.

### 15.7 Reprodutibilidade do ambiente

`requirements.txt` fixa a versão de pandas, mas não fixa versões de CapyMOA, NumPy e scikit-learn. Os CSVs também não registram o ambiente completo da execução. Para reprodutibilidade do TCC, é importante registrar as versões efetivamente usadas, Java, parâmetros, comando de execução e commit, sem assumir que o ambiente atual é idêntico ao histórico.

## 16. O que é possível afirmar e o que não é

| Afirmação | Parecer |
|---|---|
| Os arquivos são coerentes com atraso e Random Sampling implementados | Sustentada pelas verificações realizadas |
| O atraso reduz fortemente o desempenho médio neste benchmark | Sustentada descritivamente |
| Ataques curtos são especialmente prejudicados | Sustentada; duração e prevalência mudam conjuntamente |
| Orçamentos baixos podem deixar de consultar ataques | Sustentada pela reconstrução das seleções |
| Mais rótulos sem atraso melhoram o desempenho médio | Sustentada para os quatro modelos nesta execução |
| Mais rótulos atrasados sempre melhoram o desempenho | Contradita por diversas configurações |
| LB é estatisticamente superior a ARF | Não demonstrada |
| Active learning informado supera supervisão completa | Não testada; a estratégia utilizada é Random Sampling |
| Os resultados medem identificação correta das famílias | Falsa: as métricas são binárias |
| O estudo histórico é ponta a ponta totalmente causal | Não: o pré-processamento usa estatísticas futuras |
| O treino binário necessariamente será melhor | Ainda não testada |
| Um orçamento de 1% economiza 99% dos rótulos de todo o cenário | Falsa quando o treino inicial é incluído |
| As cinco repetições representam cinco tráfegos independentes | Falsa |

## 17. Recomendações antes da próxima execução

Prioridade principal: corrigir ou definir explicitamente o protocolo de pré-processamento causal antes de investir em uma nova execução completa.

Uma alternativa simples é aprender a imputação e a normalização somente no prefixo inicial de treino e manter essas transformações fixas no restante do fluxo. Valores posteriores podem ficar fora do intervalo do treino; isso precisa ser uma decisão documentada, não motivo para reajustar silenciosamente com dados futuros.

Uma alternativa mais complexa é pré-processamento incremental causal, mas mudanças de escala exigem cuidado para não tornar incompatíveis as features das instâncias guardadas na fila e o estado do modelo. Não basta substituir o scaler por outro com atualização online sem tratar essa consistência.

Também recomendo:

1. Preservar a execução multiclasse como histórico exploratório, identificando suas limitações.
2. Registrar treinamento binário e avaliação binária explicitamente nos novos resultados.
3. Salvar métricas por repetição, sementes, contagens por classe, parâmetros, versões e commit.
4. Validar, antes da execução completa, a equivalência de A com C a 100%, de B com C a atraso zero, e da referência totalmente supervisionada com um pipeline alinhado.
5. Analisar os resultados separadamente por duração de região e prevalência, além das médias globais.
6. Separar custo total de rótulos, consultas na avaliação e entregas durante o fluxo.
7. Considerar um controle congelado após o treino inicial e repetições de construção de cenários, caso isso seja aprovado como extensão do desenho experimental.
8. Se o objetivo for medir o efeito do alvo binário, comparar versões binária e multiclasse sob o mesmo pré-processamento causal. Comparar a nova versão causal binária apenas com o histórico offline multiclasse mistura duas mudanças.

Nenhuma dessas alterações foi implementada nesta análise. Elas são recomendações para decidir o desenho da próxima etapa.

## 18. Sugestão de redação para o TCC

### 18.1 Resultados principais

> Na configuração com disponibilidade integral e imediata de rótulos, Leveraging Bagging e Adaptive Random Forest obtiveram os maiores F1 médios entre os 12 cenários, com 93,09% e 92,35%, respectivamente. A introdução de atraso de 1% do tamanho total dos cenários reduziu esses valores para 70,94% e 61,35%. A degradação foi especialmente acentuada nos cenários de ataques curtos, nos quais o atraso pode ultrapassar a duração de uma região inteira de ataque, impedindo que os rótulos dessa região contribuam para sua própria detecção online.

### 18.2 Orçamento e desbalanceamento

> A redução do orçamento de consultas por Random Sampling produziu um compromisso entre custo de rotulagem e desempenho. Entretanto, os menores orçamentos foram particularmente limitados em cenários com baixa prevalência de ataque. Em Adaptação_25, o orçamento de 1% resultou em apenas 0,6 ataque consultado por execução, em média, e três das cinco repetições não consultaram nenhum ataque na etapa de avaliação. O treino inicial integralmente supervisionado foi mantido em todas as condições e, por isso, a economia total de rótulos não corresponde diretamente ao complemento do orçamento nominal.

### 18.3 Interação e limitações

> A combinação entre atraso e orçamento revelou comportamento não monotônico em algumas configurações: uma maior quantidade de rótulos atrasados não garantiu maior desempenho. Esse resultado indica dependência da trajetória de atualização, mas não permite identificar o mecanismo interno responsável sem instrumentação adicional. A análise deve ainda ser interpretada à luz do treinamento multiclasse com métricas binárias e do ajuste offline do pré-processamento sobre o cenário completo. Assim, os resultados constituem evidência exploratória para orientar uma avaliação causal posterior, e não uma estimativa definitiva de desempenho em produção.

Os textos acima são uma proposta de redação dos resultados locais. Devem ser integrados à metodologia, às referências e às regras de apresentação adotadas pelo TCC.

## 19. Rastreabilidade e reprodução da análise

Script diagnóstico temporário utilizado nesta sessão:

`/tmp/tcc_analysis_20261007.py`

Ele lê os CSVs existentes e os rótulos dos cenários, reproduz as seleções aleatórias e imprime tabelas e verificações. Não importa os classificadores, não treina modelos e não sobrescreve os resultados originais.

Esse script está em `/tmp` e não foi incorporado ao repositório; sua disponibilidade não é garantida depois da limpeza dos arquivos temporários. O relatório é o artefato permanente adicionado ao projeto. O commit de código consultado na análise é `cdb6f88`, além dos commits históricos mencionados acima.

Para executar novamente no ambiente local:

```bash
cd /home/tomas/repos/anomaly-and-classification-data-stream
.venv/bin/python /tmp/tcc_analysis_20261007.py
```

Para reproduzir as tabelas principais independentemente do script temporário, o código abaixo pode ser executado a partir da raiz do projeto, sem treinar nenhum modelo:

```python
from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

root = Path("output/ClassificationLabeling")
files = sorted(root.glob("*_20261001_101354_cumulative.csv"))
assert len(files) == 12
data = pd.concat(
    [pd.read_csv(path, sep=";") for path in files],
    ignore_index=True,
)
assert len(data) == 1632

# C contém as 24 combinações únicas: não contar A e B outra vez.
unique = data.loc[data["Experiment"].eq("C")].copy()
assert len(unique) == 1152

experiment_a = unique.loc[unique["Label_Budget_Percentage"].eq(100)]
experiment_b = unique.loc[unique["Delay_Percentage"].eq(0)]

print("A: F1 médio entre os 12 cenários")
print(experiment_a.groupby(
    ["Delay_Percentage", "Model"]
)["F1_avg"].mean().unstack().round(2))

print("B: F1 médio entre os 12 cenários")
print(experiment_b.groupby(
    ["Label_Budget_Percentage", "Model"]
)["F1_avg"].mean().unstack().round(2))

print("C: F1 médio para cada modelo")
for model, rows in unique.groupby("Model"):
    print(model)
    print(rows.groupby(
        ["Delay_Percentage", "Label_Budget_Percentage"]
    )["F1_avg"].mean().unstack().round(2))
```

Fontes locais principais:

- `output/ClassificationLabeling/*_20261001_101354_cumulative.csv`: métricas acumuladas e contagens de rótulos;
- `output/ClassificationLabeling/*_20261001_101354_prequential.csv`: métricas e erros por janela;
- `data/15k/<categoria>/<cenário>.csv`: composição e ordem dos rótulos;
- `src/Classification/Labeling.py`: treino inicial, fila, consultas, repetição e agregação;
- `src/Data/Processor.py`: imputação, normalização e codificação do alvo;
- `src/Classification/Models.py`: configurações dos modelos;
- `src/Results/Metrics.py`: cálculo das métricas binárias;
- `plot_classification_labeling_results.py`: agregação e seleção de resultados para gráficos;
- `run_classification_labeling.py` em `e51942c`: protocolo da execução histórica;
- `Classification.ipynb` em `850debe`: protocolo da tabela padrão antiga discutida na auditoria anterior.

Os arquivos históricos não possuem uma coluna explícita de modo de treinamento. Sua identificação como multiclasse depende da origem da execução e do código histórico, não do default binário do código atual.

### Situação dos gráficos existentes

Há 78 entradas no manifesto em `output/ClassificationLabeling/plots-multclass/manifest.csv`: 72 figuras individuais, seis por cenário, e seis agregadas.

Após a renomeação da pasta, os 78 caminhos antigos registrados no manifesto não existem; os 78 arquivos existem ao substituir `/plots/` por `/plots-multclass/`. Essa mudança de pasta não altera os CSVs nem as conclusões numéricas. O manifesto precisa ser atualizado caso seja utilizado para abrir os gráficos automaticamente.

As 78 figuras do manifesto são de desempenho, erros acumulados, heatmap e custo; não incluem figuras temporais detalhadas de FP/FN. Os CSVs prequential preservam os erros por janela e permitiram o estudo temporal deste relatório, mas os arquivos históricos não registram consultas e entregas por janela.

## 20. Referências de apoio

- IENCO, Dino; ŽLIOBAITĖ, Indrė; PFAHRINGER, Bernhard. [High density-focused uncertainty sampling for active learning over evolving stream data](https://proceedings.mlr.press/v36/ienco14.html). PMLR, v. 36, p. 133–148, 2014.
- CASTELLANI, Andrea; SCHMITT, Sebastian; HAMMER, Barbara. [Stream-based Active Learning with Verification Latency in Non-stationary Environments](https://arxiv.org/abs/2204.06822). 2022. DOI da publicação associada: 10.1007/978-3-031-15937-4_22.
- SCIKIT-LEARN. [Common pitfalls and recommended practices — Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Documentação oficial. Consulta em 7 de outubro de 2026.

Essas referências dão contexto metodológico. Os valores numéricos apresentados no relatório foram calculados a partir dos arquivos locais do projeto, não extraídos dos artigos.
