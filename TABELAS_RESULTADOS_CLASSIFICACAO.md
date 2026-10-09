# Tabelas dos resultados de classificação

Gerado em 2026-10-08, usando apenas resultados já existentes.

Este índice reúne todos os resultados acumulados dos experimentos A/B/C das duas execuções completas analisadas e dá acesso a todos os resultados por janela. Escopo: classificação, configuração `Default_FullFeatures`. Não inclui detecção de anomalias, experimentos padrões com outro protocolo, resultados otimizados ou seleção de features inexistentes nessas execuções. Os CSVs e gráficos originais não foram alterados.

## 1. Execuções e cobertura

| Treinamento | Exec_ID | Avaliação | Cenários | Modelos | Repetições | Linhas A/B/C | Condições únicas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binário | 20261007_130844 | binária | 12 | 4 | 5 | 1632 | 1152 |
| multiclasse | 20261001_101354 | binária | 12 | 4 | 5 | 1632 | 1152 |

A usa atrasos de 0%, 1%, 5% e 10%, com todos os rótulos. B usa orçamentos de 1%, 3%, 5%, 10%, 30% e 100%, sem atraso. C cruza essas opções. Cada cenário/modelo tem 34 linhas A/B/C, mas apenas 24 condições únicas. Não contar as linhas equivalentes como novos experimentos independentes. O resultado histórico adicional `20261001_091516` não é incluído.

## 2. Como ler as tabelas

Valores: média ± DP das repetições; DP populacional (`ddof=0`). F1/precisão/recall em %, MCC entre −1 e 1, FP/FN em número de erros e tempo em segundos. FP/FN médios podem ser fracionários. Arredondamento apenas para exibição; os CSVs consolidados preservam a precisão.

Nas tabelas individuais, o DP descreve a variação entre as repetições. Nos resumos gerais, o DP descreve a variação ENTRE CENÁRIOS, e a média dá o mesmo peso a cada cenário; não é F1 após concatenar as previsões. DP não é margem de erro nem intervalo de confiança. Um DP exibido como 0,00 pode ser um valor muito pequeno arredondado.

Δ = binário − multiclasse. F1/precisão/recall maiores são melhores; FP/FN menores são melhores. O alvo positivo avaliado é ATTACK nos dois modos. Treinamento multiclasse não significa avaliação multiclasse.

Atraso (%) é proporcional ao total do cenário, não apenas ao trecho avaliado. Orçamento (%) é probabilidade de consulta; não inclui o treino inicial totalmente rotulado. Não há condição com orçamento zero.

## 3. Referência: todos os rótulos, atraso zero

Média não ponderada entre cenários; DP entre cenários. ΔF1 em pontos percentuais.

| Modelo | F1 multiclasse (%) | F1 binário (%) | ΔF1 (p.p.) |
| --- | --- | --- | --- |
| ARF | 92,35 ± 7,29 | 92,71 ± 6,89 | +0,37 |
| LB | 93,09 ± 5,63 | 93,20 ± 6,44 | +0,11 |
| HAT | 90,42 ± 6,97 | 91,46 ± 6,64 | +1,04 |
| HT | 87,43 ± 8,18 | 89,07 ± 6,78 | +1,64 |

## 4. Todos os resumos por experimento

| Experimento | Binário | Multiclasse |
| --- | --- | --- |
| A | [Treinamento binário](output/ClassificationLabeling/tables/resumos/binary_experimento_A.md) | [Treinamento multiclasse](output/ClassificationLabeling/tables/resumos/multiclass_experimento_A.md) |
| B | [Treinamento binário](output/ClassificationLabeling/tables/resumos/binary_experimento_B.md) | [Treinamento multiclasse](output/ClassificationLabeling/tables/resumos/multiclass_experimento_B.md) |
| C | [Treinamento binário](output/ClassificationLabeling/tables/resumos/binary_experimento_C.md) | [Treinamento multiclasse](output/ClassificationLabeling/tables/resumos/multiclass_experimento_C.md) |

## 5. Todos os resultados por cenário e modelo

Cada tabela individual contém os quatro modelos, os três experimentos e todas as métricas acumuladas.

| Cenário | Total | Treino inicial | Avaliação | Binário | Multiclasse | Comparação |
| --- | --- | --- | --- | --- | --- | --- |
| Consistência_25 | 15.075 | 3.775 | 11.300 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Consistencia_25.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Consistencia_25.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Consistencia_25.md) |
| Consistência_200 | 15.600 | 3.950 | 11.650 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Consistencia_200.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Consistencia_200.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Consistencia_200.md) |
| Consistência_1000 | 18.000 | 4.750 | 13.250 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Consistencia_1000.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Consistencia_1000.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Consistencia_1000.md) |
| Generalização_25 | 15.075 | 3.775 | 11.300 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Generalizacao_25.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Generalizacao_25.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Generalizacao_25.md) |
| Generalização_200 | 15.600 | 3.950 | 11.650 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Generalizacao_200.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Generalizacao_200.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Generalizacao_200.md) |
| Generalização_1000 | 18.000 | 4.750 | 13.250 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Generalizacao_1000.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Generalizacao_1000.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Generalizacao_1000.md) |
| Adaptação_25 | 15.075 | 3.775 | 11.300 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Adaptacao_25.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Adaptacao_25.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Adaptacao_25.md) |
| Adaptação_200 | 15.600 | 3.950 | 11.650 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Adaptacao_200.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Adaptacao_200.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Adaptacao_200.md) |
| Adaptação_1000 | 18.000 | 4.750 | 13.250 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Adaptacao_1000.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Adaptacao_1000.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Adaptacao_1000.md) |
| Recorrência_25 | 15.072 | 2.154 | 12.918 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Recorrencia_25.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Recorrencia_25.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Recorrencia_25.md) |
| Recorrência_200 | 15.600 | 2.242 | 13.358 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Recorrencia_200.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Recorrencia_200.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Recorrencia_200.md) |
| Recorrência_1000 | 18.000 | 2.642 | 15.358 | [A/B/C binário](output/ClassificationLabeling/tables/binary/Recorrencia_1000.md) | [A/B/C multiclasse](output/ClassificationLabeling/tables/multiclass/Recorrencia_1000.md) | [Comparação completa](output/ClassificationLabeling/tables/comparacao/Recorrencia_1000.md) |

## 6. Custos de rotulagem completos

[Todas as consultas, entregas e custos, por cenário e condição](output/ClassificationLabeling/tables/rotulagem.md).

## 7. Resultados por janela e arquivos de origem

As tabelas completas por janela já estão nos CSVs prequenciais abaixo. Cada arquivo contém todos os modelos e A/B/C; não há seleção de melhores janelas. `Window_Index` identifica a janela; `Instance` é a posição final (base zero); `Window_Instances` é o tamanho efetivo. A última janela pode ser menor. Filtre `Experiment`, `Model`, `Delay_Percentage` e `Label_Budget_Percentage` para acompanhar uma condição.

F1/precisão/recall/FP/FN por janela são média e DP das repetições, não métricas acumuladas. A versão binária inclui consultas, entregas e pendências por janela; esses campos não foram registrados no legado multiclasse e não foram preenchidos artificialmente. Janelas sem ataques podem ter F1/recall zero mesmo com previsões benignas corretas.

| Cenário | Binário: acumulado | Binário: janelas | Multiclasse: acumulado | Multiclasse: janelas |
| --- | --- | --- | --- | --- |
| Consistência_25 | [Acumulado](output/ClassificationLabeling/Consist%C3%AAncia_25_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Consist%C3%AAncia_25_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Consist%C3%AAncia_25_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Consist%C3%AAncia_25_Default_FullFeatures_20261001_101354_prequential.csv) |
| Consistência_200 | [Acumulado](output/ClassificationLabeling/Consist%C3%AAncia_200_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Consist%C3%AAncia_200_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Consist%C3%AAncia_200_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Consist%C3%AAncia_200_Default_FullFeatures_20261001_101354_prequential.csv) |
| Consistência_1000 | [Acumulado](output/ClassificationLabeling/Consist%C3%AAncia_1000_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Consist%C3%AAncia_1000_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Consist%C3%AAncia_1000_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Consist%C3%AAncia_1000_Default_FullFeatures_20261001_101354_prequential.csv) |
| Generalização_25 | [Acumulado](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_25_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_25_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_25_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_25_Default_FullFeatures_20261001_101354_prequential.csv) |
| Generalização_200 | [Acumulado](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_200_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_200_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_200_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_200_Default_FullFeatures_20261001_101354_prequential.csv) |
| Generalização_1000 | [Acumulado](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_1000_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_1000_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_1000_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Generaliza%C3%A7%C3%A3o_1000_Default_FullFeatures_20261001_101354_prequential.csv) |
| Adaptação_25 | [Acumulado](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_25_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_25_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_25_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_25_Default_FullFeatures_20261001_101354_prequential.csv) |
| Adaptação_200 | [Acumulado](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_200_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_200_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_200_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_200_Default_FullFeatures_20261001_101354_prequential.csv) |
| Adaptação_1000 | [Acumulado](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_1000_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_1000_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_1000_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Adapta%C3%A7%C3%A3o_1000_Default_FullFeatures_20261001_101354_prequential.csv) |
| Recorrência_25 | [Acumulado](output/ClassificationLabeling/Recorr%C3%AAncia_25_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Recorr%C3%AAncia_25_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Recorr%C3%AAncia_25_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Recorr%C3%AAncia_25_Default_FullFeatures_20261001_101354_prequential.csv) |
| Recorrência_200 | [Acumulado](output/ClassificationLabeling/Recorr%C3%AAncia_200_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Recorr%C3%AAncia_200_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Recorr%C3%AAncia_200_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Recorr%C3%AAncia_200_Default_FullFeatures_20261001_101354_prequential.csv) |
| Recorrência_1000 | [Acumulado](output/ClassificationLabeling/Recorr%C3%AAncia_1000_Default_FullFeatures_binaryTraining_20261007_130844_cumulative.csv) | [Por janela](output/ClassificationLabeling/Recorr%C3%AAncia_1000_Default_FullFeatures_binaryTraining_20261007_130844_prequential.csv) | [Acumulado](output/ClassificationLabeling/Recorr%C3%AAncia_1000_Default_FullFeatures_20261001_101354_cumulative.csv) | [Por janela](output/ClassificationLabeling/Recorr%C3%AAncia_1000_Default_FullFeatures_20261001_101354_prequential.csv) |

## 8. Planilhas CSV para filtrar ou importar no TCC

Separador `;`, codificação UTF-8 com BOM, números com ponto decimal e sem arredondamento deliberado. Ao importar no Excel/LibreOffice, use essas opções e trate `Exec_ID` como texto. Os caminhos de origem são relativos à pasta do CSV consolidado.

| Planilha | Linhas | Conteúdo |
| --- | --- | --- |
| [Acumulados completos (A/B/C)](output/ClassificationLabeling/tables/acumulados_completos.csv) | 3264 | Todas as médias/DP e metadados originais de A/B/C, mais custos derivados. |
| [Condições únicas (sem repetir A/B/C)](output/ClassificationLabeling/tables/condicoes_unicas.csv) | 2304 | 24 condições por cenário/modelo/modo; Experiments_Applicable identifica A/B/C equivalentes. |
| [Comparação binário menos multiclasse](output/ClassificationLabeling/tables/comparacao_binario_multiclasse.csv) | 1152 | Condições emparelhadas; todas as médias/DP dos dois modos e diferenças das médias. |
| [Resumos por modo e experimento](output/ClassificationLabeling/tables/resumos_por_experimento.csv) | 272 | Média e DP entre cenários, em colunas explicitamente separadas das métricas individuais. |
| [Custos de rotulagem](output/ClassificationLabeling/tables/custos_rotulagem.csv) | 288 | Um registro por cenário/atraso/orçamento; custo comum aos modelos e modos. |

## 9. Limitações para o TCC

Esta organização não corrige o protocolo histórico: há pré-processamento com estatísticas globais do cenário, condições dependentes, poucos ataques em alguns cenários e execuções realizadas em dias distintos. Não foram recuperados resultados individuais das cinco repetições; os arquivos existentes registram apenas média e DP. Não inferir significância estatística apenas dos deltas ou DP destas tabelas.

[ANALISE_RESULTADOS_BINARIOS_TCC.md](ANALISE_RESULTADOS_BINARIOS_TCC.md)

[ANALISE_RESULTADOS_TCC.md](ANALISE_RESULTADOS_TCC.md)

## 10. Como regenerar

Execute na raiz do repositório. Apenas as tabelas derivadas e este índice serão reescritos; não há treinamento nem dependência do Java/CapyMOA.

```bash
.venv/bin/python tabulate_classification_labeling_results.py \
  --binary-exec-id 20261007_130844 \
  --multiclass-exec-id 20261001_101354
```

O gerador confere grades completas, duplicações, equivalência A/B/C e compatibilidade entre os modos antes de escrever. Os IDs explícitos evitam misturar execuções parciais ou contar arquivos históricos adicionais. Use `--help` para mudar as pastas, o índice ou a configuração.
