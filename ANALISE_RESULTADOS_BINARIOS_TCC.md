# Análise dos experimentos com treinamento binário e comparação com treinamento multiclasse

Data da análise: 8 de outubro de 2026.

Este relatório analisa os resultados binários já produzidos, compara-os às condições correspondentes da execução multiclasse e apresenta implicações para o TCC. Não foram executados novos treinamentos, alterados os CSVs ou modificada a implementação dos experimentos. A detecção de anomalias permanece fora do escopo.

## 1. Resumo e parecer

A execução binária apresenta resultados coerentes com o protocolo implementado. As 1.431 verificações automáticas realizadas passaram, incluindo a reprodução das consultas aleatórias e da dinâmica de entrega de rótulos por janela.

O treinamento binário melhora o F1 médio da referência totalmente supervisionada nos quatro modelos, mas os ganhos são pequenos na média e não se repetem em todas as condições. Considerando todo o grid de atraso e orçamento, ARF e LB apresentam ganhos médios, HAT fica próximo da equivalência e HT apresenta uma pequena perda média.

O principal resultado científico não é que a classificação binária vence sempre. É que a formulação do alvo modifica a trajetória de aprendizado e seus efeitos dependem do modelo, do orçamento e da sequência de famílias de ataque. Em algumas condições, há ganhos e perdas de mais de 20 pontos percentuais, mesmo quando a média global muda pouco.

A mudança para treinamento binário não resolve a dificuldade com ataques curtos, a escassez de consultas de ataques raros ou a utilização de estatísticas futuras no pré-processamento. Essa última limitação, identificada na análise anterior, permanece nas duas execuções.

Para o objetivo de detectar BENIGN versus ATTACK, o treinamento binário é uma formulação diretamente alinhada ao problema. Isso é uma justificativa de desenho da tarefa, não uma demonstração de superioridade universal de desempenho.

## 2. Execuções comparadas e unidade de análise

| Aspecto | Histórico multiclasse | Execução binária |
|---|---|---|
| Exec_ID | `20261001_101354` | `20261007_130844` |
| Alvo de treinamento | BENIGN e famílias de ataque | BENIGN e ATTACK |
| Avaliação | Binária | Binária |
| Configuração | Default_FullFeatures | Default_FullFeatures |
| Features | 77 | 77 |
| Cenários | 12 | 12 |
| Modelos | ARF, LB, HAT, HT | ARF, LB, HAT, HT |
| Repetições | 5 por condição única | 5 por condição única |
| Sementes de consulta reconstruídas | 42 a 46 | 42 a 46 |
| Janela | 100 instâncias | 100 instâncias |
| Treino inicial | Primeiro trecho benigno e primeira região de ataque | Mesmo prefixo |
| Pré-processamento | MinMaxScaler e mediana globais | Mesmo procedimento |

Os resultados estão em `output/ClassificationLabeling/`. Foram utilizados somente os 12 pares de CSVs de cada execução completa. O resultado adicional antigo de Consistência_25, de `20261001_091516`, não foi incluído.

Cada execução contém 1.632 linhas acumuladas e 204.952 linhas por janela. São 34 condições lógicas por cenário/modelo: quatro de A, seis de B e 24 de C. Há apenas 24 combinações únicas de atraso e orçamento, pois A e B estão contidos em C.

A comparação principal utiliza as 1.152 células únicas de C em cada execução e as emparelha por:

`Dataset + Model + Scenario + Delay_Percentage + Label_Budget_Percentage`

Cada célula já contém a média e o desvio padrão de cinco repetições. Portanto, 1.152 células não são 1.152 observações independentes: modelos, cenários, condições e sementes são compartilhados.

O termo média entre cenários, utilizado neste documento, significa média não ponderada das médias dos 12 cenários. Não significa F1-macro entre classes, nem F1 calculado após concatenar todas as previsões.

F1, precisão e recall estão em porcentagem. As diferenças binário menos multiclasse são apresentadas em pontos percentuais, p.p. Um delta positivo favorece o treinamento binário.

## 3. Por que esta comparação é mais alinhada que a comparação com a tabela padrão antiga

As duas execuções A/B/C usam o mesmo prefixo de treino, o mesmo trecho avaliado, os mesmos cenários e a mesma grade de condições. As contagens de consultas, entregas durante o fluxo e entregas após o término são iguais entre os modos.

A inspeção do Git entre `e51942c` e `bf64f73` não mostra alterações nos cenários de `data/15k`, nos parâmetros de `src/Classification/Models.py`, no cálculo de `src/Results/Metrics.py` ou em `requirements.txt`.

As mudanças relevantes incluem o alvo binário, a preservação das famílias originais para gráficos, o registro explícito do protocolo nos CSVs, os contadores de rótulos por janela e a geração dos gráficos detalhados. A seleção aleatória, o corte do treino inicial, o atraso e a ordem de atualização foram mantidos.

Isso sustenta uma comparação descritiva bem alinhada do efeito da formulação do alvo. Ainda não constitui isolamento experimental absoluto: as execuções aconteceram em momentos diferentes e não registram todo o ambiente de software e hardware.

Essa comparação não utiliza como referência a tabela padrão antiga com reaproveitamento de modelos entre passagens, nem os resultados padrões com warmup de 2.000 instâncias. Essas questões estão discutidas na [análise histórica multiclasse](/home/tomas/repos/anomaly-and-classification-data-stream/ANALISE_RESULTADOS_TCC.md).

### 3.1 A diferença não está na definição das métricas

Na execução antiga, o modelo aprendia a distinguir famílias, mas a avaliação convertia qualquer previsão de ataque em 1. Assim, Syn previsto como DNS já contava como detecção correta de ataque.

Na execução nova, o modelo aprende diretamente o alvo ATTACK. Portanto, a diferença não é somente agrupar as previsões depois: esse agrupamento já existia na avaliação antiga. A mudança ocorre nas classes utilizadas para treinar, nas estatísticas por classe e potencialmente nas decisões e estruturas aprendidas.

Um exemplo conceitual: se uma distribuição de escores multiclasse atribui 0,40 a BENIGN, 0,25 a DNS, 0,20 a Syn e 0,15 a LDAP, BENIGN pode vencer individualmente, embora a soma dos ataques seja 0,60. Esse exemplo ilustra o problema de dividir a evidência entre famílias; não representa probabilidades recuperadas dos modelos deste projeto. Treinar um modelo binário também não equivale simplesmente a somar as probabilidades do modelo antigo, pois o próprio aprendizado muda.

As famílias originais são preservadas como metadados para interpretar os gráficos. O código passa as instâncias com alvo binário ao treinamento, não as famílias originais. Assim, o rótulo LDAP apresentado em um gráfico não significa que o classificador binário aprendeu a prever LDAP.

## 4. Auditoria dos resultados e controles de equivalência

As 1.431 verificações incluíram:

- cobertura dos 12 cenários, quatro modelos e todas as condições;
- cinco repetições, janela 100 e modos de treinamento/avaliação esperados;
- ausência de valores ausentes nas tabelas carregadas;
- inexistência de duplicações de chaves e limites válidos das métricas;
- igualdade de A com C a 100% e de B com C a atraso zero;
- conservação entre consultas, entregas no fluxo e entregas finais;
- percentuais efetivos e atrasos compatíveis com os totais originais;
- recall compatível com os positivos dos cenários e os FN registrados;
- soma dos FP/FN das janelas igual aos totais acumulados;
- cobertura integral do trecho avaliado e alinhamento das janelas entre execuções;
- reprodução da média e do DP de consultas com as sementes 42 a 46;
- reprodução da média e do DP de consultas, entregas e pendências por janela na execução binária;
- saldo da fila por janela e correspondência entre a última pendência e as entregas após o fim;
- equivalência das métricas de Consistência entre os dois modos.

Passar nessas verificações demonstra consistência dentro dos aspectos auditados. Não prova ausência de qualquer erro, não reconstrói todas as previsões originais e não elimina limitações do desenho experimental.

### 4.1 Consistência funciona como controle negativo

Os cenários de Consistência contêm somente BENIGN e DrDoS_DNS. A versão chamada multiclasse já tinha duas classes nesses cenários. Transformar DNS em ATTACK modifica o significado do nome, mas não divide nem reúne diferentes famílias.

As 288 células únicas de Consistência, três tamanhos × quatro modelos × 24 condições, apresentam igualdade exata de todas as médias e DP das métricas acumuladas comparadas. As métricas por janela também coincidem.

Esse resultado é esperado e fortalece a evidência de que não houve uma alteração generalizada de consultas, métricas ou posição das janelas entre as execuções.

### 4.2 Antes da primeira família nova, os resultados também coincidem

Nos outros nove cenários, foram identificadas 27.360 células de janela de C inteiramente anteriores à primeira família de ataque diferente de DNS. As médias e DP de F1, precisão, recall, FP e FN coincidem exatamente entre os modos nesse trecho.

As diferenças aparecem posteriormente, quando as trajetórias de atualização passam a envolver famílias que são separadas no treino multiclasse e agrupadas no binário. Esse alinhamento temporal é compatível com o efeito da mudança do alvo. Não identifica, sozinho, o mecanismo interno responsável por cada diferença.

## 5. Composição dos cenários e fatores de confusão

Os dados não mudaram entre as execuções. Sua composição continua sendo:

| Cenário | Treino inicial | Instâncias avaliadas | Ataques avaliados | Prevalência de ataque (%) | Classes no treino multiclasse |
|---|---:|---:|---:|---:|---:|
| Adaptação_25 | 3.775 | 11.300 | 46 | 0,41 | 3 |
| Adaptação_200 | 3.950 | 11.650 | 393 | 3,37 | 3 |
| Adaptação_1000 | 4.750 | 13.250 | 1.933 | 14,59 | 3 |
| Consistência_25 | 3.775 | 11.300 | 47 | 0,42 | 2 |
| Consistência_200 | 3.950 | 11.650 | 327 | 2,81 | 2 |
| Consistência_1000 | 4.750 | 13.250 | 1.859 | 14,03 | 2 |
| Generalização_25 | 3.775 | 11.300 | 48 | 0,42 | 3 |
| Generalização_200 | 3.950 | 11.650 | 395 | 3,39 | 3 |
| Generalização_1000 | 4.750 | 13.250 | 1.935 | 14,60 | 3 |
| Recorrência_25 | 2.154 | 12.918 | 58 | 0,45 | 4 |
| Recorrência_200 | 2.242 | 13.358 | 483 | 3,62 | 4 |
| Recorrência_1000 | 2.642 | 15.358 | 2.162 | 14,08 | 4 |

A última coluna indica as classes existentes no cenário e no espaço de classes do modelo, não que todas estejam presentes no prefixo inicial. O prefixo contém BENIGN e DNS; outras famílias aparecem depois. O treino binário tem duas classes em todos os casos.

As sequências de ataque continuam sendo DNS–DNS–DNS em Consistência, DNS–LDAP–DNS em Generalização, DNS–Syn–DNS em Adaptação e DNS–Syn–LDAP–DNS–Syn–LDAP em Recorrência.

O tamanho nominal do bloco não corresponde sempre à quantidade efetiva de ataques. Há benignos em algumas regiões extraídas dos arquivos de ataque. Além disso, Recorrência utiliza seis regiões com spans aproximadamente 12, 100 ou 500, enquanto as outras categorias utilizam três regiões com spans 25, 200 ou 1000.

Ao comparar tamanhos e categorias, mudam duração, prevalência, quantidade de positivos no treino e extensão do prefixo inicial. Portanto, uma diferença entre categorias não isola somente a natureza da mudança de distribuição. A comparação binário versus multiclasse dentro de cada cenário é mais controlada que uma comparação direta entre categorias diferentes.

## 6. Treinamento binário: referência integral e imediata

### 6.1 F1 por cenário

Referência: atraso zero e orçamento de 100%. Valores médios de cinco repetições, em porcentagem.

| Cenário | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| Adaptação_25 | 78,23 | 84,82 | 75,95 | 76,92 |
| Adaptação_200 | 94,74 | 89,94 | 94,67 | 86,00 |
| Adaptação_1000 | 98,90 | 97,74 | 97,33 | 84,44 |
| Consistência_25 | 90,69 | 95,14 | 88,64 | 88,64 |
| Consistência_200 | 93,19 | 93,62 | 92,21 | 91,59 |
| Consistência_1000 | 98,91 | 98,77 | 97,84 | 94,96 |
| Generalização_25 | 88,68 | 95,46 | 89,13 | 89,13 |
| Generalização_200 | 97,11 | 97,58 | 97,19 | 95,19 |
| Generalização_1000 | 98,99 | 99,09 | 98,60 | 96,40 |
| Recorrência_25 | 80,01 | 76,43 | 81,63 | 76,29 |
| Recorrência_200 | 94,43 | 92,10 | 90,18 | 91,95 |
| Recorrência_1000 | 98,70 | 97,67 | 94,13 | 97,34 |

LB lidera seis cenários, ARF cinco e HAT um. HAT passa a liderar Recorrência_25, com 81,63%. HT não lidera nenhuma referência individual, embora apresente ganhos relevantes na comparação de alguns cenários.

### 6.2 Médias entre os 12 cenários

| Modelo | F1 (%) | Precisão (%) | Recall (%) | MCC | Taxa de FP (%) | Tempo médio (s) |
|---|---:|---:|---:|---:|---:|---:|
| ARF | 92,71 | 92,62 | 93,01 | 0,9262 | 0,1250 | 28,38 |
| LB | 93,20 | 97,05 | 90,28 | 0,9325 | 0,0937 | 24,76 |
| HAT | 91,46 | 96,62 | 87,37 | 0,9148 | 0,1264 | 2,21 |
| HT | 89,07 | 96,13 | 83,57 | 0,8902 | 0,2076 | 2,02 |

A taxa de FP é calculada como FP dividido pelos benignos avaliados em cada cenário, antes da média entre cenários. Ela não é o complemento da precisão.

LB tem o maior F1 médio e a menor taxa média de FP; ARF tem o maior recall médio. Portanto, LB é uma referência forte quando a prioridade inclui reduzir alarmes falsos, enquanto ARF é uma referência forte para sensibilidade aos ataques. Essa é uma leitura dos resultados observados, não uma recomendação universal de implantação.

A diferença média de F1 entre LB e ARF é aproximadamente 0,48 p.p. Não foi demonstrada significância estatística dessa diferença.

HAT e HT são muito mais rápidos nesta execução. HAT alcança F1 relativamente próximo dos ensembles, mas com recall menor. O tempo médio deve ser interpretado com as ressalvas da seção de custo computacional.

## 7. Comparação da referência binária com a multiclasse

### 7.1 Ganho médio por modelo

| Modelo | F1 multiclasse (%) | F1 binário (%) | Delta F1 (p.p.) | Recall multiclasse (%) | Recall binário (%) |
|---|---:|---:|---:|---:|---:|
| ARF | 92,35 | 92,71 | +0,37 | 92,22 | 93,01 |
| LB | 93,09 | 93,20 | +0,11 | 89,66 | 90,28 |
| HAT | 90,42 | 91,46 | +1,04 | 85,67 | 87,37 |
| HT | 87,43 | 89,07 | +1,64 | 81,68 | 83,57 |

Os quatro modelos melhoram em F1 médio e recall médio nesta referência. A precisão média de ARF e LB cai um pouco: de 92,71% para 92,62%, e de 97,41% para 97,05%, respectivamente. HAT e HT apresentam aumento de precisão média.

Assim, a melhora dos ensembles não significa redução simultânea de todos os tipos de erro em todos os cenários. Há um compromisso entre sensibilidade e alarmes falsos.

Entre os 48 pares da referência, 20 favorecem o binário, dez favorecem o multiclasse e 18 são iguais. Essas contagens são descritivas, não um teste de superioridade.

### 7.2 Delta de F1 por cenário

Binário menos multiclasse, em p.p.:

| Cenário | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| Adaptação_25 | -0,33 | -0,79 | 0,00 | 0,00 |
| Adaptação_200 | -0,75 | +3,61 | 0,00 | -0,12 |
| Adaptação_1000 | -0,01 | +1,41 | -0,09 | -0,61 |
| Consistência_25 | 0,00 | 0,00 | 0,00 | 0,00 |
| Consistência_200 | 0,00 | 0,00 | 0,00 | 0,00 |
| Consistência_1000 | 0,00 | 0,00 | 0,00 | 0,00 |
| Generalização_25 | -0,22 | 0,00 | 0,00 | 0,00 |
| Generalização_200 | +1,63 | +0,42 | +0,79 | +0,14 |
| Generalização_1000 | +0,02 | +0,40 | +1,71 | +0,97 |
| Recorrência_25 | +3,24 | -3,98 | +3,74 | +10,27 |
| Recorrência_200 | +0,55 | -0,26 | +2,71 | +4,21 |
| Recorrência_1000 | +0,26 | +0,53 | +3,60 | +4,82 |

O maior ganho da referência é HT em Recorrência_25, de 66,02% para 76,29%, ou +10,27 p.p. O maior prejuízo é LB nesse mesmo cenário, de 80,41% para 76,43%, ou -3,98 p.p.

Logo, nem mesmo dentro de um único cenário o alvo binário ajuda necessariamente todos os modelos. Para ARF, os três cenários de Adaptação apresentam pequenas perdas na referência, apesar do ganho médio global.

## 8. Experimento A binário: somente atraso

### 8.1 Resultados absolutos

F1 médio entre os 12 cenários, com 100% de consultas:

| Atraso nominal | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| 0% | 92,71 | 93,20 | 91,46 | 89,07 |
| 1% | 63,45 | 71,00 | 54,59 | 57,27 |
| 5% | 40,54 | 41,01 | 42,32 | 40,56 |
| 10% | 31,45 | 33,62 | 43,74 | 37,45 |

O atraso continua sendo altamente prejudicial. Com 1%, ARF perde 29,26 p.p. em relação à sua referência binária, LB perde 22,20 p.p., HAT perde 36,87 p.p. e HT perde 31,80 p.p.

Com atraso de 10%, o F1 de ARF cai para 31,45%, acompanhado de precisão média de 30,66% e recall de 53,92%. Sua taxa média de FP aumenta de 0,1250% para 6,8471%. A degradação envolve tanto ataques não detectados quanto alarmes falsos.

HAT tem o maior F1 médio com 5% e 10% de atraso, mas isso não significa melhor detecção de todos os ataques: com 10%, seu recall médio é 38,99%, abaixo dos 53,92% de ARF e 53,59% de LB. Sua precisão média é maior, 61,46%. O ranking por F1 incorpora esse compromisso.

### 8.2 Diferenças em relação ao multiclasse

Delta de F1 médio, em p.p.:

| Atraso nominal | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| 0% | +0,37 | +0,11 | +1,04 | +1,64 |
| 1% | +2,10 | +0,06 | -1,08 | +2,55 |
| 5% | +2,72 | +0,86 | +0,59 | +2,53 |
| 10% | +2,56 | +0,72 | +0,10 | +1,55 |

ARF e HT apresentam ganhos médios em todas as condições de A. LB muda pouco. HAT perde com atraso de 1%, apesar de melhorar na referência sem atraso.

Esses ganhos não eliminam a queda causada pelo atraso. Por exemplo, ARF binário com 10% melhora 2,56 p.p. frente ao multiclasse correspondente, mas permanece 61,26 p.p. abaixo de sua própria referência sem atraso.

### 8.3 Duração dos ataques e atraso relativo

Média de F1 entre quatro categorias e quatro modelos:

| Tamanho nominal | Atraso 0% | Atraso 1% | Atraso 5% | Atraso 10% |
|---|---:|---:|---:|---:|
| 25 | 84,74 | 38,54 | 21,66 | 17,61 |
| 200 | 93,23 | 58,47 | 43,99 | 37,69 |
| 1000 | 96,86 | 87,73 | 57,68 | 54,40 |

O atraso é `round(N × proporção)`, com N incluindo o treino inicial. Para os tamanhos 25, 200 e 1000, o atraso de 1% é aproximadamente 151, 156 e 180 posições, respectivamente.

Um atraso de 151 ultrapassa uma região de 25 posições, e ultrapassa ainda mais as regiões de 12 posições de Recorrência_25. O modelo não pode utilizar rótulos da própria região antes que ela termine. Esse limite de disponibilidade de informação permanece, independentemente de agrupar os ataques em uma classe.

O efeito do tamanho também está confundido com prevalência e quantidade de positivos no treino. A tabela não demonstra um efeito isolado da duração.

## 9. Experimento B binário: somente Random Sampling

### 9.1 Desempenho e orçamento

F1 médio entre os 12 cenários, sem atraso:

| Orçamento | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| 1% | 47,82 | 48,80 | 56,31 | 48,57 |
| 3% | 59,26 | 62,64 | 59,01 | 51,25 |
| 5% | 65,96 | 68,50 | 61,67 | 56,99 |
| 10% | 73,76 | 76,18 | 66,72 | 63,28 |
| 30% | 83,94 | 84,90 | 82,37 | 78,57 |
| 100% | 92,71 | 93,20 | 91,46 | 89,07 |

Os quatro modelos melhoram na média à medida que o orçamento aumenta. Com 30%, LB mantém aproximadamente 91,1% do F1 de sua referência integral, mas perde 8,30 p.p.; ARF perde 8,78 p.p.

Com 1%, HAT tem o maior F1 médio, 56,31%, embora fique muito abaixo dos 91,46% da sua referência. Nos ensembles, um recall relativamente alto pode coexistir com precisão baixa: ARF com 1% tem recall médio de 75,23% e precisão de 42,98%; LB tem 71,66% e 44,26%.

Portanto, reduzir consultas compromete tanto a aquisição de exemplos de ataque quanto o feedback corretivo de benignos.

### 9.2 Comparação com o multiclasse

Delta de F1 médio, em p.p.:

| Orçamento | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| 1% | +0,40 | +0,51 | +0,06 | -0,64 |
| 3% | +1,84 | +1,12 | +0,79 | -0,13 |
| 5% | +0,76 | +1,44 | -0,03 | -0,78 |
| 10% | +1,29 | +1,60 | +0,90 | -1,96 |
| 30% | +1,22 | +1,50 | +2,28 | -0,63 |
| 100% | +0,37 | +0,11 | +1,04 | +1,64 |

Um contraste importante: HT melhora com 100%, mas piora em todos os cinco orçamentos reduzidos, na média dos cenários. Assim, a conclusão da referência totalmente supervisionada não deve ser automaticamente estendida ao aprendizado com poucas consultas.

ARF e LB apresentam ganhos médios em todos os orçamentos de B, mas existem perdas individuais. Em Adaptação_1000 com HT e orçamento de 1%, o F1 cai de 74,35% para 63,33%. Os FN médios diminuem de 671,4 para 551,0, mas os FP aumentam de 226,6 para 1.067,0. O modelo detecta mais ataques e, mesmo assim, piora em F1 por produzir muito mais alarmes falsos.

Esse caso mostra por que a comparação precisa incluir FP/FN e precisão/recall, não somente F1.

### 9.3 Ataques raros continuam pouco consultados

Média de F1 entre categorias e modelos:

| Tamanho nominal | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 25 | 11,36 | 19,64 | 25,19 | 37,22 | 64,05 | 84,74 |
| 200 | 59,40 | 68,29 | 75,00 | 80,59 | 87,47 | 93,23 |
| 1000 | 80,36 | 86,19 | 89,65 | 92,15 | 95,82 | 96,86 |

As seleções de instâncias são exatamente as mesmas nos dois modos. Em Adaptação_25, o orçamento de 1% continua consultando apenas 0,6 ataque avaliado por repetição, em média; três das cinco repetições não consultam nenhum ataque da avaliação.

Isso não significa ausência de treino rotulado: o prefixo inicial continua integralmente supervisionado.

Com k ataques e probabilidade de consulta p, a chance de nenhuma consulta é `(1 − p)^k`. Para 46 ataques e p = 1%, ela é aproximadamente 62,98%. O alvo binário não muda essa probabilidade porque a estratégia de seleção não utiliza as previsões ou as classes.

O problema principal dos menores cenários continua sendo escassez de exemplos positivos consultados, prevalência muito baixa e falta de feedback suficientemente frequente. Treinar com duas classes não fornece os rótulos que não foram consultados.

## 10. Experimento C binário: resultados completos

F1 médio entre os 12 cenários. Linhas: atraso nominal. Colunas: orçamento.

### 10.1 ARF

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 47,82 | 59,26 | 65,96 | 73,76 | 83,94 | 92,71 |
| 1% | 44,74 | 52,74 | 58,16 | 60,71 | 63,47 | 63,45 |
| 5% | 38,41 | 42,47 | 43,20 | 42,47 | 38,95 | 40,54 |
| 10% | 35,19 | 37,88 | 37,77 | 37,22 | 33,63 | 31,45 |

### 10.2 LB

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 48,80 | 62,64 | 68,50 | 76,18 | 84,90 | 93,20 |
| 1% | 45,35 | 56,25 | 59,75 | 64,13 | 68,68 | 71,00 |
| 5% | 39,39 | 44,42 | 45,51 | 47,39 | 46,80 | 41,01 |
| 10% | 36,35 | 39,70 | 39,77 | 41,11 | 41,01 | 33,62 |

### 10.3 HAT

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 56,31 | 59,01 | 61,67 | 66,72 | 82,37 | 91,46 |
| 1% | 55,18 | 54,90 | 56,21 | 57,07 | 53,80 | 54,59 |
| 5% | 54,89 | 52,69 | 53,48 | 53,44 | 44,18 | 42,32 |
| 10% | 55,89 | 52,99 | 53,34 | 54,03 | 47,31 | 43,74 |

### 10.4 HT

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | 48,57 | 51,25 | 56,99 | 63,28 | 78,57 | 89,07 |
| 1% | 45,81 | 45,45 | 48,61 | 50,46 | 54,02 | 57,27 |
| 5% | 40,53 | 38,18 | 39,61 | 41,39 | 40,00 | 40,56 |
| 10% | 41,29 | 37,28 | 37,28 | 40,32 | 38,44 | 37,45 |

### 10.5 Interação entre atraso e consultas

A relação entre orçamento e F1 deixa de ser monotônica quando há atraso. Com atraso de 10%, HAT apresenta 55,89% de F1 médio a 1%, contra 43,74% a 100%. LB tem 41,11% a 10% de consultas, contra 33,62% a 100%.

Esses resultados mostram que mais atualizações atrasadas não são necessariamente melhores nesta execução. Não mostram que orçamento baixo seja sempre preferível: sem atraso, os maiores orçamentos têm maior F1 médio, e a referência de HAT é 91,46%, muito acima dos resultados com atraso.

Também não basta olhar a melhor célula de cada linha e declarar um orçamento ótimo. Essa escolha é posterior à observação dos dados, depende da grade testada e precisa de validação em cenários adicionais.

A literatura de active learning com verification latency destaca que o efeito da disponibilidade tardia precisa ser considerado conjuntamente com mudanças no fluxo. Isso contextualiza a interação observada, sem provar qual mecanismo interno causou cada inversão. [Castellani, Schmitt e Hammer, 2022](https://arxiv.org/abs/2204.06822).

## 11. Experimento C: diferenças em relação ao multiclasse

As tabelas mostram delta de F1 médio em p.p. Cada célula compara condições alinhadas nos 12 cenários, usando as médias não arredondadas antes da subtração.

### 11.1 ARF

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | +0,40 | +1,84 | +0,76 | +1,29 | +1,22 | +0,37 |
| 1% | +0,34 | +1,41 | +0,59 | +1,26 | +4,07 | +2,10 |
| 5% | +0,34 | +0,41 | +0,96 | +1,47 | +3,22 | +2,72 |
| 10% | +0,11 | +0,22 | +0,30 | +0,58 | +2,21 | +2,56 |

ARF apresenta ganho médio em todas as 24 condições, embora não em todos os cenários individuais. O maior ganho médio é de 4,07 p.p. com atraso de 1% e orçamento de 30%.

### 11.2 LB

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | +0,51 | +1,12 | +1,44 | +1,60 | +1,50 | +0,11 |
| 1% | +0,15 | +0,47 | +0,62 | +0,80 | +1,97 | +0,06 |
| 5% | -0,35 | -0,33 | -0,30 | +0,54 | +2,59 | +0,86 |
| 10% | -0,26 | -0,15 | -0,18 | +0,75 | +2,44 | +0,72 |

LB melhora principalmente com orçamento pelo menos 10%, ou atraso baixo. Em atrasos de 5% e 10%, os três menores orçamentos apresentam pequenas perdas médias.

### 11.3 HAT

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | +0,06 | +0,79 | -0,03 | +0,90 | +2,28 | +1,04 |
| 1% | -0,18 | -0,27 | -0,45 | +0,86 | -0,94 | -1,08 |
| 5% | -0,12 | -0,14 | +0,05 | +1,15 | -1,50 | +0,59 |
| 10% | -0,21 | +0,26 | +0,09 | +1,40 | -1,11 | +0,10 |

O saldo médio pequeno de HAT esconde resultados muito diferentes entre cenários. Há melhoras fortes em Generalização e perdas fortes em algumas condições de Recorrência e Adaptação.

### 11.4 HT

| Atraso | 1% | 3% | 5% | 10% | 30% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| 0% | -0,64 | -0,13 | -0,78 | -1,96 | -0,63 | +1,64 |
| 1% | -0,54 | -1,62 | -1,47 | -1,73 | -2,26 | +2,55 |
| 5% | -0,16 | -0,81 | -0,32 | -0,44 | -1,87 | +2,53 |
| 10% | -0,08 | -0,74 | -0,65 | -0,24 | -1,97 | +1,55 |

HT melhora na média com orçamento de 100% em todos os atrasos, mas piora na média em todas as 20 condições de orçamento reduzido. É o exemplo mais claro de que uma conclusão sobre o alvo depende também da disponibilidade de rótulos.

### 11.5 Saldo nas 1.152 células únicas

Cada modelo tem 288 pares: 12 cenários × 24 condições. Foi utilizada tolerância de 1e-9 p.p. para a contagem de empates.

| Modelo | Delta médio no grid (p.p.) | Pares melhores no binário | Pares piores no binário | Empates |
|---|---:|---:|---:|---:|
| ARF | +1,28 | 141 | 73 | 74 |
| LB | +0,69 | 110 | 82 | 96 |
| HAT | +0,15 | 65 | 73 | 150 |
| HT | -0,45 | 62 | 93 | 133 |
| Total | +0,42 | 378 | 321 | 453 |

A média total atribui o mesmo peso a todos os modelos, cenários e condições do grid. Esse peso é uma escolha descritiva, não uma representação da frequência dessas condições em produção.

As medianas dos deltas por modelo são zero, em parte porque Consistência contribui com 288 empates. Excluindo essa categoria que já tinha duas classes, o ganho médio total é aproximadamente 0,56 p.p.; a heterogeneidade e as perdas individuais continuam presentes.

Não se pode interpretar as contagens como um teste estatístico com 1.152 amostras independentes.

### 11.6 Saldo por categoria e modelo

Delta médio entre todas as 24 condições e os três tamanhos:

| Categoria | ARF | LB | HAT | HT |
|---|---:|---:|---:|---:|
| Adaptação | +1,33 | +0,23 | -1,03 | -2,44 |
| Consistência | 0,00 | 0,00 | 0,00 | 0,00 |
| Generalização | +0,92 | +0,85 | +1,91 | +3,03 |
| Recorrência | +2,87 | +1,69 | -0,29 | -2,39 |

Generalização apresenta ganho médio nos quatro modelos. Recorrência melhora nos ensembles, mas piora ligeiramente em HAT e mais em HT quando todo o grid é considerado. Isso contrasta com os ganhos das árvores na referência totalmente supervisionada de Recorrência.

## 12. Estudos temporais: onde os ganhos e perdas acontecem

Os CSVs por janela foram alinhados às regiões dos rótulos originais. Nas regiões analisadas, cada janela com positivos contém ataques de uma única família; isso permite associar os FN à família. Essa atribuição mede recall de detecção sobre aquela família, não identificação correta da família pelo modelo.

FP em trechos benignos não foi atribuído causalmente a uma família de ataque.

### 12.1 Ganho de HAT em Generalização_1000: melhor retorno de DNS

Condição: atraso de 5%, equivalente a 900 posições, e orçamento de 100%.

| Medida | Multiclasse | Binário |
|---|---:|---:|
| F1 acumulado (%) | 47,56 | 66,11 |
| FP | 52 | 54 |
| FN | 1.315 | 953 |
| FN no bloco LDAP, 1.000 ataques | 902 | 903 |
| Recall de detecção no bloco LDAP (%) | 9,80 | 9,70 |
| FN no retorno DNS, 935 ataques | 413 | 50 |
| Recall de detecção no retorno DNS (%) | 55,83 | 94,65 |

O ganho de 18,54 p.p. de F1 não resulta de uma melhor detecção inicial da família nova LDAP: ela permanece quase toda perdida nos dois modos. A melhora ocorre principalmente na região posterior DNS.

Esse achado é importante para a redação do TCC. Não seria correto resumir esse caso como treinamento binário aprendeu melhor o ataque desconhecido. A evidência mostra melhor desempenho no retorno da família previamente observada, depois de uma trajetória de atualização diferente.

Uma hipótese é que a separação ou reunião das famílias afete o estado aprendido após as atualizações tardias. Sem logs de árvores, folhas e detectores de drift, não é possível determinar o mecanismo exato.

### 12.2 Perda de HAT em Recorrência_200: falha na última ocorrência LDAP

Condição: atraso de 1%, equivalente a 156 posições, e orçamento de 100%.

| Medida | Multiclasse | Binário |
|---|---:|---:|
| F1 acumulado (%) | 49,31 | 21,79 |
| FP | 6 | 16 |
| FN | 323 | 422 |

Recall de detecção por região avaliada:

| Região | Família | Ataques | Multiclasse (%) | Binário (%) |
|---|---|---:|---:|---:|
| 2 | Syn | 98 | 0,00 | 0,00 |
| 3 | LDAP | 100 | 0,00 | 0,00 |
| 4 | DNS | 85 | 71,76 | 71,76 |
| 5 | Syn | 100 | 1,00 | 0,00 |
| 6 | LDAP | 100 | 98,00 | 0,00 |

Dos 99 FN adicionais do binário, 98 estão na última ocorrência LDAP e um na segunda ocorrência Syn. O déficit de F1 de 27,52 p.p. decorre principalmente da perda de detecção de um ataque recorrente, com aumento adicional dos FP.

As primeiras ocorrências Syn e LDAP são perdidas por ambos. O atraso de 156 supera a duração de suas regiões de aproximadamente 100 posições, mas os rótulos dessas ocorrências chegam antes das ocorrências posteriores. Portanto, a falha na última LDAP não pode ser explicada simplesmente dizendo que seu tipo nunca foi rotulado; é preciso investigar o estado mantido pelo modelo após o aprendizado intermediário.

O alvo agrupado pode facilitar a transferência entre ataques em alguns casos e prejudicar a retenção de diferenças úteis em outros. Essa é uma hipótese compatível com os resultados, não um mecanismo demonstrado.

O gráfico binário detalhado desse caso pode ser consultado em [FP/FN e dinâmica dos rótulos de HAT em Recorrência_200](/home/tomas/repos/anomaly-and-classification-data-stream/output/ClassificationLabeling/plots-binary/stream/binary/Default_FullFeatures/Recorrência_200/20261007_130844/HoeffdingAdaptiveTree/experiment_A_delay_1pct_labels_100pct_FP_FN_Labeling.png).

### 12.3 HT em Recorrência_25: ganho com supervisão completa, perda com 30%

Com atraso de 5% e 100% de consultas, o F1 de HT muda de 0,00% para 26,87%. A versão multiclasse perde os 58 ataques avaliados. A binária perde 49 e detecta nove, todos na segunda ocorrência Syn, que contém dez ataques. Não há FP na condição binária.

Apesar do delta grande, o recall acumulado binário é apenas 15,52%. Portanto, essa melhora não representa desempenho satisfatório para o cenário inteiro; representa recuperação localizada em uma região de ataque.

Com atraso de 1% e orçamento de 30%, a direção se inverte:

- multiclasse: F1 de 28,71% ± 14,44 p.p., FP médio zero e FN médio 47,8;
- binário: F1 zero, FP médio 2,0 e FN médio 58,0.

O multiclasse detecta, em média, três ataques na segunda Syn e 7,2 na segunda LDAP. O binário não detecta nenhum nessas regiões. As consultas são as mesmas, mas a trajetória de atualização com rótulos parciais é diferente daquela com supervisão completa.

O mesmo déficit de 28,71 p.p. aparece para HT a 30% nos atrasos de 5% e 10%. Essas três células não devem ser tratadas como três evidências independentes de um fenômeno: elas compartilham cenário, sementes e trajetórias fortemente relacionadas.

### 12.4 ARF em Recorrência_25: ganho por menos FN

Com atraso de 1% e 100% de consultas, o F1 muda de 28,77% ± 6,76 p.p. para 46,11% ± 2,46 p.p.

Os FP médios permanecem em 16,2; os FN caem de 45,4 para 35,8. A melhora vem da sensibilidade, não de reduzir alarmes falsos:

- retorno DNS: recall de 41,67% para 83,33%;
- segunda Syn: de 0% para 4%;
- segunda LDAP: de 63,33% para 98,33%.

As primeiras ocorrências Syn e LDAP continuam com recall zero. O benefício aparece nas ocorrências posteriores, depois de existir oportunidade de receber rótulos das regiões anteriores.

### 12.5 Adaptação_25: treinamento binário não resolve o ataque curto e pode aumentar a instabilidade

Com atraso de 1% e 100% de consultas, os quatro modelos continuam deixando passar os 23 ataques Syn dessa região. Não há feedback da própria região disponível durante sua duração.

Em ARF, o F1 multiclasse de 36,99% ± 0,70 p.p. passa a 29,01% ± 14,52 p.p. no binário. Os FN na região DNS posterior aumentam de 1,0 para 5,6; os FP totais ficam próximos, 51,0 e 51,6.

A perda se concentra no retorno DNS, acompanhada de maior variabilidade entre repetições. Não é uma mudança na quantidade de ataques Syn perdidos, que continua igual.

LB também piora, de 59,95% para 57,62%, principalmente pelo aumento dos FP de 5,4 para 8,4, mantendo 24 FN. HAT e HT continuam com F1 zero nessa condição.

Esse caso reforça que reduzir o número de classes não garante maior estabilidade nem elimina a limitação temporal de aquisição dos rótulos.

### 12.6 Hipóteses para investigar as diferenças internas

Uma hipótese concreta vem das configurações de HAT e HT em `src/Classification/Models.py`: ambos utilizam `NaiveBayesAdaptive` nas folhas e `GaussianNumericAttributeClassObserver` para atributos numéricos. A documentação descreve esses parâmetros como o mecanismo de predição das folhas e o observador das estatísticas numéricas por classe. [Documentação de Hoeffding Tree](https://capymoa.org/api/modules/capymoa.classifier.HoeffdingTree.html), [documentação de Hoeffding Adaptive Tree](https://capymoa.org/v0.15.1/api/modules/capymoa.classifier.HoeffdingAdaptiveTree.html).

Reunir DNS, Syn e LDAP em ATTACK pode reduzir a fragmentação de exemplos entre classes e facilitar o compartilhamento de evidência. Por outro lado, famílias com distribuições diferentes passam a contribuir para as mesmas estatísticas de uma classe. Em componentes baseados em aproximações gaussianas, uma distribuição reunida pode ser mais difícil de representar que distribuições separadas. Também podem mudar os critérios de divisão e as decisões de predição das folhas.

Essa é uma explicação possível para o fato de o binário ajudar em alguns cenários e prejudicar em outros. Não demonstra que o mecanismo gaussiano ou Naive Bayes estava efetivamente decidindo nas folhas responsáveis pelos erros observados; o modo adaptativo e os estados das folhas não foram registrados.

Nos modelos com mecanismos de adaptação, mudanças nos erros de previsão também podem alterar a trajetória desses mecanismos. Os CSVs não contêm os logs necessários para afirmar quantas detecções de drift, substituições ou divisões ocorreram. A hipótese deve orientar instrumentação, não ser apresentada como causa confirmada no TCC.

## 13. O que os novos contadores de rótulos permitem verificar

Os resultados binários acrescentam consultas, entregas e pendências por janela. A reconstrução confirmou suas médias e DP em todas as combinações únicas.

O saldo médio da fila obedece a:

`pendentes_fim_da_janela = pendentes_fim_anterior + consultados_na_janela − entregues_na_janela`

Por exemplo, em Recorrência_200 com atraso de 156 e orçamento de 100%, cada janela completa consulta 100 rótulos. Após o período inicial de preenchimento da fila, cada janela também recebe 100 rótulos, mas são de instâncias anteriores. A fila mantém 156 pendências ao fim das janelas estabilizadas.

Assim, igualdade entre consultas e entregas por janela não significa atraso zero. Há igual vazão de solicitações e respostas, mas com identidades defasadas.

Nesse caso, 13.358 rótulos são consultados; 13.202 chegam durante o fluxo e 156 depois do término. As atualizações finais não alteram as previsões já avaliadas.

Nos orçamentos menores, as pendências variam entre janelas e sementes. Sua contagem não é uma cota fixa. Esses dados ajudam a distinguir ausência de consulta, atraso de chegada e incapacidade de beneficiar a avaliação antes do fim.

Os CSVs multiclasse antigos não persistem os mesmos contadores por janela, mas suas contagens acumuladas coincidem com as binárias, e a seleção pode ser reconstruída a partir das sementes.

## 14. Custo total de rotulagem e compromisso com desempenho

O orçamento p incide somente sobre a avaliação. O prefixo inicial continua totalmente rotulado nas duas execuções.

`custo_total_percentual = 100 × (treino_inicial + consultas_na_avaliação) / N`

As quantidades e os custos são iguais entre modos:

| Orçamento nominal | Consulta efetiva na avaliação, média (%) | Custo total no cenário, média (%) |
|---|---:|---:|
| 1% | 1,01 | 23,58 |
| 3% | 2,91 | 25,04 |
| 5% | 4,97 | 26,64 |
| 10% | 10,05 | 30,55 |
| 30% | 30,28 | 46,18 |
| 100% | 100,00 | 100,00 |

Com 30% e atraso zero, LB apresenta F1 médio de 84,90% utilizando aproximadamente 46,18% dos rótulos do cenário inteiro, contra 93,20% com todos os rótulos. É um compromisso interessante para estudo, mas não há um limiar operacional de F1 ou recall definido para declarar essa configuração suficiente.

Na média dos cenários de tamanho 25, o F1 com 30% é somente 64,05%, contra 84,74% com 100%. Uma média global pode esconder essa insuficiência nos ataques mais curtos.

Com orçamento de 1%, não é correto anunciar economia de 99% dos rótulos considerando todo o cenário: o custo total médio é de 23,58%. Essas medidas representam quantidades por execução, não custo monetário nem tempo de um anotador humano.

O atraso, isoladamente, não reduz solicitações. Em A, todas as instâncias avaliadas são consultadas; parte dos rótulos apenas chega tarde demais para beneficiar a avaliação.

## 15. Custo computacional: resultado observado, não efeito causal isolado

Na referência sem atraso e com 100% de consultas:

| Modelo | Tempo multiclasse (s) | Tempo binário (s) |
|---|---:|---:|
| ARF | 21,92 | 28,38 |
| LB | 21,28 | 24,76 |
| HAT | 2,08 | 2,21 |
| HT | 1,88 | 2,02 |

A execução binária registrou tempos maiores nessa referência. Não é seguro concluir que o alvo binário seja intrinsecamente mais lento.

Um controle importante: somente nos três cenários de Consistência, que mantêm duas classes e métricas idênticas, ARF passa de 20,63 s para 26,97 s e LB de 20,45 s para 24,19 s. Logo, nem toda diferença de tempo pode ser atribuída à reunião das famílias de ataque.

As execuções ocorreram em momentos diferentes, não registram toda a carga do sistema e a instrumentação por janela mudou. Outros fatores, incluindo ambiente e trajetória interna dos modelos, podem influenciar os tempos. Uma comparação causal de desempenho computacional exige medições controladas e repetidas no mesmo ambiente.

Dentro da execução binária, reduzir consultas diminui o tempo dos ensembles: ARF passa de 28,38 s a 100% para 11,38 s a 1%; LB, de 24,76 s para 10,13 s. Entretanto, todas as instâncias continuam previstas e avaliadas, e o prefixo integral continua sendo treinado.

Os cronômetros incluem treino inicial, avaliação, atualizações, cálculos por janela e entrega final das pendências. Não incluem todo o pré-processamento e não equivalem apenas à latência de predição. A produção dos gráficos ocorre fora do cronômetro específico de cada repetição.

A soma de `Time_avg × Runs` somente nas condições únicas de C é aproximadamente 13,52 horas no binário e 11,77 horas no multiclasse. Somar A, B e C integralmente duplicaria condições reutilizadas. Essas somas não representam necessariamente o tempo total de parede das execuções.

## 16. Variabilidade e interpretação dos desvios padrão

Os `*_std` individuais medem a dispersão de cinco execuções sobre o mesmo fluxo, com `ddof=0`. Não são cinco capturas independentes de tráfego, nem intervalos de confiança de 95%.

Na referência binária, HAT e HT mantêm DP de F1 zero em todos os cenários. ARF apresenta DP médio de 0,49 p.p. e LB de 0,62 p.p. entre as respectivas células de referência. Esses números descrevem variabilidade de repetição, não incerteza de generalização para tráfego novo.

Em C, há configurações instáveis:

| Cenário e modelo | Atraso | Orçamento | F1 binário médio ± DP (%) |
|---|---:|---:|---:|
| Adaptação_200, HAT | 5% | 30% | 36,07 ± 29,53 |
| Adaptação_1000, HAT | 10% | 30% | 36,11 ± 29,48 |
| Adaptação_1000, HAT | 5% | 30% | 40,69 ± 24,95 |
| Recorrência_200, HAT | 5% | 30% | 18,40 ± 22,90 |

Essas médias não devem ser apresentadas isoladamente como desempenho previsível ou estável. Os CSVs agregados não permitem reconstruir a distribuição das cinco métricas, identificar modalidades ou realizar fielmente um teste pareado por semente.

Há 36 células únicas com F1 zero em HAT binário, contra 38 no multiclasse; HT tem 39 em ambos; ARF e LB não têm células com F1 médio zero. A igualdade na quantidade de zeros de HT não significa que sejam exatamente as mesmas condições ou que os resultados tenham permanecido iguais.

Nos gráficos agregados, o DP representa diferenças entre médias dos cenários, não o DP das cinco repetições de uma célula. Já os gráficos temporais individuais mostram a dispersão por janela entre repetições. Essas duas fontes de variabilidade precisam ser diferenciadas nas legendas do TCC.

Uma janela sem ataques recebe F1 e recall zero por convenção, mesmo se todos os benignos forem classificados corretamente. Por isso, a interpretação temporal deve combinar presença de ataques, FP e FN. Somar FP/FN entre janelas é válido; somar ou promediar F1 das janelas não reconstrói o F1 acumulado.

## 17. Limitações que permanecem

### 17.1 Pré-processamento não causal

A normalização continua usando `scaler.fit_transform(X)` sobre todo o cenário. A imputação por mediana também utiliza todas as linhas antes do corte do treino inicial. Essas partes do código não foram corrigidas entre as duas execuções.

Na análise anterior, foram identificadas de 37 a 43 features com mínimo ou máximo global diferente do prefixo de treino e uma feature com valores ausentes cuja mediana global difere da inicial, em cada cenário. Os cenários e esse procedimento permanecem iguais.

A orientação metodológica é ajustar as transformações somente com dados de treino e aplicar os parâmetros aprendidos aos dados posteriores. Ajustar com dados futuros da avaliação constitui vazamento de informação. [Documentação oficial do scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Isso não demonstra quanto cada F1 foi alterado, nem que todos os resultados necessariamente aumentaram. O efeito quantitativo ainda precisa ser medido. O pré-processamento comum permite uma comparação exploratória entre os alvos, mas não torna o estudo ponta a ponta causal ou garante a preservação dos rankings após uma correção.

### 17.2 Cenários construídos e dependência

Os cenários reutilizam fontes e procedimentos de construção. Seus tamanhos, prevalências e prefixos variam conjuntamente. As médias não devem ser extrapoladas diretamente para qualquer rede nem utilizadas como se os 12 cenários ou as 1.152 células fossem independentes.

Os nomes Generalização, Adaptação e Recorrência descrevem a intenção da construção; não comprovam isoladamente uma modalidade matemática específica de concept drift.

### 17.3 Poucas repetições e ausência de informações por semente

Cinco sementes oferecem uma medida inicial de variabilidade, mas os CSVs preservam somente médias e DP. Não há armazenamento suficiente de previsões por instância, métricas individuais, decisões internas ou alarmes de drift para explicar causalmente cada ganho ou perda.

Não foram calculados p-valores ou demonstrada superioridade estatística do alvo binário nesta análise. Uma comparação estatística adequada precisa respeitar o emparelhamento e a dependência entre condições.

### 17.4 Não houve comparação entre estratégias informadas de active learning

O método de consulta continua sendo Random Sampling com probabilidade constante por instância. Não usa incerteza, diversidade, previsão de ataque ou detecção de mudança para priorizar solicitações.

Os resultados não demonstram desempenho superior de active learning informado nem que uma estratégia mais seletiva produziria os mesmos ganhos ou perdas.

### 17.5 Outros controles e configurações ainda não avaliados

O grid não contém orçamento de 0% após o treino inicial. Esse controle permitiria comparar atualizações escassas ou atrasadas com um modelo congelado, mas sua inclusão seria uma extensão do desenho experimental.

Os resultados analisados continuam restritos a Default_FullFeatures. Não há evidência desta execução para comparar hiperparâmetros otimizados ou conjuntos reduzidos de features.

As versões completas do ambiente, Java e hardware não são registradas nos CSVs, e várias dependências de `requirements.txt` não estão fixadas. A reprodutibilidade precisa ser fortalecida antes de uma versão definitiva do estudo.

## 18. Conclusões sustentadas e recomendações

### 18.1 O que os dados sustentam

1. A execução binária é coerente com as verificações realizadas sobre consultas, atrasos, métricas e janelas.
2. Na referência integral e imediata, o alvo binário melhora o F1 médio dos quatro modelos, sem vencer em todos os cenários.
3. No grid completo, o ganho médio é modesto, aproximadamente 0,42 p.p., e varia fortemente entre modelos e condições.
4. ARF melhora em todas as 24 condições quando se considera a média dos cenários; HT perde nas 20 condições com orçamento reduzido e ganha nas quatro com 100%.
5. O alvo binário pode melhorar a detecção de ataques recorrentes em algumas trajetórias e prejudicá-la em outras.
6. Atraso e consultas raras continuam sendo limitações dominantes; reduzir classes não substitui informação rotulada disponível no momento adequado.
7. A comparação mostra detecção de ataque, não identificação de famílias.

### 18.2 O que não se deve concluir

- que o treinamento binário é sempre melhor;
- que um ganho médio pequeno estabelece superioridade estatística;
- que uma queda de F1 resulta necessariamente de menos ataques detectados, pois FP também podem dominar a perda;
- que um grande delta garante desempenho absoluto satisfatório;
- que rótulos atrasados são sempre prejudiciais ou que menos consultas são sempre melhores;
- que a execução binária resolveu o vazamento de pré-processamento;
- que diferenças de tempo entre as duas datas isolam o custo da formulação do alvo;
- que todos os cenários são observações independentes de tráfego real.

### 18.3 Prioridades para a próxima etapa

A principal prioridade metodológica é implementar e validar um pré-processamento causal. Para uma comparação final dos alvos, devem ser executados binário e multiclasse sob esse mesmo procedimento corrigido, mantendo cenário, consultas, sementes e parâmetros alinhados.

Uma opção simples é ajustar imputação e normalização somente no prefixo inicial e manter os parâmetros fixos depois. Se for escolhido pré-processamento incremental, é necessário preservar a compatibilidade da representação das instâncias guardadas na fila com o estado do modelo.

Recomendações complementares:

- salvar métricas por repetição, sementes e contagens de consultas por classe/região;
- registrar previsões e rótulos recebidos por instância para casos diagnósticos;
- instrumentar divisões, folhas ou alarmes de drift nos casos de ganhos e perdas mais fortes;
- registrar commit, comando, versões, Java e informações do ambiente;
- apresentar resultados estratificados por duração e prevalência, além das médias globais;
- incluir um controle sem atualizações após o treino inicial, caso aprovado como extensão;
- validar os mesmos controles de equivalência antes de uma nova execução completa;
- manter os resultados atuais como histórico exploratório, sem sobrescrevê-los.

Nenhuma dessas alterações foi implementada como parte desta análise.

## 19. Sugestão de texto para o TCC

### 19.1 Comparação geral

> A alteração do treinamento multiclasse para binário manteve o protocolo de avaliação binária, os cenários e as consultas de rótulos alinhados. Na referência com disponibilidade integral e imediata de rótulos, os quatro classificadores apresentaram aumento do F1 médio, com diferenças entre 0,11 e 1,64 ponto percentual. Entretanto, a vantagem não foi uniforme: ao considerar as 24 combinações de atraso e orçamento, ARF e Leveraging Bagging apresentaram ganhos médios de 1,28 e 0,69 ponto percentual, enquanto Hoeffding Tree apresentou perda média de 0,45 ponto percentual. Esses valores descrevem o benchmark avaliado e não estabelecem superioridade estatística universal do alvo binário.

### 19.2 Interpretação temporal

> A análise por janela mostrou que ganhos acumulados podem resultar de melhor detecção de ataques recorrentes, e não da primeira ocorrência de uma família nova. Em Generalização_1000, com HAT, atraso de 5% e consultas integrais, o ganho de F1 de 18,54 pontos percentuais decorreu principalmente do retorno DNS: seu recall aumentou de 55,83% para 94,65%, enquanto o recall da região LDAP permaneceu próximo de 10%. Em contraste, em Recorrência_200 com atraso de 1%, o treino binário perdeu a detecção da última região LDAP, reduzindo seu recall de 98% para zero. Assim, o efeito do alvo depende da trajetória de atualização e não pode ser interpretado somente pela média global.

### 19.3 Limitações

> O treinamento binário aproxima o objetivo de aprendizado da tarefa de distinguir tráfego benigno de ataque, mas não elimina as restrições de disponibilidade de rótulos. Nos cenários curtos, o atraso pode ultrapassar a duração de uma região inteira, e Random Sampling com orçamento baixo pode não consultar nenhum ataque da avaliação. Os resultados também permanecem sujeitos ao ajuste offline do pré-processamento sobre o cenário completo. Portanto, devem ser tratados como evidência exploratória e acompanhados de uma avaliação posterior com transformações causais e registro das métricas individuais por repetição.

Esses parágrafos são propostas de redação dos resultados locais, a serem integradas à metodologia e às normas do TCC.

## 20. Rastreabilidade, reprodução e arquivos de gráficos

O código consultado corresponde a `bf64f73`; a origem multiclasse foi verificada em `e51942c`. A implementação e os cenários estavam sem mudanças locais no início desta análise. As renomeações das pastas de gráficos feitas pelo usuário foram preservadas.

O script diagnóstico temporário utilizado é `/tmp/analyze_binary_vs_multiclass_tcc.py`. Ele lê os arquivos existentes, reconstrói consultas/entregas e produz as verificações e tabelas; não importa classificadores e não treina modelos. Sua disponibilidade não é garantida depois da limpeza de `/tmp`.

As tabelas principais podem ser reproduzidas a partir da raiz do projeto com o código abaixo, sem depender desse script temporário:

```python
from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
root = Path("output/ClassificationLabeling")
patterns = {
    "multiclass": "*_20261001_101354_cumulative.csv",
    "binary": "*_binaryTraining_20261007_130844_cumulative.csv",
}
results = {}
for mode, pattern in patterns.items():
    files = sorted(root.glob(pattern))
    assert len(files) == 12
    data = pd.concat(
        [pd.read_csv(path, sep=";") for path in files],
        ignore_index=True,
    )
    assert len(data) == 1632
    # C cobre as 24 condições únicas: não contar A e B outra vez.
    unique = data.loc[data["Experiment"].eq("C")].copy()
    assert len(unique) == 1152
    results[mode] = unique
    print(mode)
    for model, rows in unique.groupby("Model"):
        print(model)
        print(rows.groupby(
            ["Delay_Percentage", "Label_Budget_Percentage"]
        )["F1_avg"].mean().unstack().round(2))

keys = [
    "Dataset", "Model", "Scenario",
    "Delay_Percentage", "Label_Budget_Percentage",
]
paired = results["binary"].merge(
    results["multiclass"], on=keys,
    suffixes=("_binary", "_multiclass"),
    validate="one_to_one",
)
delta_f1 = (
    paired["F1_avg_binary"] - paired["F1_avg_multiclass"]
)
print("Delta médio por modelo no grid completo")
print(delta_f1.groupby(paired["Model"]).mean().round(4))
```

### 20.1 Figuras disponíveis

O manifesto em `output/ClassificationLabeling/plots-binary/manifest.csv` tem 126 entradas: 72 figuras individuais de desempenho/erro/custo, seis agregadas e 48 figuras temporais de comparação das condições de C, uma por cenário/modelo.

A execução `20261007_130844` também possui 1.632 gráficos detalhados de FP/FN e dinâmica dos rótulos em `plots-binary/stream/binary/`. Há oito figuras de verificações anteriores com outros Exec_IDs, que não foram tratadas como resultados adicionais desta análise.

Após a renomeação da pasta, os caminhos do manifesto ainda apontam para `/plots/`. Nenhum dos 126 caminhos antigos existe, mas os 126 arquivos são encontrados ao substituir esse trecho por `/plots-binary/`. Isso não altera os números dos CSVs; apenas deixa as referências do manifesto desatualizadas. O manifesto e as pastas não foram modificados nesta análise.

Para o TCC, as figuras temporais dos estudos da seção 12 são mais informativas sobre o mecanismo temporal que uma única média global. Recomenda-se acompanhá-las das quantidades reais de ataques e dos atrasos em instâncias, não somente do percentual nominal.

### 20.2 Fontes locais principais

- CSVs acumulados e prequential dos dois Exec_IDs em `output/ClassificationLabeling/`;
- rótulos dos cenários em `data/15k/`;
- `run_classification_labeling.py`: seleção de alvo, cenários e factories;
- `src/Classification/Labeling.py`: treino inicial, consultas, fila, repetições e agregação;
- `src/Classification/Models.py`: parâmetros dos classificadores;
- `src/Data/Processor.py`: pré-processamento e codificação dos rótulos;
- `src/Results/Metrics.py`: métricas binárias;
- `plot_classification_labeling_results.py`: agregações e figuras;
- `ANALISE_RESULTADOS_TCC.md`: análise histórica e limitações previamente identificadas.

### 20.3 Referências metodológicas

- CASTELLANI, Andrea; SCHMITT, Sebastian; HAMMER, Barbara. [Stream-based Active Learning with Verification Latency in Non-stationary Environments](https://arxiv.org/abs/2204.06822). 2022.
- SCIKIT-LEARN. [Common pitfalls and recommended practices — Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Documentação oficial. Consulta em 8 de outubro de 2026.
- CAPYMOA. [HoeffdingTree](https://capymoa.org/api/modules/capymoa.classifier.HoeffdingTree.html) e [HoeffdingAdaptiveTree](https://capymoa.org/v0.15.1/api/modules/capymoa.classifier.HoeffdingAdaptiveTree.html). Documentação dos parâmetros dos classificadores. A versão documentada não deve ser confundida com um registro da versão utilizada na execução histórica.

As referências dão contexto metodológico. Todos os valores numéricos do relatório foram calculados a partir dos arquivos locais do projeto.
