---
project: eda
task: regression
dataset: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
team:
  - Cynthia Naoko Yasutake
  - Davi Peter Bastian Nehls
ai_use: "used for planning and writing support"
---
# 1. EDA — Análise Exploratória

!!! abstract "Entrega 1 de 3 do [Projeto](../index.md)"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

!!! info "Equipe"

    | Nome completo | GitHub |
    |---------------|--------|
    | Cynthia Naoko Yasutake | CYNahko|
    | Davi Peter Bastian Nehls | dpnnpd |
    | | |

    Dataset, decisões e status: [página do projeto](../index.md).

!!! tip "O que esta entrega decide"

    O EDA não é um álbum de gráficos: é onde a equipe **escolhe o dataset** e descobre o que
    vai atrapalhar o treino depois — desbalanceamento, vazamento, escalas incompatíveis com a
    ativação, ausências não aleatórias. Cada achado aqui deve virar uma linha do plano de
    pré-processamento no fim da página, e é esse plano que as duas entregas
    seguintes executam.

    As aulas de **Classes → Data** no
    [site da disciplina](https://insper.github.io/ann-dl/){:target='_blank'} dão a estrutura:
    tipos, distribuições, qualidade, desbalanceamento, vazamento, split e pré-processamento.

## 1. Inspeção inicial

### A - Dicionário de dados

### B - Qualidade

### C - Alvo

### D - Treino e teste

## 2. Análise univariada

### A - Numéricas

### B - Categóricas

## 3. Análise bivariada e multivariada

### A - Numérica x numérica

### B - Categórica x alvo

### C - Numérica x categórica

## 4. Pré-processamento

### A - Estratégias

### B - Redução de dimensionalidade

### C - Pipeline

## 5. Síntese

## 6. Qualidade dos dados

| # | Resumo dos resultados | Valor |
|---|---------|-------|
| 1 | Dataset, tarefa e alvo | |
| 2 | Instâncias x features (numéricas / categóricas) | |
| 3 | Coluna com mais faltantes e seu percentual | |
| 4 | Colunas descartadas e o motivo | |
| 5 | Classe minoritária (%) - ou média e mediana do alvo | |
| 6 | Tamanho do treino e do teste | |
| 7 | Par de numéricas mais correlacionado e o valor | |
| 8 | Linhas afetadas pela estratégia de outliers | |
| 9 | Variância explicada por PC1 + PC2 | |
| 10 | `shape` do treino e do teste após o pipeline | |

## Conclusão

O que o dataset permite e o que ele impede. Se algum achado inviabiliza a tarefa pretendida,
é aqui que a equipe muda de rumo — ainda dá tempo.

## Referências