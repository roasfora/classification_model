# Projeto de Classificação de Vendas

Este projeto tem como objetivo prever se um produto será vendido ou não com base em um conjunto de dados. Para isso, foram testados diversos modelos de classificação, e os resultados foram comparados para identificar o melhor desempenho.

## 📋 Descrição do Projeto

O projeto consiste em:
1. **Carregamento e pré-processamento dos dados**: O dataset foi carregado e preparado para análise.
2. **Treinamento de modelos**: Foram testados vários modelos de classificação, incluindo:
   - Regressão Logística
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Árvore de Decisão
   - Random Forest
   - Gradient Boosting
   - AdaBoost
   - Naive Bayes
3. **Otimização de hiperparâmetros**: Utilizou-se `GridSearchCV` para ajustar os hiperparâmetros de cada modelo.
4. **Avaliação de desempenho**: As métricas de avaliação (Acurácia, Precisão, Recall e F1-Score) foram calculadas para cada modelo.
5. **Visualização dos resultados**: Os resultados foram salvos em uma imagem (`metricas_modelos.png`) para fácil visualização.

## 📊 Resultados

Os resultados dos modelos, após a otimização de hiperparâmetros, são apresentados abaixo:

| Modelo                  | Melhores Parâmetros                     | Acurácia | Precisão | Recall | F1-Score |
|-------------------------|-----------------------------------------|----------|----------|--------|----------|
| Logistic Regression     | `{'C': 0.1}`                           | 0.85     | 0.83     | 0.80   | 0.81     |
| K-Nearest Neighbors     | `{'n_neighbors': 5}`                   | 0.82     | 0.81     | 0.78   | 0.79     |
| Support Vector Machine  | `{'C': 10, 'kernel': 'rbf'}`           | 0.86     | 0.84     | 0.82   | 0.83     |
| Decision Tree           | `{'max_depth': 20}`                    | 0.84     | 0.82     | 0.81   | 0.81     |
| Random Forest           | `{'max_depth': 20, 'n_estimators': 200}`| 0.87     | 0.85     | 0.83   | 0.84     |
| Gradient Boosting       | `{'learning_rate': 0.1, 'n_estimators': 200}`| 0.88 | 0.86     | 0.84   | 0.85     |
| AdaBoost                | `{'n_estimators': 200}`                 | 0.86     | 0.84     | 0.82   | 0.83     |
| Naive Bayes             | `{}`                                    | 0.80     | 0.78     | 0.75   | 0.76     |

A imagem com as métricas foi salva como `metricas_modelos.png`.

## 🛠️ Como Executar o Projeto

1. **Pré-requisitos**:
   - Python 3.x
   - Bibliotecas: `pandas`, `scikit-learn`, `matplotlib`

2. **Instalação das dependências**:
   ```bash
   pip install pandas scikit-learn matplotlib