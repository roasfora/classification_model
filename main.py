# Importando bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# 🔹 Caminho do dataset local
dataset_path = r"C:\Users\isabe\Desktop\EDIT\EDIT\Final Project\LTP_PROJECT\dataset.csv"

# Carregando o dataset
df_forced = pd.read_csv(dataset_path)

# Verificando as primeiras linhas do dataset
print("📊 Primeiras 5 linhas do dataset:")
print(df_forced.head())

# Separando features e target
X = df_forced.drop(columns=["sold", "sku", "brand", "store"])  # Ajuste conforme necessário
y = df_forced["sold"]

# Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista de modelos e hiperparâmetros para otimização
models_params = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "Support Vector Machine": (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}), 
    "Decision Tree": (DecisionTreeClassifier(), {'max_depth': [None, 10, 20, 30]}),
    "Random Forest": (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    "Gradient Boosting": (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    "AdaBoost": (AdaBoostClassifier(), {'n_estimators': [50, 100, 200]}),
    "Naive Bayes": (GaussianNB(), {})  # Sem hiperparâmetros para otimizar
}

# Dicionário para armazenar os resultados
results = {}

# Testando cada modelo com GridSearchCV para ajuste de hiperparâmetros
for name, (model, params) in models_params.items():
    print(f"🔄 Treinando {name}...")
    grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Calculando métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        "Best Params": grid_search.best_params_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# Convertendo resultados para DataFrame
results_df = pd.DataFrame.from_dict(results, orient='index')

# Exibindo os resultados
print("\n📊 Resultados dos Modelos com Otimização de Hiperparâmetros:")
print(results_df)

# Salvando as métricas em uma imagem
plt.figure(figsize=(12, 8))
plt.axis('off')  # Desativa os eixos
plt.table(cellText=results_df.values,
          colLabels=results_df.columns,
          rowLabels=results_df.index,
          loc='center',
          cellLoc='center',
          colColours=['#f3f3f3'] * len(results_df.columns),
          rowColours=['#f3f3f3'] * len(results_df.index))
plt.title("Métricas dos Modelos de Classificação", fontsize=16, pad=20)
plt.savefig("metricas_modelos.png", bbox_inches='tight', dpi=300)  # Salva a imagem
plt.show()

print("✅ Imagem salva como 'metricas_modelos.png'.")