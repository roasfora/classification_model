# 📦 Importando bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import plot_tree

# 📁 Caminho do dataset
dataset_path = r"C:\Users\isabe\Desktop\EDIT\EDIT\Final Project\LTP_PROJECT\dataset.csv"

# 📊 Carregando e visualizando o dataset
df = pd.read_csv(dataset_path)
print("📊 Primeiras 5 linhas do dataset:")
print(df.head())

# 🎯 Separando features e target
X = df.drop(columns=["sold", "sku", "brand", "store"])
y = df["sold"]

# 🔀 Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔍 Definindo o modelo e hiperparâmetros para GridSearch
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 🧠 Treinando com GridSearchCV
print("🔄 Treinando Random Forest com GridSearchCV...")
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print(f"\n🔧 Melhores hiperparâmetros encontrados: {grid_search.best_params_}")

# ✅ Fazendo predições no treino e teste
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

# 🧮 Métricas no treino
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# 🧮 Métricas no teste
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# 📋 Exibindo métricas no console
print("\n📈 Métricas do Random Forest:")
print(f"{'Métrica':<10} | {'Treino':<10} | {'Teste':<10}")
print("-" * 35)
print(f"Accuracy   | {train_accuracy:.4f}    | {test_accuracy:.4f}")
print(f"Precision  | {train_precision:.4f}    | {test_precision:.4f}")
print(f"Recall     | {train_recall:.4f}    | {test_recall:.4f}")
print(f"F1-Score   | {train_f1:.4f}    | {test_f1:.4f}")

# 📌 Salvando as métricas em uma imagem
metrics_df = pd.DataFrame({
    "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Treino": [train_accuracy, train_precision, train_recall, train_f1],
    "Teste": [test_accuracy, test_precision, test_recall, test_f1]
})

plt.figure(figsize=(10, 4))
plt.axis('off')
plt.table(cellText=metrics_df.values,
          colLabels=metrics_df.columns,
          loc='center',
          cellLoc='center',
          colColours=['#f3f3f3'] * 3)
plt.title("📋 Métricas - Treino vs Teste (Random Forest)", fontsize=14, pad=10)
plt.savefig("metricas_treino_teste_rf.png", bbox_inches='tight', dpi=300)
plt.show()

# 🔲 Matriz de confusão (teste)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("📌 Matriz de Confusão - Random Forest (Teste)")
plt.savefig("matriz_confusao_rf.png", dpi=300)
plt.show()

# 📊 Importância das features
importances = best_rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Importância")
plt.title("📊 Importância das Features - Random Forest")
plt.tight_layout()
plt.savefig("importancia_features_rf.png", dpi=300)
plt.show()

# 🌳 Visualização de uma árvore da floresta
plt.figure(figsize=(20, 10))
plot_tree(best_rf.estimators_[0],
          feature_names=X.columns,
          class_names=[str(cls) for cls in best_rf.classes_],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("🌳 Uma Árvore do Random Forest")
plt.savefig("arvore_individual_rf.png", dpi=300)
plt.show()

print("\n✅ Todas as imagens foram salvas com sucesso.")
