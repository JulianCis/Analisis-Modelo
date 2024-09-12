import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Cargar el archivo y leerlo
data = pd.read_csv('bankloan.csv')

# Eliminar columnas irrelevantes para el modelo
# En este caso, el ID no aporta nada, ni tampoco el zip code ya que no tenemos algo con qué comparar
# si el zip code determina alguna zona donde haya mayor nivel socioeconómico
data.drop(['ID', 'ZIP.Code'], axis=1, inplace=True)

print(data)
# Separar variables independientes X, el cual sólo tiene datos numéricos y la variable objetivo y, en este caso, Personal.Loan
X = data.drop('Personal.Loan', axis=1)
y = data['Personal.Loan']

# Dividir el conjunto de datos ya limpios en entrenamiento y prueba (80% para entrenamiento, y 20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Dividir el conjunto de entrenamiento en entrenamiento y validación (80% para entrenamiento, y 20% para validación)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Creamos el modelo de random forest con una misma semilla para tener los mismos valores cuando lo corremos
random_forest = RandomForestClassifier(random_state=42)

# Definir los hiperparámetros, podemos ajustar los estimadores (número de árboles) y la profundidad máxima de cada uno 
# de los árboles de decisión dentro del random forest
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Usamos grid search para buscar los mejores hiperparámetros usando los datos de validación que habíamos apartado
# Le subimos el cross validation para evitar overfitting
random_search = RandomizedSearchCV(estimator=random_forest, param_distributions=param_grid, 
                                    n_iter=50, cv=5, scoring='accuracy', random_state=42)

# Entrenamos al modelo con los datos de entrenamiento
random_search.fit(X_train, y_train)

# Obtenemos los mejores hiperparámetros y con base en esto, igual el mejor modelo
best_rf = random_search.best_estimator_
best_params = random_search.best_params_
best_score = random_search.best_score_

# Imprimimos Los mejores hiperparámetros, sacados con grid search
print("Mejores hiperparámetros:", best_params)
print("Mejor puntuación de validación usando grid search:", best_score)

# Evaluar el modelo con el conjunto de entrenamiento
train_score = best_rf.score(X_train, y_train)
print("Puntuación de entrenamiento:", train_score)

# Evaluar el modelo con el conjunto de validación
val_score = best_rf.score(X_val, y_val)
print("Puntuación de validación:", val_score)

# Evaluar el modelo con el conjunto de prueba
test_score = best_rf.score(X_test, y_test)
print("Puntuación de prueba:", test_score)

# Hacemos predicciones con validación
y_val_pred = best_rf.predict(X_val)

# Hacemos predicciones con prueba
y_test_pred = best_rf.predict(X_test)

# Usamos la matriz de confusión para el conjunto de validación
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
print("Matriz de confusión (validación):\n", conf_matrix_val)

# Matriz de confusión para el conjunto de prueba
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Matriz de confusión (prueba):\n", conf_matrix_test)

# Evaluación con métricas adicionales como acccuracy, precision, recall y f1, es importante saber que la evaluación funciona
# como un conjunto de todas las métricas
print("Evaluación del conjunto de prueba:\n")
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Imprimimos las métricas
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Reporte de clasificación más detallado
classification_rep = classification_report(y_test, y_test_pred)
print("Reporte de clasificación:\n", classification_rep)

# Información de los conjuntos de datos
print("\nDataset completo:")
print(data.head())
print("Tamaño del dataset:")
print(data.shape)

print("\nConjunto de entrenamiento:")
print(X_train.head())
print("Tamaño del conjunto de entrenamiento:")
print(X_train.shape)

print("\nConjunto de validación")
print(X_val.head())
print("Tamaño del conjunto de validación:")
print(X_val.shape)

print("\nConjunto de prueba")
print(X_test.head())
print("Tamaño del conjunto de prueba:")
print(X_test.shape)


# Gráficas:

# Crear un DataFrame para las puntuaciones de entrenamiento y validación, para encontrar el sesgo 
scores_df = pd.DataFrame({
    'Conjunto': ['Entrenamiento', 'Validación'],
    'Puntuación': [train_score, val_score]
})

# Graficar las puntuaciones de precisión para el conjunto de entrenamiento y validación
plt.figure(figsize=(6, 4))
sns.barplot(x='Conjunto', y='Puntuación', data=scores_df, palette='viridis')

# Añadir etiquetas y título a la gráfica
plt.title('Puntuaciones de Precisión: Entrenamiento vs. Validación')
plt.xlabel('Conjunto de Datos')
plt.ylabel('Precisión')
plt.ylim(0, 1)  # Limitar el eje Y entre 0 y 1
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añadir una cuadrícula al fondo
plt.show()

# Crear un DataFrame para las puntuaciones de entrenamiento, validación y prueba
scores_df = pd.DataFrame({
    'Conjunto': ['Entrenamiento', 'Validación', 'Prueba'],
    'Puntuación': [train_score, val_score, test_score]
})

# Graficar las puntuaciones de precisión para los conjuntos de entrenamiento, validación y prueba
plt.figure(figsize=(8, 6))
sns.barplot(x='Conjunto', y='Puntuación', data=scores_df, palette='viridis')

# Añadir etiquetas y título a la gráfica
plt.title('Puntuaciones de Precisión en Entrenamiento, Validación y Prueba')
plt.xlabel('Conjunto de Datos')
plt.ylabel('Precisión')
plt.ylim(0, 1)  # Limitar el eje Y entre 0 y 1
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Añadir una cuadrícula al fondo
plt.show()

# Graficar la matriz de confusión para el conjunto de validación
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión (Validación)')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')

# Graficar la matriz de confusión para el conjunto de prueba
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión (Prueba)')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')

plt.tight_layout()
plt.show()

# Datos para las métricas
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

# Crear un DataFrame para facilitar la visualización
metrics_df = pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor'])

# Graficar las métricas
plt.figure(figsize=(8, 6))
sns.barplot(x='Métrica', y='Valor', data=metrics_df, palette='viridis')
plt.title('Métricas de Evaluación del Modelo')
plt.ylim(0, 1)  # Para que el gráfico muestre las métricas en el rango de 0 a 1
plt.show()

# Extraer los resultados del GridSearchCV
results = random_search.cv_results_

# Crear un DataFrame con los resultados
results_df = pd.DataFrame({
    'params': results['params'],
    'mean_test_score': results['mean_test_score'],
    'std_test_score': results['std_test_score']
})

# Descomponer los hiperparámetros para facilitar la visualización
results_df['n_estimators'] = results_df['params'].apply(lambda x: x['n_estimators'])
results_df['max_depth'] = results_df['params'].apply(lambda x: x['max_depth'])

# Graficar la convergencia de los hiperparámetros
plt.figure(figsize=(14, 7))

# Graficar la puntuación media de validación en función del número de estimadores
plt.subplot(1, 2, 1)
sns.lineplot(x='n_estimators', y='mean_test_score', hue='max_depth', data=results_df, marker='o')
plt.title('Convergencia en función de n_estimators')
plt.xlabel('Número de Estimadores')
plt.ylabel('Puntuación Media de Validación')
plt.legend(title='Max Depth')

# Graficar la puntuación media de validación en función de la profundidad máxima
plt.subplot(1, 2, 2)
sns.lineplot(x='max_depth', y='mean_test_score', hue='n_estimators', data=results_df, marker='o')
plt.title('Convergencia en función de max_depth')
plt.xlabel('Profundidad Máxima')
plt.ylabel('Puntuación Media de Validación')
plt.legend(title='Número de Estimadores')

plt.tight_layout()
plt.show()

