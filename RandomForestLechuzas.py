# random_forest_lechuzas_15_classes_with_metrics.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar los datos
df = pd.read_csv('lechuzasdataset.csv')
df = df.dropna()

# Ajustar los rangos de 'Potencia' para 15 clases
bins = [
    -3500, -3000, -2500, -2000, -1500, -1000, -500, -250, 0,
    2, 4, 6, 8, 10, 12
]
labels = [
    '<-3000', '-3000 to -2500', '-2500 to -2000', '-2000 to -1500',
    '-1500 to -1000', '-1000 to -500', '-500 to -250', '-250 to 0',
    '0 to 2', '2 to 4', '4 to 6', '6 to 8', '8 to 10', '10 to 12'
]
df['PotenciaCat'] = pd.cut(df['Potencia'], bins=bins, labels=labels)

# Separar las características y el objetivo
X = df[['Radiacion', 'Temperatura', 'Temperatura panel']]
y = df['Potencia']
y_cat = df['PotenciaCat']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
    X, y, y_cat, test_size=0.2, random_state=42
)

# Crear el modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test)

# Convertir las predicciones y el objetivo de prueba a categorías
y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels).add_categories(['Unknown']).fillna('Unknown')
y_cat_test = y_cat_test.cat.add_categories(['Unknown']).fillna('Unknown')

# Calcular métricas de evaluación
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE

# Imprimir resultados
print("=== Resultados del Modelo (Lechuzas - Regresión) ===")
print(f"Precisión del modelo (R²): {r2:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Error Porcentual Absoluto Medio (MAPE): {mape:.2f}%\n")

# Interpretación
print("Interpretación de las métricas:")
print(f"1. Precisión del modelo (R²): {r2:.2f}")
print("   - Explica aproximadamente el porcentaje de la variabilidad en 'Potencia' que es explicada por el modelo.")
print(f"2. Error Cuadrático Medio (MSE): {mse:.2f}")
print("   - Representa el promedio del error cuadrático.")
print(f"3. Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print("   - Muestra el error promedio en las mismas unidades de 'Potencia'.")
print(f"4. Error Absoluto Medio (MAE): {mae:.2f}")
print("   - Representa el error promedio absoluto entre predicciones y valores reales.")

# Matriz de confusión (con categorías)
conf_matrix = confusion_matrix(y_cat_test, y_pred_cat, labels=labels)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.title('Matriz de Confusión (Random Forest - Lechuzas - Potencia)')
plt.show()
