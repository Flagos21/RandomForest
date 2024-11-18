# random_forest_lechuzas.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar los datos
df = pd.read_csv('lechuzasdataset.csv')
df = df.dropna()

# Ajustar los rangos de 'Potencia' en función de los valores observados
bins = [-3500, -1500, -500, 0, 5, 12]  # Ajustado para cubrir el rango observado
labels = ['<-1500', '-1500 to -500', '-500 to 0', '0 to 5', '5 to 12']
df['PotenciaCat'] = pd.cut(df['Potencia'], bins=bins, labels=labels)

# Separar las características y el objetivo
X = df[['Radiacion', 'Temperatura', 'Temperatura panel']]
y = df['Potencia']
y_cat = df['PotenciaCat']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(X, y, y_cat, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test)

# Convertir las predicciones y el objetivo de prueba a categorías
y_pred_cat = pd.cut(y_pred, bins=bins, labels=labels).add_categories(['Unknown']).fillna('Unknown')
y_cat_test = y_cat_test.cat.add_categories(['Unknown']).fillna('Unknown')

# Calcular precisión como R² y el error cuadrático medio
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Imprimir resultados
print("=== Resultados del Modelo (Lechuzas - Regresión) ===")
print(f"Precisión del modelo (R²): {r2:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}\n")

# Matriz de confusión (con categorías)
conf_matrix = confusion_matrix(y_cat_test, y_pred_cat, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.title('Matriz de Confusión (Random Forest - Lechuzas - Potencia)')
plt.show()
