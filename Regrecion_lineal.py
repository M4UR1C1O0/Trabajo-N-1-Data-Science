from Limpieza import df # Importamos el dataframe limpio creado en Limpieza.py

# Librerias necesarias
import pandas as pd 
from sklearn.linear_model import LinearRegression # Importamos el modelo de regresión lineal
import matplotlib.pyplot as plt                   # Librería para los graficos

# Prediccion de fallecidos usando regresion lineal multiple

# Definimos las variables independientes (X) y la variable dependiente (y)
X = df[['Siniestros','Población', 'Año','Tasa motorización','Lesionados - Graves', 'Lesionados - Leves', 'Lesionados - Menos graves',]]
y = df['Fallecidos']

# Creamos el modelo de regresión lineal multiple y lo ajustamos a los datos
modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

# Mostramos los coeficientes del modelo, la interseccion y la precision (R^2)
print(f"Coeficientes: {modelo.coef_}") 
print(f"Interseccion: {modelo.intercept_}")
print(f"Precision (R^2): {modelo.score(X, y)}")

# Grafico de los valores reales vs los valores predichos
plt.figure(figsize=(7,5))
plt.scatter(y, y_pred, color='blue', alpha=0.7, label='Prediccion vs Real')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Prediccion perfecta')
plt.xlabel('Fallecidos reales')
plt.ylabel('Fallecidos predichos')
plt.title('Regresión lineal multiple: Prediccion vs Real')
plt.legend()
plt.tight_layout() 

# Guardamos el grafico para su posterior visualizacion
plt.savefig('Imagenes/regresion_lineal_multiple.png')