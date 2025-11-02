from Limpieza_profunda import df # Importamos el dataframe limpio creado en limpieza.py

#Librereias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Librerias necesarias para la regresion lineal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Definimos las variables dependientes e independientes
X = ['Parque vehicular', 'Población', 'Siniestros', 'Año']
y = 'Fallecidos'

#Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.2, random_state=42)


#Creamos el modelo de regresion lineal y lo entrenamos
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#Realizamos las predicciones en el conjunto de prueba 
y_pred = modelo.predict(X_test)

#Revisamos los coeficeintes del modelo
print("Coeficientes del modelo:", modelo.coef_)
print("Intersección del modelo:", modelo.intercept_)
print("Maen Absolute Error (MAE):", np.mean(np.abs(y_test - y_pred)))
print("Precisión del modelo (R^2):", modelo.score(X_test, y_test))