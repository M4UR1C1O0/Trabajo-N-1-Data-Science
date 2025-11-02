from Limpieza import df  # Importamos el dataframe limpio creado en Limpieza.py

# Librerias necesarias
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Prediccion de fallecidos usando regresion lineal multiple

# Definimos las variables independientes (X) y la variable dependiente (y)
X = df[[
    'Siniestros', 
    'Población', 
    'Año', 
    'Tasa motorización', 
    'Lesionados - Graves', 
    'Lesionados - Leves', 
    'Lesionados - Menos graves'
    ]]
y = df['Fallecidos']

# Preparamos los datos para entrenar y probar el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Creamos el modelo de regresion lineal multiple y lo ajustamos con datos de entrenamiento
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizamos predicciones sobre el conjunto de prueba
y_pred_test = modelo.predict(X_test)

# Mostramos los coeficientes del modelo, la interseccion y la precision (R^2)
print(f"Coeficientes: {modelo.coef_}")
print(f"Interseccion: {modelo.intercept_}")
print(f"Precision en test (R^2): {modelo.score(X_test, y_test)}")

# Grafico de los valores reales vs los valores predichos
# Extraemos los años para etiquetar los puntos
anos = df.loc[X_test.index, 'Año']

# Distinguimos los puntos con fallecidos mayores a 1700
mask = y_test <= 1700

# Creamos el grafico
plt.figure(figsize=(10, 6))
plt.scatter(y_test[mask], y_pred_test[mask], color='royalblue', alpha=0.85, label='Fallecidos <= 1700')
plt.scatter(y_test[~mask], y_pred_test[~mask], color='crimson', s=70, label='Fallecidos > 1700')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Prediccion perfecta')

# Anotamos cada punto con su año correspondiente
for i, txt in enumerate(anos):
    plt.annotate(str(txt), (y_test.iloc[i], y_pred_test[i]), fontsize=9, color='dimgray', alpha=0.8)

plt.xlabel('Fallecidos reales')
plt.ylabel('Fallecidos predichos')
plt.title('Regresion lineal multiple: Prediccion vs Real')
plt.legend()
plt.grid(True, ls='--', lw=0.7, alpha=0.5)
plt.tight_layout()

# Guardamos el grafico para su posterior visualizacion
plt.savefig('Imagenes/regresion_lineal_multiple.png')
plt.close()

'''
# Prediccion de fallecidos para el año 2025
# Hacemos un ejemplo con valores para 2025 manualmente
valores_2025 = [
    76000,      # Siniestros
    18500000,   # Poblacion
    2025,       # Año
    8.0,        # Tasa motorizacion
    7100,       # Lesionados - Graves
    32000,      # Lesionados - Leves
    34000       # Lesionados - Menos graves
]

# Realizamos la prediccion para 2025
prediccion_2025 = modelo.predict([valores_2025])
print(f"Prediccion de fallecidos para 2025: {int(prediccion_2025[0])}")
'''

# Guardamos las predicciones de todos los años en un archivo CSV
# Realizamos predicciones sobre todos los datos historicos
y_pred_all = modelo.predict(X)

# Creamos un DataFrame con los años y las predicciones
df_pred = pd.DataFrame({
    'Año': list(df['Año']),
    'Fallecidos_predichos': list(y_pred_all)
})

# Guardamos el archivo CSV con las predicciones (opcional)
df_pred.to_csv('Datos/fallecidos_predichos_por_año.csv', index=False)
