from tensorflow import keras
# TensorFlow es el motor de bajo nivel que realiza los calculos matematicos complejos de manera eficiente
# y Keras corre sobre Tensorflow, permitiendo la creacion y experimentacion rapida de modelos de Deep Learning.
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paso 1: Cargar y Preparar los Datos
archivo = 'Datos/datos_sin_subcategoria.csv'
df = pd.read_csv(archivo, encoding='utf-8')

#Paso 2: Construir la Arquitectura del Modelo
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(100, activation='relu', input_shape=(7,)),   # Capa de entrada con 7 features
    Dense(50, activation='relu'),                       # Capa oculta
    Dense(1)                                            # Capa de salida para regresión
])

#Paso 3: Compilar el Modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nResumen de la arquitectura:")
model.summary()

#Paso 4: Entrenar el Modelo
print("Iniciando entrenamiento...")
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))
print("Entrenamiento completado.")

#Paso 5: Evaluar el Rendimiento
print("\nEvaluando el modelo con datos de prueba...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida (Loss) en el conjunto de prueba: {loss:.4f}")
print(f"Exactitud (Accuracy) en el conjunto de prueba: {accuracy:.4f}")

#Paso 6: Hacer Predicciones
X_new = X_test[:3] # Tomamos las primeras 3 imágenes de prueba
y_proba = model.predict(X_new)
y_pred_classes = y_proba.argmax(axis=-1)

print("\nPredicciones para las primeras 3 muestras de prueba:")
for i, pred in enumerate(y_proba):
    print(f"Muestra {i+1}: Predicción de fallecidos = {pred[0]:.2f}")
plt.ylabel('Fallecidos predichos')
plt.title('Fallecidos Reales vs Predichos')
plt.grid(True, ls='--', lw=0.7, alpha=0.5)