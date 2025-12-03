from tensorflow import keras
# TensorFlow es el motor de bajo nivel que realiza los calculos matematicos complejos de manera eficiente
# y Keras corre sobre Tensorflow, permitiendo la creacion y experimentacion rapida de modelos de Deep Learning.
from keras.models import Sequential
from keras.layers import Dense

# Cargamos el dataset, ya dividido en entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = keras.datasets.#########

# Definimos el modelo secuencial
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)), #capa
    Dense(4, activation='softmax'),
])

# Compilamos el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
metrics=['accuracy'])

# Mostramos resumen del modelo
model.summary()
