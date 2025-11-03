# Librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Carga los datos y normaliza nombres
archivo = 'Datos/datos_sin_subcategoria.csv'
df = pd.read_csv(archivo, encoding='utf-8')
df.columns = [c.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace(' ','_') for c in df.columns]

<<<<<<< HEAD
# Seleccion de variables predictoras (las mas relevantes segun analisis previo)
selected_features = [
    'Siniestros',
    'Lesionados - Graves',
    'Indicadores cada 100.000 habitantes - Siniestralidad',
    'Parque vehicular',
    'Población',
    'Tasa motorización'
]
=======
# Crear taza y variable binaria
df['tasa_siniestros_poblacion'] = df['siniestros'] / df['poblacion'] * 10000
mediana = df['siniestros'].median()
df['alta_accidentalidad'] = (df['siniestros'] > mediana).astype(int)
features = ['tasa_siniestros_poblacion']
df_modelo = df.dropna(subset=features + ['alta_accidentalidad']).copy()
>>>>>>> 2b9b368db3c632959d0650533a524ad4caf2ac29

# Entrenar modelo
feature = 'tasa_siniestros_poblacion'
X = df[[feature]]
y = df['alta_accidentalidad']
modelo_simple = LogisticRegression()
modelo_simple.fit(X, y)

# Predecir probabilidad continua para cada año
probs = modelo_simple.predict_proba(X)[:,1]

# Grafico de la probabilidad predicha vs la tasa de siniestros por 10.000 habitantes
plt.figure(figsize=(8,5))
plt.scatter(df[feature], probs, s=80, c=probs, cmap='viridis', label='Probabilidad predicha')
plt.xlabel("Tasa de siniestros por 10.000 hab.")
plt.ylabel("Probabilidad de alta accidentalidad (predicha)")
plt.title("Regresión logística: probabilidad de alta accidentalidad vs tasa")
plt.colorbar(label='Probabilidad predicha')
plt.tight_layout()

<<<<<<< HEAD
# Guardar y mostrar grafico
plt.savefig('Imagenes/curva_logistica_prediccion.png', dpi=300)
plt.show()
=======
# Guardamos el grafico para su posterior visualizacion
plt.savefig("Imagenes/probs_logistica_vs_tasa.png")
>>>>>>> 2b9b368db3c632959d0650533a524ad4caf2ac29
