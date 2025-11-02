import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. Cargar y preparar los datos
df = pd.read_csv('Datos/datos_sin_subcategoria.csv')

# Seleccion de variables predictoras (las mas relevantes segun analisis previo)
selected_features = [
    'Siniestros',
    'Lesionados - Graves',
    'Indicadores cada 100.000 habitantes - Siniestralidad',
    'Parque vehicular',
    'Poblacion',
    'Tasa motorizacion'
]

# Crear variable objetivo: Alta mortalidad (por encima de la mediana)
target_median = df['Fallecidos'].median()
df['Alta_Mortalidad_Real'] = (df['Fallecidos'] > target_median).astype(int)

X = df[selected_features]
y = df['Alta_Mortalidad_Real']

# =============================================================================
# 2. Estandarizacion de variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# 3. Entrenamiento del modelo de regresion logistica
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# =============================================================================
# 4. Obtener predicciones del modelo para los datos reales
x_real = X_scaled[:, 1]                   # Lesionados - Graves (ya estandarizado)
y_real = model.predict_proba(X_scaled)[:, 1]  # Probabilidad predicha para cada año

# Clasificacion del modelo para cada año (alta mortalidad si > 0.5)
pred_class = (y_real > 0.5).astype(int)

# =============================================================================
# 5. Calcular y graficar la curva logistica suavizada
X_curve = np.linspace(x_real.min() - 0.3, x_real.max() + 0.3, 300)
# Utilizar medias historicas para las otras variables
X_curve_full = np.column_stack([
    np.full_like(X_curve, X_scaled[:, 0].mean()),   # Siniestros
    X_curve,                                        # Lesionados - Graves (variable principal)
    np.full_like(X_curve, X_scaled[:, 2].mean()),   # Indicador siniestralidad
    np.full_like(X_curve, X_scaled[:, 3].mean()),   # Parque vehicular
    np.full_like(X_curve, X_scaled[:, 4].mean()),   # Poblacion
    np.full_like(X_curve, X_scaled[:, 5].mean()),   # Tasa motorizacion
])
y_curve = model.predict_proba(X_curve_full)[:, 1]

# =============================================================================
# 6. Visualizacion profesional del grafico

plt.figure(figsize=(12, 8))

# Curva logistica del modelo
plt.plot(X_curve, y_curve, color='black', lw=2, label='Curva logistica ')

# Puntos clasificados como "Alta mortalidad" 
plt.scatter(x_real[pred_class==1], y_real[pred_class==1], 
            c='firebrick', s=85, edgecolor='gray', linewidths=1.2, alpha=0.95, 
            label='Predicho: Alta mortalidad')

# Puntos clasificados como "Baja mortalidad" 
plt.scatter(x_real[pred_class==0], y_real[pred_class==0], 
            c='mediumseagreen', s=85, edgecolor='gray', linewidths=1.2, alpha=0.95, 
            label='Predicho: Baja mortalidad')

# Linea de decision
plt.axhline(0.5, color='orange', ls='--', lw=2, label='Linea decision (0.5)')

# Etiquetas y titulos
plt.xlabel('Lesionados - Graves (Estandarizado)', fontsize=13, fontweight='bold')
plt.ylabel('Probabilidad de Alta Mortalidad', fontsize=13, fontweight='bold')
plt.title('Clasificacion Predicha por la Regresion Logistica (1972-2024)', fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Guardar y mostrar grafico
plt.savefig('curva_logistica_prediccion.png', dpi=300)
plt.show()
