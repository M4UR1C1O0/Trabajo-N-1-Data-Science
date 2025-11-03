import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('Datos/datos_sin_subcategoria.csv')
fallecidos_col = 'Fallecidos'
for col in df.columns:
    if col != 'Año':
        df[col] = pd.to_numeric(df[col], errors='coerce')

mediana = df[fallecidos_col].median()
df['alta_mortalidad'] = (df[fallecidos_col] > mediana).astype(int)

features = ['Siniestros', 'Lesionados - Graves', 
            'Indicadores cada 100.000 habitantes - Siniestralidad',
            'Parque vehicular', 'Población', 'Tasa motorización']
X = df[features]
y = df['alta_mortalidad']
mask = X.notnull().all(axis=1) & y.notnull()
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

arbol = DecisionTreeClassifier(max_depth=4, min_samples_split=8, min_samples_leaf=5, random_state=42)
arbol.fit(X_train_scaled, y_train)

plt.figure(figsize=(20, 12))
plot_tree(arbol, feature_names=features, class_names=['Baja', 'Alta'], 
          filled=True, rounded=True, fontsize=11)
plt.title('Árbol de Decisión: Mortalidad en Accidentes (Chile 1972-2024)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('arbol_decision_mejorado.png', dpi=300, bbox_inches='tight')
plt.show()
