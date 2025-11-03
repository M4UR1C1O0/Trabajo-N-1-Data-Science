import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Cargar datos
df = pd.read_csv('Datos/datos_sin_subcategoria.csv')
for col in df.columns:
    if col != 'Año':
        df[col] = pd.to_numeric(df[col], errors='coerce')

mediana = df['Fallecidos'].median()
df['alta_mortalidad'] = (df['Fallecidos'] > mediana).astype(int)

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

# Crear un arbol visual interpretable con etiquetas claras
fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Funcion para dibujar nodos
def draw_node(ax, x, y, text, node_type='decision', width=2.5, height=0.8):
    colors = {
        'decision': '#87CEEB',      # Azul cielo
        'question': '#FFD700',      # Dorado
        'baja': '#90EE90',          # Verde claro
        'alta': '#FF6B6B'           # Rojo
    }
    
    color = colors.get(node_type, '#CCCCCC')
    
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(box)
    
    # Añadir texto con multiples lineas si es necesario
    ax.text(x, y, text, ha='center', va='center', fontsize=10, 
            fontweight='bold', wrap=True, multialignment='center')

# Funcion para dibujar flechas
def draw_arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y + 0.2, label, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8))

# NODO RAiZ
draw_node(ax, 5, 9, "¿Lesionados Graves\nbajo promedio?", 'question', width=2.8, height=1)

# NIVEL 2 - IZQUIERDA (Lesionados bajos)
draw_node(ax, 2.5, 7.2, "¿Muy pocos\nlesionados?", 'question', width=2.5, height=0.9)
draw_arrow(ax, 3.8, 8.6, 2.8, 7.8, "Si")

# NIVEL 2 - DERECHA (Lesionados altos)
draw_node(ax, 7.5, 7.2, "¿Tasa de\nsiniestralidad alta?", 'question', width=2.5, height=0.9)
draw_arrow(ax, 6.2, 8.6, 7.2, 7.8, "NO")

# NIVEL 3 - IZQUIERDA IZQUIERDA
draw_node(ax, 1.2, 5.5, "BAJA\nmortalidad\n", 'baja', width=1.8, height=0.9)
draw_arrow(ax, 1.8, 6.7, 1.4, 6, "Si")
ax.text(0.9, 6.2, "Años muy\nseguros", fontsize=8, style='italic', ha='center')

# NIVEL 3 - IZQUIERDA DERECHA
draw_node(ax, 3.8, 5.5, "¿Pocos\nsiniestros\nese año?", 'question', width=2.2, height=1)
draw_arrow(ax, 3.2, 6.7, 3.9, 6.1, "NO")

# NIVEL 3 - DERECHA IZQUIERDA
draw_node(ax, 6.2, 5.5, "ALTA\nmortalidad\n", 'alta', width=1.8, height=0.9)
draw_arrow(ax, 6.8, 6.7, 6.4, 6, "Si")
ax.text(5.8, 6.2, "Severidad\nalta", fontsize=8, style='italic', ha='center')

# NIVEL 3 - DERECHA DERECHA
draw_node(ax, 8.8, 5.5, "BAJA\nmortalidad\n", 'baja', width=1.8, height=0.9)
draw_arrow(ax, 8.2, 6.7, 8.6, 6, "NO")
ax.text(9.2, 6.2, "Tasa\nnormalizada", fontsize=8, style='italic', ha='center')


draw_node(ax, 2.5, 3.8, "ALTA\nmortalidad\n", 'alta', width=1.8, height=0.9)
draw_arrow(ax, 3.2, 5.0, 2.8, 4.4, "Si")
ax.text(1.9, 4.6, "Pocos eventos,\nmuy graves", fontsize=8, style='italic', ha='center')

draw_node(ax, 5.1, 3.8, "BAJA\nmortalidad\n", 'baja', width=1.8, height=0.9)
draw_arrow(ax, 4.4, 5.0, 5.2, 4.4, "NO")
ax.text(5.9, 4.6, "Muchos eventos,\nmenos graves", fontsize=8, style='italic', ha='center')

# Titulo y leyenda
ax.text(5, 9.8, "Arbol de Decision: Prediccion de Mortalidad en Accidentes Viales (Chile 1972-2024)", 
        fontsize=14, fontweight='bold', ha='center')

# Leyenda
legend_elements = [
    mpatches.Patch(facecolor='#FFD700', edgecolor='black', label='Pregunta / Decision'),
    mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='Prediccion: BAJA mortalidad'),
    mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Prediccion: ALTA mortalidad')
]
ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig('arbol_decision_explicativo.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()