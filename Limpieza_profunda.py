import pandas as pd
import numpy as np

archivo = 'Datos/datos_sin_subcategoria.csv'
df = pd.read_csv(archivo, encoding='utf-8')

problemas = []

# Chequeo duplicados
if df.duplicated().sum() > 0:
    problemas.append("Filas duplicadas (completas)")
    print(" Filas duplicadas:")
    print(df[df.duplicated(keep=False)])

if 'ano' in df.columns and df.duplicated(subset=['ano']).sum() > 0:
    problemas.append("Años duplicados")
    print(" Años duplicados:")
    print(df[df.duplicated(subset=['ano'], keep=False)][['ano']])

# Filas completamente vacias
if df.isnull().all(axis=1).sum() > 0:
    problemas.append("Filas completamente vacias (todos los valores nulos)")
    print(" Filas completamente vacias (muestra):")
    print(df[df.isnull().all(axis=1)])

# Nulos
null_cols = df.isnull().sum()[df.isnull().sum() > 0]
if not null_cols.empty:
    problemas.append(f"Columnas con nulos: {null_cols.to_dict()}")
    for col in null_cols.index:
        print(f" Filas con nulo en columna '{col}':")
        print(df[df[col].isnull()][[col]])

# Negativos
for col in df.select_dtypes('number').columns:
    nneg = (df[col] < 0).sum()
    if nneg > 0:
        problemas.append(f"Columna '{col}': {nneg} valores negativos")
        print(f" Filas con negativo en '{col}':")
        print(df[df[col] < 0][[col]])

# Rango de año
if 'ano' in df.columns:
    fuera = df[(df['ano'] < 1972) | (df['ano'] > 2024)]
    if not fuera.empty:
        problemas.append(f"Años fuera de rango [1972-2024]: {fuera['ano'].tolist()}")
        print(" Filas con años fuera de rango:")
        print(fuera[['ano']])

# Coherencia suma victimas
if set(['fallecidos','total_lesionados','total_victimas']).issubset(df.columns):
    mask = ~np.isclose(df['total_victimas'], df['fallecidos'] + df['total_lesionados'])
    incoh = mask.sum()
    if incoh > 0:
        problemas.append(f"Incoherencia en suma de victimas: {incoh} filas")
        print(" Filas con incoherencia suma:")
        print(df[mask][['ano','fallecidos','total_lesionados','total_victimas']])


print("\nREPORTE DE PROBLEMAS ENCONTRADOS ")
if problemas:
    for p in problemas:
        print(" ", p)
    print("\nCorrige exactamente las FILAS/COLUMNAS mostradas arriba antes de continuar el analisis.")
else:
    print("No se encontraron problemas. El archivo esta correcto y normalizado")