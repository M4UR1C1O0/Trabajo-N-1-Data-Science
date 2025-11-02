import pandas as pd
import numpy as np

archivo = 'Datos/datos_sin_subcategoria.csv'
df = pd.read_csv(archivo, encoding='utf-8')
df.columns = [col.lower() for col in df.columns]
problemas = []

# ==================== VALIDAR AÑO ====================
if 'año' in df.columns:
    año_norm = df['año'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    vacios = df['año'].isnull() | (año_norm == '') | (año_norm == 'nan')
    valido = año_norm.str.match(r'^\d{4}$')
    invalido = ~vacios & ~valido
    
    for idx in df.index[invalido]:
        problemas.append({'tipo': 'año - no valido', 'columna': 'año', 'fila': idx, 'valor': df.loc[idx, 'año'], 'desc': 'texto/simbolos'})
    for idx in df.index[vacios]:
        problemas.append({'tipo': 'año - vacio', 'columna': 'año', 'fila': idx, 'valor': df.loc[idx, 'año'], 'desc': 'nan/vacio'})
    
    if invalido.any(): print(f"\ninvalidos: {invalido.sum()}")
    if vacios.any(): print(f"vacios: {vacios.sum()}")
    
    if valido.any(): 
        df.loc[valido, 'año'] = año_norm[valido].astype(int)
        dup = df.loc[valido].duplicated(subset=['año']).sum()
        if dup > 0:
            print(f"duplicados: {dup}")
            for idx in df.loc[valido][df.loc[valido].duplicated(subset=['año'], keep=False)].index:
                problemas.append({'tipo': 'año - duplicado', 'columna': 'año', 'fila': idx, 'valor': df.loc[idx, 'año'], 'desc': 'repetido'})

# ==================== VALIDAR NUMERICAS ====================
def validar(col, permitir_neg=False):
    if col not in df.columns: return
    col_norm = pd.to_numeric(df[col], errors='coerce')
    nulos = col_norm.isnull()
    negativos = (col_norm < 0) & ~nulos if not permitir_neg else pd.Series([False] * len(df))
    no_num = ~df[col].astype(str).str.strip().str.match(r'^-?\d+\.?\d*$') & ~df[col].isnull() & (df[col].astype(str).str.strip() != '') & (df[col].astype(str).str.strip() != 'nan')
    
    for idx in df.index[no_num]: problemas.append({'tipo': f'{col} - no numerico', 'columna': col, 'fila': idx, 'valor': df.loc[idx, col], 'desc': 'texto'})
    for idx in df.index[nulos]: problemas.append({'tipo': f'{col} - vacio', 'columna': col, 'fila': idx, 'valor': 'nan', 'desc': 'vacio'})
    for idx in df.index[negativos]: problemas.append({'tipo': f'{col} - negativo', 'columna': col, 'fila': idx, 'valor': col_norm.loc[idx], 'desc': 'negativo'})
    
    print(f"\n{col}: no_num={no_num.sum()} nulos={nulos.sum()} neg={negativos.sum()}")
    df[col] = col_norm

cols = ['siniestros', 'fallecidos', 'lesionados - graves', 'lesionados - menos graves', 'lesionados - leves',
        'total lesionados', 'total victimas', 'parque vehicular', 'poblacion', 'tasa motorizacion',
        'vehiculos cada 100 habitantes', 'indicadores cada 10.000 vehiculos - siniestralidad',
        'indicadores cada 10.000 vehiculos - mortalidad', 'indicadores cada 10.000 vehiculos - morbilidad',
        'indicadores cada 100.000 habitantes - siniestralidad', 'indicadores cada 100.000 habitantes - mortalidad',
        'indicadores cada 100.000 habitantes - morbilidad', 'fallecidos cada 100 siniestros', 'siniestros por cada fallecido']

print("\n" + "="*80)
print("validaciones")
print("="*80)
for col in cols: validar(col)

# ==================== CHEQUEOS GENERALES ====================
print(f"\n{'='*80}\nchequeos generales\n{'='*80}")

vac = df[df.isnull().all(axis=1)].index
if len(vac) > 0:
    for idx in vac: problemas.append({'tipo': 'estructura - fila vacia', 'columna': 'todas', 'fila': idx, 'valor': 'vacia', 'desc': 'sin datos'})
    print(f"\nfilas vacias: {len(vac)}")

dup = df[df.duplicated(keep=False)].index
if len(dup) > 0:
    for idx in dup: problemas.append({'tipo': 'estructura - duplicada', 'columna': 'todas', 'fila': idx, 'valor': 'dup', 'desc': 'igual a otra'})
    print(f"filas duplicadas: {len(dup)}")

if all(c in df.columns for c in ['lesionados - graves', 'lesionados - menos graves', 'lesionados - leves', 'total lesionados']):
    inco = df[~np.isclose(df['total lesionados'], df['lesionados - graves'] + df['lesionados - menos graves'] + df['lesionados - leves'], equal_nan=True)]
    if len(inco) > 0:
        for idx in inco.index: problemas.append({'tipo': 'coherencia - lesionados', 'columna': 'total lesionados', 'fila': idx, 'valor': f"t={inco.loc[idx,'total lesionados']}", 'desc': 'suma incorrecta'})
        print(f"incoherencias lesionados: {len(inco)}")

if all(c in df.columns for c in ['fallecidos', 'total lesionados', 'total victimas']):
    inco = df[~np.isclose(df['total victimas'], df['fallecidos'] + df['total lesionados'], equal_nan=True)]
    if len(inco) > 0:
        for idx in inco.index: problemas.append({'tipo': 'coherencia - victimas', 'columna': 'total victimas', 'fila': idx, 'valor': f"t={inco.loc[idx,'total victimas']}", 'desc': 'suma incorrecta'})
        print(f"incoherencias victimas: {len(inco)}")

# ==================== REPORTE ====================
print(f"\n{'='*80}\nreporte final\n{'='*80}\n")
if problemas:
    print(f"total: {len(problemas)} errores\n")
    print(f"{'fila':<6} {'columna':<35} {'valor':<15} {'descripcion':<20}")
    print("-" * 80)
    for p in problemas:
        print(f"{p['fila']:<6} {p['columna']:<35} {str(p['valor']):<15} {p['desc']:<20}")
    print("\ncorrige los datos en tu CSV segun las filas y columnas indicadas\n")
else:
    print("datos validados correctamente, listo para analisis\n")
