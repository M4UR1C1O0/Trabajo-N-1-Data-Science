import pandas as pd

# Cargar CSV original
archivo = 'Datos/EvolucionsiniestrostransitoChile-1972-2024.csv'
nombres = [
    'Año','Siniestros','Fallecidos','Lesionados - Graves','Lesionados - Menos graves','Lesionados - Leves','Total lesionados',
    'Total víctimas','Tasa motorización','Vehículos cada 100 habitantes','Parque vehicular','Población',
    'Indicadores cada 10.000 vehículos - Siniestralidad','Indicadores cada 10.000 vehículos - Mortalidad','Indicadores cada 10.000 vehículos - Morbilidad',
    'Indicadores cada 100.000 habitantes - Siniestralidad','Indicadores cada 100.000 habitantes - Mortalidad','Indicadores cada 100.000 habitantes - Morbilidad',
    'Fallecidos cada 100 siniestros','Siniestros por cada fallecido'
]

df = pd.read_csv(archivo, sep=';', encoding='latin-1', skiprows=4, names=nombres, dtype=str)

# Quita puntos de miles (menos la columna Año)
for col in df.columns:
    if col != 'Año':
        df[col] = df[col].str.replace('.', '', regex=False)

# Elimina filas que no sean año de 4 digitos legitimo
filtro_anios = df['Año'].fillna('').str.match(r'^\d{4}$')
df = df[filtro_anios].copy()

# Resetea índice
df = df.reset_index(drop=True)

# (opcional) Convierte columnas a float para estadísticas y gráficos
for col in df.columns:
    if col != 'Año':
        df[col] = df[col].str.replace(',', '.').astype(float)
df['Año'] = df['Año'].astype(int)

# (opcional) Exporta CSV limpio
df.to_csv('Datos/datos_sin_subcategoria.csv', index=False, encoding='utf-8')
print("Listo: solo años, sin puntos, ni filas basura, y todo numérico.")
