#Librerias
import pandas as pd
import numpy as np

#1.- Definimos los nombres de las columnas
#Estos nombres se construyeron combinando:
#Línea 3 del CSV: Encabezados principales
#Línea 4 del CSV: Subcategorías

nombres_columnas = ['Año','Siniestros','Fallecidos','Lesionados - Graves','Lesionados - Menos graves','Lesionados - Leves','Total lesionados','Total víctimas','Tasa motorización',
                    'Vehículos cada 100 habitantes','Parque vehicular','Población','Indicadores cada 10.000 vehículos - Siniestralidad','Indicadores cada 10.000 vehículos - Mortalidad',
                    'Indicadores cada 10.000 vehículos - Morbilidad','Indicadores cada 100.000 habitantes - Siniestralidad','Indicadores cada 100.000 habitantes - Mortalidad',
                    'Indicadores cada 100.000 habitantes - Morbilidad','Fallecidos cada 100 siniestros','Siniestros por cada fallecido']

#2.- Cargamos los datos desde el archivo CSV
patharchivo = 'Datos/EvolucionsiniestrostransitoChile-1972-2024.csv'
df = pd.read_csv(patharchivo,sep=';',encoding='latin-1',skiprows=4, # Codificación del archivo # Saltar título y encabezados múltiples # Asignar nombres personalizados
                 names=nombres_columnas,na_values=['', ' '])

#3.- Filtrar filas invalidas y ordenar por año
#Mantenemos solo filas donde Año es un número
df_limpio = df[df['Año'].notna()].copy()
df_limpio = df_limpio[
    df_limpio['Año'].apply(
        lambda x: str(x).replace('.', '').replace(',', '').isdigit() 
        if pd.notna(x) else False
    )
].copy()

df_limpio['Año'] = df_limpio['Año'].astype(int)
df_limpio = df_limpio.sort_values('Año').reset_index(drop=True)

#4.- Convertir formatos numericos a float estándar
def convertir_numero(valor):

    #Convierte formato (26.727 = 26727, 3,5 = 3.5) a float estándar
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)):
        return float(valor)
    
    valor_str = str(valor).strip().replace('.', '').replace(',', '.')
    
    try:
        return float(valor_str)
    except:
        return np.nan

#Aplicamos la funcion anterior a todas las columnas excepto Año
for col in df_limpio.columns:
    if col != 'Año':
        df_limpio[col] = df_limpio[col].apply(convertir_numero)

#5.- Guardamos el archivo limpio sin subcategorías
nuevo_archivo = 'Datos/datos_sin_subcategorias.csv'
df_limpio.to_csv(nuevo_archivo, index=False, encoding='utf-8')

#print(df)
#print(df_limpio)