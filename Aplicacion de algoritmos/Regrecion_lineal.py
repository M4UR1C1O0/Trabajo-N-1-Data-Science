from Limpieza import df_limpio # Importamos el dataframe limpio creado en limpieza.py

#Librereias
import pandas as pd
import numpy as np

#Librerias necesarias para la regresion lineal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Definimos las variables dependientes e independientes
X = ['Parque vehicular', 'Población']