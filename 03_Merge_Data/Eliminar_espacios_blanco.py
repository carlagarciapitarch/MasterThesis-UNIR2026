#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
import openpyxl
df = pd.read_excel("merge_all_databases.xlsx")
#print(df)

# Filtrar filas donde columnas estén vacias
df_limpio = df.dropna(subset=["SMILES", "IC50/EC50(microM)"])

# Guardar el resultado en un nuevo archivo
df_limpio.to_excel("merge_all_sin_blancos.xlsx", index=False)

# El documento BindingDB_filtrado_nulos lo he cambiado a BindingDB_Data_processing para 
# que esté todo igual
