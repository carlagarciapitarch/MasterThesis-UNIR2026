#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
import openpyxl
df = pd.read_excel("DRAVP_Data_processing.xlsx")
print(df)

# Filtrar filas donde la columna 'Target_Organism' contenga 'DENV'.

filtro = df[df["Target_Organism"].str.contains("DENV", case=False, na=False) &
            (df["Activity"].str.contains("IC50|EC50", case=False, na=False))]
print(filtro)

# Guardar el resultado en un nuevo archivo
filtro.to_excel("DRAVP_filtrado_denv.xlsx", index=False)

# El documento DenvInD_filtrado_nulos lo he cambiado a DenvInD_Data_processing para 
# que esté todo igual
