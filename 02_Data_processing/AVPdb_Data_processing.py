#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
import openpyxl
df = pd.read_excel("AVPdb_Data_processing.xlsx")
print(df)

# Filtrar filas donde la columna 'Virus' contenga 'dengue'
filtro = df[df["Virus"].str.contains("dengue", case=False, na=False)]

# Guardar el resultado en un nuevo archivo
filtro.to_excel("filtrado_dengue.xlsx", index=False)

# El documento filtrado_dengue lo he cambiado a AVPdb_Data_processing para 
# que esté todo igual
