#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd

#cambiamos valores de IC50 por 0 y 1. 0 es si es igual o menor a 5 microM y 1 si es mayor a 5 microM. 
def actividad_activo_inactivo(valor):
  if valor <= 5:
    return 0
  else:
    return 1

df_pept = pd.read_excel("peptidos_sin_duplicados.xlsx")
df_mol_peq = pd.read_excel("mol_peq_sin_duplicados.xlsx")

df_pept['Actividad(0/1)'] = df_pept['IC50/EC50(microM)'].apply(actividad_activo_inactivo)
df_mol_peq['Actividad(0/1)'] = df_mol_peq['IC50/EC50(microM)'].apply(actividad_activo_inactivo)

print(df_pept['Actividad(0/1)'].value_counts())
print(df_mol_peq['Actividad(0/1)'].value_counts())

df_pept.to_excel("peptidos_con_0_1_5uM.xlsx")
df_mol_peq.to_excel("mol_peq_con_0_1_5uM.xlsx")