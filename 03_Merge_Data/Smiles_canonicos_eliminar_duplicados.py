#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:49:29 2026

@author: carla
"""

import pandas as pd
from rdkit import Chem

def smiles_canonico(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


df = pd.read_excel("merge_all_databases.xlsx")
# Crear columna de SMILES canónico
df['SMILES_canonico'] = df['SMILES'].apply(smiles_canonico)

#mirar si hay duplicados
duplicados = df[df.duplicated(subset=['SMILES_canonico', 'IC50/EC50(microM)'], keep=False)]
print(duplicados)


#eliminadar entradas con igual smile canonico e igual valor de inhibicion porque serían duplicados
df_sin_dup = df.drop_duplicates(subset=['SMILES_canonico', 'IC50/EC50(microM)'])
df_sin_dup.to_excel("todos_datos_limpios.xlsx", index=False)

#cambiamos valores de IC50 por 0 y 1. 0 es si es igual o menor a 1 microM y 1 si es mayor a 1 microM. 
def actividad_activo_inactivo(valor):
  if valor <= 1:
    return 0
  else:
    return 1

df = pd.read_excel("todos_datos_limpios.xlsx")

df['Actividad(0/1)'] = df['IC50/EC50(microM)'].apply(actividad_activo_inactivo)
print(df)
print(df['Actividad(0/1)'].value_counts())
df.to_excel("todos_datos_con_0_1.xlsx")
