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


# Cargar datasets
df_main = pd.read_excel("mi_dataset.xlsx")
df_ext = pd.read_excel("dataset_externo.xlsx")

# Canonicalizar
df_main["smiles_can"] = df_main["smiles"].apply(canonicalize)
df_ext["smiles_can"] = df_ext["smiles"].apply(canonicalize)

# Eliminar duplicados
df_filtrado = df_main[~df_main["smiles_can"].isin(df_ext["smiles_can"])]

# (opcional) eliminar columna auxiliar
df_filtrado = df_filtrado.drop(columns=["smiles_can"])

# Guardar
df_filtrado.to_excel("mi_dataset_filtrado.xlsx", index=False)











df_peptidos = pd.read_excel("df_peptidos.xlsx")
df_mol_peq = pd.read_excel("df_mol_pequenas.xlsx")

# Crear columna de SMILES canónico
df_peptidos['SMILES_canonico'] = df_peptidos['SMILES'].apply(smiles_canonico)
df_mol_peq['SMILES_canonico'] = df_mol_peq['SMILES'].apply(smiles_canonico)

#mirar si hay duplicados en peptidos
duplicados_pept = df_peptidos[df_peptidos.duplicated(subset=['SMILES_canonico', 'IC50/EC50(microM)'], keep=False)]
print(duplicados_pept)

#mirar si hay duplicados en mol pequeñas
duplicados_mol_peq = df_mol_peq[df_mol_peq.duplicated(subset=['SMILES_canonico', 'IC50/EC50(microM)'], keep=False)]
print(duplicados_mol_peq)


#eliminadar entradas con igual smile canonico e igual valor de inhibicion porque serían duplicados
df_pept_sin_dup = df_peptidos.drop_duplicates(subset=['SMILES_canonico', 'IC50/EC50(microM)']) # peptidos
df_mol_peq_sin_dup = df_mol_peq.drop_duplicates(subset=['SMILES_canonico', 'IC50/EC50(microM)']) # mol pequeñas

df_pept_sin_dup.to_excel("peptidos_sin_duplicados.xlsx", index=False)
df_mol_peq_sin_dup.to_excel("mol_peq_sin_duplicados.xlsx", index=False)

#cambiamos valores de IC50 por 0 y 1. 0 es si es igual o menor a 5 microM y 1 si es mayor a 5 microM. 
def actividad_activo_inactivo(valor):
  if valor <= 5:
    return 0
  else:
    return 1

df = pd.read_excel("todos_datos_limpios.xlsx")

df['Actividad(0/1)'] = df['IC50/EC50(microM)'].apply(actividad_activo_inactivo)
print(df)
print(df['Actividad(0/1)'].value_counts())
df.to_excel("todos_datos_con_0_1_5uM.xlsx")
