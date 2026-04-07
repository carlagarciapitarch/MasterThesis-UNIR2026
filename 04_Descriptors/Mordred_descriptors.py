#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:21:25 2026

@author: carla
"""

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

# Cargar datos
df = pd.read_excel("todos_datos_con_0_1.xlsx")

# Convertir SMILES → moléculas RDKit
df["RDKit"] = df["SMILES_canonico"].apply(
    lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None
)

# Eliminar SMILES inválidos
df = df[df["RDKit"].notna()].copy()

# Inicializar calculadora
calc = Calculator(descriptors)

# Calcular descriptores (esto sí funciona con pandas)
desc = calc.map(df["RDKit"])
desc = pd.DataFrame([d.asdict() for d in desc])


# Unir resultados
df_final = pd.concat(
    [df.reset_index(drop=True), desc.reset_index(drop=True)],
    axis=1
)

# Guardar
df_final.to_excel("todos_datos_con_descriptores.xlsx", index=False)

