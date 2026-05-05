#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:21:25 2026

@author: carla
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef, roc_curve, auc
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


# =========================
# CARGA DE DATOS
# =========================

df = pd.read_excel("datos_finales_molpeq_para_ML_0,9.xlsx")

X = df.drop(columns=["Actividad(0/1)"])
y = df["Actividad(0/1)"].map({0: 1, 1: 0})  # 1 = activo
#cambio esto porque antes tenia 0 activo y 1 inactivo pero quiero los modelos con probabilidad de calcular activo y usa por defecto 1.

# =========================
# SPLIT: TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# MÉTRICAS
# =========================

def evaluate_model(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "specificity": tn / (tn + fp),
        "sensitivity": tp / (tp + fn),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }

# =========================
# CV CONFIG
# =========================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "mcc": "matthews_corrcoef"
}

# =========================
# MODELOS
# =========================

models = {

    "RF": (
        Pipeline([
            ("model", RandomForestClassifier(class_weight="balanced"))
        ]),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10],
        }
    ),

    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, class_weight="balanced"))
        ]),
        {
            "model__C": [0.1, 1, 10],
        }
    ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        {
            "model__n_neighbors": [3, 5, 7],
        }
    ),

    "NB": (
        Pipeline([
            ("model", GaussianNB())
        ]),
        {}
    ),

    "LR": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        {
            "model__C": [0.1, 1, 10],
        }
    ),

    "AdaBoost": (
        Pipeline([
            ("model", AdaBoostClassifier())
        ]),
        {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1],
        }
    ),

    "GB": (
        Pipeline([
            ("model", GradientBoostingClassifier())
        ]),
        {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5],
        }
    ),

    "ExtraTrees": (
        Pipeline([
            ("model", ExtraTreesClassifier(class_weight="balanced"))
        ]),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10],
        }
    ),

    "MLP": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(max_iter=1000))
        ]),
        {
            "model__hidden_layer_sizes": [(50,), (100,)],
        }
    ),

    "XGB": (
        Pipeline([
            ("model", XGBClassifier(eval_metric='logloss'))
        ]),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.01, 0.1],
        }
    )
}

# =========================
# GRID SEARCH
# =========================

results_test = {}
best_models = {}
best_params = {}

with pd.ExcelWriter("gridsearch_resultados.xlsx") as writer:

    for name, (pipe, params) in models.items():
        print(f"Entrenando {name}...")
        
        grid = GridSearchCV(
            pipe,
            params,
            cv=cv,
            scoring=scoring,
            refit="roc_auc",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_models[name] = best_model
        best_params[name] = grid.best_params_

        # Guardar CV completo
        df_grid = pd.DataFrame(grid.cv_results_)
        df_grid.to_excel(writer, sheet_name=name[:31], index=False)

        # Evaluación en TEST
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        results_test[name] = evaluate_model(y_test, y_pred, y_prob)

# =========================
# RESULTADOS
# =========================

df_results = pd.DataFrame(results_test).T
df_results = df_results.sort_values(by="roc_auc", ascending=False)

df_results.to_excel("resultados_test.xlsx")
pd.DataFrame(best_params).T.to_excel("mejores_parametros.xlsx")

print("\nResultados en TEST:")
print(df_results)

# Mejor modelo
best_model_name = df_results.index[0]
print("\nMejor modelo:", best_model_name)

# =========================
# ROC GLOBAL
# =========================

plt.figure(figsize=(8,6))

for name, model in best_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC={results_test[name]['roc_auc']:.3f})")

plt.plot([0,1], [0,1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test Set")
plt.legend()
plt.grid()
plt.savefig("ROC_global.png", dpi=150)
plt.show()

print("Figura guardada: ROC_global.png")

# =========================
# ROC INDIVIDUALES
# =========================

for name, model in best_models.items():
    plt.figure(figsize=(7,5))
    
    # TRAIN
    y_prob_train = model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
    auc_train = auc(fpr_train, tpr_train)
    
    # TEST
    y_prob_test = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    auc_test = auc(fpr_test, tpr_test)
    
    plt.plot(fpr_train, tpr_train, label=f"Train (AUC={auc_train:.3f})")
    plt.plot(fpr_test, tpr_test, label=f"Test (AUC={auc_test:.3f})")
    
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    
    plt.title(f"ROC - {name}")
    plt.legend()
    plt.grid()
    
    filename = f"ROC_{name}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
   
    plt.show()
    plt.close()
   
print(f"Guardado: {filename}")





# =========================
# INTERPRETACIÓN DEL MODELO (SHAP)
# =========================

import shap
import numpy as np

print(f'Calculando SHAP para el modelo: {best_model_name}...')

# =========================
# PREPARAR DATOS
# =========================

# Si usas pipeline con scaler:
if hasattr(best_model, "named_steps") and "scaler" in best_model.named_steps:
    X_shap = best_model.named_steps["scaler"].transform(X_train)
else:
    X_shap = X_train.copy()

FEATURES = X_train.columns

# =========================
# SHAP SEGÚN MODELO
# =========================

if best_model_name in ["RF", "XGB", "GB", "ExtraTrees"]:
    
    # Extraer modelo real del pipeline
    model = best_model.named_steps["model"] if "model" in best_model.named_steps else best_model
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # Para clasificación binaria
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

else:
    # Para modelos tipo SVM, KNN, LR, etc.
    
    # Reducir tamaño por coste computacional
    X_sample = X_shap[:100]
    
    explainer = shap.KernelExplainer(
        best_model.predict_proba,
        X_sample
    )
    
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    X_shap = X_sample

print("SHAP calculado")

# =========================
# PLOT
# =========================

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=FEATURES,
    max_display=20,
    show=False
)

plt.title(f"SHAP Summary - {best_model_name}")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.show()

print("Figura guardada: shap_summary.png")

# =========================
# IMPORTANCIA GLOBAL
# =========================

importance_df = pd.DataFrame({
    "descriptor": FEATURES,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values(by="mean_abs_shap", ascending=False)

importance_df.to_excel("shap_importance.xlsx", index=False, engine="openpyxl")

print("\nTop 10 descriptores más importantes:")
print(importance_df.head(10))

#df_prueba = df.to_excel("df_prueba.xlsx", engine="openpyxl")


# =========================
# Applicability Domain DURANTE ENTRENAMIENTO PARA CONSTRUIR DOMINIO DE APLICABILIDAD
# =========================
#kNN-AD
# Una molécula está dentro del dominio si está “cerca” de moléculas del training set.

from sklearn.neighbors import NearestNeighbors
import numpy as np

# =========================
# 1. PREPARAR TRAINING SET
# =========================

if "scaler" in best_model.named_steps:
    X_train_ad = best_model.named_steps["scaler"].transform(X_train)
else:
    X_train_ad = X_train.values


# =========================
# 2. FIT DEL MODELO kNN
# =========================

nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_train_ad)

# =========================
# 3. DISTANCIAS EN TRAINING (para definir umbral)
# =========================

distances_train, _ = nn.kneighbors(X_train_ad)

# media de la distancia a vecinos
mean_dist_train = distances_train.mean(axis=1)

# umbral (regla clásica)
threshold = mean_dist_train.mean() + 2 * mean_dist_train.std()

print(f"Threshold AD: {threshold:.4f}")

