#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para imputar valores nulos usando KNNImputer en el dataset "Board_Summary_Fusionado.csv"
procesándolo en chunks para no cargar todo el dataset en memoria.

Se ajusta el imputador en un sample de 5000 filas y luego se aplica en chunks de 10,000 filas.
El resultado se guarda en "KNN_Board_Summary_Filtered.csv".
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def main():
    # ------------------------------
    # 1. Ajustar el imputador en un sample representativo
    # ------------------------------
    sample_file = "Board_Summary_Fusionado.csv"
    print("Cargando un sample de 5000 filas...")
    sample_df = pd.read_csv(sample_file, nrows=5000, low_memory=False)
    sample_df.columns = sample_df.columns.str.strip()  # Limpiar nombres de columnas

    # Definir las columnas a imputar
    ml_columns = [
        "Time to Retirement",
        "Age (Yrs)",
        "Avg. Yrs on Other Quoted Boards",
        "Total Number of Quoted Boards to Date"
    ]

    # Convertir estas columnas a numérico (errores se convierten en NaN)
    for col in ml_columns:
        sample_df[col] = pd.to_numeric(sample_df[col], errors="coerce")

    # Inicializar y ajustar el KNNImputer con 5 vecinos
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_imputer.fit(sample_df[ml_columns])
    print("KNN Imputer ajustado en el sample.")

    # ------------------------------
    # 2. Procesar el dataset completo en chunks
    # ------------------------------
    chunksize = 10000  # Ajusta este valor según la memoria disponible
    imputed_chunks = []
    
    print("Procesando el dataset completo en chunks...")
    for chunk in pd.read_csv(sample_file, chunksize=chunksize, low_memory=False):
        # Limpiar nombres de columnas
        chunk.columns = chunk.columns.str.strip()
        
        # Convertir las columnas de interés a numérico
        for col in ml_columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        
        # Aplicar el imputador previamente ajustado
        chunk[ml_columns] = knn_imputer.transform(chunk[ml_columns])
        
        # Almacenar el chunk imputado
        imputed_chunks.append(chunk)
        print("Procesado un chunk de tamaño:", len(chunk))
    
    # Concatenar todos los chunks imputados
    df_imputed = pd.concat(imputed_chunks, ignore_index=True)
    print("Todos los chunks han sido concatenados. Total filas:", len(df_imputed))
    
    # ------------------------------
    # 3. Guardar el dataset final imputado
    # ------------------------------
    output_file = "KNN_Board_Summary_Filtered.csv"
    df_imputed.to_csv(output_file, index=False)
    print("KNN Imputation completada y guardada en '{}'.".format(output_file))


if __name__ == "__main__":
    main()
                                               