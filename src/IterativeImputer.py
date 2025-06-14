#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para imputar valores nulos usando IterativeImputer en el dataset "Board_Summary_Fusionado.csv"
procesándolo en chunks para no cargar todo el dataset en memoria.

Se ajusta el imputador en un sample de 5000 filas y luego se aplica en chunks de 10,000 filas.
El resultado se guarda en "II_Board_Summary_Filtered.csv".
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Habilitar IterativeImputer
from sklearn.impute import IterativeImputer

def main():
    sample_file = "Board_Summary_Fusionado.csv"
    
    # 1. Ajustar el imputador en una muestra representativa
    print("Cargando un sample de 5000 filas para ajustar IterativeImputer...")
    sample_df = pd.read_csv(sample_file, nrows=5000, low_memory=False)
    sample_df.columns = sample_df.columns.str.strip()  # Limpiar nombres de columnas

    # Definir las columnas a imputar (excluyendo "Nationality Mix" según lo acordado)
    ml_columns = [
        "Time to Retirement",
        "Age (Yrs)",
        "Avg. Yrs on Other Quoted Boards",
        "Total Number of Quoted Boards to Date"
    ]

    # Convertir estas columnas a numérico (errores se convierten en NaN)
    for col in ml_columns:
        sample_df[col] = pd.to_numeric(sample_df[col], errors="coerce")

    # Inicializar y ajustar el IterativeImputer con una semilla para reproducibilidad
    iter_imputer = IterativeImputer(random_state=42)
    iter_imputer.fit(sample_df[ml_columns])
    print("IterativeImputer ajustado en el sample.")

    # 2. Procesar el dataset completo en chunks para no cargarlo entero en memoria
    chunksize = 10000  # Puedes ajustar este valor según la memoria disponible
    imputed_chunks = []
    print("Procesando el dataset completo en chunks...")
    for chunk in pd.read_csv(sample_file, chunksize=chunksize, low_memory=False):
        # Limpiar nombres de columnas
        chunk.columns = chunk.columns.str.strip()

        # Convertir las columnas de interés a numérico
        for col in ml_columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        
        # Aplicar el imputador previamente ajustado a las columnas seleccionadas
        chunk[ml_columns] = iter_imputer.transform(chunk[ml_columns])
        
        # Almacenar el chunk imputado
        imputed_chunks.append(chunk)
        print("Procesado un chunk de tamaño:", len(chunk))

    # 3. Concatenar todos los chunks imputados
    df_imputed = pd.concat(imputed_chunks, ignore_index=True)
    print("Todos los chunks han sido concatenados. Total filas:", len(df_imputed))

    # 4. Guardar el dataset final imputado en un archivo CSV
    output_file = "II_Board_Summary_Filtered.csv"
    df_imputed.to_csv(output_file, index=False)
    print("Iterative Imputer completado y guardado en '{}'.".format(output_file))

if __name__ == "__main__":
    main()