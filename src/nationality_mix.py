import pandas as pd

# Cargar el dataset
df = pd.read_csv("Board_Summary_Fusionado.csv", low_memory=False)
df.columns = df.columns.str.strip()  # Limpiar espacios en los nombres de las columnas

# Mostrar un resumen de la variable "Nationality Mix"
print("Antes de imputar, valores nulos en 'Nationality Mix':", df["Nationality Mix"].isnull().sum())
print("Valores únicos (sin nulos):", df["Nationality Mix"].dropna().unique())

# Calcular la moda de la columna "Nationality Mix"
mode_nationality = df["Nationality Mix"].mode()[0]
print("La moda de 'Nationality Mix' es:", mode_nationality)

# Imputar los valores nulos con la moda
df["Nationality Mix"].fillna(mode_nationality, inplace=True)

# Verificar que ya no queden nulos
print("Después de imputar, valores nulos en 'Nationality Mix':", df["Nationality Mix"].isnull().sum())

# Guardar el dataset modificado en un nuevo archivo CSV
df.to_csv("Board_Summary_NationalityImputed.csv", index=False)
print("Imputación de 'Nationality Mix' completada y guardada en 'Board_Summary_NationalityImputed.csv'.")

num_german = df_fusionado[df_fusionado["Nationality Mix"] == "German"].shape[0]
print("Número de filas con 'German' en Nationality Mix:", num_german)
