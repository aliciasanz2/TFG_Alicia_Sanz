import pandas as pd

# Cargar el archivo Excel
df = pd.read_excel("Board_Summary_Fusionado_Excel.xlsx")

# Suponiendo que las columnas A a AK corresponden a las primeras 37 columnas del DataFrame
# Se filtran las filas que estén completas en esas columnas
df_completas = df[df.iloc[:, :37].notna().all(axis=1)]

# Mostrar la cantidad de filas completas y una vista previa de ellas
print("Número de filas completas (columnas A a DL):", len(df_completas))
print(df_completas.head())
