import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('Board_Summary_Fusionado.csv', low_memory=False)

# Escribir el DataFrame en un archivo Excel (.xlsx)
df.to_excel('Board_Summary_Fusionado_Excel.xlsx', index=False)


