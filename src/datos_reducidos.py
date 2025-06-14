import csv
from pathlib import Path
import pandas as pd

# Rutas absolutas de entrada/salida
base_dir    = Path(r"C:/Users/Alicia/Desktop/TFG - INF/TFG_AliciaSanz")
csv_dir     = base_dir / "data" / "processed"
excel_dir   = base_dir / "data" / "raw" / "xslx"
output_dir  = csv_dir / "reduced"

# Crear carpeta de salida
output_dir.mkdir(parents=True, exist_ok=True)

# 1) Reducir todos los CSV a 50 líneas
for csv_path in csv_dir.glob("*.csv"):
    out_csv = output_dir / csv_path.name
    print(f"Reduciendo CSV: {csv_path.name} → {out_csv.name}")
    with csv_path.open(newline='', encoding='utf-8') as fin, \
         out_csv.open('w', newline='', encoding='utf-8') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for i, row in enumerate(reader):
            writer.writerow(row)
            if i >= 19:
                break

# 2) Reducir todos los XLSX a 50 filas y exportar a XLSX
for xlsx_path in excel_dir.glob("*.xlsx"):
    out_xlsx = output_dir / xlsx_path.name
    print(f"Reduciendo XLSX: {xlsx_path.name} → {out_xlsx.name}")
    df = pd.read_excel(xlsx_path, nrows=20)
    df.to_excel(out_xlsx, index=False)

print(" Todos los archivos grandes han sido reducidos y están en:")
print(output_dir)
