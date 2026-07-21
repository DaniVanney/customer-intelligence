# Customer 360 transaccional

## Objetivo

Customer 360 transforma las líneas de transacción de Online Retail II en una
tabla analítica con una fila por cliente.

El dataset se utilizará como base para customer scoring y para la construcción
posterior de observaciones temporales del modelo de inactividad.

## Archivo

`data/processed/customer_360.parquet`

El archivo se genera con:

```powershell
python scripts/build_customer_360.py