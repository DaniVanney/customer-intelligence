# Customer scoring

## Objetivo

El customer scoring busca ordenar y segmentar a los clientes según su comportamiento transaccional hasta el snapshot del `2011-12-10`.

Para este análisis se utilizaron dos enfoques. Primero se construyó un scoring RFM como referencia comercial. Después se desarrolló una segmentación con K-Means para incorporar más características del comportamiento de cada cliente.

## Archivos

El proceso utiliza `data/processed/customer_360.parquet` y genera `data/processed/customer_scoring.parquet`.

El análisis del modelo se encuentra en `notebooks/01_customer_segmentation.ipynb` y la implementación reutilizable en `src/customer_intelligence/scoring/clustering.py`.

El dataset final se genera con:

```powershell
python scripts/build_customer_scoring.py



```

## Scoring RFM

RFM utiliza tres variables: recencia, frecuencia y valor monetario.

`recency_days` representa los días desde la última compra efectiva, `order_count` la cantidad de pedidos y `total_spend` el gasto histórico después de reconciliar las cancelaciones identificables.

Cada dimensión recibe un score entre 1 y 5 mediante rangos percentiles. Una recencia menor recibe un score mayor. En frecuencia y valor monetario, los valores más altos reciben mejores scores.

El score total se calcula como:

```text
rfm_score = recency_score + frequency_score + monetary_score
```

A partir de estos valores se generan ocho segmentos comerciales: Valor estratégico, Incorporación reciente, Lealtad consolidada, Alto valor, Potencial de desarrollo, Riesgo de inactividad, Inactividad prolongada y Base general.

RFM se conserva como una referencia fácil de interpretar y explicar a usuarios de negocio.

## Segmentación con K-Means

Para ampliar el análisis se entrenó un modelo K-Means con cinco variables de Customer 360: recencia, cantidad de pedidos, gasto total, antigüedad y cantidad de productos únicos.

`total_items` y `average_order_value` se utilizaron para describir los perfiles, pero quedaron fuera del entrenamiento porque contienen información muy relacionada con el gasto y la cantidad de pedidos.

Se aplicó una transformación `log1p` a las variables con mayor asimetría y luego se estandarizaron las cinco variables. Esto permite que las distancias calculadas por K-Means no queden dominadas por las diferencias de escala.

Se compararon modelos de 2 a 8 clusters. La solución de cuatro clusters fue elegida por el equilibrio entre estabilidad, tamaño de los grupos e interpretación comercial. Obtuvo un silhouette score de `0.294` y un ARI medio de estabilidad de `0.997`.

## Resultados

| Segmento ML | Clientes | Clientes (%) | Gasto (%) |
|---|---:|---:|---:|
| Alto valor consolidado | 1.231 | 21,01 | 74,01 |
| Valor intermedio en seguimiento | 1.749 | 29,85 | 16,67 |
| Desarrollo reciente | 1.244 | 21,23 | 6,17 |
| Baja actividad prolongada | 1.635 | 27,91 | 3,16 |

Alto valor consolidado concentra el 74,01% del gasto histórico con el 21,01% de los clientes. Este grupo representa la principal prioridad de retención.

Valor intermedio en seguimiento reúne clientes con una relación más estable, aunque con mayor tiempo desde su última compra. Desarrollo reciente contiene clientes con poca antigüedad y margen para ampliar la relación. Baja actividad prolongada agrupa clientes con baja frecuencia, bajo valor y recencia elevada.

Las acciones asociadas a estos perfiles se incorporan en la columna `recommended_action` del dataset final.

## Comparación entre RFM y K-Means

El Adjusted Rand Index entre ambas segmentaciones fue de `0.455`. La coincidencia es clara en los perfiles de mayor valor y menor actividad, mientras que los clientes intermedios se distribuyen de forma diferente.

Esta diferencia se debe a que K-Means también considera la antigüedad y la diversidad de productos. RFM mantiene una lectura sencilla basada en tres dimensiones y K-Means agrega una perspectiva multivariable.

## Validaciones

El dataset final contiene 5.859 clientes únicos y 19 columnas. No presenta valores faltantes ni clientes duplicados. Todos los clientes tienen scores RFM, un segmento RFM, un cluster, un segmento ML y una acción sugerida.

La implementación completa cuenta con 41 pruebas automatizadas. Siete corresponden específicamente al pipeline de clustering y validan su reproducibilidad, la asignación de perfiles y el tratamiento de entradas inválidas.

## Limitaciones

Los resultados corresponden a un único corte histórico. Los perfiles pueden cambiar cuando se incorporen nuevos datos o se analice otra población de clientes.

Las acciones sugeridas representan criterios iniciales de priorización. Su impacto comercial tendría que medirse posteriormente mediante campañas o experimentos controlados.