# Predicción de inactividad de clientes

## Objetivo

El modelo estima el riesgo de que un cliente deje de comprar durante los próximos 90 días. Su finalidad es ordenar la cartera, identificar clientes críticos y facilitar acciones de retención.

Online Retail II no contiene bajas contractuales. En este proyecto, el churn se representa mediante inactividad de compra durante un período definido.

## Archivos

El dataset temporal se encuentra en `data/processed/inactivity_modeling.parquet` y se genera con:

```powershell
python scripts/build_inactivity_dataset.py
```

El análisis del modelo está desarrollado en `notebooks/02_inactivity_prediction.ipynb`. La implementación reutilizable se encuentra en `src/customer_intelligence/churn/model.py`.

La tabla final de alertas se genera con:

```powershell
python scripts/build_inactivity_risk.py
```

El resultado se guarda en `data/processed/customer_inactivity_risk.parquet`.

## Definición del objetivo

Un cliente es elegible cuando realizó al menos una compra efectiva durante los 180 días anteriores a una fecha de corte.

`is_inactive` toma el valor verdadero cuando ese cliente no realiza ninguna compra efectiva durante los 90 días posteriores. Las cancelaciones identificables se reconcilian antes de construir las variables y la etiqueta.

Los clientes con más de 180 días sin comprar quedan fuera del modelo predictivo porque su inactividad ya es observable. Esos perfiles permanecen disponibles en la segmentación de customer scoring.

## Dataset temporal

Se construyeron cinco snapshots históricos. Cada fila representa el comportamiento de un cliente en una fecha determinada y utiliza solamente información disponible antes de esa fecha.

| Snapshot | Clientes | Inactivos | Tasa de inactividad |
|---|---:|---:|---:|
| 2010-09-01 | 2.824 | 1.073 | 38,00% |
| 2010-12-01 | 3.466 | 2.151 | 62,06% |
| 2011-03-01 | 3.337 | 1.924 | 57,66% |
| 2011-06-01 | 2.655 | 1.326 | 49,94% |
| 2011-09-10 | 2.781 | 1.062 | 38,19% |

El dataset contiene 15.063 observaciones y 5.018 clientes distintos. Un cliente puede aparecer en varios snapshots porque su comportamiento cambia con el tiempo.

## Variables del modelo

El modelo utiliza recencia, antigüedad, cantidad de pedidos, gasto histórico, variedad de productos, valor promedio del pedido y actividad durante los últimos 30 y 90 días.

Las variables con distribuciones muy asimétricas reciben una transformación `log1p`. Después, todas las variables son estandarizadas dentro de un pipeline de Scikit-Learn.

`total_items` e `items_last_90d` se conservaron para análisis de perfiles, pero quedaron fuera del entrenamiento por su fuerte relación con otras variables monetarias y de actividad.

## Validación temporal

Los primeros tres snapshots se utilizaron para entrenamiento, el snapshot del `2011-06-01` para validación y el del `2011-09-10` como prueba final.

También se realizaron validaciones móviles utilizando distintos cortes históricos. Esta estrategia reproduce mejor el uso futuro del modelo que una separación aleatoria.

Se compararon un baseline, una regresión logística y un Random Forest. La regresión logística fue seleccionada por su equilibrio entre rendimiento, estabilidad e interpretación.

| Modelo | F1 medio | ROC AUC medio | Average Precision media | Brier medio |
|---|---:|---:|---:|---:|
| Regresión logística | 0,637 | 0,774 | 0,789 | 0,221 |
| Random Forest | 0,636 | 0,759 | 0,770 | 0,224 |
| Baseline | 0,466 | 0,500 | 0,566 | 0,264 |

## Resultado final

El umbral de alerta se seleccionó en validación exigiendo un recall mínimo del 80%. El valor elegido fue `0.45`.

| Métrica de prueba | Resultado |
|---|---:|
| Accuracy | 0,646 |
| Precision | 0,523 |
| Recall | 0,841 |
| F1 | 0,645 |
| ROC AUC | 0,759 |
| Average Precision | 0,638 |
| Brier score | 0,206 |

El modelo detectó 893 de los 1.062 clientes inactivos del período de prueba. Generó 816 falsas alertas y dejó sin detectar 169 casos.

Frente a una regresión que utiliza solamente recencia, el modelo completo mejoró ROC AUC de `0.655` a `0.759` y redujo la población alertada de `84,43%` a `61,45%`.

## Capacidad de priorización

| Clientes revisados | Inactivos encontrados | Precisión | Captura de inactivos | Lift |
|---|---:|---:|---:|---:|
| Top 10% | 205 | 73,48% | 19,30% | 1,92 |
| Top 20% | 387 | 69,48% | 36,44% | 1,82 |
| Top 30% | 526 | 62,99% | 49,53% | 1,65 |

La cantidad histórica de pedidos fue la variable con mayor importancia predictiva. La antigüedad del cliente y la actividad durante los últimos 90 días aportaron información adicional.

## Niveles de riesgo

Los scores se agrupan en cuatro niveles:

| Nivel | Rango | Uso sugerido |
|---|---:|---|
| Bajo | Menor que 0,30 | Seguimiento regular |
| Moderado | 0,30 a 0,45 | Monitorear la evolución de actividad |
| Alto | 0,45 a 0,65 | Contactar con una acción de retención |
| Crítico | 0,65 o más | Intervención prioritaria de retención |

Los niveles alto y crítico generan una alerta.

## Snapshot operativo

El modelo final se entrenó con los cinco snapshots etiquetados y puntuó a los clientes elegibles al `2011-12-10`.

| Nivel | Clientes | Participación | Alertas | Recencia mediana | Pedidos medianos |
|---|---:|---:|---:|---:|---:|
| Crítico | 878 | 25,34% | 878 | 70 | 2 |
| Alto | 950 | 27,42% | 950 | 46 | 4 |
| Moderado | 646 | 18,64% | 0 | 30 | 7 |
| Bajo | 991 | 28,60% | 0 | 13 | 14 |

Se puntuaron 3.465 clientes y se generaron 1.828 alertas. La columna `risk_rank` permite ordenar la cartera desde el mayor hasta el menor riesgo.

## Validaciones

La tabla operativa contiene un cliente por fila, no presenta valores faltantes y conserva únicamente clientes elegibles. Todos los scores se encuentran entre 0 y 1, las alertas respetan el umbral definido y cada cliente posee un nivel de riesgo y una acción sugerida.

El proyecto cuenta con 51 pruebas automatizadas. Diez corresponden al dataset temporal y al modelo de inactividad.

## Limitaciones

La tasa de inactividad cambió entre los períodos históricos. En validación, el riesgo promedio estimado fue de 49,08% frente a una tasa observada de 49,94%. En prueba, el riesgo estimado fue de 49,87% y la tasa observada de 38,19%.

Esta diferencia indica que los valores requieren seguimiento y recalibración cuando se incorporen datos recientes. La salida se utiliza como score de priorización y sus niveles permiten organizar la capacidad comercial disponible.

Los resultados pertenecen a un dataset histórico de comercio minorista. Las acciones de retención deberán evaluarse posteriormente mediante campañas o experimentos controlados.