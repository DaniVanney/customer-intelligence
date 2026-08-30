# Forecast de oportunidades comerciales

## Objetivo

Este módulo estima la probabilidad de conversión y el valor esperado de las oportunidades comerciales que se encuentran en etapa de negociación. El resultado permite ordenar el pipeline según su aporte económico esperado y obtener una proyección agregada de ventas.

El análisis utiliza el historial de oportunidades cerradas para evaluar el comportamiento del modelo y luego genera el forecast sobre las oportunidades abiertas.

## Datos utilizados

El dataset de modelado contiene 6.711 oportunidades cerradas: 4.238 ganadas y 2.473 perdidas. La tasa histórica de conversión es de 63,15%.

Las oportunidades se separaron de forma temporal:

| Periodo | Uso | Oportunidades |
| --- | --- | ---: |
| 2017 Q1-Q2 | Entrenamiento | 2.679 |
| 2017 Q3 | Validación y calibración | 2.047 |
| 2017 Q4 | Prueba final | 1.985 |

Esta separación respeta el orden cronológico y permite evaluar el modelo sobre un periodo posterior al utilizado durante su desarrollo.

Las variables disponibles antes del cierre son el producto, el agente comercial, la oficina regional, el mes de incorporación al pipeline y el precio de venta. Los datos asociados al resultado final, como la etapa de cierre, la fecha de cierre y el valor cerrado, se excluyen de los predictores.

## Desarrollo del modelo

Se evaluaron una regresión logística y un Random Forest para estimar la probabilidad de venta. Ambos modelos mostraron una capacidad limitada para distinguir oportunidades ganadas y perdidas de manera individual.

La regresión logística obtuvo mejores resultados generales y ofrece una interpretación más sencilla. Sus probabilidades se calibraron con un periodo posterior al entrenamiento para ajustar la tasa pronosticada a la conversión observada.

En el trimestre de prueba se obtuvieron los siguientes resultados:

| Métrica | Resultado |
| --- | ---: |
| ROC AUC | 0,544 |
| Average Precision | 0,655 |
| Brier Score | 0,240 |
| Tasa de conversión estimada | 61,43% |
| Tasa de conversión real | 60,25% |

La capacidad de clasificación individual es moderada. Sin embargo, el modelo mejora el ordenamiento comercial: el 10% de oportunidades con mayor probabilidad alcanzó una conversión de 71,86% y un lift de 1,19 frente al promedio del trimestre.

## Estimación del valor esperado

También se evaluaron modelos de regresión para estimar el valor de una oportunidad ganada. El precio de venta disponible en el CRM ya explica la mayor parte de ese valor, por lo que el Random Forest no produjo una mejora material en el forecast final.

La estimación seleccionada utiliza la siguiente fórmula:

`valor esperado = probabilidad de conversión × precio de venta`

En el trimestre de prueba, esta metodología proyectó un pipeline de 2.872.809,47 frente a un resultado real de 2.802.496,00. El error agregado fue de 2,51%.

## Forecast operativo

El modelo final se entrenó con las oportunidades cerradas hasta el tercer trimestre de 2017 y se calibró con los resultados del cuarto trimestre. Luego se aplicó sobre las 1.589 oportunidades que permanecían en etapa `Engaging`.

El forecast al 31 de diciembre de 2017 produjo:

| Indicador | Resultado |
| --- | ---: |
| Oportunidades evaluadas | 1.589 |
| Ventas esperadas | 949,7 |
| Tasa de conversión esperada | 59,77% |
| Pipeline esperado | 2.313.461,50 |

Las oportunidades se ordenan mediante `forecast_rank` y se agrupan en cuatro niveles de prioridad según su valor esperado. El 30% con mayor prioridad concentra aproximadamente el 69% del pipeline proyectado.

El resultado se guarda en:

`data/processed/opportunity_forecast.parquet`

Este dataset contiene la probabilidad de conversión, el valor esperado, el ranking comercial y la prioridad asignada a cada oportunidad.

## Limitaciones

La información disponible explica bien el valor económico de los productos, pero contiene poca señal para anticipar qué oportunidad concreta se convertirá en venta. Las probabilidades deben interpretarse como una estimación de riesgo y priorización, especialmente útil al analizar grupos de oportunidades y el pipeline agregado.

La evaluación temporal permite mostrar esta limitación de forma transparente y evita presentar una precisión mayor a la que permiten los datos.
