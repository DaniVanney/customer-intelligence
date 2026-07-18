# Evaluación de factibilidad de los datos

## Decisión

Ambos datasets son adecuados para el proyecto de portfolio, siempre que los
problemas predictivos se definan respetando las limitaciones de los datos
disponibles.

## Inteligencia de clientes

### Fuente

Online Retail II contiene 1.067.371 líneas de transacciones registradas entre
diciembre de 2009 y diciembre de 2011.

### Evidencia

- 5.942 clientes identificados.
- 5.878 clientes con al menos una compra válida.
- 4.255 clientes con al menos dos facturas de compra.
- Tasa de clientes recurrentes: 72,39%.
- 243.007 filas sin identificador de cliente.
- 19.494 filas correspondientes a cancelaciones.
-  34.335 copias exactas excedentes identificadas.

Provisionalmente, una compra válida se define cuando cumple con un identificador de cliente conocido, una cantidad positiva, un precio positivo y una factura que no esté marcada como cancelación.

Durante el procesamiento se conservará la primera aparición de cada fila exactamente duplicada y se eliminarán las copias excedentes. Los archivosoriginales permanecerán sin modificaciones.

### Etiqueta de inactividad

Un cliente es elegible si realizó una compra durante los 180 días anteriores
a una fecha de corte. El objetivo es predecir inactividad cuando el cliente no
realiza ninguna compra válida durante los 90 días posteriores.

| Fecha de corte | Clientes elegibles | Clientes inactivos | Tasa de inactividad |
|---|---:|---:|---:|
| 2010-09-01 | 2.831 | 1.078 | 38,08% |
| 2010-12-01 | 3.479 | 2.159 | 62,06% |
| 2011-03-01 | 3.349 | 1.933 | 57,72% |
| 2011-06-01 | 2.659 | 1.328 | 49,94% |
| 2011-09-10 | 2.787 | 1.066 | 38,25% |

El volumen y la distribución de las clases son suficientes para desarrollar
customer scoring y un modelo de inactividad con validación temporal.

Este objetivo representa inactividad de compra, no churn contractual.

## Inteligencia de oportunidades

### Fuente

El dataset CRM contiene 8.800 oportunidades comerciales:

- 4.238 ganadas.
- 2.473 perdidas.
- 1.589 en etapa `Engaging`.
- 500 en etapa `Prospecting`.

La población disponible para el entrenamiento supervisado contiene 6.711
oportunidades cerradas, con una tasa de éxito del 63,15%.

### Punto de predicción

El modelo inicial puntuará oportunidades que se encuentren en la etapa
`Engaging`. Las oportunidades en `Prospecting` serán excluidas porque sus 500
registros no contienen una fecha de interacción.

El modelo de probabilidad de éxito utilizará solamente información disponible
antes del cierre. No utilizará `deal_stage`, `close_date` ni `close_value` como
variables predictoras.

### Valor esperado del pipeline

El valor esperado de una oportunidad se calculará como:

`probabilidad de ganar × valor estimado si se gana`

El modelo condicional de valor podrá entrenarse con las 4.238 oportunidades
ganadas que poseen un valor de cierre positivo.

### Evaluación temporal

| Trimestre de cierre | Perdidas | Ganadas |
|---|---:|---:|
| 2017Q1 | 116 | 531 |
| 2017Q2 | 778 | 1.254 |
| 2017Q3 | 790 | 1.257 |
| 2017Q4 | 789 | 1.196 |

La evaluación final utilizará una separación cronológica, no una división
aleatoria de los datos.

### Decisiones de calidad de datos

- Normalizar `GTXPro` como `GTX Pro` durante el procesamiento.
- Conservar todos los archivos originales sin modificaciones.
- Evitar que el modelo principal dependa de atributos de la cuenta, porque
  1.088 oportunidades en `Engaging` no tienen una cuenta identificada.
- Utilizar atributos de productos y equipos comerciales solamente después de
  validar correctamente sus relaciones.
- Considerar esperados los valores y fechas de cierre ausentes en oportunidades
  abiertas, en lugar de tratarlos como errores.

## Alcance final

El alcance aprobado del proyecto es:

1. Customer scoring basado en comportamiento transaccional.
2. Predicción de inactividad de compra durante los próximos 90 días.
3. Predicción de la probabilidad de ganar una oportunidad en `Engaging`.
4. Estimación del valor condicional y del valor esperado del pipeline.

Las dos fuentes representan casos de negocio independientes y no serán unidas
a nivel de registros.