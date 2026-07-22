# Customer scoring RFM

## Objetivo

El customer scoring prioriza clientes según su comportamiento transaccional
hasta el snapshot del `2011-12-10`.

La metodología utiliza:

- recencia;
- frecuencia;
- valor monetario.

El resultado no es un modelo predictivo. Es una clasificación descriptiva y
reproducible que permite orientar acciones comerciales.

## Archivo

`data/processed/customer_scoring.parquet`

Se genera con:

```powershell
python scripts/build_customer_scoring.py

Variables RFM
Recency
recency_days representa los días transcurridos desde la última compra
efectiva.
Una recencia menor recibe un score mayor.
Frequency
order_count representa la cantidad de pedidos efectivos del cliente.
Una frecuencia mayor recibe un score mayor.
Monetary
total_spend representa el gasto efectivo después de reconciliar
cancelaciones identificables.
Un gasto mayor recibe un score mayor.
Asignación de scores
Cada dimensión recibe un score entre 1 y 5 mediante rangos percentiles.
Score	Interpretación relativa
1	Nivel inferior
2	Nivel bajo
3	Nivel medio
4	Nivel alto
5	Nivel superior

Los empates reciben el mismo rango promedio. Por esa razón, los grupos no
necesariamente contienen exactamente 20 % de los clientes.
El score total se calcula como:
rfm_score = recency_score + frequency_score + monetary_score
Su rango posible es de 3 a 15.
Segmentos
Las reglas se evalúan en el orden presentado. Una vez asignado un segmento,
las reglas posteriores no modifican al cliente.
Prioridad	Segmento	Regla principal
1	Valor estratégico	R >= 4, F >= 4 y M >= 4
2	Incorporación reciente	R >= 4 y F <= 2
3	Lealtad consolidada	R >= 3 y F >= 4
4	Alto valor	R >= 3 y M >= 4
5	Potencial de desarrollo	R >= 4
6	Riesgo de inactividad	R <= 2 y F >= 3 o M >= 3
7	Inactividad prolongada	R <= 2
8	Base general	Casos restantes

Resultados
La tabla contiene 5.859 clientes y ocho segmentos.
Segmento	Clientes	Clientes (%)	Gasto (%)
Valor estratégico	1.262	21,54	68,94
Lealtad consolidada	604	10,31	10,39
Riesgo de inactividad	985	16,81	10,36
Alto valor	273	4,66	3,68
Inactividad prolongada	1.359	23,20	2,08
Base general	566	9,66	1,69
Incorporación reciente	506	8,64	1,57
Potencial de desarrollo	304	5,19	1,29

El segmento Valor estratégico concentra 68,94 % del gasto con 21,54 % de los
clientes.
El segmento Riesgo de inactividad conserva 10,36 % del gasto histórico, por
lo que representa la principal oportunidad de recuperación.
Inactividad prolongada contiene 23,20 % de los clientes, pero solamente
2,08 % del gasto.
Acciones recomendadas
Segmento	Acción principal
Valor estratégico	Retención personalizada, beneficios y acceso prioritario
Incorporación reciente	Onboarding y estímulo de una segunda compra
Lealtad consolidada	Fidelización, venta cruzada y reconocimiento
Alto valor	Protección de valor y ofertas personalizadas
Potencial de desarrollo	Incrementar frecuencia y variedad de productos
Riesgo de inactividad	Campañas de recuperación priorizadas por valor
Inactividad prolongada	Reactivación automatizada y de bajo costo
Base general	Comunicación regular y seguimiento de evolución

Validaciones
El dataset final cumple:
5.859 clientes únicos;
16 columnas;
cero valores faltantes;
scores individuales entre 1 y 5;
score total consistente con la suma de R, F y M;
ocho segmentos representados.
Limitaciones
Los scores son relativos a la población y al snapshot. Si cambia la población
o se actualizan las transacciones, también pueden cambiar los percentiles.
Los segmentos representan reglas comerciales, no efectos causales. Las
acciones recomendadas deberían validarse mediante experimentos o campañas
controladas antes de afirmar que producen un incremento de ventas.