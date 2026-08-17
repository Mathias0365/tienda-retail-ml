# Modelado de Datos y Analytics para Retail — Grupo 5

## Descripción

Proyecto final de análisis y predicción de la demanda para una cadena de tiendas
de consumo masivo. A partir de un dataset de 76,000 registros de venta se aplica
todo el ciclo de modelado de datos:

EDA (distribución, precios vs competencia, correlaciones, estacionalidad) →
limpieza y tratamiento (outliers, log1p, escalado) → (One-Hot) → reducción de dimensionalidad (PCA) → segmentación (K-Means + jerárquico)
→ modelos predictivos (regresión lineal y clasificación Alta/Baja) → selección
del mejor modelo → recomendaciones de negocio y fase piloto.

Incluye un **dashboard interactivo** (Plotly Dash) que permite explorar los
hallazgos, simular escenarios y consultar la validación del modelo.

## Integrantes (Grupo 5)

- Cabanillas Yovera Roger Arnold
- Salazar Cristóbal Jonathan Walter
- Solano Torres Anthony Dimas
- Quispe Mendoza Josué Isaías
- Candela Ortiz Sebastian Matias



## Requisitos e instalación

Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

Dependencias principales: `dash`, `dash-bootstrap-components`, `plotly`,
`pandas`, `numpy`, `scikit-learn`.

## Cómo ejecutar el dashboard

```bash
python app_retail.py
```

Luego abre en el navegador: **http://127.0.0.1:8050**

> Nota: la primera carga tarda unos 15 segundos (la app entrena los modelos al
> abrir). Después, la interfaz responde al instante.

## Explicación del dashboard 

El dashboard está organizado en 6 pestañas que cuentan la historia completa:
del estado de los datos a la recomendación de negocio.

### 1. El Contexto
Punto de partida. Cuatro KPIs (registros, demanda promedio, unidades vendidas
y precio promedio frente a la competencia), una franja de **calidad de datos**
(0% de nulos, 16 variables, outliers conservados) y tres visuales: qué categoría
mueve más unidades, cómo se comporta la demanda por región y la **demanda por mes**
(estacionalidad: pico en agosto ~120 uds, valle en febrero ~72 uds).

### 2. Los Hallazgos
Las anomalías del negocio: la relación precio propio vs competencia (r = 0.98),
el efecto combinado de promoción y epidemia (+28 uds vs −43 uds) y el impacto
de la epidemia por categoría.

### 3. Simulador (IA)
El corazón práctico del dashboard. Configuras un escenario (categoría, región,
tienda, clima, estacionalidad, precio, descuento, inventario, orden, precio de
la competencia, promoción y epidemia) y la app responde en lenguaje de negocio:
unidades esperadas, nivel (Baja/Media/Alta), una lectura de qué significa y la
cantidad sugerida a ordenar (demanda + 10% de margen). La app abre con un
escenario de ejemplo listo para la demostración (Groceries · 158 uds · MEDIA ·
orden ≈ 170).

### 4. Modelos
Evidencia de cómo se eligió la mejor opción, presentada en lenguaje de negocio:
comparación de **aciertos y fiabilidad (de 100)** entre la opción estándar
(76/86) y la opción reforzada (85/93), la tabla de qué hace cada opción y su
resultado (explica el 56% / acierta 76 / acierta 85 de cada 100) y los
**4 criterios de selección** (prueba a libro cerrado, explicable, confiable y
que responda la pregunta del negocio).

### 5. Detalles técnicos
¿Cuánto confiar en los números?: confiabilidad de la predicción (explica el 56%
de por qué cambia la demanda, sin mirar las ventas ya hechas), error promedio
±24 uds, el gráfico real vs esperado, qué influye más en la demanda
(coeficientes: la promoción y alimentación suben; muebles y epidemia restan) y
la importancia de variables de la opción reforzada (top 10).

### 6. Conclusiones y siguientes pasos
El cierre: 4 conclusiones clave del análisis, las acciones recomendadas para el
negocio y el plan de fase piloto (implementación en 2 tiendas, KPIs de éxito y
retroalimentación del modelo).

## Resultados principales del modelo

| Regresión Lineal | Predecir demanda (unidades) | R² / RMSE / MAE | 0.56 / 31.1 / 23.9 |
| Regresión Logística | Clasificar Alta vs Baja | Accuracy / AUC | 76.2% / 0.856 |
| Random Forest | Clasificar Alta vs Baja | Accuracy / AUC | 84.8% / 0.932 |

- **Segmentación:** K-Means con k=8 (validado por silhouette 0.268 y jerárquico).
- **Hallazgos clave:** promoción suma ≈ +28 uds; epidemia resta ≈ −43 uds;
  el 62% de los registros no tiene orden de reposición incluso con demanda alta.


