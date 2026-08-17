from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score

#### DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'TIENDA RETAIL.csv'), sep=';')
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')

#### MODELO (mismo enfoque que PA1.ipynb)
cat_cols = ['Categoria', 'Region', 'Condiciones Climaticas', 'Estacionalidad', 'Store ID']
encoder = OneHotEncoder(drop='first', sparse_output=False)
cat_encoded = encoder.fit_transform(df[cat_cols])
cat_names = encoder.get_feature_names_out(cat_cols)

# Sin 'Unidad Vendida': es la demanda realizada y generaría fuga de datos
num_cols = ['Inventory Level', 'Unidades Ordenadas',
            'Precio', 'Descuento', 'Promoción',
            'Precios de la competencia', 'Epidemic']

X = np.hstack([df[num_cols].values, cat_encoded])
y = df['Demanda'].values
all_cols = list(num_cols) + list(cat_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model_global = LinearRegression().fit(X_train, y_train)
y_pred_test = model_global.predict(X_test)
score_r2 = r2_score(y_test, y_pred_test)

# CLASIFICACIÓN: Alta vs Baja demanda (mismo enfoque que PA1.ipynb)
mediana_demanda = float(np.median(df['Demanda']))
y_bin = (df['Demanda'].values > mediana_demanda).astype(int)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X, y_bin, test_size=0.3, random_state=101, stratify=y_bin)
scaler_clf = StandardScaler()
Xb_train_s = scaler_clf.fit_transform(Xb_train)
Xb_test_s = scaler_clf.transform(Xb_test)

modelo_log = LogisticRegression(random_state=101).fit(Xb_train_s, yb_train)
acc_log = accuracy_score(yb_test, modelo_log.predict(Xb_test_s))
auc_log = roc_auc_score(yb_test, modelo_log.predict_proba(Xb_test_s)[:, 1])

modelo_rf = RandomForestClassifier(n_estimators=200, random_state=101).fit(Xb_train, yb_train)
acc_rf = accuracy_score(yb_test, modelo_rf.predict(Xb_test))
auc_rf = roc_auc_score(yb_test, modelo_rf.predict_proba(Xb_test)[:, 1])

imp_rf = pd.DataFrame({'Variable': all_cols, 'Importancia': modelo_rf.feature_importances_})
imp_rf = imp_rf.sort_values('Importancia', ascending=False).head(10).sort_values('Importancia')

DEMANDA_MIN = int(df['Demanda'].min())
DEMANDA_MAX = int(df['Demanda'].max())

def encode_row(categoria, region, clima, estacion, store):
    """Convierte las selecciones categóricas en el bloque one-hot (drop='first')."""
    vals = [categoria, region, clima, estacion, store]
    out = []
    for cats, v in zip(encoder.categories_, vals):
        if v == cats[0]:
            out.extend([0.0] * (len(cats) - 1))
        else:
            vec = [0.0] * (len(cats) - 1)
            vec[list(cats[1:]).index(v)] = 1.0
            out.extend(vec)
    return out

#### GRÁFICOS ESTÁTICOS (Capítulos 1, 2 y 4)

# CAP 1: KPIs
total_registros = len(df)
demanda_prom = round(df['Demanda'].mean(), 1)
vendidas_total = int(df['Unidad Vendida'].sum())
precio_prom = round(df['Precio'].mean(), 2)
precio_comp_prom = round(df['Precios de la competencia'].mean(), 2)
diff_precio = round((precio_prom - precio_comp_prom) / precio_comp_prom * 100, 1)

# CAP 1: Demanda por categoría
df_cat = df.groupby('Categoria')['Demanda'].mean().sort_values().reset_index()
cat_top = df_cat.loc[df_cat['Demanda'].idxmax()]
fig_cat = px.bar(df_cat, x='Demanda', y='Categoria', orientation='h',
                 color='Categoria', color_discrete_sequence=px.colors.qualitative.Pastel,
                 title=f"¿Qué mueve más unidades? {cat_top['Categoria']} lidera (~{cat_top['Demanda']:.0f} uds)",
                 labels={'Demanda': 'Demanda promedio'})
fig_cat.update_layout(showlegend=False, height=320)

# CAP 1: Demanda por región
df_reg = df.groupby('Region')['Demanda'].mean().reset_index()
reg_top = df_reg.loc[df_reg['Demanda'].idxmax()]
fig_reg = px.bar(df_reg, x='Region', y='Demanda', color='Region',
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title=f"Demanda por región: {reg_top['Region']} es la más fuerte (~{reg_top['Demanda']:.0f} uds)")
fig_reg.update_layout(showlegend=False, height=320)

# CAP 1: Demanda por mes (serie temporal)
df_mes = df.groupby(df['Fecha'].dt.to_period('M'))['Demanda'].mean().reset_index()
df_mes['Mes'] = df_mes['Fecha'].astype(str)
pico_mes = df_mes.loc[df_mes['Demanda'].idxmax()]
fig_tiempo = px.line(df_mes, x='Mes', y='Demanda', markers=True,
                     title=f"La venta no es pareja en el tiempo: el pico fue {pico_mes['Mes']} "
                           f"(~{pico_mes['Demanda']:.0f} uds)")
fig_tiempo.update_layout(height=320, showlegend=False)

# CAP 2: Precio vs competencia
muestra = df.sample(min(5000, len(df)), random_state=42)
corr_precio = df['Precio'].corr(df['Precios de la competencia'])
fig_pc = px.scatter(muestra, x='Precios de la competencia', y='Precio',
                    color='Categoria', opacity=0.6,
                    title=f"Seguimos a la competencia (r = {corr_precio:.2f})",
                    labels={'Precios de la competencia': 'Precio competencia', 'Precio': 'Precio retail'})
fig_pc.update_layout(height=400)

# CAP 2: Promoción + Epidemia
promo_eff = df.groupby(['Promoción', 'Epidemic'])['Demanda'].mean().reset_index()
promo_eff['Promoción'] = promo_eff['Promoción'].map({0: 'Sin promoción', 1: 'Con promoción'})
promo_eff['Epidemic'] = promo_eff['Epidemic'].map({0: 'Sin epidemia', 1: 'Con epidemia'})
fig_pe = px.bar(promo_eff, x='Promoción', y='Demanda', color='Epidemic', barmode='group',
                title='Promoción y epidemia: ¿aliados o enemigos?',
                color_discrete_map={'Sin epidemia': '#4c78a8', 'Con epidemia': '#e45756'})
fig_pe.update_layout(height=400)

# CAP 2: Efecto de la epidemia por categoría (% de cambio)
epi_cat = df.groupby(['Categoria', 'Epidemic'])['Demanda'].mean().unstack()
epi_cat['Cambio'] = ((epi_cat[1] - epi_cat[0]) / epi_cat[0] * 100).round(1)
epi_cat = epi_cat.reset_index().sort_values('Cambio')
fig_epi = px.bar(epi_cat, x='Cambio', y='Categoria', orientation='h',
                 color='Cambio',
                 color_continuous_scale=['#e45756', '#f4a582', '#72b7b2'],
                 range_color=[-80, 10],
                 title='Impacto de la epidemia en la demanda por categoría (%)')
fig_epi.update_layout(showlegend=False, height=400)

# CAP 4: Validación (Real vs Predicho)
fig_validation = px.scatter(x=y_test, y=y_pred_test,
                            labels={'x': 'Demanda real', 'y': 'Demanda esperada'},
                            title="Qué tan cerca está la predicción de la realidad",
                            opacity=0.4, color_discrete_sequence=['#4c78a8'])
fig_validation.add_shape(type='line', x0=y_test.min(), y0=y_test.min(),
                         x1=y_test.max(), y1=y_test.max(),
                         line=dict(color='red', dash='dash'))
fig_validation.update_layout(height=400)

# CAP 4: Coeficientes del modelo
cdf = pd.DataFrame(model_global.coef_, all_cols, columns=['Coeff'])
cdf = cdf.sort_values('Coeff', key=lambda s: s.abs())
cdf_top = cdf.tail(15)
fig_coef = px.bar(cdf_top, x='Coeff', y=cdf_top.index, orientation='h',
                  color='Coeff', color_continuous_scale='RdYlGn',
                  title='Qué influye más en la demanda')
fig_coef.update_layout(showlegend=False, height=500)

# CAP 4: Comparación de clasificadores (en lenguaje de negocio: aciertos y fiabilidad de 100)
fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(name='Opción estándar', x=['Aciertos (de 100)', 'Fiabilidad (de 100)'],
                          y=[acc_log * 100, auc_log * 100],
                          text=[f'{acc_log:.0%}', f'{auc_log:.0%}'],
                          textposition='outside', marker_color='#4c78a8'))
fig_comp.add_trace(go.Bar(name='Opción reforzada', x=['Aciertos (de 100)', 'Fiabilidad (de 100)'],
                          y=[acc_rf * 100, auc_rf * 100],
                          text=[f'{acc_rf:.0%}', f'{auc_rf:.0%}'],
                          textposition='outside', marker_color='#72b7b2'))
fig_comp.update_layout(barmode='group', height=320,
                       title='¿Qué opción anticipa mejor la demanda?',
                       yaxis_range=[0, 100], yaxis_tickformat='.0f')

# CAP 5: Importancia de variables (Random Forest, top 10)
fig_imp = px.bar(imp_rf, x='Importancia', y='Variable', orientation='h',
                 color='Importancia', color_continuous_scale='Blues',
                 title='Qué influye más en la demanda')
fig_imp.update_layout(showlegend=False, height=420)

#### DEFINIR APP
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "TIENDA RETAIL: Análisis y Predicción de la Demanda"
server = app.server

#### LAYOUT (Narrativa en capítulos)
app.layout = dbc.Container(fluid=True, children=[

    dbc.Row([
        dbc.Col(html.H1("TIENDA RETAIL: Análisis y Predicción de la Demanda",
                        className="text-center text-primary my-4"), width=12),
        dbc.Col(html.P("76,000 registros. 5 tiendas. Una pregunta: ¿qué impulsa la demanda?",
                       className="text-center text-muted lead"), width=12),
    ]),
    html.Hr(),

    dbc.Tabs([

        # ---- CAPÍTULO 1: EL CONTEXTO ----
        dbc.Tab(label="1. El Contexto", children=[
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Registros", className="text-muted"),
                    html.H3(f"{total_registros:,}".replace(",", "."), className="text-primary"),
                    html.P("5 tiendas · 25 meses", className="text-muted mb-0", style={"fontSize": "12px"}),
                ]), className="text-center shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Demanda promedio", className="text-muted"),
                    html.H3(f"{demanda_prom:,.1f}".replace(",", "."), className="text-primary"),
                    html.P("≈ unidades por producto", className="text-muted mb-0", style={"fontSize": "12px"}),
                ]), className="text-center shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Unidades vendidas", className="text-muted"),
                    html.H3(f"{vendidas_total:,}".replace(",", "."), className="text-primary"),
                    html.P("volumen total 2022–2024", className="text-muted mb-0", style={"fontSize": "12px"}),
                ]), className="text-center shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Precio promedio", className="text-muted"),
                    html.H3(f"${precio_prom:,.2f}".replace(",", "."), className="text-primary"),
                    html.P(f"{abs(diff_precio):.0f}% {'bajo' if diff_precio < 0 else 'sobre'} la competencia",
                           className="text-muted mb-0", style={"fontSize": "12px"}),
                ]), className="text-center shadow-sm"), md=3),
            ], className="mt-4"),

            dbc.Alert(
                "Calidad de datos: 0% de valores nulos · 16 variables (9 numéricas y 7 categóricas) · "
                "outliers de precio conservados (productos premium).",
                color="info", className="mt-2"),

            dbc.Row([
                dbc.Col([
                    html.H4("¿Dónde está el negocio?", className="mt-3"),
                    html.P("Antes de hablar de modelos, entendamos el terreno. La demanda no es homogénea: "
                           "unas categorías venden mucho más que otras y las regiones no se comportan igual."),
                    dbc.Alert("Conclusión: identificar los productos y zonas que mueven el negocio es el punto de partida.",
                              color="primary"),
                ], md=4, className="d-flex flex-column justify-content-center"),
                dbc.Col(dcc.Graph(figure=fig_cat, style={'height': '360px'}), md=4),
                dbc.Col(dcc.Graph(figure=fig_reg, style={'height': '360px'}), md=4),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    html.H4("La venta no es pareja en el tiempo", className="mt-3"),
                    html.P("Hay meses que venden mucho más que otros. Conocer el ritmo del año "
                           "permite anticipar cuándo surtir más y cuándo relajar el inventario."),
                ], md=4, className="d-flex flex-column justify-content-center"),
                dbc.Col(dcc.Graph(figure=fig_tiempo), md=8),
            ], className="mt-4"),
        ]),

        # ---- CAPÍTULO 2: LOS HALLAZGOS (Datos atípicos / contradicciones) ----
        dbc.Tab(label="2. Los Hallazgos", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Tres anomalías que cambian el plan", className="mt-3"),
                    html.P("Los números cuentan algo distinto a lo que el negocio espera: los precios no "
                           "marcan rumbo propio, la promoción depende del contexto y la epidemia no "
                           "golpea parejo."),
                    dbc.Alert("Conclusión: la epidemia devastó la demanda en 4 de 5 categorías, "
                              "pero Clothing creció +1.25%.",
                              color="danger"),
                ], md=4, className="d-flex flex-column justify-content-center"),
                dbc.Col([
                    html.H5("1 · Precios: sin estrategia propia (r ≈ 0.98)", className="mt-3"),
                    dcc.Graph(figure=fig_pc),
                ], md=8),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    html.H5("2 · Promoción vs epidemia: suman +28, restan −43", className="mt-3"),
                    dcc.Graph(figure=fig_pe),
                ], md=6),
                dbc.Col([
                    html.H5("3 · La epidemia no es pareja", className="mt-3"),
                    dcc.Graph(figure=fig_epi),
                ], md=6),
            ], className="mt-4"),
        ]),

        # ---- CAPÍTULO 3: SIMULADOR (IA) ----
        dbc.Tab(label="3. Simulador (IA)", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Configura el escenario", className="m-0 text-white text-center"),
                                       className="bg-primary"),
                        dbc.CardBody([
                            html.Label("1. Categoría"),
                            dcc.Dropdown(id='sim-categoria',
                                         options=[{"label": c, "value": c} for c in sorted(df['Categoria'].unique())],
                                         value='Groceries'),
                            html.Label("2. Región", className="mt-3"),
                            dcc.Dropdown(id='sim-region',
                                         options=[{"label": r, "value": r} for r in sorted(df['Region'].unique())],
                                         value='East'),
                            html.Label("3. Tienda", className="mt-3"),
                            dcc.Dropdown(id='sim-store',
                                         options=[{"label": s, "value": s} for s in sorted(df['Store ID'].unique())],
                                         value='S001'),
                            html.Label("4. Condiciones climáticas", className="mt-3"),
                            dcc.Dropdown(id='sim-clima',
                                         options=[{"label": c, "value": c} for c in sorted(df['Condiciones Climaticas'].unique())],
                                         value='Sunny'),
                            html.Label("5. Estacionalidad", className="mt-3"),
                            dcc.Dropdown(id='sim-estacion',
                                         options=[{"label": e, "value": e} for e in sorted(df['Estacionalidad'].unique())],
                                         value='Spring'),

                            html.Label("6. Precio ($)", className="mt-3"),
                            dcc.Slider(id='sim-precio', min=5, max=230, step=1, value=60,
                                       marks={5: '5', 115: '115', 230: '230'}),
                            html.Label("7. Descuento (%)", className="mt-3"),
                            dcc.Slider(id='sim-descuento', min=0, max=25, step=1, value=15,
                                       marks={0: '0', 25: '25'}),
                            html.Label("8. Inventario disponible", className="mt-3"),
                            dcc.Slider(id='sim-inventario', min=0, max=2300, step=50, value=500,
                                       marks={0: '0', 1150: '1150', 2300: '2300'}),
                            html.Label("9. Unidades ordenadas", className="mt-3"),
                            dcc.Slider(id='sim-ordenadas', min=0, max=1600, step=50, value=150,
                                       marks={0: '0', 800: '800', 1600: '1600'}),
                            html.Label("10. Precio de la competencia ($)", className="mt-3"),
                            dcc.Slider(id='sim-competencia', min=5, max=260, step=1, value=70,
                                       marks={5: '5', 130: '130', 260: '260'}),

                            dbc.Row([
                                dbc.Col(html.Label("Promoción")),
                                dbc.Col(html.Label("Epidemia")),
                            ], className="mt-3"),
                            dbc.Row([
                                dbc.Col(dcc.RadioItems(id='sim-promo', inline=True,
                                                        options=[{"label": "No", "value": 0}, {"label": "Sí", "value": 1}],
                                                        value=1)),
                                dbc.Col(dcc.RadioItems(id='sim-epi', inline=True,
                                                        options=[{"label": "No", "value": 0}, {"label": "Sí", "value": 1}],
                                                        value=0)),
                            ]),

                            dbc.Button("Predecir demanda", id='btn-predict', color="success",
                                       className="w-100 mt-4"),
                        ])
                    ], className="shadow")
                ], md=4),
                dbc.Col(html.Div(id='sim-output'), md=8),
            ], className="mt-4")
        ]),

        # ---- CAPÍTULO 4: MODELOS (Comparación y criterios) ----
        dbc.Tab(label="4. Modelos", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Cómo llegamos a las conclusiones", className="mt-3"),
                    html.P("Probamos tres maneras de analizar el historial y las comparamos con datos "
                           "que la herramienta nunca había visto (una prueba a libro cerrado), para "
                           "asegurar que el resultado se repite en la vida real."),
                    dbc.Alert("Para decidir cuánto ordenar usamos la opción más precisa; para explicar "
                              "qué mueve la venta, la más transparente.",
                              color="success"),
                ], md=5),
                dbc.Col(dcc.Graph(figure=fig_comp), md=7),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col(dbc.Table([
                    html.Thead(html.Tr([html.Th("Qué hace"), html.Th("Cómo"),
                                        html.Th("Resultado")])),
                    html.Tbody([
                        html.Tr([html.Td("Explica qué pesa en la venta"), html.Td("Análisis transparente"),
                                 html.Td("Explica el 56% de los cambios")]),
                        html.Tr([html.Td("Anticipa si la demanda será alta o baja"), html.Td("Regresión Logística"),
                                 html.Td("Acierta 76 de cada 100")]),
                        html.Tr([html.Td("Anticipa si la demanda será alta o baja"), html.Td("Random Forest (opción reforzada)"),
                                 html.Td("Acierta 85 de cada 100")]),
                    ]),
                ], bordered=True, hover=True, striped=True, size="sm", className="mt-4"), md=7),
                dbc.Col([
                    html.H5("Por qué elegimos así", className="mt-4"),
                    html.P("1. Probado con datos que nunca había visto (examen a libro cerrado).", className="mb-1"),
                    html.P("2. Que se pueda explicar al negocio.", className="mb-1"),
                    html.P("3. Que sea confiable sin complicar la operación.", className="mb-1"),
                    html.P("4. Que responda la pregunta: ¿cuánto voy a vender y cuándo preocuparme?", className="mb-1"),
                    dbc.Alert("Elegimos la opción más confiable y que se pueda explicar.",
                              color="info", className="mt-2"),
                ], md=5),
            ], className="mt-4"),
        ]),

        # ---- CAPÍTULO 5: DETALLES TÉCNICOS ----
        dbc.Tab(label="5. Detalles técnicos", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("¿Cuánto podemos confiar en los números?", className="mt-3"),
                    html.P("Le pedimos al sistema que anticipara ventas que nunca había visto. La línea roja "
                           "es la predicción perfecta: mientras más cerca estén los puntos, mejor."),
                    dbc.Alert(f"Confiabilidad: la predicción explica el {score_r2:.0%} de por qué cambia la demanda. "
                              "Es un resultado honesto: el sistema no miró las ventas ya hechas.",
                              color="success"),
                ], md=4, className="d-flex flex-column justify-content-center"),
                dbc.Col(dcc.Graph(figure=fig_validation), md=8),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    html.H5("Qué mueve la venta", className="mt-3"),
                    html.P("Cada barra muestra cuánto influye cada factor en la demanda: hacia un lado, "
                           "más venta; hacia el otro, menos."),
                    dcc.Graph(figure=fig_coef),
                ], md=12, className="mt-4"),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_imp), md=12, className="mt-4"),
            ]),
        ]),

        # ---- CAPÍTULO 6: CONCLUSIONES Y SIGUIENTES PASOS ----
        dbc.Tab(label="6. Conclusiones y siguientes pasos", children=[
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("1 · Datos y EDA", className="text-primary"),
                    html.P("Sin nulos, precios alineados a la competencia (r=0.98) y demanda concentrada en la mediana (100 uds)."),
                ]), className="h-100 shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("2 · Estructura comercial", className="text-primary"),
                    html.P("K-Means (k=8 validado) reveló 'Sobreinventario' (capital atascado) y 'Máxima demanda' (el más rentable)."),
                ]), className="h-100 shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("3 · Modelo elegido", className="text-primary"),
                    html.P("Random Forest es el mejor predictor (84.8% accuracy, AUC 0.932); la regresión honesta explica el 56% de la demanda."),
                ]), className="h-100 shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("4 · Decisiones accionables", className="text-primary"),
                    html.P("Promocionar sin epidemia (+28 uds) y evitar descuentos agresivos en muebles (mayor impacto negativo)."),
                ]), className="h-100 shadow-sm"), md=3),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    html.H4("Acciones recomendadas", className="mt-4"),
                    html.Ul([
                        html.Li("Liberar capital del cluster 'Sobreinventario' (liquidaciones / devoluciones a proveedores)."),
                        html.Li("Focalizar promociones en épocas sin epidemia para maximizar el retorno."),
                        html.Li("Usar Random Forest para predecir demanda y generar alertas tempranas de reabastecimiento."),
                        html.Li("Evitar descuentos agresivos en muebles: no responden a promociones."),
                        html.Li("Revisar reposición: 62% de los registros no tienen orden, incluso con demanda alta."),
                    ]),
                ], md=6),
                dbc.Col([
                    html.H4("Plan de fase piloto", className="mt-4"),
                    html.Ul([
                        html.Li("Implementar el modelo (Random Forest) en 2 tiendas durante 1 trimestre."),
                        html.Li("KPI de éxito: reducción de sobreinventario y precisión de la predicción (RMSE / accuracy)."),
                        html.Li("Medir el impacto de las promociones focalizadas antes del despliegue total."),
                        html.Li("Retroalimentar el modelo con los datos del piloto para su recalibración."),
                    ]),
                ], md=6),
            ], className="mt-4"),
        ]),
    ], className="mt-2"),
], style={"max-width": "1400px"})

#### CALLBACKS
@app.callback(
    Output('sim-output', 'children'),
    Input('btn-predict', 'n_clicks'),
    State('sim-categoria', 'value'),
    State('sim-region', 'value'),
    State('sim-store', 'value'),
    State('sim-clima', 'value'),
    State('sim-estacion', 'value'),
    State('sim-precio', 'value'),
    State('sim-descuento', 'value'),
    State('sim-inventario', 'value'),
    State('sim-ordenadas', 'value'),
    State('sim-competencia', 'value'),
    State('sim-promo', 'value'),
    State('sim-epi', 'value'),
)
def predecir(n_clicks, categoria, region, store, clima, estacion,
             precio, descuento, inventario, ordenadas, competencia, promo, epi):
    if not n_clicks:
        return html.Div([
            html.H3("Configura un escenario y presiona 'Predecir demanda'",
                    className="text-center text-muted mt-5"),
            html.P("La consultora aprendió de 76,000 registros de ventas.",
                   className="text-center text-muted"),
        ])

    fila_num = [inventario, ordenadas, precio, descuento, promo, competencia, epi]
    fila_row = fila_num + encode_row(categoria, region, clima, estacion, store)
    pred = model_global.predict(np.array([fila_row]))[0]
    pred = max(DEMANDA_MIN, min(pred, DEMANDA_MAX))

    if pred < 150:
        nivel, color, color_hex = "BAJA", "success", "#27ae60"
        lectura = "Hay espacio para crecer: puedes empujar con promoción sin riesgo de quedarte con sobrestock."
    elif pred < 250:
        nivel, color, color_hex = "MEDIA", "warning", "#f1c40f"
        lectura = "Demanda estable: prepara el inventario cerca de lo esperado y monitorea semanalmente."
    else:
        nivel, color, color_hex = "ALTA", "danger", "#e74c3c"
        lectura = "Alta rotación: asegura el inventario antes del pico, o perderás ventas."
    orden_sugerido = int(round(pred * 1.1, -1))

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pred,
        title={'text': "Demanda predicha"},
        number={'suffix': " uds"},
        gauge={'axis': {'range': [DEMANDA_MIN, DEMANDA_MAX]},
               'bar': {'color': color_hex},
               'steps': [
                   {'range': [DEMANDA_MIN, 150], 'color': "#d5f5e3"},
                   {'range': [150, 250], 'color': "#fef9e7"},
                   {'range': [250, DEMANDA_MAX], 'color': "#fadbd8"}]}
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))

    fig_hist = px.histogram(df, x='Demanda', nbins=30,
                            title='Posición en el mercado',
                            color_discrete_sequence=['#bdc3c7'])
    fig_hist.add_vline(x=pred, line_color="#e74c3c", line_dash="dash", line_width=3)
    fig_hist.update_layout(height=280, margin=dict(t=40, b=20), showlegend=False)

    return html.Div([
        html.H2(f"{pred:,.0f} unidades", className="text-center", style={"color": color_hex}),
        dbc.Badge(nivel, color=color, className="d-block mx-auto fs-5 py-2 px-4"),
        html.P(lectura, className="text-center mt-3", style={"fontSize": "16px"}),
        dbc.Alert(
            f"Para no quedarte sin stock, ordena ≈ {orden_sugerido:,.0f} unidades "
            f"(demanda esperada + 10% de margen de seguridad).",
            color="success", className="mt-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_gauge), md=6),
            dbc.Col(dcc.Graph(figure=fig_hist), md=6),
        ]),
        dbc.Alert(f"Escenario configurado: {categoria} en {region} (tienda {store}), precio ${precio:,.0f} "
                  f"con {descuento}% de descuento, competencia ${competencia:,.0f}. "
                  f"Promoción: {'sí' if promo else 'no'} | Epidemia: {'sí' if epi else 'no'}.",
                  color="light", className="mt-2"),
    ])

if __name__ == '__main__':
    app.run(debug=True, port=8050)
