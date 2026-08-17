from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#### DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'TIENDA RETAIL.csv'), sep=';')
df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')

#### MODELO (mismo enfoque que PA1.ipynb)
cat_cols = ['Categoria', 'Region', 'Condiciones Climaticas', 'Estacionalidad', 'Store ID']
encoder = OneHotEncoder(drop='first', sparse_output=False)
cat_encoded = encoder.fit_transform(df[cat_cols])
cat_names = encoder.get_feature_names_out(cat_cols)

num_cols = ['Inventory Level', 'Unidad Vendida', 'Unidades Ordenadas',
            'Precio', 'Descuento', 'Promoción',
            'Precios de la competencia', 'Epidemic']

X = np.hstack([df[num_cols].values, cat_encoded])
y = df['Demanda'].values
all_cols = list(num_cols) + list(cat_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model_global = LinearRegression().fit(X_train, y_train)
y_pred_test = model_global.predict(X_test)
score_r2 = r2_score(y_test, y_pred_test)

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

# CAP 1: Demanda por categoría
df_cat = df.groupby('Categoria')['Demanda'].mean().sort_values().reset_index()
fig_cat = px.bar(df_cat, x='Demanda', y='Categoria', orientation='h',
                 color='Categoria', color_discrete_sequence=px.colors.qualitative.Pastel,
                 title='¿Qué mueve más unidades? (Demanda promedio)',
                 labels={'Demanda': 'Demanda promedio'})
fig_cat.update_layout(showlegend=False, height=320)

# CAP 1: Demanda por región
df_reg = df.groupby('Region')['Demanda'].mean().reset_index()
fig_reg = px.bar(df_reg, x='Region', y='Demanda', color='Region',
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title='Demanda promedio por región')
fig_reg.update_layout(showlegend=False, height=320)

# CAP 2: Precio vs competencia
muestra = df.sample(min(5000, len(df)), random_state=42)
fig_pc = px.scatter(muestra, x='Precios de la competencia', y='Precio',
                    color='Categoria', opacity=0.6,
                    title='¿Seguimos a la competencia o inventamos?',
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
                            labels={'x': 'Demanda Real', 'y': 'Demanda Predicha'},
                            title=f'Precisión del modelo (R² = {score_r2:.2f})',
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
                  title='¿Qué variables pesan en la predicción?')
fig_coef.update_layout(showlegend=False, height=500)

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
                ]), className="text-center shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Demanda promedio", className="text-muted"),
                    html.H3(f"{demanda_prom:,.1f}".replace(",", "."), className="text-primary"),
                ]), className="text-center shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Unidades vendidas", className="text-muted"),
                    html.H3(f"{vendidas_total:,}".replace(",", "."), className="text-primary"),
                ]), className="text-center shadow-sm"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H6("Precio promedio", className="text-muted"),
                    html.H3(f"${precio_prom:,.2f}".replace(",", "."), className="text-primary"),
                ]), className="text-center shadow-sm"), md=3),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    html.H4("¿Dónde está el negocio?", className="mt-3"),
                    html.P("Antes de hablar de modelos, entendamos el terreno. La demanda no es homogénea: "
                           "unas categorías venden mucho más que otras y las regiones no se comportan igual."),
                    dbc.Alert("Insight: Identificar los productos y zonas que mueven el negocio es el punto de partida.",
                              color="primary"),
                ], md=4, className="d-flex flex-column justify-content-center"),
                dbc.Col(dcc.Graph(figure=fig_cat, style={'height': '360px'}), md=4),
                dbc.Col(dcc.Graph(figure=fig_reg, style={'height': '360px'}), md=4),
            ], className="mt-4"),
        ]),

        # ---- CAPÍTULO 2: LOS HALLAZGOS (Datos atípicos / contradicciones) ----
        dbc.Tab(label="2. Los Hallazgos", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Los datos se contradicen", className="mt-3"),
                    html.P("Aquí está la parte interesante. Los números revelan anomalías que el negocio "
                           "debería revisar: promociones que no venden, precios desalineados y una epidemia "
                           "que casi todos resienten... excepto una categoría."),
                    dbc.Alert("Insight clave: La epidemia devastó la demanda en 4 de 5 categorías, "
                              "pero Clothing creció +1.25%.",
                              color="danger"),
                ], md=4, className="d-flex flex-column justify-content-center"),
                dbc.Col(dcc.Graph(figure=fig_pc), md=8),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_pe), md=6),
                dbc.Col(dcc.Graph(figure=fig_epi), md=6),
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
                                         value='Electronics'),
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
                            dcc.Slider(id='sim-precio', min=5, max=230, step=1, value=70,
                                       marks={5: '5', 115: '115', 230: '230'}),
                            html.Label("7. Descuento (%)", className="mt-3"),
                            dcc.Slider(id='sim-descuento', min=0, max=25, step=1, value=10,
                                       marks={0: '0', 25: '25'}),
                            html.Label("8. Inventario disponible", className="mt-3"),
                            dcc.Slider(id='sim-inventario', min=0, max=2300, step=50, value=300,
                                       marks={0: '0', 1150: '1150', 2300: '2300'}),
                            html.Label("9. Unidades ordenadas", className="mt-3"),
                            dcc.Slider(id='sim-ordenadas', min=0, max=1600, step=50, value=90,
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
                                                        value=0)),
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

        # ---- CAPÍTULO 4: VALIDACIÓN ----
        dbc.Tab(label="4. Validación", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("¿Por qué confiar en la IA?", className="mt-3"),
                    html.P("Sometimos el modelo a una prueba ciega: le pedimos predecir demandas que nunca "
                           "había visto. La línea roja representa la predicción perfecta."),
                    dbc.Alert(f"El modelo explica el {score_r2:.0%} de la variabilidad de la demanda (R²).",
                              color="success"),
                ], md=4, className="d-flex flex-column justify-content-center"),
                dbc.Col(dcc.Graph(figure=fig_validation), md=8),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    html.H5("Lo que más pesa", className="mt-3"),
                    html.P("Los coeficientes del modelo (cada barra es cuánto influye cada variable en la demanda)."),
                    dcc.Graph(figure=fig_coef),
                ], md=12, className="mt-4"),
            ]),
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
            html.P("El simulador usa el modelo entrenado con 76,000 registros.",
                   className="text-center text-muted"),
        ])

    fila_num = [inventario, 0, ordenadas, precio, descuento, promo, competencia, epi]
    fila_row = fila_num + encode_row(categoria, region, clima, estacion, store)
    pred = model_global.predict(np.array([fila_row]))[0]
    pred = max(0, min(pred, DEMANDA_MAX))

    color = "#27ae60" if pred < 150 else "#f1c40f" if pred < 250 else "#e74c3c"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pred,
        title={'text': "Demanda predicha"},
        number={'suffix': " uds"},
        gauge={'axis': {'range': [DEMANDA_MIN, DEMANDA_MAX]},
               'bar': {'color': color},
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

    nivel = "baja" if pred < 150 else "media" if pred < 250 else "alta"

    return html.Div([
        html.H2(f"{pred:,.0f} unidades", className="text-center text-primary"),
        html.P(f"Demanda {nivel} para el escenario configurado.", className="text-center text-muted"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_gauge), md=6),
            dbc.Col(dcc.Graph(figure=fig_hist), md=6),
        ]),
        dbc.Alert(f"Escenario: {categoria} en {region} (tienda {store}), precio ${precio:,.0f} "
                  f"con {descuento}% de descuento, competencia ${competencia:,.0f}. "
                  f"Promoción: {'sí' if promo else 'no'} | Epidemia: {'sí' if epi else 'no'}.",
                  color="info", className="mt-2"),
    ])

if __name__ == '__main__':
    app.run(debug=True, port=8050)
