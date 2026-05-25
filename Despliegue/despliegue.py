import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from tensorflow import keras
from scipy import stats

# =========================
# Cargar modelos y artefactos
# =========================

# Modelo de clasificación (inglés) - características del colegio
modelo_clf = joblib.load("modelo_rf.pkl")
encoder = joblib.load("encoder.pkl")
clases = ["a-", "a1", "a2", "b+", "b1"]

# Modelo de regresión por características del colegio (Random Forest)
modelo_colegio = joblib.load("modelo_rf.pkl")

# Modelo de regresión (puntaje global por variables sociales y familiares)
modelo_reg = keras.models.load_model("modelo_final.keras")
scaler = joblib.load("scaler.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

# RMSE del modelo de regresión
RMSE_MODELO = 40.153157

# Variables para clasificación inglés
variables_clasificacion = [
    "cole_area_ubicacion",
    "cole_bilingue",
    "cole_calendario",
    "cole_caracter",
    "cole_depto_ubicacion",
    "cole_genero",
    "cole_jornada",
    "cole_naturaleza",
    "cole_sede_principal"
]

# Variables one-hot encoding de educación para regresión
educacion_madre_cols = [col for col in columnas_modelo if col.startswith("fami_educacionmadre_")]
educacion_padre_cols = [col for col in columnas_modelo if col.startswith("fami_educacionpadre_")]

# Mapeos de educación para regresión
educacion_opciones = {
    "Primaria incompleta": "primaria incompleta",
    "Primaria completa": "primaria completa",
    "Secundaria incompleta": "secundaria incompleta",
    "Secundaria completa": "secundaria completa",
    "Técnica o tecnológica incompleta": "tecnica o tecnologica incompleta",
    "Técnica o tecnológica completa": "tecnica o tecnologica completa",
    "Profesional incompleta": "profesional incompleta",
    "Profesional completa": "profesional completa",
    "Postgrado": "postgrado"
}

# =========================
# App
# =========================

app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial", "backgroundColor": "#f4f6f8", "padding": "30px"},
    children=[
        html.H1(
            "Herramientas de predicción - Saber 11",
            style={"textAlign": "center", "color": "#1f3c88"}
        ),

        html.P(
            "Análisis educativo avanzado para instituciones y coordinadores académicos.",
            style={"textAlign": "center", "fontSize": "18px"}
        ),

        # TABS
        dcc.Tabs(
            id="tabs",
            value="tab-1",
            children=[
                # TAB 1: Clasificación de desempeño en inglés
                dcc.Tab(
                    label="Características del Colegio - Desempeño en Inglés",
                    value="tab-1",
                    style={"padding": "20px"},
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "35% 65%",
                                "gap": "25px",
                                "marginTop": "30px"
                            },
                            children=[
                                # Panel izquierdo - Clasificación
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "padding": "25px",
                                        "borderRadius": "15px",
                                        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                                    },
                                    children=[
                                        html.H3("Características del colegio"),

                                        html.Label("Área de ubicación"),
                                        dcc.Dropdown(
                                            id="area",
                                            options=[
                                                {"label": "Urbano", "value": "urbano"},
                                                {"label": "Rural", "value": "rural"}
                                            ],
                                            value="urbano"
                                        ),

                                        html.Br(),

                                        html.Label("Colegio bilingüe"),
                                        dcc.Dropdown(
                                            id="bilingue",
                                            options=[
                                                {"label": "Sí", "value": "si"},
                                                {"label": "No", "value": "no"}
                                            ],
                                            value="no"
                                        ),

                                        html.Br(),

                                        html.Label("Calendario"),
                                        dcc.Dropdown(
                                            id="calendario",
                                            options=[
                                                {"label": "A", "value": "a"},
                                                {"label": "B", "value": "b"}
                                            ],
                                            value="a"
                                        ),

                                        html.Br(),

                                        html.Label("Carácter"),
                                        dcc.Dropdown(
                                            id="caracter",
                                            options=[
                                                {"label": "Académico", "value": "academico"},
                                                {"label": "Técnico", "value": "tecnico"},
                                                {"label": "Técnico/académico", "value": "tecnico/academico"}
                                            ],
                                            value="academico"
                                        ),

                                        html.Br(),

                                        html.Label("Departamento"),
                                        dcc.Dropdown(
                                            id="depto",
                                            options=[
                                                {"label": "Bogotá", "value": "bogota"},
                                                {"label": "Antioquia", "value": "antioquia"},
                                                {"label": "Valle", "value": "valle"},
                                                {"label": "Cundinamarca", "value": "cundinamarca"},
                                                {"label": "Santander", "value": "santander"},
                                                {"label": "Atlántico", "value": "atlantico"}
                                            ],
                                            value="bogota"
                                        ),

                                        html.Br(),

                                        html.Label("Género del colegio"),
                                        dcc.Dropdown(
                                            id="genero",
                                            options=[
                                                {"label": "Mixto", "value": "mixto"},
                                                {"label": "Femenino", "value": "femenino"},
                                                {"label": "Masculino", "value": "masculino"}
                                            ],
                                            value="mixto"
                                        ),

                                        html.Br(),

                                        html.Label("Jornada"),
                                        dcc.Dropdown(
                                            id="jornada",
                                            options=[
                                                {"label": "Mañana", "value": "manana"},
                                                {"label": "Tarde", "value": "tarde"},
                                                {"label": "Noche", "value": "noche"},
                                                {"label": "Completa", "value": "completa"},
                                                {"label": "Sabatina", "value": "sabatina"},
                                                {"label": "Única", "value": "unica"}
                                            ],
                                            value="manana"
                                        ),

                                        html.Br(),

                                        html.Label("Naturaleza"),
                                        dcc.Dropdown(
                                            id="naturaleza",
                                            options=[
                                                {"label": "Oficial", "value": "oficial"},
                                                {"label": "No oficial", "value": "no oficial"}
                                            ],
                                            value="oficial"
                                        ),

                                        html.Br(),

                                        html.Label("Sede principal"),
                                        dcc.Dropdown(
                                            id="sede",
                                            options=[
                                                {"label": "Sí", "value": "si"},
                                                {"label": "No", "value": "no"}
                                            ],
                                            value="si"
                                        ),

                                        html.Br(),

                                        html.Button(
                                            "Predecir desempeño",
                                            id="boton",
                                            n_clicks=0,
                                            style={
                                                "width": "100%",
                                                "padding": "12px",
                                                "backgroundColor": "#1f3c88",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "10px",
                                                "fontSize": "16px"
                                            }
                                        )
                                    ]
                                ),

                                # Panel derecho - Clasificación
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "padding": "25px",
                                        "borderRadius": "15px",
                                        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                                    },
                                    children=[
                                        html.H3("Resultado de la predicción"),

                                        html.Div(
                                            id="resultado",
                                            style={
                                                "fontSize": "42px",
                                                "fontWeight": "bold",
                                                "color": "#1f3c88",
                                                "textAlign": "center",
                                                "margin": "30px"
                                            }
                                        ),

                                        dcc.Graph(id="grafico_probabilidades"),

                                        html.Div(
                                            id="interpretacion",
                                            style={
                                                "fontSize": "17px",
                                                "marginTop": "20px",
                                                "backgroundColor": "#eef2ff",
                                                "padding": "15px",
                                                "borderRadius": "10px"
                                            }
                                        )
                                    ]
                                )
                            ]
                        ),

                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "15px",
                                "marginTop": "25px",
                                "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                            },
                            children=[
                                html.H3("Variables más relevantes del modelo"),
                                html.P("El modelo Random Forest identificó como más relevantes: jornada escolar, calendario académico, naturaleza del colegio y condición bilingüe.")
                            ]
                        )
                    ]
                ),

                # TAB 2: Regresión puntaje global
                dcc.Tab(
                    label="Puntaje Global - Variables Sociales y Familiares",
                    value="tab-2",
                    style={"padding": "20px"},
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "35% 65%",
                                "gap": "25px",
                                "marginTop": "30px"
                            },
                            children=[
                                # Panel izquierdo - Regresión
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "padding": "25px",
                                        "borderRadius": "15px",
                                        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                                    },
                                    children=[
                                        html.H3("Información del estudiante"),

                                        html.Label("Edad (años)"),
                                        dcc.Input(
                                            id="edad",
                                            type="number",
                                            min=14,
                                            max=25,
                                            value=17,
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "marginBottom": "20px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #d0d0d0"
                                            }
                                        ),

                                        html.Label("Estrato de vivienda"),
                                        dcc.Dropdown(
                                            id="estrato",
                                            options=[
                                                {"label": "Estrato 1", "value": 1},
                                                {"label": "Estrato 2", "value": 2},
                                                {"label": "Estrato 3", "value": 3},
                                                {"label": "Estrato 4", "value": 4},
                                                {"label": "Estrato 5", "value": 5},
                                                {"label": "Estrato 6", "value": 6}
                                            ],
                                            value=3,
                                            style={"marginBottom": "20px"}
                                        ),

                                        html.Label("Personas en el hogar"),
                                        dcc.Input(
                                            id="personas_hogar",
                                            type="number",
                                            min=1,
                                            max=15,
                                            value=4,
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "marginBottom": "20px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #d0d0d0"
                                            }
                                        ),

                                        html.Label("Cuartos en el hogar"),
                                        dcc.Input(
                                            id="cuartos_hogar",
                                            type="number",
                                            min=1,
                                            max=20,
                                            value=4,
                                            style={
                                                "width": "100%",
                                                "padding": "10px",
                                                "marginBottom": "20px",
                                                "borderRadius": "5px",
                                                "border": "1px solid #d0d0d0"
                                            }
                                        ),

                                        html.H4("Servicios y bienes en el hogar"),

                                        html.Label("¿Tiene acceso a Internet?"),
                                        dcc.RadioItems(
                                            id="internet",
                                            options=[
                                                {"label": "  Sí", "value": 1},
                                                {"label": "  No", "value": 0}
                                            ],
                                            value=1,
                                            style={"marginBottom": "15px"}
                                        ),

                                        html.Label("¿Tiene computador?"),
                                        dcc.RadioItems(
                                            id="computador",
                                            options=[
                                                {"label": "  Sí", "value": 1},
                                                {"label": "  No", "value": 0}
                                            ],
                                            value=1,
                                            style={"marginBottom": "15px"}
                                        ),

                                        html.Label("¿Tiene automóvil?"),
                                        dcc.RadioItems(
                                            id="automovil",
                                            options=[
                                                {"label": "  Sí", "value": 1},
                                                {"label": "  No", "value": 0}
                                            ],
                                            value=0,
                                            style={"marginBottom": "15px"}
                                        ),

                                        html.Label("¿Tiene lavadora?"),
                                        dcc.RadioItems(
                                            id="lavadora",
                                            options=[
                                                {"label": "  Sí", "value": 1},
                                                {"label": "  No", "value": 0}
                                            ],
                                            value=1,
                                            style={"marginBottom": "20px"}
                                        ),

                                        html.H4("Educación de los padres"),

                                        html.Label("Educación de la madre"),
                                        dcc.Dropdown(
                                            id="educacion_madre",
                                            options=[{"label": k, "value": v} for k, v in educacion_opciones.items()],
                                            value="secundaria completa",
                                            style={"marginBottom": "20px"}
                                        ),

                                        html.Label("Educación del padre"),
                                        dcc.Dropdown(
                                            id="educacion_padre",
                                            options=[{"label": k, "value": v} for k, v in educacion_opciones.items()],
                                            value="secundaria completa",
                                            style={"marginBottom": "20px"}
                                        ),

                                        html.Button(
                                            "Predecir puntaje global",
                                            id="boton_regresion",
                                            n_clicks=0,
                                            style={
                                                "width": "100%",
                                                "padding": "12px",
                                                "backgroundColor": "#1f3c88",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "10px",
                                                "fontSize": "16px",
                                                "fontWeight": "bold",
                                                "marginTop": "10px"
                                            }
                                        )
                                    ]
                                ),

                                # Panel derecho - Regresión
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "padding": "25px",
                                        "borderRadius": "15px",
                                        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                                    },
                                    children=[
                                        html.H3("Predicción del puntaje global"),

                                        html.Div(
                                            id="resultado_puntaje",
                                            style={
                                                "fontSize": "48px",
                                                "fontWeight": "bold",
                                                "color": "#1f3c88",
                                                "textAlign": "center",
                                                "margin": "30px 0",
                                                "padding": "20px",
                                                "backgroundColor": "#f0f4ff",
                                                "borderRadius": "10px"
                                            }
                                        ),

                                        dcc.Graph(id="grafico_probabilidades_reg"),

                                        html.Div(
                                            id="interpretacion_reg",
                                            style={
                                                "fontSize": "16px",
                                                "marginTop": "20px",
                                                "backgroundColor": "#eef2ff",
                                                "padding": "15px",
                                                "borderRadius": "10px",
                                                "lineHeight": "1.6"
                                            },
                                            children=html.Div()
                                        )
                                    ]
                                )
                            ]
                        ),

                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "15px",
                                "marginTop": "25px",
                                "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                            },
                            children=[
                                html.H3("Sobre el modelo de regresión"),
                                html.P(
                                    "Este modelo de regresión neuronal predice el puntaje global esperado en Saber 11 "
                                    "a partir de 15 variables socioeconómicas y familiares. El modelo explica cómo factores como el estrato, "
                                    "educación de los padres, y acceso a tecnología impactan el desempeño académico."
                                )
                            ]
                        )
                    ]
                ),

                # TAB 3: Regresión puntaje global por características del colegio
                dcc.Tab(
                    label="Puntaje Global - Características del Colegio",
                    value="tab-3",
                    style={"padding": "20px"},
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "35% 65%",
                                "gap": "25px",
                                "marginTop": "30px"
                            },
                            children=[
                                # Panel izquierdo - Regresión colegio
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "padding": "25px",
                                        "borderRadius": "15px",
                                        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                                    },
                                    children=[
                                        html.H3("Características del colegio"),

                                        html.Label("Área de ubicación"),
                                        dcc.Dropdown(
                                            id="area_colegio",
                                            options=[
                                                {"label": "Urbano", "value": "urbano"},
                                                {"label": "Rural", "value": "rural"}
                                            ],
                                            value="urbano"
                                        ),

                                        html.Br(),

                                        html.Label("Colegio bilingüe"),
                                        dcc.Dropdown(
                                            id="bilingue_colegio",
                                            options=[
                                                {"label": "Sí", "value": "si"},
                                                {"label": "No", "value": "no"}
                                            ],
                                            value="no"
                                        ),

                                        html.Br(),

                                        html.Label("Calendario"),
                                        dcc.Dropdown(
                                            id="calendario_colegio",
                                            options=[
                                                {"label": "A", "value": "a"},
                                                {"label": "B", "value": "b"}
                                            ],
                                            value="a"
                                        ),

                                        html.Br(),

                                        html.Label("Carácter"),
                                        dcc.Dropdown(
                                            id="caracter_colegio",
                                            options=[
                                                {"label": "Académico", "value": "academico"},
                                                {"label": "Técnico", "value": "tecnico"},
                                                {"label": "Técnico/académico", "value": "tecnico/academico"}
                                            ],
                                            value="academico"
                                        ),

                                        html.Br(),

                                        html.Label("Departamento"),
                                        dcc.Dropdown(
                                            id="depto_colegio",
                                            options=[
                                                {"label": "Bogotá", "value": "bogota"},
                                                {"label": "Antioquia", "value": "antioquia"},
                                                {"label": "Valle", "value": "valle"},
                                                {"label": "Cundinamarca", "value": "cundinamarca"},
                                                {"label": "Santander", "value": "santander"},
                                                {"label": "Atlántico", "value": "atlantico"}
                                            ],
                                            value="bogota"
                                        ),

                                        html.Br(),

                                        html.Label("Género del colegio"),
                                        dcc.Dropdown(
                                            id="genero_colegio",
                                            options=[
                                                {"label": "Mixto", "value": "mixto"},
                                                {"label": "Femenino", "value": "femenino"},
                                                {"label": "Masculino", "value": "masculino"}
                                            ],
                                            value="mixto"
                                        ),

                                        html.Br(),

                                        html.Label("Jornada"),
                                        dcc.Dropdown(
                                            id="jornada_colegio",
                                            options=[
                                                {"label": "Mañana", "value": "manana"},
                                                {"label": "Tarde", "value": "tarde"},
                                                {"label": "Noche", "value": "noche"},
                                                {"label": "Completa", "value": "completa"},
                                                {"label": "Sabatina", "value": "sabatina"},
                                                {"label": "Única", "value": "unica"}
                                            ],
                                            value="manana"
                                        ),

                                        html.Br(),

                                        html.Label("Naturaleza"),
                                        dcc.Dropdown(
                                            id="naturaleza_colegio",
                                            options=[
                                                {"label": "Oficial", "value": "oficial"},
                                                {"label": "No oficial", "value": "no oficial"}
                                            ],
                                            value="oficial"
                                        ),

                                        html.Br(),

                                        html.Label("Sede principal"),
                                        dcc.Dropdown(
                                            id="sede_colegio",
                                            options=[
                                                {"label": "Sí", "value": "si"},
                                                {"label": "No", "value": "no"}
                                            ],
                                            value="si"
                                        ),

                                        html.Br(),

                                        html.Button(
                                            "Predecir puntaje global",
                                            id="boton_colegio",
                                            n_clicks=0,
                                            style={
                                                "width": "100%",
                                                "padding": "12px",
                                                "backgroundColor": "#1f3c88",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "10px",
                                                "fontSize": "16px",
                                                "fontWeight": "bold",
                                                "marginTop": "10px"
                                            }
                                        )
                                    ]
                                ),

                                # Panel derecho - Regresión colegio
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "padding": "25px",
                                        "borderRadius": "15px",
                                        "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                                    },
                                    children=[
                                        html.H3("Predicción del puntaje global"),

                                        html.Div(
                                            id="resultado_colegio",
                                            style={
                                                "fontSize": "48px",
                                                "fontWeight": "bold",
                                                "color": "#1f3c88",
                                                "textAlign": "center",
                                                "margin": "30px 0",
                                                "padding": "20px",
                                                "backgroundColor": "#f0f4ff",
                                                "borderRadius": "10px"
                                            }
                                        ),

                                        dcc.Graph(id="grafico_probabilidades_colegio"),

                                        html.Div(
                                            id="interpretacion_colegio",
                                            style={
                                                "fontSize": "16px",
                                                "marginTop": "20px",
                                                "backgroundColor": "#eef2ff",
                                                "padding": "15px",
                                                "borderRadius": "10px",
                                                "lineHeight": "1.6"
                                            },
                                            children=html.Div()
                                        )
                                    ]
                                )
                            ]
                        ),

                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "20px",
                                "borderRadius": "15px",
                                "marginTop": "25px",
                                "boxShadow": "0 4px 10px rgba(0,0,0,0.1)"
                            },
                            children=[
                                html.H3("Sobre el modelo de regresión por características del colegio"),
                                html.P(
                                    "Este modelo de regresión predice el puntaje global esperado en Saber 11 "
                                    "a partir de características del establecimiento educativo. El modelo explica cómo factores como "
                                    "la jornada, calendario académico, naturaleza del colegio y ubicación impactan el desempeño académico."
                                )
                            ]
                        )
                    ]
                )
            ]
        ),

    ]
)


# Funciones auxiliares para regresión
def calcular_indices(personas_hogar, cuartos_hogar, internet, computador, automovil, lavadora):
    """Calcula densidad_hogar, indice_tecnologico, indice_bienes_hogar"""
    densidad = personas_hogar / max(cuartos_hogar, 1)
    indice_tec = internet + computador
    indice_bienes = automovil + lavadora
    return densidad, indice_tec, indice_bienes


def crear_columnas_dummy(educacion_madre, educacion_padre):
    """Crea las columnas dummy para educación madre y padre"""
    dummy_dict = {}
    
    for col in educacion_madre_cols:
        dummy_dict[col] = 0
    for col in educacion_padre_cols:
        dummy_dict[col] = 0
    
    col_madre = f"fami_educacionmadre_{educacion_madre}"
    col_padre = f"fami_educacionpadre_{educacion_padre}"
    
    if col_madre in dummy_dict:
        dummy_dict[col_madre] = 1
    if col_padre in dummy_dict:
        dummy_dict[col_padre] = 1
    
    return dummy_dict


def calcular_probabilidades(prediccion, rmse):
    """Calcula probabilidades aproximadas usando distribución normal"""
    sigma = rmse
    
    prob_bajo = stats.norm.cdf(250, loc=prediccion, scale=sigma)
    prob_medio = stats.norm.cdf(300, loc=prediccion, scale=sigma) - prob_bajo
    prob_alto = stats.norm.cdf(350, loc=prediccion, scale=sigma) - prob_bajo - prob_medio
    prob_superior = 1 - prob_bajo - prob_medio - prob_alto
    
    return {
        "Bajo (<250)": max(0, min(1, prob_bajo)),
        "Medio (250-300)": max(0, min(1, prob_medio)),
        "Alto (300-350)": max(0, min(1, prob_alto)),
        "Superior (>350)": max(0, min(1, prob_superior))
    }


# CALLBACK 1: Clasificación de desempeño en inglés
@app.callback(
    Output("resultado", "children"),
    Output("grafico_probabilidades", "figure"),
    Output("interpretacion", "children"),
    Input("boton", "n_clicks"),
    State("area", "value"),
    State("bilingue", "value"),
    State("calendario", "value"),
    State("caracter", "value"),
    State("depto", "value"),
    State("genero", "value"),
    State("jornada", "value"),
    State("naturaleza", "value"),
    State("sede", "value")
)
def predecir_clasificacion(n_clicks, area, bilingue, calendario, caracter, depto, genero, jornada, naturaleza, sede):

    entrada = pd.DataFrame([{
        "cole_area_ubicacion": area,
        "cole_bilingue": bilingue,
        "cole_calendario": calendario,
        "cole_caracter": caracter,
        "cole_depto_ubicacion": depto,
        "cole_genero": genero,
        "cole_jornada": jornada,
        "cole_naturaleza": naturaleza,
        "cole_sede_principal": sede
    }])

    entrada_encoded = encoder.transform(entrada)

    prediccion = modelo_clf.predict(entrada_encoded)[0]
    probabilidades = modelo_clf.predict_proba(entrada_encoded)[0]

    nivel_predicho = clases[prediccion]

    df_prob = pd.DataFrame({
        "Nivel": clases,
        "Probabilidad": probabilidades
    })

    fig = px.bar(
        df_prob,
        x="Nivel",
        y="Probabilidad",
        title="Probabilidad por nivel de desempeño",
        text=df_prob["Probabilidad"].round(2)
    )

    fig.update_layout(
        yaxis_tickformat=".0%",
        plot_bgcolor="white"
    )

    interpretacion = (
        f"Según las características ingresadas, el estudiante tendría mayor probabilidad "
        f"de alcanzar un nivel {nivel_predicho.upper()} en inglés."
    )

    return nivel_predicho.upper(), fig, interpretacion


# CALLBACK 2: Regresión puntaje global
@app.callback(
    Output("resultado_puntaje", "children"),
    Output("grafico_probabilidades_reg", "figure"),
    Output("interpretacion_reg", "children"),
    Input("boton_regresion", "n_clicks"),
    State("edad", "value"),
    State("estrato", "value"),
    State("personas_hogar", "value"),
    State("cuartos_hogar", "value"),
    State("internet", "value"),
    State("computador", "value"),
    State("automovil", "value"),
    State("lavadora", "value"),
    State("educacion_madre", "value"),
    State("educacion_padre", "value")
)
def predecir_regresion(n_clicks, edad, estrato, personas_hogar, cuartos_hogar, 
                       internet, computador, automovil, lavadora, 
                       educacion_madre, educacion_padre):

    if n_clicks == 0:
        return "Ingresa valores y presiona el botón", {}, "Aguardando predicción..."

    if not all([edad, estrato, personas_hogar, cuartos_hogar]):
        return "Error", {}, "Por favor completa todos los campos."

    # Calcular índices
    densidad, indice_tec, indice_bienes = calcular_indices(
        personas_hogar, cuartos_hogar, internet, computador, automovil, lavadora
    )

    # Crear diccionario de entrada
    entrada_dict = {
        "edad": edad,
        "fami_estratovivienda_num": estrato,
        "fami_personashogar_num": personas_hogar,
        "fami_cuartoshogar_num": cuartos_hogar,
        "densidad_hogar": densidad,
        "fami_tieneinternet": internet,
        "fami_tienecomputador": computador,
        "fami_tieneautomovil": automovil,
        "fami_tienelavadora": lavadora,
        "indice_tecnologico": indice_tec,
        "indice_bienes_hogar": indice_bienes
    }

    # Agregar columnas dummy de educación
    dummy_educacion = crear_columnas_dummy(educacion_madre, educacion_padre)
    entrada_dict.update(dummy_educacion)

    # Crear DataFrame con el orden correcto de columnas
    entrada_df = pd.DataFrame([entrada_dict])
    
    # Asegurar que todas las columnas del modelo estén presentes
    for col in columnas_modelo:
        if col not in entrada_df.columns:
            entrada_df[col] = 0

    # Seleccionar solo las columnas del modelo en el orden correcto
    entrada_df = entrada_df[columnas_modelo]

    # Escalar
    entrada_scaled = scaler.transform(entrada_df)

    # Predecir
    prediccion = modelo_reg.predict(entrada_scaled, verbose=0)[0][0]

    # Calcular probabilidades aproximadas
    probabilidades = calcular_probabilidades(prediccion, RMSE_MODELO)

    # Crear gráfico
    df_prob = pd.DataFrame({
        "Rango de puntaje": list(probabilidades.keys()),
        "Probabilidad": [prob * 100 for prob in probabilidades.values()]
    })

    fig = px.bar(
        df_prob,
        x="Rango de puntaje",
        y="Probabilidad",
        title="Probabilidad estimada por rango de puntaje global",
        labels={"Probabilidad": "Probabilidad (%)"},
        text=[f"{prob:.1f}%" for prob in df_prob["Probabilidad"]],
        color="Rango de puntaje",
        color_discrete_sequence=["#e8f4f8", "#b3d9e8", "#7eb8d6", "#1f3c88"]
    )

    fig.update_layout(
        plot_bgcolor="white",
        xaxis_title="Rango de puntaje global",
        yaxis_title="Probabilidad (%)",
        showlegend=False,
        hovermode="x unified"
    )

    fig.update_traces(textposition="outside")

    # Determinar rango con mayor probabilidad
    rango_principal = max(probabilidades, key=probabilidades.get)
    prob_principal = probabilidades[rango_principal] * 100

    # Generar interpretación
    interpretacion = (
        f"El modelo estima un puntaje global esperado de **{prediccion:.0f} puntos**. "
        f"Bajo las condiciones socioeconómicas y familiares ingresadas, existe mayor probabilidad ({prob_principal:.1f}%) "
        f"de ubicarse en el rango **{rango_principal}**."
    )

    # Formato del resultado
    resultado_texto = f"{prediccion:.0f} puntos"

    return resultado_texto, fig, dcc.Markdown(interpretacion)


# CALLBACK 3: Regresión puntaje global por características del colegio
@app.callback(
    Output("resultado_colegio", "children"),
    Output("grafico_probabilidades_colegio", "figure"),
    Output("interpretacion_colegio", "children"),
    Input("boton_colegio", "n_clicks"),
    State("area_colegio", "value"),
    State("bilingue_colegio", "value"),
    State("calendario_colegio", "value"),
    State("caracter_colegio", "value"),
    State("depto_colegio", "value"),
    State("genero_colegio", "value"),
    State("jornada_colegio", "value"),
    State("naturaleza_colegio", "value"),
    State("sede_colegio", "value")
)
def predecir_colegio(n_clicks, area, bilingue, calendario, caracter, depto, genero, jornada, naturaleza, sede):

    if n_clicks == 0:
        return "Ingresa valores y presiona el botón", {}, "Aguardando predicción..."

    entrada = pd.DataFrame([{
        "cole_area_ubicacion": area,
        "cole_bilingue": bilingue,
        "cole_calendario": calendario,
        "cole_caracter": caracter,
        "cole_depto_ubicacion": depto,
        "cole_genero": genero,
        "cole_jornada": jornada,
        "cole_naturaleza": naturaleza,
        "cole_sede_principal": sede
    }])

    entrada_encoded = encoder.transform(entrada)

    # Predecir puntaje global
    prediccion = modelo_colegio.predict(entrada_encoded)[0]

    # Calcular probabilidades aproximadas
    RMSE_COLEGIO = 40.0
    probabilidades = calcular_probabilidades(prediccion, RMSE_COLEGIO)

    # Crear gráfico
    df_prob = pd.DataFrame({
        "Rango de puntaje": list(probabilidades.keys()),
        "Probabilidad": [prob * 100 for prob in probabilidades.values()]
    })

    fig = px.bar(
        df_prob,
        x="Rango de puntaje",
        y="Probabilidad",
        title="Probabilidad estimada por rango de puntaje global",
        labels={"Probabilidad": "Probabilidad (%)"},
        text=[f"{prob:.1f}%" for prob in df_prob["Probabilidad"]],
        color="Rango de puntaje",
        color_discrete_sequence=["#e8f4f8", "#b3d9e8", "#7eb8d6", "#1f3c88"]
    )

    fig.update_layout(
        plot_bgcolor="white",
        xaxis_title="Rango de puntaje global",
        yaxis_title="Probabilidad (%)",
        showlegend=False,
        hovermode="x unified"
    )

    fig.update_traces(textposition="outside")

    # Determinar rango con mayor probabilidad
    rango_principal = max(probabilidades, key=probabilidades.get)
    prob_principal = probabilidades[rango_principal] * 100

    # Generar interpretación
    interpretacion = (
        f"El modelo estima un puntaje global esperado de **{prediccion:.0f} puntos** para estudiantes en colegios con características similares. "
        f"Existe una probabilidad de **{prob_principal:.1f}%** de ubicarse en el rango **{rango_principal}**."
    )

    # Formato del resultado
    resultado_texto = f"{prediccion:.0f} puntos"

    return resultado_texto, fig, dcc.Markdown(interpretacion)


if __name__ == "__main__":
    app.run(debug=True)