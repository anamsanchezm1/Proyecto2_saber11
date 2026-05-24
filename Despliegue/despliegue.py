import joblib
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State

# =========================
# Cargar modelo y encoder
# =========================

modelo = joblib.load("modelo_rf.pkl")
encoder = joblib.load("encoder.pkl")

clases = ["a-", "a1", "a2", "b+", "b1"]

variables_modelo = [
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

# =========================
# App
# =========================

app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Arial", "backgroundColor": "#f4f6f8", "padding": "30px"},
    children=[
        html.H1(
            "Predicción de desempeño en inglés - Saber 11",
            style={"textAlign": "center", "color": "#1f3c88"}
        ),

        html.P(
            "Herramienta para coordinadores académicos basada en características del establecimiento educativo.",
            style={"textAlign": "center", "fontSize": "18px"}
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "35% 65%",
                "gap": "25px",
                "marginTop": "30px"
            },
            children=[

                # Panel izquierdo
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

                # Panel derecho
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
)


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
def predecir(n_clicks, area, bilingue, calendario, caracter, depto, genero, jornada, naturaleza, sede):

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

    prediccion = modelo.predict(entrada_encoded)[0]
    probabilidades = modelo.predict_proba(entrada_encoded)[0]

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


if __name__ == "__main__":
    app.run(debug=True)