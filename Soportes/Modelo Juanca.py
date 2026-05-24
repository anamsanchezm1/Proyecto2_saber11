import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

ruta = "/Users/juancamilohernandezjacobo/Desktop/saber11_limpio_base.csv"
df = pd.read_csv(ruta)

columnas_X = [
    "cole_area_ubicacion",
    "cole_naturaleza",
    "cole_calendario",
    "cole_jornada",
    "cole_caracter",
    "cole_bilingue",
    "cole_genero"
]

X = df[columnas_X]
y = df["punt_global"]

datos = pd.concat([X, y], axis=1).dropna()

X = datos[columnas_X]
y = datos["punt_global"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train.copy()
X_test = X_test.copy()

for col in columnas_X:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

preprocesador = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_X)
    ]
)

X_train_procesado = preprocesador.fit_transform(X_train)
X_test_procesado = preprocesador.transform(X_test)

if hasattr(X_train_procesado, "toarray"):
    X_train_procesado = X_train_procesado.toarray()

if hasattr(X_test_procesado, "toarray"):
    X_test_procesado = X_test_procesado.toarray()