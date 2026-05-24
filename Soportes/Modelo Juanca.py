
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

ruta = "/Users/juancamilohernandezjacobo/Desktop/saber11_limpio_base.csv"
df = pd.read_csv(ruta)

X = [
    "periodo",
    "cole_area_ubicacion",
    "cole_naturaleza",
    "cole_calendario",
    "cole_jornada",
    "cole_caracter",
    "cole_bilingue",
    "cole_genero"
]

y = "punt_global"


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)