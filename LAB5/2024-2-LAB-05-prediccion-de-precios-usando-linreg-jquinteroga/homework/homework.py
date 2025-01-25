#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
#_________________________________________________Step 0 Librerias______________________________________________________
import pandas as pd
import os
import pickle
import gzip
import json
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

#_________________________________________________Step 1 Limpieza_______________________________________________________

# Función para cargar datos desde un archivo CSV comprimido
def load_data(file_path, index_col=False, compression="zip"):
    return pd.read_csv(file_path, index_col=index_col, compression=compression)

# Función para calcular la columna 'Age' basada en el año actual
def calculate_age(data, current_year=2021, year_column='Year'):
    data['Age'] = current_year - data[year_column]
    return data

# Función para eliminar columnas no deseadas
def drop_columns(data, columns_to_drop):
    return data.drop(columns=columns_to_drop)

# Cargar datasets de prueba y entrenamiento
test_data = load_data("../files/input/test_data.csv.zip")
train_data = load_data("../files/input/train_data.csv.zip")

# Calcular la columna 'Age' para ambos datasets
test_data = calculate_age(test_data)
train_data = calculate_age(train_data)

# Eliminar las columnas 'Year' y 'Car_Name' de ambos datasets
columns_to_drop = ['Year', 'Car_Name']
test_data = drop_columns(test_data, columns_to_drop)
train_data = drop_columns(train_data, columns_to_drop)

#________________________________________Step 2 División datasets_______________________________________________________

x_train=train_data.drop(columns="Present_Price")
y_train=train_data["Present_Price"]


x_test=test_data.drop(columns="Present_Price")
y_test=test_data["Present_Price"]
#________________________________________Step 3 Creación Pipeline_______________________________________________________

# Función para definir las columnas categóricas y numéricas
def define_features(X_train, categorical_cols):
    return [col for col in X_train.columns if col not in categorical_cols]

# Función para crear el preprocesador
def create_preprocessor(categorical_cols, numerical_cols):
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols),
            ('scaler', MinMaxScaler(), numerical_cols)
        ]
    )

# Función para crear el pipeline
def create_pipeline(preprocessor, feature_selection_func=f_regression, k_best_features=None):
    steps = [
        ("preprocessor", preprocessor),
        ('feature_selection', SelectKBest(feature_selection_func, k=k_best_features)),
        ('classifier', LinearRegression())
    ]
    return Pipeline(steps)

# Definir las columnas categóricas y numéricas
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_features = define_features(x_train, categorical_features)

# Crear el preprocesador
preprocessor = create_preprocessor(categorical_features, numerical_features)

# Crear el pipeline
pipeline = create_pipeline(preprocessor, feature_selection_func=f_regression, k_best_features=None)


#________________________________________Step 4 Optimización parametros ________________________________________________
# Función para definir la cuadrícula de hiperparámetros
def define_param_grid():
    return {
        'feature_selection__k': range(1, 15),  # Número de características a seleccionar
        'classifier__fit_intercept': [True, False],  # Si se debe ajustar la intersección
        'classifier__positive': [True, False]  # Si se deben forzar coeficientes positivos
    }

# Función para crear el modelo de búsqueda de hiperparámetros
def create_grid_search_model(pipeline, param_grid, cv=10, scoring="neg_mean_absolute_error", n_jobs=-1):
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )

# Definir la cuadrícula de hiperparámetros
param_grid = define_param_grid()
# Crear el modelo de búsqueda de hiperparámetros
model = create_grid_search_model(pipeline, param_grid)
# Entrenar el modelo
model.fit(x_train, y_train)

#____________________________________________Step 5 Guardar modelo _____________________________________________________

# Función para guardar un modelo comprimido en un archivo
def save_compressed_model(model, file_path):
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Guardar el modelo comprimido
    with gzip.open(file_path, "wb") as file:
        pickle.dump(model, file)


# Ruta del directorio de modelos
models_dir = '../files/models'

# Ruta del archivo comprimido
compressed_model_path = os.path.join(models_dir, 'model.pkl.gz')

# Guardar el modelo comprimido
save_compressed_model(model, compressed_model_path)

#_______________________________________________Step 6 Matrices ________________________________________________________



# Función para calcular métricas
def calculate_metrics(y_true, y_pred, dataset_name):
    return {
        'type': 'metrics',
        'dataset': dataset_name,
        'r2': float(r2_score(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'mad': float(median_absolute_error(y_true, y_pred))
    }


# Función para guardar métricas en un archivo JSON
def save_metrics_to_file(metrics_list, output_dir, filename):
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Guardar métricas en el archivo
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        for metrics in metrics_list:
            f.write(json.dumps(metrics) + '\n')


# Función principal para calcular y guardar métricas
def calculate_and_save_metrics(model, X_train, X_test, y_train, y_test, output_dir='../files/output',
                               filename='metrics.json'):
    # Hacer predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular métricas para train y test
    metrics_train = calculate_metrics(y_train, y_train_pred, 'train')
    metrics_test = calculate_metrics(y_test, y_test_pred, 'test')

    # Guardar métricas en un archivo JSON
    save_metrics_to_file([metrics_train, metrics_test], output_dir, filename)


# Ejemplo de uso
calculate_and_save_metrics(model, x_train, x_test, y_train, y_test)