# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
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
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

#_______________________________________________________________________________________________________________________
#_________________________________________________Step 0 Librerias______________________________________________________
import pandas as pd
import numpy as np
import os
import pickle
import gzip
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

#_________________________________________________Step 1 Limpieza_______________________________________________________
data_train = pd.read_csv('../files/input/train_data.csv.zip', index_col = False, compression = "zip")
data_test = pd.read_csv("../files/input/test_data.csv.zip", index_col = False, compression = "zip")

# Función para renombrar columnas
def rename_columns(data, old_name, new_name):
    data.rename(columns={old_name: new_name}, inplace=True)

# Función para eliminar columnas
def drop_columns(data, columns_to_drop):
    data.drop(columns=columns_to_drop, inplace=True)

# Función para limpiar valores en una columna
def clean_column(data, column_name, condition, replacement):
    data[column_name] = data[column_name].apply(lambda x: replacement if condition(x) else x)

# Renombrar columnas en ambos datasets
rename_columns(data_train, "default payment next month", "default")
rename_columns(data_test, "default payment next month", "default")

# Eliminar la columna 'ID' en ambos datasets
drop_columns(data_train, ['ID'])
drop_columns(data_test, ['ID'])

# Limpiar las columnas 'EDUCATION' y 'MARRIAGE' en ambos datasets
for column in ['EDUCATION', 'MARRIAGE']:
    clean_column(data_train, column, lambda x: x <= 0, np.nan)
    clean_column(data_test, column, lambda x: x <= 0, np.nan)

# Asegurar que los valores en 'EDUCATION' no sean mayores que 4
clean_column(data_train, 'EDUCATION', lambda x: x > 4, 4)
clean_column(data_test, 'EDUCATION', lambda x: x > 4, 4)

# Eliminar filas con valores nulos en ambos datasets
data_train.dropna(inplace=True)
data_test.dropna(inplace=True)
#________________________________________Step 2 División datasets_______________________________________________________
x_train = data_train.drop(columns=['default'])
y_train = data_train["default"]
x_test = data_test.drop(columns=['default'])
y_test = data_test["default"]

#________________________________________Step 3 Creación Pipeline_______________________________________________________
# Definir las columnas categóricas
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']

# Función para crear el preprocesador
def create_preprocessor(categorical_cols):
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )

# Función para crear el clasificador
def create_classifier(random_state=42):
    return RandomForestClassifier(random_state=random_state)

# Función para crear el pipeline
def create_pipeline(preprocessor, classifier):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', classifier)
    ])

# Crear el preprocesador
preprocesamiento = create_preprocessor(categorical_features)

# Crear el clasificador
clf = create_classifier(random_state=42)

# Crear el pipeline
pipeline = create_pipeline(preprocesamiento, clf)
#________________________________________Step 4 Optimización parametros ________________________________________________

param_grid = {
  'clf__n_estimators': [200],
  'clf__max_depth': [45],
  'clf__min_samples_split': [8],
  'clf__max_features': ['sqrt']
}

modelo = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy', n_jobs=-1)

modelo.fit(x_train, y_train)
#____________________________________________Step 5 Guardar modelo _____________________________________________________

dir_path = '../files/models'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    with gzip.open('../files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(modelo, f)
else:
    with gzip.open('../files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(modelo, f)


#_______________________________________________Step 6 Matrices ________________________________________________________
# Función para calcular métricas
def calculate_metrics(y_true, y_pred, dataset_name):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

# Función para guardar métricas en un archivo JSON
def save_metrics_to_file(metrics_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for metrics in metrics_list:
            json.dump(metrics, f, ensure_ascii=False)
            f.write('\n')

# Obtener predicciones
y_train_pred = modelo.predict(x_train)
y_test_pred = modelo.predict(x_test)

# Calcular métricas para train y test
train_metrics = calculate_metrics(y_train, y_train_pred, "train")
test_metrics = calculate_metrics(y_test, y_test_pred, "test")

# Guardar métricas en un archivo JSON
output_path = '../files/output/metrics.json'
save_metrics_to_file([train_metrics, test_metrics], output_path)
#___________________________________Step 7 Calculo de Confusión ________________________________________________________
# Función para crear un diccionario de matriz de confusión
def create_confusion_matrix_dict(cm, dataset_name):
    return {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {'predicted_0': int(cm[0, 0]), 'predicted_1': int(cm[0, 1])},
        'true_1': {'predicted_0': int(cm[1, 0]), 'predicted_1': int(cm[1, 1])}
    }

# Función para guardar diccionarios en un archivo JSON
def append_to_json_file(data_list, output_path):
    with open(output_path, 'a', encoding='utf-8') as f:
        for data in data_list:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

# Calcular matrices de confusión
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Crear diccionarios de matrices de confusión
train_cm_dict = create_confusion_matrix_dict(train_cm, 'train')
test_cm_dict = create_confusion_matrix_dict(test_cm, 'test')

# Guardar diccionarios en el archivo JSON
output_path = '../files/output/metrics.json'
append_to_json_file([train_cm_dict, test_cm_dict], output_path)