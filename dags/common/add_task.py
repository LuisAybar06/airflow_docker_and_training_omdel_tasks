from airflow.decorators import task, dag
import logging
from datetime import datetime, timedelta

@task
def sum_task(x,y):
    print(f"Task arg: x={x}, y={y}")
    return x+y

@task.virtualenv(
    task_id="virtualenv_python", requirements=["pandas", "numpy", "scikit-learn"], system_site_packages=False
)
def task_virtualenv():
    import numpy as np
    from sklearn.model_selection import train_test_split
    import sys
    print("Python Version")
    print(sys.version)
    print("-----------")
    
    print("Init")
    x = np.arange(1, 25).reshape(12, 2)
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print('Train:')
    print(x_train)
    print('Test:')
    print(x_test)
    print("Finished")


@task.virtualenv(
    task_id="training_model", 
    requirements=[
        "pandas", 
        "numpy", 
        "scikit-learn",
        "joblib"
        ], 
    system_site_packages=False
)
def task_trainmodel():
    import sys
    from datetime import datetime
    from sklearn.linear_model import Lasso
    import pandas as pd
    import joblib

    PATH_COMMON = '../'
    sys.path.append(PATH_COMMON)

    X_train = pd.read_csv('/opt/airflow/dags/data/inputs/xtrain.csv') 
    y_train = pd.read_csv('/opt/airflow/dags/data/inputs/ytrain.csv') 

    # Seleccionar características
    features = pd.read_csv('/opt/airflow/dags/data/inputs/selected_features.csv')
    features = features['0'].to_list()

    X_train = X_train[features]

    # Configurar el modelo
    lin_model = Lasso(alpha=0.001, random_state=0)

    # Entrenar el modelo
    lin_model.fit(X_train, y_train)

    # Guardar el modelo
    joblib.dump(lin_model, '/opt/airflow/dags/data/model/linear_regression.joblib')

    print("Modelo Guardado")


@task.virtualenv(
    task_id="prediction_model", 
    requirements=[
        "pandas", 
        "numpy", 
        "scikit-learn",
        "joblib"
        ], 
    system_site_packages=False
)
def task_predictionnmodel():
    import sys
    from datetime import datetime
    from sklearn.linear_model import Lasso
    from joblib import load
    import pandas as pd
    import joblib

    PATH_COMMON = '../'
    sys.path.append(PATH_COMMON)

    X_train = pd.read_csv('/opt/airflow/dags/data/inputs/xtrain.csv')
    classifier = load("/opt/airflow/dags/data/model/linear_regression.joblib")

    # Seleccionar características
    features = pd.read_csv('/opt/airflow/dags/data/inputs/selected_features.csv')
    features = features['0'].to_list()
    X_train = X_train[features]


    #Realizamos la prediccion
    predictions = classifier.predict(X_train)

    predictions = pd.DataFrame(predictions, columns=['prediction'])
    #Guardamos el resultado
    predictions.to_csv('/opt/airflow/dags/data/outputs/predictions.csv', index=False)

    print("Prediccion Generada")




