import sys
from airflow.decorators import dag, task
from datetime import datetime

PATH_COMMON = '../'
sys.path.append(PATH_COMMON)

from common.add_task import task_virtualenv

@task.docker(image="datapath:latest")
def task_docker():

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


@dag(
    dag_id='taskflow_dag_v1-4',
    start_date=datetime(2022, 1, 1),
    schedule=None
)
def mydag():
    
    task_virtualenv()
    task_docker()


first_dag = mydag()