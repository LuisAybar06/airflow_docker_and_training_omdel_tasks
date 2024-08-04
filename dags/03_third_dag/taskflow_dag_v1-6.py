import sys
from airflow.decorators import dag, task
from datetime import datetime

PATH_COMMON = '../'
sys.path.append(PATH_COMMON)

from common.add_task import task_predictionnmodel


@dag(
    dag_id='taskflow_dag_v1-6',
    start_date=datetime(2022, 1, 1),
    schedule=None
)
def mydag():
    
    task_predictionnmodel()


first_dag = mydag()