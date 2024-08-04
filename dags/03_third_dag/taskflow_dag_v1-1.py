import sys
from airflow.decorators import dag
from datetime import datetime

PATH_COMMON = '../'
sys.path.append(PATH_COMMON)

from common.add_task import sum_task

@dag(
    dag_id='taskflow_dag_v1-1',
    start_date=datetime(2022, 1, 1), 
    schedule=None
)
def mydag():
    start = sum_task(1, 2)
    for i in range(3):
        start = sum_task(start, i + 1)

first_dag = mydag()