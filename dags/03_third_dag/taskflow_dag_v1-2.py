import sys
from airflow.decorators import dag
from datetime import datetime

PATH_COMMON = '../'
sys.path.append(PATH_COMMON)

from common.add_task import task_virtualenv

@dag(
    dag_id='taskflow_dag_v1-2',
    start_date=datetime(2022, 1, 1),
    schedule=None
)
def mydag():

    task_virtualenv()


first_dag = mydag()