from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.hooks.S3_hook import S3Hook

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from typing import Any, Dict, Literal

from datetime import datetime, timedelta
import pandas as pd
import json
import io
import pickle
import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    'owner': 'Simon Matrenok', 
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
        dag_id='Simon_Matrenok',
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        tags=["mlops"],
        default_args=DEFAULT_ARGS,
)

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)



def init() -> Dict[str, Any]:
    configure_mlflow()

    experiment_name = "Simon Matrenok"
    if not mlflow.get_experiment_by_name(experiment_name):
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name="matrs01") as parent_run:
        timestamp = datetime.now().strftime("%Y%m%d %H:%M")
        metrics = {
            'pipeline_start_time': timestamp,
            'experiment_id': experiment_id,
            'parent_run_id': parent_run.info.run_id,  
        }
        return metrics

def get_data(**kwargs) -> Dict[str, Any]:
    start_time = datetime.now().strftime("%Y%m%d %H:%M")

    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    end_time = datetime.now().strftime("%Y%m%d %H:%M")

    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='init')
    metrics['data_load_start_time'] = start_time
    metrics['data_load_end_time'] = end_time
    metrics['data_size'] = df.shape

    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    df.to_pickle(filebuffer)
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"Simon_Matrenok/datasets/raw_data.pkl",
        bucket_name=BUCKET,
        replace=True,
    )

    return metrics

def prepare_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='get_data')


    start_time = datetime.now().strftime("%Y%m%d %H:%M")

    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(
        key=f"Simon_Matrenok/datasets/raw_data.pkl", 
        bucket_name=BUCKET
    )
    df = pd.read_pickle(file)

    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    end_time = datetime.now().strftime("%Y%m%d %H:%M")


    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train, X_test, y_train, y_test],
    ):
        filebuffer = io.BytesIO()
        pickle.dump(data, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"Simon_Matrenok/datasets/{name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

    metrics['data_processing_start_time'] = start_time
    metrics['data_processing_end_time'] = end_time
    metrics['features'] = X.columns.tolist()

    return metrics

def train_model(model_name, **kwargs) -> Dict[str, Any]:
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='prepare_data')

    s3_hook = S3Hook("s3_connection")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=f"Simon_Matrenok/{model_name}/datasets/{name}.pkl",
            bucket_name=BUCKET,
        )
        data[name] = pd.read_pickle(file)


    model = models[model_name]
    experiment_id = metrics['experiment_id']
    parent_run_id = metrics['parent_run_id']

    configure_mlflow()

    start_time = datetime.now().strftime("%Y%m%d %H:%M")

    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name, nested=True, parent_run_id=parent_run_id) as run:
        model.fit(data["X_train"], data["y_train"])

        end_time = datetime.now().strftime("%Y%m%d %H:%M")
        
        signature = infer_signature(data["X_test"], model.predict(data["X_test"]))
        model_info = mlflow.sklearn.log_model(model, f'{model_name}_model', signature=signature)
        
        eval_df = pd.DataFrame(data["X_test"], columns=metrics['features'])
        eval_df["target"] = data["y_test"]
        eval_df = eval_df.dropna()
        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )
        
        metrics['training_start_time'] = start_time
        metrics['training_end_time'] = end_time
        metrics[f'{model_name}_run_id'] = run.info.run_id 

    return {model_name: metrics}


def save_results(**kwargs) -> None:
    ti = kwargs['ti']
    model_results = {}
    
    for model_name in model_names:
        result = ti.xcom_pull(task_ids=f'train_model_{model_name}')
        model_results.update(result)  

    s3_hook = S3Hook("s3_connection")
    date = datetime.now().strftime("%Y_%m_%d_%H")
    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(model_results).encode())
    filebuffer.seek(0)
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"Simon_Matrenok/results/{date}.json",
        bucket_name=BUCKET,
        replace=True,
    )


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag, provide_context=True)

task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag, provide_context=True)

training_model_tasks = []
for model_name in model_names:
    training_model_tasks.append(
        PythonOperator(
            task_id=f"train_model_{model_name}",
            python_callable=train_model,
            op_kwargs={'model_name': model_name},  
            dag=dag,
            provide_context=True
        )
    )

task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag, provide_context=True)

task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results