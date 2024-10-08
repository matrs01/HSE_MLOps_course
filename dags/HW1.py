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



BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    'owner': 'Simon Matrenok', 
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}
model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


def create_dag(dag_id: str, m_name: Literal["random_forest", "linear_regression", "desicion_tree"]):

    ####### DAG STEPS #######

    def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d %H:%M")
        return {'pipeline_start_time': timestamp, 'model_name': m_name}

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
            key=f"SimonMatrenok/{metrics['model_name']}/datasets/raw_data.pkl",
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
            key=f"SimonMatrenok/{metrics['model_name']}/datasets/raw_data.pkl", 
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
                key=f"SimonMatrenok/{metrics['model_name']}/datasets/{name}.pkl",
                bucket_name=BUCKET,
                replace=True,
            )

        metrics['data_processing_start_time'] = start_time
        metrics['data_processing_end_time'] = end_time
        metrics['features'] = X.columns.tolist()

        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='prepare_data')

        s3_hook = S3Hook("s3_connection")
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(
                key=f"SimonMatrenok/{metrics['model_name']}/datasets/{name}.pkl",
                bucket_name=BUCKET,
            )
            data[name] = pd.read_pickle(file)


        start_time = datetime.now().strftime("%Y%m%d %H:%M")

        model = models[metrics['model_name']]
        model.fit(data["X_train"], data["y_train"])

        end_time = datetime.now().strftime("%Y%m%d %H:%M")

        prediction = model.predict(data["X_test"])

        metrics["r2_score"] = r2_score(data["y_test"], prediction)
        metrics["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
        metrics["mae"] = median_absolute_error(data["y_test"], prediction)

        metrics['training_start_time'] = start_time
        metrics['training_end_time'] = end_time

        return metrics


    def save_results(**kwargs) -> None:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='train_model')

        s3_hook = S3Hook("s3_connection")
        date = datetime.now().strftime("%Y_%m_%d_%H")
        filebuffer = io.BytesIO()
        filebuffer.write(json.dumps(metrics).encode())
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"SimonMatrenok/{metrics['model_name']}/results/{date}.json",
            bucket_name=BUCKET,
            replace=True,
        )

    ####### INIT DAG #######

    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        tags=["mlops"],
        default_args=DEFAULT_ARGS,
    )

    with dag:
        task_init = PythonOperator(task_id="init", python_callable=init, op_kwargs={'m_name': m_name}, dag=dag)

        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag, provide_context=True)

        task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag, provide_context=True)

        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag, provide_context=True)

        task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag, provide_context=True)

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():
    create_dag(f"Simon_Matrenok_{model_name}", model_name)