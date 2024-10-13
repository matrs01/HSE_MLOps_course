from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository


data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


experiment_name = "Simon Matrenok"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

model_names = ["random_forest", "linear_regression", "desicion_tree"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
    ]))


with mlflow.start_run(run_name="matrs01") as parent_run:
    for model_name in models.keys():
        with mlflow.start_run(run_name=model_name, nested=True) as child_run:
            model = models[model_name]
            model.fit(X_train, y_train)
        
            signature = infer_signature(X_test, model.predict(X_test))
            model_info = mlflow.sklearn.log_model(model, f'{model_name}_model', signature=signature)

            eval_df = pd.DataFrame(X_test, columns=X.columns)
            eval_df["target"] = y_test
            eval_df = eval_df.dropna()

            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )