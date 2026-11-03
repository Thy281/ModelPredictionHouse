# %%
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5001/")

registered_models = mlflow.search_registered_models(
    filter_string="name = 'model_price'"
)

if not registered_models or not registered_models[0].latest_versions:
    raise ValueError("O modelo 'model_price' não foi encontrado no MLflow Registry.")

latest_version = max(
    registered_models[0].latest_versions,
    key = lambda version: int(version.version),
)

model = mlflow.sklearn.load_model(
    f"models:/model_price/{latest_version.version}"
)

data_path = Path(__file__).resolve().parent.parent / "data" / "abt.csv"
df = pd.read_csv(data_path, parse_dates=["date"])

dados_novos = df.loc[df["date"] > pd.Timestamp("2014-06-26")].copy()
features = list(model.feature_names_in_)

dados_novos["predicted_price"] = model.predict(dados_novos[features])

predicoes = dados_novos[["date", "predicted_price"]]
print(predicoes.head())

# %%
