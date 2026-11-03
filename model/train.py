# %%
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import tree
from feature_engine import discretisation,  encoding
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import ensemble
from sklearn.dummy import DummyRegressor

mlflow.set_tracking_uri("http://127.0.0.1:5001/")
mlflow.set_experiment(experiment_id = "2")

df = pd.read_csv("../data/abt.csv", parse_dates = ["date"])
df.head()
# %%
# Out-Of-Time

oot = df[df['date'] > "2014-06-26"].copy()
df_train = df[df['date'] <= "2014-06-26"].copy()

# %%
# Features

features = df_train.columns[2:-1]
target = 'price'

X = df_train[features].drop(columns = ["street", "sqft_basement", "city", "statezip", "view", "yr_built"])
y = df_train[target]

# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    random_state = 42,
    test_size = 0.2,
)

print("Taxa variável resposta Geral:", y.mean())
print("Taxa variável resposta Treino:", y_train.mean())
print("Taxa variável resposta Test:", y_test.mean())

# %%
# Explore

X_train.isna().sum().sort_values(ascending = False)
# %%
# Modify / Model

variaveis_discretizar = [
    "sqft_living",
    "sqft_lot",
    "sqft_above",
]

variaveis_onehot = variaveis_discretizar

params = {
    "modelo__max_depth": [3, 5, 10, None],
    "modelo__min_samples_leaf": [1, 5, 10],
    "modelo__n_estimators": [100, 200],
}

modelo_pipeline = Pipeline([
    ("Discretizacao", discretisation.DecisionTreeDiscretiser(
        variables = variaveis_discretizar,
        bin_output = "bin_number",
        regression = True,
        random_state = 42,
    )),
    ("OneHot", encoding.OneHotEncoder(
        variables = variaveis_onehot,
        ignore_format = True,
    )),
    ("modelo", ensemble.RandomForestRegressor(
        random_state = 42,
    )),
])

grid = model_selection.GridSearchCV(
    estimator = modelo_pipeline,
    param_grid = params,
    cv = 5,
    scoring = "neg_mean_absolute_error",
    n_jobs = -2,
    verbose = 4,
)

with mlflow.start_run(
    run_name = str(modelo_pipeline.named_steps["modelo"].__class__.__name__
)):
    mlflow.sklearn.autolog(registered_model_name = "model_price")
    grid.fit(X_train, y_train)

    # Assess
    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)
    y_oot_pred = grid.predict(oot[features].drop(columns = ["street", "sqft_basement", "city", "statezip", "view", "yr_built"]))

    mae_train = metrics.mean_absolute_error(y_train, y_train_pred)
    mae_test = metrics.mean_absolute_error(y_test, y_test_pred)
    mae_oot = metrics.mean_absolute_error(oot[target], y_oot_pred)

    print("\nMelhores parâmetros:", grid.best_params_)
    print(f"MAE no Treino: {mae_train:.2F}")
    print(f"MAE no Teste: {mae_test:.2F}")
    print(f"MAE no OOT: {mae_oot:.2F}")

    mlflow.log_metrics({
        "mae_train":mae_train,
        "mae_test":mae_test,
        "mae_oot":mae_oot,
    })
# %%
plt.bar(
    ["Treino", "Teste", "OOT"],
    [mae_train, mae_test, mae_oot],
)
plt.ylabel("MAE")
plt.show()

# %%
