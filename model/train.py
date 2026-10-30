# %%
import pandas as pd
from sklearn import model_selection
from sklearn import tree
from feature_engine import discretisation,  encoding
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import ensemble

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

X, y = df_train[features].drop(columns = ["street"]), df_train[target]

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
    "sqft_basement",
]

variaveis_onehot = ["city", "statezip"] + variaveis_discretizar

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
    ("modelo", ensemble.RandomForestRegressor(random_state = 42)),
])

grid = model_selection.GridSearchCV(
    estimator = modelo_pipeline,
    param_grid = params,
    cv = 5,
    scoring = "neg_mean_absolute_error",
    n_jobs = -1,
    verbose = 2,
)

grid.fit(X_train, y_train)

# Assess
y_train_pred = grid.predict(X_test)
y_train_proba = grid.predict_proba(X_train)[:,1]



print("Melhores parâmetros:", grid.best_params_)
print("MAE no teste:", metrics.mean_absolute_error(y_test, y_test_pred))
# %%
