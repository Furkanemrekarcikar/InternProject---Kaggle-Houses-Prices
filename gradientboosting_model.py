import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from adaboost_model import r2_score
from feature_eng import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

train_processed_df = full_df[full_df["is_train"] == 1].drop("is_train", axis=1)
test_processed_df = full_df[full_df["is_train"] == 0].drop("is_train", axis=1)

train_processed_df = train_processed_df.drop(columns=["MSSubClassfreq", "Neighborhoodfreq","Exterior1stfreq","Exterior2ndfreq", "GarageYrBltfreq"],axis=1)
train_processed_df["SalePrice"] = train_labels



X = train_processed_df.drop("SalePrice", axis=1)
y = train_processed_df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X.head(5)

for col in cat_but_car:
    print(f"{col}: {X_train[col].dtype}")

for col in ['MSSubClass', 'Neighborhood', 'Exterior1st', 'Exterior2nd', 'GarageYrBlt']:
    train_processed_df[col] = train_processed_df[col].astype(str)

train_processed_df["Neighborhood"].dtype

for col in cat_but_car:
    means = y_train.groupby(X_train[col]).mean()

    X_train[col] = X_train[col].map(means)
    X_test[col] = X_test[col].map(means)
    X_test[col] = X_test[col].map(means)

X_train.head(5)

gbm_model = GradientBoostingRegressor(random_state=15)

gmb_grid_params = {
    "learning_rate": [0.1,1],
    "n_estimators": [100, 500],
    "subsample": [0.5, 0.8],
    "max_depth": [2,4],
    "min_samples_split": [2,4]
}

grid_gbm = GridSearchCV(estimator=gbm_model, param_grid=gmb_grid_params, cv=5, n_jobs=-1,verbose=1)
grid_gbm.fit(X_train, y_train)

grid_gbm.best_params_
grid_gbm.best_score_  #0.86

y_pred = grid_gbm.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)
print(rmse)   # 30.497
print(mae)    # 15.107
print(r2_score)