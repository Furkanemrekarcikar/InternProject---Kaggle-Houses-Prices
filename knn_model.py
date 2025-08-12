import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from feature_eng import *
import numpy as np

pd.set_option('future.no_silent_downcasting', True)


X = train_processed_df.drop(["SalePrice"], axis=1)
y = train_processed_df["SalePrice"]

X["GarageYrBlt"] = X["GarageYrBlt"].replace("NA", np.nan)
X["GarageYrBlt"] = X["GarageYrBlt"].astype(float)
X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["GarageYrBlt"].median())

X.head(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)


knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("RMSE:", rmse) # 39918
print("MAE :", mae)  # 24805

knn_model = KNeighborsRegressor()
knn_model.get_params()

knn_params = {"n_neighbors" : range(2,50)}

knn_grid_best = GridSearchCV(estimator=knn_model,
                             param_grid=knn_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=1).fit(X_train_scaled, y_train)
knn_grid_best.best_params_  # n = 11
knn_grid_best.best_score_   #0.748
knn_final = knn_model.set_params(**knn_grid_best.best_params_).fit(X_train_scaled, y_train)
y_pred = knn_final.predict(X_test_scaled)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
mae_final = mean_absolute_error(y_test, y_pred)
r2_scoree = r2_score(y_test, y_pred)

print("RMSE:", rmse_final)
print("MAE :", mae_final)
print("R2 score:", r2_scoree)  # 0.80

