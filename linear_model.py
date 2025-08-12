from sklearn.model_selection import train_test_split, cross_val_score
from feature_eng import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np


X = train_processed_df.drop("SalePrice", axis=1)
y = train_processed_df["SalePrice"]

X["GarageYrBlt"] = X["GarageYrBlt"].replace("NA", np.nan)
X["GarageYrBlt"] = X["GarageYrBlt"].astype(float)
X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["GarageYrBlt"].median())


X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)
print("RMSE: ", rmse)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("MAE: ", mae)
print("RMSE: ", rmse)
print("MAPE: ", mae)

neg_mse_scores = cross_val_score(reg_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

rmse_scores = np.sqrt(-neg_mse_scores)

print("Katlama Başına RMSE:", rmse_scores)
print("Ortalama RMSE:", rmse_scores.mean())