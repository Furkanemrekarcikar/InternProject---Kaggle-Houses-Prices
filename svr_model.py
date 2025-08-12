import numpy as np
from feature_eng import *
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import pandas as pd


for col in cat_cols:
    le = LabelEncoder()
    full_df[col] = le.fit_transform(full_df[col])
for col in cat_but_car:
    freq = full_df[col].value_counts() / len(full_df)
    full_df[col + 'freq'] = full_df[col].map(freq)

full_df_final = full_df.drop(columns=['Neighborhood', 'Exterior1st', 'Exterior2nd'])
full_df_final.head(5)

train_processed_df = full_df_final[full_df_final["is_train"] == 1].drop("is_train", axis=1)
test_processed_df = full_df_final[full_df_final["is_train"] == 0].drop("is_train", axis=1)

train_processed_df.head(5)

train_processed_df["SalePrice"] = train_labels
train_processed_df.astype({col: float for col in train_processed_df.select_dtypes(include='bool').columns})

X = train_processed_df.drop("SalePrice", axis=1)
y = train_processed_df["SalePrice"]

X["GarageYrBlt"] = X["GarageYrBlt"].replace("NA", np.nan)
X["GarageYrBlt"] = X["GarageYrBlt"].astype(float)
X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["GarageYrBlt"].median())

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)

svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
y_pred = svr_model.predict(X_test_scaled)

rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
mae_final = mean_absolute_error(y_test, y_pred)

print("rmse: ",rmse_final)
print("mae: ", mae_final)

grid_param = {
    "C" : [0.1, 1, 10, 100, 1000],
    "gamma" : [0.1, 1, 10, 100, 1000],
    "kernel": ["linear", "rbf", "poly"]
}

grid_search = GridSearchCV(estimator=svr_model, param_grid=grid_param, cv=5 , n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

grid_search.best_params_
grid_search.best_score_  # 0.79

svr_model_final = svr_model.set_params(**grid_search.best_params_).fit(X_train_scaled, y_train)
y_pred = svr_model.predict(X_test_scaled)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
mae_final = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

print("rmse: ",rmse_final)  #26563
print("mae: ", mae_final)   #17385
print("r2:", r2)  #0.81


