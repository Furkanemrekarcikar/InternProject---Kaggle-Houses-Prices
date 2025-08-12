import numpy as np
from feature_eng import *
from sklearn.tree import DecisionTreeRegressor
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



cart_model = DecisionTreeRegressor(random_state=17)

car_grid_params = {'max_depth': [2, 4, 6, 8, 10, 12, 14],
                   'min_samples_split': [2, 4, 6, 8, 12, 14],
                   'min_samples_leaf': [2, 4, 6, 8, 10 ,12],
                   'max_leaf_nodes': [2, 4, 6, 8, 10, 12, 14],
                   'max_features': [2, 4, 6, 8,10,12,14]
                                                                }
cart_grid = GridSearchCV(estimator=cart_model, param_grid= car_grid_params, cv=5, n_jobs=-1, verbose=1)
cart_grid.fit(X_train, y_train)  #61740 fits

cart_grid.best_params_
cart_grid.best_score_  #0.7503

cart_model_final = cart_model.set_params(**cart_grid.best_params_).fit(X_train, y_train)

y_pred = cart_model_final.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("rmse: ", rmse)  # 40.383
print("mae: ", mae)    # 28752