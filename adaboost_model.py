import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from feature_eng import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Kategorik sütunları LabelEncoder ile dönüştür
for col in cat_cols:  # cat_cols: kategorik sütun adları listesi
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

ada_model = AdaBoostRegressor(random_state=17)

ada_model_params = {
    "learning_rate": [0.1,1],
    "n_estimators": [100,200]
}

ada_grid = GridSearchCV(estimator=ada_model, param_grid=ada_model_params, cv=5, n_jobs=-1, verbose=1)
ada_grid.fit(X_train, y_train)

ada_grid.best_params_
ada_grid.best_score_  #0.80

ada_model_final = ada_model.set_params(**ada_grid.best_params_)

ada_model_final.fit(X_train, y_train)
y_pred = ada_model_final.predict(X_test)

ada_model_final.score(X_test, y_test)   # 0.81

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)

print(rmse)   #33.673
print(mae)    #23.881
print(r2_score)  #81.58