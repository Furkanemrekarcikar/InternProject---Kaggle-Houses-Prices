import numpy as np
import xgboost as xgb
from feature_eng import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

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

xgbm_model = xgb.XGBRegressor(random_state=42)

xgbm_model_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [3,5],
    "n_estimators": [100,400],
    "colsample_bytree": [0.1,0.6]

}

xgbm_grid = GridSearchCV(estimator=xgbm_model,param_grid=xgbm_model_params, cv=3, n_jobs=-1, verbose=1)
xgbm_grid.fit(X_train, y_train)

xgbm_grid.best_params_
xgbm_grid.best_score_  #0.8926

y_pred = xgbm_grid.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("rmse: ",rmse)  #35280
print("mae: ",mae) #27601
print("r2: ",r2)   #0.837
