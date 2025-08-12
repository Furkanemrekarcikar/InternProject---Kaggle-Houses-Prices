import numpy as np
from lightgbm import LGBMRegressor
from feature_eng import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
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

lightgbm_model = LGBMRegressor(random_state=30)

lightgbm_model_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'num_leaves': [15, 30]
}

lightgbm_model_grid = GridSearchCV(estimator=lightgbm_model, param_grid=lightgbm_model_params,cv=5,n_jobs=-1,verbose=1)
lightgbm_model_grid.fit(X_train, y_train)

lightgbm_model_grid.best_params_
lightgbm_model_grid.best_score_  #0.88

lightgbm_model_final = lightgbm_model.set_params(**lightgbm_model_grid.best_params_)

lightgbm_model_final.fit(X_train, y_train)
lightgbm_model_final.score(X_test, y_test)   #0.79

y_pred = lightgbm_model_final.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(rmse)  #39.000
print(mae)   #21.855
print(r2)    #0.7939