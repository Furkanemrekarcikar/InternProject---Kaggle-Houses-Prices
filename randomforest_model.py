import numpy as np
from sklearn.ensemble import RandomForestRegressor
from feature_eng import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import shap

X = train_processed_df.drop(["SalePrice"], axis=1)
y = train_processed_df["SalePrice"]

X["GarageYrBlt"] = X["GarageYrBlt"].replace("NA", np.nan)
X["GarageYrBlt"] = X["GarageYrBlt"].astype(float)
X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["GarageYrBlt"].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X.head(5)

rf_model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4]
}

rf_model_grid = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5,n_jobs=-1,verbose=1)
rf_model_grid.fit(X_train, y_train)

rf_model_grid.best_params_
rf_model_grid.best_score_   #0.85

rf_model_final = rf_model.set_params(**rf_model_grid.best_params_)

rf_model_final.fit(X_train, y_train)

X_sample = X_test.sample(200, random_state=42)
explainer = shap.TreeExplainer(rf_model_final)
shap_values = explainer.shap_values(X_sample)
explainer = shap.TreeExplainer(rf_model_final, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_sample, approximate=True)
shap.summary_plot(shap_values, X_sample, plot_type="bar")
shap.summary_plot(shap_values, X_sample)


force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_sample.iloc[0, :]
)

shap.save_html("force_plot.html", force_plot)


y_pred = rf_model_final.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(rmse) #29530
print(mae)  #17560
print(r2)   #0.88