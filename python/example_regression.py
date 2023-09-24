from StageML import StageMLRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

california_dataset = fetch_california_housing()
california_df = pd.DataFrame(california_dataset.data, columns=california_dataset.feature_names)
california_df['MedHouseVal'] = california_dataset.target

def mse(y_true, y_pred):
    """
    Calculates MSE between y_true and y_pred.
    """
    # Calculate MSE 
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse
#New Model
stage_ml_regressor = StageMLRegressor(objective = "rmse")

#
features = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude" ]
target_col = "MedHouseVal"

X_train, X_test, y_train, y_test = train_test_split(california_df[features], california_df[target_col], test_size=0.2, random_state=42)



stage_ml_regressor.fit_glm(
            X = X_train,
            y = y_train,
            features=features)

stage_ml_regressor.fit_gam(
            X = X_train,
            y = y_train,
            features=features)

stage_ml_regressor.fit_ga2m(
            X = X_train,
            y = y_train,
            features=features)

stage_ml_regressor.fit_lgbm(
            X = X_train,
            y = y_train,
            features=features)


mse(y_test, stage_ml_regressor.predict(X= X_test,model_name="glm_model"))
mse(y_test, stage_ml_regressor.predict(X= X_test,model_name="gam_model"))
mse(y_test, stage_ml_regressor.predict(X= X_test,model_name="ga2m_model"))
mse(y_test, stage_ml_regressor.predict(X= X_test,model_name="lgbm_model"))

gam_model = StageMLRegressor()
gam_model.fit_ga2m(
            X = X_train,
            y = y_train,
            features=features)
#Testing
mse(y_test, stage_ml_regressor.predict(X= X_test,model_name="gam_model"))
mse(y_test, gam_model.predict(X= X_test,model_name="gam_model"))






init_score = stage_ml_regressor.predict(X= california_df[features],model_name="glm_model")


model_name, init_score = self._prepare_fit(X,features,model_name)
# Fit Models
from interpret.glassbox import ExplainableBoostingClassifier,ExplainableBoostingRegressor # EBM

model = ExplainableBoostingRegressor(
    max_bins=52,
    interactions = 0,
    objective = stage_ml_regressor.ebm_objective
    )
model.fit(X=california_df[features], 
          y = california_df["MedHouseVal"],
          sample_weight = None,
          init_score = init_score)
#Update Object
a = model.predict(X= california_df[features])
a + init_score
self._update_model_features(model_name,features)

model.term_scores_[0]
stage_ml_regressor.models["gam_model"].term_scores_[0]
