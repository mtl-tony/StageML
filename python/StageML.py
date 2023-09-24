# Core data science packages
import numpy as np
import pandas as pd
from typing import Optional
from copy import deepcopy
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 

# Model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Explainability
import eli5
from pdpbox import pdp
import shap 

# Linear Models
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, LinearRegression,LogisticRegression #GLM
from sklearn.ensemble import GradientBoostingClassifier # GBM 
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
# Explainable Boosting Machines
from interpret.glassbox import ExplainableBoostingClassifier,ExplainableBoostingRegressor # EBM
# Gradient Boosting Machine
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor

# Ignoring warnings
from abc import ABC, abstractmethod
from utils.inv_link import inv_log, inv_identity, inv_logistic



class StageML:
    """
    An abstract base class for Staged Machine Learning models.

    The purpose of this model is to handle different stages of model fitting and
    prediction, including linear models, Generalized Additive Models (GAM), 
    Generalized Additive Model of location, scale and shape (GA2M), and LightGBM.

    Parameters
    ----------
    feature_names: list
        List of feature names.
    n_jobs: int, optional
        Number of jobs to run in parallel. Negative integers are interpreted as 
        following joblib's formula (n_cpus + 1 + n_jobs), just like scikit-learn. 
        Eg: -2 means using all threads except 1.
    feature_types: list, optional
        List of feature types.
    random_state: int, optional
        Random state for reproducibility. None uses device_random and generates 
        non-repeatable sequences.

    Attributes
    ----------
    models: dict
        Dictionary to store fitted models.
    features: dict
        Dictionary to store features used in each model.
    model_stages: list
        List to maintain the order of model stages.
    """
    def __init__(
        self, 
        feature_names:list,
        n_jobs: Optional[int] = -2,
        feature_types = None,
        random_state = 42):

        self.feature_names = feature_names
        self.feature_types = feature_types
        self.n_jobs = n_jobs
        self.random_state = random_state
        #Models
        self.models = {}
        self.features = {}
        self.model_stages=[]

    def _validate_data(self,X,features):
        missing_cols = set(features) - set(X.columns)
        if len(missing_cols) > 0:
            raise Exception("Missing columns: {}".format(list(missing_cols)))
        else:
            print("All required columns present")

    def _check_model_name(self,model_name):
        index = 0
        #Updating Model Name if
        if model_name in self.models.keys():
            print("Model name already exists. Updating Model Name with index")
        while model_name in self.models.keys():
            model_name = f"{model_name}_{index}"
            index += 1
        return model_name

    def _objective_handler(self):
        match self.objective:
            case "rmse":
                self.linear_model = LinearRegression()
                self.inv_link = inv_identity
            case "poisson":
                self.linear_model = PoissonRegressor()
                self.inv_link = inv_log
            case "gamma":
                self.linear_model = TweedieRegressor(power = 1, link = "log")
                self.inv_link = inv_log
            case "logloss":
                self.linear_model = LogisticRegression()
                self.inv_link = inv_logistic

    def _can_fit_glm(self):
        #Can only fit if it's first model
        return len(self.models) == 0

    def _get_feature_types(self, X):
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return numeric_features, categorical_features

    @abstractmethod
    def fit_glm():
        pass

    @abstractmethod
    def fit_gam():
        pass

    @abstractmethod
    def fit_ga2m():
        pass

    @abstractmethod
    def fit_lgbm():
        pass

class StageMLRegressor(StageML):
    """
    A Regressor model for Staged Machine Learning.

    The purpose of this model is to handle different stages of model fitting and
    prediction, including linear models, Generalized Additive Models (GAM), 
    Generalized Additive Model of location, scale and shape (GA2M), and LightGBM.

    Parameters
    ----------
    feature_names: list
        List of feature names.
    n_jobs: int, optional
        Number of jobs to run in parallel.
    objective: str, optional
        Objective function for the model.
    feature_types: list, optional
        List of feature types.
    random_state: int, optional
        Random state for reproducibility.

    Attributes
    ----------
    models: dict
        Dictionary to store fitted models.
    features: dict
        Dictionary to store features used in each model.
    model_stages: list
        List to maintain the order of model stages.
    objective: str
        Objective function for the model.
    linear_model: object
        Instance of linear model.
    inv_link: function
        Inverse link function.
    """
    def __init__(
        self, 
        feature_names:list,
        n_jobs: Optional[int] = -2,
        objective : str = "rmse",
        feature_types = None,
        random_state = 42):

        super().__init__(
            feature_names = feature_names,
            n_jobs = n_jobs,
            feature_types = feature_types,
            random_state = random_state)
        #Setting the Objective
        self.objective = objective
        self._objective_handler()

    def _prepare_init_score(self,X):
        if len(self.models) == 0:
            init_score = None
        else:
            init_score = np.zeros(X.shape[0] )
            for model_name in self.model_stages:
                init_score += self.inv_link(self.models[model_name].predict(X[self.features[model_name]]))
        return init_score
    
    def _prepare_fit(self,X,features,model_name):
        #Validate_Data
        self._validate_data(X,features)
        #Init Score
        init_score = self._prepare_init_score(X=X)
        #Update Model Name
        model_name = self._check_model_name(model_name)
        return model_name, init_score

    def fit_glm(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
            sample_weight = None,
            model_name:str = "glm_model"):
        if self._can_fit_glm():
            #Prepare Fitting
            model_name, init_score = self._prepare_fit(X,features,model_name)
            #Numeric + Categoric Features
            numeric_features, categorical_features = self._get_feature_types(X[features])
            #Fit Model
            # One-hot encoding for categorical features
            preprocessor = make_column_transformer(
                (StandardScaler(), ~X[features].columns.isin(categorical_features)),
                (OneHotEncoder(), categorical_features)
            )
            # Create pipeline with preprocessor and model

            self.models[model_name] = make_pipeline(preprocessor, deepcopy(self.linear_model))
            self.models[model_name].fit(X = X[features],y = y,sample_weight = sample_weight)
            self.features[model_name] = X.columns.tolist()
            self.model_stages = self.model_stages + [model_name]
        else:
            print("Not Fitting GLM. No init_score Parameter for this model, can only be first model fit")

    def fit_gam(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
            sample_weight = None,
            model_name:str = "gam_model"):
        #Prepare Fitting
        model_name, init_score = self._prepare_fit(X,features,model_name)
        # Fit Models
        self.models[model_name] = ExplainableBoostingRegressor(
            max_bins=52,
            interactions = 0,
            objective = self.objective
            )
        self.models[model_name].fit(X= X[features], y = y,sample_weight = sample_weight,init_score = init_score)
        #Update Object
        self.features = X.columns.tolist()
        self.model_stages = self.model_stages + [model_name]

    def fit_ga2m(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
            sample_weight = None,
            model_name:str = "ga2m_model",
            interactions = 10):
        #Prepare Fitting
        model_name, init_score = self._prepare_fit(X,features,model_name)
        # Fit Models
        self.models[model_name] = ExplainableBoostingRegressor(
            max_interaction_bins=10,
            interactions = interactions,
            objective = self.objective,
            exclude = list(range(len(features)))
            )
        self.models[model_name].fit(X= X[features], y = y,sample_weight = sample_weight,init_score = init_score)
        #Update Object
        self.features = X.columns.tolist()
        self.model_stages = self.model_stages + [model_name]

    def fit_lgbm(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
            sample_weight = None,
            model_name:str = "lgbm_model"):
        #Prepare Fitting
        model_name, init_score = self._prepare_fit(X,features,model_name)
        # Fit Models
        self.models[model_name] = LGBMRegressor(
            objective = self.objective            
            )
        self.models[model_name].fit(X= X[features], y = y,sample_weight = sample_weight,init_score = init_score)
        #Update Object
        self.features = X.columns.tolist()
        self.model_stages = self.model_stages + [model_name]