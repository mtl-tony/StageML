# Core data science packages
import numpy as np
import pandas as pd
from typing import Optional
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

# Machine learning models
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson, Gaussian, Binomial,Gamma
from statsmodels.genmod.families.links import log, identity, logit
from statsmodels.genmod import GLM
from sklearn.linear_model import LogisticRegression, SGDClassifier # GLM
from sklearn.ensemble import GradientBoostingClassifier # GBM 
from interpret.glassbox import ExplainableBoostingClassifier,ExplainableBoostingRegressor # EBM

from sklearn.linear_model import PoissonRegressor, TweedieRegressor, LinearRegression
from copy import deepcopy


# Additional useful modeling packages
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor

# Ignoring warnings
from abc import ABC, abstractmethod



class StageML:
    """A model for Staged ML
    
    Parameters
    ----------
    feature_names : list of str, default=None
        List of feature names.
    feature_types : list of FeatureType, default=None
    n_jobs : int, default=-2
        Number of jobs to run in parallel. Negative integers are interpreted as following joblib's formula
        (n_cpus + 1 + n_jobs), just like scikit-learn. Eg: -2 means using all threads except 1.
    random_state : int or None, default=42
        Random state. None uses device_random and generates non-repeatable sequences.
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
    def fit_gbm():
        pass

class StageMLRegressor(StageML):
    """A Regressor model for Staged ML"""
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

    def _objective_handler(self):
        match self.objective:
            case "rmse":
                self.linear_model = LinearRegression()
            case "poisson":
                self.linear_model = PoissonRegressor()
            case "gamma":
                self.linear_model = TweedieRegressor(power = 1, link = "log")
    def _check_model_name(self,model_name):
        index = 0
        #Updating Model Name if
        if model_name in self.models.keys():
            print("Model name already exists. Updating Model Name with index")
        while model_name in self.models.keys():
            model_name = f"{model_name}_{index}"
            index += 1
        return model_name
    
    def _validate_data(self,X,features):
        missing_cols = set(features) - set(X.columns)
        if len(missing_cols) > 0:
            raise Exception("Missing columns: {}".format(list(missing_cols)))
        else:
            print("All required columns present")

    def _prepare_init_score(self,X):
        if len(self.models) == 0:
            return None

    def _can_fit_glm(self):
        #Can only fit if it's first model
        return len(self.models) == 0
    
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
            model_name:str = "glm_model"):
        if self._can_fit_glm():
            #Prepare Fitting
            model_name, init_score = self._prepare_fit(X,features,model_name)
            #Fit Model
            self.models[model_name] = deepcopy(self.linear_model)
            self.models[model_name].fit(X = X[features],y = y)
            self.features[model_name] = X.columns.tolist()
            self.model_stages = self.model_stages + [model_name]
        else:
            print("Not Fitting GLM. No init_score Parameter for this model, can only be first model fit")

    def fit_gam(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
            model_name:str = "gam_model"):
        #Prepare Fitting
        model_name, init_score = self._prepare_fit(X,features,model_name)
        # Fit Models
        self.models[model_name] = ExplainableBoostingRegressor(
            max_bins=52,
            interactions = 0,
            objective = self.objective
            )
        self.models[model_name].fit(X= X[features], y = y,init_score = init_score)
        #Update Object
        self.features = X.columns.tolist()
        self.model_stages = self.model_stages + [model_name]

    def fit_ga2m(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
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
        self.models[model_name].fit(X= X[features], y = y,init_score = init_score)
        #Update Object
        self.features = X.columns.tolist()
        self.model_stages = self.model_stages + [model_name]

    def fit_lgbm(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
            model_name:str = "lgbm_model"):
        #Prepare Fitting
        model_name, init_score = self._prepare_fit(X,features,model_name)
        # Fit Models
        self.models[model_name] = LGBMRegressor(
            objective = self.objective,
            init_score = init_score
            )
        self.models[model_name].fit(Xtrain= X[features], y_train = y)
        #Update Object
        self.features = X.columns.tolist()
        self.model_stages = self.model_stages + [model_name]