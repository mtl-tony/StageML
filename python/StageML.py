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
from sklearn.pipeline import make_pipeline, Pipeline
# Explainable Boosting Machines
from interpret.glassbox import ExplainableBoostingClassifier,ExplainableBoostingRegressor # EBM
# Gradient Boosting Machine
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor

# Ignoring warnings
from abc import ABC, abstractmethod
from utils.inv_link import inv_log, inv_identity, inv_logistic,link_identity,link_log,link_logistic



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
        n_jobs: Optional[int] = -2,
        feature_types = None,
        random_state = 42):

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

    def _model_exists(self, model_name):
        return model_name in self.model_stages

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
                self.ebm_objective = "rmse"
                self.link = link_identity
            case "poisson":
                self.linear_model = PoissonRegressor()
                self.inv_link = inv_log
                self.ebm_objective = "poisson_deviance"
                self.link = link_log
            case "gamma":
                self.linear_model = TweedieRegressor(power = 1, link = "log")
                self.inv_link = inv_log
                self.ebm_objective = "gamma_deviance"
                self.link = link_log
            case "logloss":
                self.linear_model = LogisticRegression()
                self.inv_link = inv_logistic
                self.ebm_objective = "log_loss"
                self.link = link_logistic

    def _can_fit_glm(self):
        #Can only fit if it's first model
        return len(self.models) == 0

    def _get_feature_types(self, X):
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return numeric_features, categorical_features
    
    def _update_model_features(self,model_name,features):
        self.features[model_name] = features
        self.model_stages = self.model_stages + [model_name]

    def _handle_train_valid_data(self,X_train, y_train, X_valid,y_valid):
        pass

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
        n_jobs: Optional[int] = -2,
        objective : str = "rmse",
        feature_types = None,
        random_state = 42):

        super().__init__(
            n_jobs = n_jobs,
            feature_types = feature_types,
            random_state = random_state)
        #Setting the Objective
        self.objective = objective
        self._objective_handler()

    def _prepare_init_score(self,X):
        """Prepares the initial score value for LightGBM model fitting.

        Parameters:
        X (pandas.DataFrame): The predictor/feature data for which initial 
        scores will be computed.
        
        Returns: 
        init_score (numpy.array): The initial score values to pass to the model fitting.
        
        This function calculates initial scores based on:
        - If first model being fit, returns None 
        - Else sums predictions from previous stage models
        - Predicts on X from each previous stage model
        - Sums predictions using self.link()
        - Returns summed array as init_scores
        """
        if len(self.models) == 0:
            init_score = None
        else:
            init_score = np.zeros(X.shape[0] )
            for model_name in self.model_stages:
                # print(model_name)
                init_score += self.link(self.models[model_name].predict(X[self.features[model_name]]))
        return init_score
    
    def _prepare_fit(self,X,features,model_name):
        """Performs validations and prepares init score before fitting model.

        Args:
            X (pd.DataFrame): Input features dataframe
            features (list): List of feature names  
            model_name (str): Name to use for the fitted model
        
        Returns:
            model_name (str): Updated model name if needed
            init_score (np.array): Initial score based on X, else None

        This method:
            - Validates X and features
            - Computes an initial score if possible based on X 
            - Checks that model_name is unique and updates it if needed

        The init_score can be used by some models that support warm starts or 
        using prior information during training.
        """
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
        """Fits a Generalized Linear Model (GLM) on the input data X and target y.

        Args:
            X (pd.DataFrame): Input features dataframe 
            y (pd.DataFrame): Target dataframe
            features (list): List of feature names to use for training
            sample_weight (pd.DataFrame): Optional weight for each sample
            model_name (str): Name to use for the fitted model

        Returns:
            None: Fits a GLM pipeline internally and stores it in self.models

        The feature columns are preprocessed using a ColumnTransformer to handle numeric and
        categorical features. A linear model is fitted on the processed features.

        The fitted GLM pipeline is stored in self.models under the key model_name. 
        Model can be accessed later for predictions using self.models[model_name].
        """
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

            self.models[model_name] = Pipeline([
                ("preprocessor",preprocessor), 
                ("linearmodel",deepcopy(self.linear_model))])
            #Fit Model
            self.models[model_name].fit(X = X[features],y = y,linearmodel__sample_weight = sample_weight)
            self._update_model_features(model_name,features)
        else:
            print("Not Fitting GLM. No init_score Parameter for this model, can only be first model fit")

    def fit_gam(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
            sample_weight = None,
            model_name:str = "gam_model"):
        """Fits a Generalized Additive Model (GAM) on the input data.

        Args:
            X (pd.DataFrame): Input features dataframe
            y (pd.DataFrame): Target dataframe 
            features (list): List of feature names to use for training
            sample_weight (pd.DataFrame): Optional weight for each sample
            model_name (str): Name to use for the fitted model
        
        Returns:
            None: Fits a GAM model internally and stores it in self.models

        This method:
            - Prepares the data by validating and generating init scores
            - Fits a ExplainableBoostingRegressor model on X and y
            - Stores the fitted model pipeline in self.models under model_name

        The fitted model can be used for predictions and insights on important
        features.
        """
        #Prepare Fitting
        model_name, init_score = self._prepare_fit(X,features,model_name)
        # Fit Models
        self.models[model_name] = ExplainableBoostingRegressor(
            max_bins=52,
            interactions = 0,
            objective = self.ebm_objective
            )
        self.models[model_name].fit(X= X[features], y = y,sample_weight = sample_weight,init_score = init_score)
        #Update Object
        self._update_model_features(model_name,features)

    def fit_ga2m(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            features:list,
            sample_weight = None,
            model_name:str = "ga2m_model",
            interactions = 10):
        """Fits a Generalized Additive Interaction Model (GA2M) on the input data.

        Args:
            X (pd.DataFrame): Input features dataframe
            y (pd.DataFrame): Target dataframe
            features (list): List of feature names to use for training
            sample_weight (pd.DataFrame): Optional weight for each sample 
            model_name (str): Name to use for the fitted model
            interactions (int): Max number of feature interactions to model

        Returns:
            None: Fits a GA2M model internally and stores it in self.models

        This method:
            - Prepares the data by validating and generating init scores
            - Fits a ExplainableBoostingRegressor model with interactions
            - Stores the fitted model pipeline in self.models under model_name

        The fitted model can be used for predictions and insights on important
        features and interactions.
        """
        #Prepare Fitting
        model_name, init_score = self._prepare_fit(X,features,model_name)
        # Fit Models
        self.models[model_name] = ExplainableBoostingRegressor(
            max_interaction_bins=10,
            interactions = interactions,
            objective = self.ebm_objective,
            exclude = list(range(len(features)))
            )
        self.models[model_name].fit(X= X[features], y = y,sample_weight = sample_weight,init_score = init_score)
        #Update Object
        self._update_model_features(model_name,features)


    def fit_lgbm(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            valid_sets:list,
            features:list,
            sample_weight = None,
            model_name:str = "lgbm_model"):
        """Fits a LightGBM model on the input data.

        Args:
            X (pd.DataFrame): Input features dataframe
            y (pd.DataFrame): Target dataframe
            valid_sets (list): List of tuples (X_valid, y_valid) 
            features (list): List of feature names to use for training
            sample_weight (pd.DataFrame): Optional weight for each sample
            model_name (str): Name to use for the fitted model

        Returns:
            None: Fits a LightGBM model internally and stores it in self.models

        This method:
            - Prepares the data by validating and generating init scores 
            - Fits a LightGBMRegressor on X and y
            - Evaluates on valid_sets during training
            - Stores fitted model in self.models under model_name

        The fitted model can be used for predictions and insights on important
        features.
        """
        #Prepare Fitting
        model_name, init_score = self._prepare_fit(X,features,model_name)
        # Fit Models
        self.models[model_name] = LGBMRegressor(
            objective = self.objective            
            )
        self.models[model_name].fit(X= X[features], y = y,sample_weight = sample_weight,init_score = init_score)
        #Update Object
        self._update_model_features(model_name,features)


    def predict(
            self,
            X:pd.DataFrame,
            model_name:str):
        """Makes predictions on new data using a fitted model.

        Args:
            X (pd.DataFrame): New data for predictions 
            model_name (str): Name of fitted model to use  

        Returns:
            y_pred (np.array): Array of model predictions on new data X 

        This makes predictions by:
        
        1. Checking if the given model_name exists
        2. Applying preprocessing and prediction steps 
        for each stage in the model pipeline
        3. Transforming predictions using the link function
        4. Returning predictions when final stage is reached

        The model must be fitted first before predictions can be made.
        """
        if self._model_exists(model_name):
            score = np.zeros(X.shape[0])
            for item in self.model_stages:
                # print(item)
                score += self.link(self.models[item].predict(X[self.features[item]]))
                if item == model_name:
                    return self.inv_link(score)
        else:
            print("Model does not exists")