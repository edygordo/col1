from os import POSIX_FADV_SEQUENTIAL
from typing import OrderedDict
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import xgboost as xgb
from sklearn.utils.validation import check_X_y, check_array
from astropy.constants import G, M_earth
import pdb

## CNN model
class cnn_model:
    def __init__(self, arct= {}, *args, **kwargs):
        self.reg = Sequential()
        self._models = OrderedDict()
        self.arct = {'hlayers': 1,
                        'ipnodes': 10,
                        'nodes': [10],
                        'activation': ['tanh']
                        }
        self.arct.update(arct)
    
    def get_model(self, model_name):
        return self._models.get(model_name)
    
    def set_model(self, model_name, model):
        """Stores the regression model by the name of the error column that it
        estimates.
        :param model_name: The name of the error column to store the model as
        :type model_name: str
        :param model: The regression model to store
        :type model: CNN model
        """
        self._models[model_name] = model
        
    def get_models(self):
        """Gets all the stored models.
        :return: A list model name, regression model pairs
        :rtype: [(str, xgboost.XGBRegressor)]
        """
        return list(self._models.items())
    
    def fit(self, X, ys, n_inputs, n_outputs):
        """Fits the underlying GBRT models on the provided training data.
        :param X: The feature matrix to use in training
        :type X: numpy.ndarray
        :param ys: The multiple target columns to use in training
        :type ys: numpy.ndarray
        :param eval_metric: The metric to use to evaluate each GBRT
            model's performance
        :type eval_metric: str
        :return: The fitted multi-output regression model
        :rtype: orbit_prediction.ml_model.ErrorGBRT
        """
        # Check that the feature matrix and target matrix are the correct sizes
        check_X_y(X, ys, multi_output=True)
        
        for target_col in ys.columns:
            y = ys[target_col]
            model = Sequential()
            ## construct input layer
            model.add(Dense(self.arct['ipnodes'], input_dim=n_inputs, kernel_initializer='he_uniform', \
                activation= self.arct['activation'][0]))
            
            ## construct hidden layers
            for l in range(0, self.arct['hlayers']): 
                model.add(Dense(self.arct['nodes'][l], activation = self.arct['activation'][l]))
            
            ## construct output layer
            model.add(Dense(n_outputs))
            model.compile(loss=tf.keras.metrics.mean_squared_error, optimizer='adam')
            model.fit(X, y, verbose=0, epochs=100)
            self.set_model(target_col, model)
        return self
    
    def predict(self, X):
        """Uses the underlying GBRT models to estimate each component of the
        physical model error.
        :param X: The feature matrix to make predictions for
        :type X: numpy.ndarray
        :return: The estimated physical model error for each component
        :rtype: numpy.ndarray
        """
        # Make sure the input matrix is the right shape
        X = check_array(X)
        # Each model predicts the error for its respective state
        # vector component
        y_preds = [m.predict(X) for (_, m) in self.get_models()]
        # Orient the error estimates as column vectors
        y_preds = np.stack(y_preds, axis=1)
        return y_preds
    
    
    def eval_models(self, X, ys):
        """Calculates the root mean squared error (RMSE) and the coefficient of
        determination (R^2) for each of the models.
        :param X: The feature matrix to use in evaluating the regression models
        :type X: numpy.ndarray
        :param y: The target columns to use in evaluating the regression models
        :type y: numpy.ndarray
        :return: Returns a DataFrame containing the evaluation metric results
        :rtype: pandas.DataFrame
        """
        evals = []
        for target_col, reg in self.get_models():
            y_hat = reg.predict(X)
            y = ys[target_col]
            rmse = metrics.mean_squared_error(y, y_hat, squared=False)
            r2 = metrics.r2_score(y, y_hat)
            std = np.std(y.values - y_hat)
            eval_dict = {'Error': target_col, 'RMSE': rmse, 'R^2': r2, 'STD': std}
            evals.append(eval_dict)
        return pd.DataFrame(evals)
    
    
## XGB model
class xgb_model:
    def __init__(self, *args, **kwargs):
        self.reg = xgb.XGBRegressor(*args, **kwargs)
        self._models = OrderedDict()
        
    def get_model(self, model_name):
        """Gets a regression model by the name of the error column that
        it estimates.
        :param model_name: The name of the error column to fetch the model for
        :type model_name: str
        :return: The model responsible for estimating the error column
        :rtype: xgboost.XGBRegressor
        """
        return self._models.get('model_name')

    def set_model(self, model_name, model):
        """Stores the regression model by the name of the error column that it
        estimates.
        :param model_name: The name of the error column to store the model as
        :type model_name: str
        :param model: The regression model to store
        :type model: xgboost.XGBRegressor
        """
        self._models[model_name] = model

    def get_models(self):
        """Gets all the stored models.
        :return: A list model name, regression model pairs
        :rtype: [(str, xgboost.XGBRegressor)]
        """
        return list(self._models.items())
    
    def fit(self, X, ys, eval_metric='rmse'):
        """Fits the underlying GBRT models on the provided training data.
        :param X: The feature matrix to use in training
        :type X: numpy.ndarray
        :param ys: The multiple target columns to use in training
        :type ys: numpy.ndarray
        :param eval_metric: The metric to use to evaluate each GBRT
            model's performance
        :type eval_metric: str
        :return: The fitted multi-output regression model
        :rtype: orbit_prediction.ml_model.ErrorGBRT
        """
        # Check that the feature matrix and target matrix are the correct sizes
        check_X_y(X, ys, multi_output=True)
        # Get the XGBoost parameters to use for each regressor
        xgb_params = self.reg.get_params()
        # Build and train a GBRT model for each target column
        for target_col in ys.columns:
            y = ys[target_col]
            reg = xgb.XGBRegressor(**xgb_params)
            reg.fit(X, y, eval_metric=eval_metric)
            self.set_model(target_col, reg)

        return self

    def predict(self, X):
        """Uses the underlying GBRT models to estimate each component of the
        physical model error.
        :param X: The feature matrix to make predictions for
        :type X: numpy.ndarray
        :return: The estimated physical model error for each component
        :rtype: numpy.ndarray
        """
        # Make sure the input matrix is the right shape
        X = check_array(X)
        # Each model predicts the error for its respective state
        # vector component
        y_preds = [m.predict(X) for (_, m) in self.get_models()]
        # Orient the error estimates as column vectors
        y_preds = np.stack(y_preds, axis=1)
        return y_preds

    def eval_models(self, X, ys):
        """Calculates the root mean squared error (RMSE) and the coefficient of
        determination (R^2) for each of the models.
        :param X: The feature matrix to use in evaluating the regression models
        :type X: numpy.ndarray
        :param y: The target columns to use in evaluating the regression models
        :type y: numpy.ndarray
        :return: Returns a DataFrame containing the evaluation metric results
        :rtype: pandas.DataFrame
        """
        evals = []
        for target_col, reg in self.get_models():
            y_hat = reg.predict(X)
            y = ys[target_col]
            rmse = metrics.mean_squared_error(y, y_hat, squared=False)
            r2 = metrics.r2_score(y, y_hat)
            std = np.std(y.values - y_hat)
            eval_dict = {'Error': target_col,
                         'RMSE': rmse, 
                         'R^2': r2,
                         'STD': std
                         }
            evals.append(eval_dict)
        return pd.DataFrame(evals)
    
    
class ensamble_nn:

    def __init__(self, dim, ref_sat, arct, *args, **kwargs):
        default_arct = {'hlayers': 1,
                        'ipnodes': 10,
                        'nodes': [10],
                        'activation': ['relu']
                        }
        default_arct.update(arct)
        self.reg = Sequential()
        
        ## constructing input layer
        self.reg.add(Dense(default_arct['ipnodes'], input_dim= dim, \
            kernel_initializer='he_uniform', activation=default_arct['activation'][0]))
        
        ## Constructing hidden layers
        for l in range(0, default_arct['hlayers']): 
            self.reg.add(Dense(default_arct['nodes'][l], activation = default_arct['activation'][l]))
         
        ## constructing output layer    
        self.reg.add(Dense(dim))
        self.reg.compile(loss=orbit_constraint(ref_sat), optimizer='adam')
        
    def fit(self, X, ys):
        self.reg.fit(X, ys, verbose=0, epochs=100)
        return self
    
    def predict(self, X):
        return self.reg.predict(X)

def orbit_constraint(params):
    params = tf.convert_to_tensor(params, dtype=tf.float32)
    def loss(y_true, y_pred):
        mu = G*M_earth        
        idx = tf.cast(y_true[0,0], tf.int32)
        #pdb.set_trace()
        r1 = params[idx, 0:3] + y_true[0, 1:4]
        v1 = params[idx, 3:6] + y_true[0, 4:7]
        c1 = tf.math.scalar_mul(2*mu, tf.math.reciprocal(tf.norm(r1))) -  \
            tf.math.square(tf.norm(v1))
        
        r2 = params[idx, 0:3] + y_pred[0, 0:3]
        v2 = params[idx, 3:6] + y_pred[0, 3:6]
        c2 = tf.math.scalar_mul(2*mu, tf.math.reciprocal(tf.norm(r2))) \
            -  tf.math.square(tf.norm(v2))
        
        #pdb.set_trace()
        return tf.math.square(c1 - c2) + tf.math.square(y_true[0,7] - y_pred[0,6])
    return loss


def train_ensamble(X, ys, n_inputs, n_outputs, arct, method = "cnn"):
    if method == "cnn":
        p_model = cnn_model(arct)
        p_model.fit(X[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']], ys, n_inputs, n_outputs)
        X_e = p_model.predict(X[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']])
    elif method == "xgb":
        default_params = {
                            'booster': 'dart',
                            'tree_method': 'hist'
            }
        p_model = xgb_model(**default_params)
        p_model.fit(X[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']].values, ys)
        X_e = p_model.predict(X[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']].values)
        
    X_e = X_e.reshape((X_e.shape[0], X_e.shape[1]))
    ref_sat = X[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
    e_model = ensamble_nn(X_e.shape[1], ref_sat, arct)
    #pdb.set_trace()
    l = ys.index.shape[0]
    y_ens = np.hstack((np.arange(0,l).reshape((l,1)), ys.values))
    e_model.fit(X_e, y_ens)
    return p_model, e_model

def ensamble_predict(parallel_model, ensamble_model, X):
    X_e = parallel_model.predict(X)
    X_e = X_e.reshape((X_e.shape[0], X_e.shape[1]))
    y = ensamble_model.predict(X_e)
    return y

def ensamble_eval(parallel_model, ensamble_model, X, ys):
    X_e = parallel_model.predict(X)
    X_e = X_e.reshape((X_e.shape[0], X_e.shape[1]))
    y_hat = ensamble_model.predict(X_e)
    #rmse = metrics.mean_squared_error(ys, y_hat, squared=False)
    #r2 = metrics.r2_score(ys, y_hat)
    rmse = np.mean(np.square(ys.values - y_hat), axis = 0)
    std = np.std(ys.values - y_hat, axis = 0)
    eval_dict = {'RMSE': rmse, 'STD': std}
    return pd.DataFrame(eval_dict)
    
    