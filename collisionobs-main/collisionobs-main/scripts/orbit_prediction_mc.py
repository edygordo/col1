import sys
sys.path.append("./modules")
import pdb
from tqdm import tqdm
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

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import *
from poliastro.constants import J2000
from poliastro.util import time_range
from datetime import datetime, timedelta
from astropy import time
from astropy import units as u
import plotly.io as pio
from mlmodels import cnn_model, xgb_model, train_ensamble, ensamble_predict, ensamble_eval
from orbit_module import tle_hist
from utilities import nn_config
pio.renderers.default = "vscode"

data = pd.read_csv('./data/training_data.csv')
test_data = pd.read_csv('./data/test_data.csv')

layer_list = [5, 10, 15, 20, 25, 30, 35, 40]
ipnodes = 10
tnodes = [8*l for l in layer_list]
activ = ['tanh']
conf = nn_config(layer_list, ipnodes, tnodes, activ).creat_config()

## training and test data
# Starlink training data
X = data[['x', 'y', 'z', 'vx', 'vy', 'vz', 'delvx', 'delvy', 'delvz']]
y = data[['rso_x', 'rso_y', 'rso_z', 'rso_vx', 'rso_vy', 'rso_vz']].values- \
    data[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
y = pd.DataFrame(y, columns=['delx', 'dely', 'delz', 'delvx', 'delvy', 'delvz'])
y['rso_m'] = data['rso_m']

n_inputs, n_outputs = 6, 1
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=123, test_size=0.01)

# Lemur test data
X_lemur = test_data[['x', 'y', 'z', 'vx', 'vy', 'vz', 'delvx', 'delvy', 'delvz']]
y_lemur = test_data[['rso_x', 'rso_y', 'rso_z', 'rso_vx', 'rso_vy', 'rso_vz']].values- \
    test_data[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
y_lemur = pd.DataFrame(y_lemur, columns=['delx', 'dely', 'delz', 'delvx', 'delvy', 'delvz'])
y_lemur['rso_m'] = test_data['rso_m']

ann_arct = {'hlayers': 5,
            'ipnodes': 10,
            'nodes': [20, 10, 10, 10, 7],
            'activation': ['tanh']*5
            }

nn_rmse = []
nn_std = []

ens_rmse = []
ens_std = []

xgb_rmse = []
xgb_std = []

ens_xgb_rmse = []
ens_xgb_std = []
# ann model evaluation
for arct in tqdm(conf):
    # parallel NN
    #pdb.set_trace()
    model_cnn = cnn_model(arct)
    model_cnn.fit(X_train[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']], y_train, n_inputs, n_outputs) # training
    eval_nn = model_cnn.eval_models(X_lemur[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']], y_lemur) # model evaluation
    nn_rmse.append(eval_nn['RMSE'].values.tolist())
    nn_std.append(eval_nn['STD'].values.tolist())

    # Ensemble NN
    model_p, model_e = train_ensamble(X_train, y_train, n_inputs, n_outputs, arct, method = "cnn") #training
    eval_ens = ensamble_eval(model_p, model_e, X_lemur[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']], y_lemur) #model evaluation
    ens_rmse.append(eval_ens['RMSE'].values.tolist())
    ens_std.append(eval_ens['STD'].values.tolist())
    
    # XGB
    default_params = {
        'booster': 'dart',
        'tree_method': 'hist'
    }

    # parallel
    model_xgb = xgb_model(**default_params)
    model_xgb.fit(X_train[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']].values, y_train)
    eval_xgb = model_xgb.eval_models(X_lemur[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']].values, y_lemur)
    xgb_rmse.append(eval_xgb['RMSE'].values.tolist())
    xgb_std.append(eval_xgb['STD'].values.tolist())
    
    # ensamble
    model_p_xgb, model_e_xgb = train_ensamble(X_train, y_train, n_inputs, n_outputs, arct, method = "xgb")
    eval_ens_xgb = ensamble_eval(model_p_xgb, model_e_xgb, X_lemur[['x', 'y', 'z', 'delvx', 'delvy', 'delvz']], y_lemur)
    ens_xgb_rmse.append(eval_ens_xgb['RMSE'].values.tolist())
    ens_xgb_std.append(eval_ens_xgb['STD'].values.tolist())

nn_rmse = pd.DataFrame(nn_rmse, columns = ['x', 'y', 'z', 'delvx', 'delvy', 'delvz', 'm'])
nn_std = pd.DataFrame(nn_std, columns = ['x', 'y', 'z', 'delvx', 'delvy', 'delvz', 'm'])
ens_rmse = pd.DataFrame(ens_rmse, columns = ['x', 'y', 'z', 'delvx', 'delvy', 'delvz', 'm'])
ens_std = pd.DataFrame(ens_std, columns = ['x', 'y', 'z', 'delvx', 'delvy', 'delvz', 'm'])
    
xgb_rmse = pd.DataFrame(xgb_rmse, columns = ['x', 'y', 'z', 'delvx', 'delvy', 'delvz', 'm'])
xgb_std = pd.DataFrame(xgb_std, columns = ['x', 'y', 'z', 'delvx', 'delvy', 'delvz', 'm'])
ens_xgb_rmse = pd.DataFrame(ens_xgb_rmse, columns = ['x', 'y', 'z', 'delvx', 'delvy', 'delvz', 'm'])
ens_xgb_std = pd.DataFrame(ens_xgb_std, columns = ['x', 'y', 'z', 'delvx', 'delvy', 'delvz', 'm'])
    
nn_rmse.to_csv('nn_rmse.csv', index = False)
nn_std.to_csv('nn_std.csv', index = False)

ens_rmse.to_csv('ens_rmse.csv', index = False)
ens_std.to_csv('ens_std.csv', index = False)

xgb_rmse.to_csv('xgb_rmse.csv', index = False)
xgb_std.to_csv('xgb_std.csv', index = False)

ens_xgb_rmse.to_csv('ens_xgb_rmse.csv', index = False)
ens_xgb_std.to_csv('ens_xgb_std.csv', index = False)