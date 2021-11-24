from numpy import float16, float32
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from skorch.regressor import NeuralNetRegressor
from get_data import BostonData, TorchDataSet
from multiprocessing import cpu_count
from sklearn.metrics import *
import torch
from ray import tune

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.machine_learning import SKLearnModelSelector

if __name__ == "__main__":
    boston = BostonData(split=True, normalize=True, shuffle=True, seed=None)

    sk_model_selector = SKLearnModelSelector(
        name="Boston",
        n_features=boston.n_features,
        n_labels=boston.n_labels,
        X_sets=boston.X_sets,
        y_sets=boston.y_sets,
        mode='regressor',
        models=["linear_reg", "knn_reg"],
        # mlflow_tracking_uri=None,
        mlflow_tracking_uri="http://localhost:5000",
        multiproc=False
        )
    sk_model_selector.sk_flow()

    
    # torch_model_selector = SKLearnModelSelector(
    #     name="Boston",
    #     tensors=boston.tensors,
    #     torch_models=torch_models,
    #     mode='regressor',
    #     mlflow_tracking_uri=None,
    #     multiproc=False
    #     )
    # torch_model_selector.flow()