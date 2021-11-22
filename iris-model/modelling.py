from get_data import IrisData
from multiprocessing import cpu_count
from sklearn.metrics import *
import torch
from ray import tune

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.machine_learning import ModelSelector

if __name__ == "__main__":
    iris = IrisData(split=True, normalize=True, shuffle=True, seed=None)

    sk_model_selector = ModelSelector(
        name="Iris",
        n_features=iris.n_features,
        n_labels=iris.n_labels,
        X_sets=iris.X_sets,
        y_sets=iris.y_sets,
        mode='classifier',
        # mlflow_tracking_uri=None,
        mlflow_tracking_uri="http://localhost:5000",
        multiproc=False
        )
    sk_model_selector.flow()