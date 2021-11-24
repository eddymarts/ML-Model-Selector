from get_data import MNISTData
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
    mnist = MNISTData(split=True, normalize=True)

    sk_model_selector = ModelSelector(
        name="MNIST",
        n_features=mnist.n_features,
        n_labels=mnist.n_labels,
        X_sets=mnist.X_sets,
        # y_sets={set: mnist.y_sets[set].detach().numpy() for set in mnist.y_sets.keys()},
        y_sets=mnist.y_sets,
        mode='classifier',
        # mlflow_tracking_uri=None,
        mlflow_tracking_uri="http://localhost:5000",
        multiproc=False
        )
    sk_model_selector.flow()