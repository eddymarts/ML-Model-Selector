from get_data import BreastCancerData
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
    breast_cancer = BreastCancerData(split=True, normalize=True, shuffle=True, seed=None)

    sk_model_selector = ModelSelector(
        name="Breast Cancer",
        n_features=breast_cancer.n_features,
        n_labels=breast_cancer.n_labels,
        X_sets=breast_cancer.X_sets,
        y_sets=breast_cancer.y_sets,
        mode='classifier',
        # mlflow_tracking_uri=None,
        mlflow_tracking_uri="http://localhost:5000",
        multiproc=False
        )
    sk_model_selector.flow()

    
    # torch_model_selector = ModelSelector(
    #     name="breast_cancer",
    #     tensors=breast_cancer.tensors,
    #     torch_models=torch_models,
    #     mode='regressor',
    #     mlflow_tracking_uri=None,
    #     multiproc=False
    #     )
    # torch_model_selector.flow()