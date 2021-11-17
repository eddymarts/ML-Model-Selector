from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import time
import numpy as np
import mlflow
from utils.multiproc import parmap
from utils.device import get_device

from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

class ModelSelector:
    def __init__(self, name, dataset, sk_models=None, torch_models=None, mode='regressor', mlflow_tracking=False, multiproc=False) -> None:
        self.dataset = dataset
        self.sk_models = sk_models
        self.torch_models = torch_models
        self.name = name
        self.mlflow_tracking = mlflow_tracking
        self.multiprocess = multiproc
        self.max_score = np.NINF
        self.device, self.num_cpu, self.num_gpu = get_device()
        if self.device == "cuda:0":
            self.num_workers = self.num_gpu
        else:
            self.num_workers = self.num_cpu

        if mode == 'regressor':
            self.scores = {
                "first": {"name": "r2", "function": r2_score},
                "second": {"name": "mse", "function": mean_squared_error}
            }
        elif mode == 'classifier':
            self.scores = {
                "first": {"name": "accuracy", "function": accuracy},
                "second": {"name": "f1", "function": f1_score}
            }
        
    def sk_tune(self, model):
        """ Turns hyperparameters of the model. """
        self.sk_tuned_models[model] = {"model": GridSearchCV(
            estimator=self.sk_models[model]["model"], param_grid=self.sk_models[model]["parameters"]
            )}
        tuning_start = time.time()
        self.sk_tuned_models[model]["model"].fit(X=self.dataset.X_sets[0], y=self.dataset.y_sets[0])
        tuning_end = time.time()

        self.sk_tuned_models[model]["tune_time"] = tuning_end - tuning_start
        self.sk_tuned_models[model]["fit_time"] = self.sk_tuned_models[model]["model"].refit_time_
        self.sk_tuned_models[model]["best_model"] = self.sk_tuned_models[model]["model"].best_estimator_
        self.sk_tuned_models[model]["best_parameters"] = self.sk_tuned_models[model]["model"].best_params_

    def sk_score(self, model):
        """ Returns the score of the tuned model for every set of the data. """
        y_pred_sets = {}
        for score in self.scores.keys():
            self.sk_tuned_models[model][self.scores[score]["name"]] = {}
            for set in range(len(self.dataset.X_sets)):
                y_pred_sets[set] = self.sk_tuned_models[model]["best_model"].predict(self.dataset.X_sets[set])

                if self.scores[score]["name"] == "Accuracy":
                    self.sk_tuned_models[model][self.scores[score]["name"]][set] = self.scores[score]["function"](
                        self.dataset.y_sets[set], y_pred_sets[set], average='weighted'
                    )
                else:
                    self.sk_tuned_models[model][self.scores[score]["name"]][set] = self.scores[score]["function"](
                        self.dataset.y_sets[set], y_pred_sets[set]
                    )
                print(f"{model} - dataset {set}: {self.scores[score]['name']} = {self.sk_tuned_models[model][self.scores[score]['name']][set]}")
        print(f"{model} - parameters: {self.sk_tuned_models[model]['best_parameters']}")

        if self.sk_tuned_models[model][self.scores['first']['name']][1] > self.max_score:
            self.best_model = self.sk_tuned_models[model]
            self.best_model["name"] = model
    
    def sk_experiment(self, model):
        self.sk_tune(model)
        self.sk_score(model)

        if self.mlflow_tracking:
            mlflow.log_metric(f"{model}-tune_time", self.sk_tuned_models[model]["tune_time"])
            mlflow.log_metric(f"{model}-fit_time", self.sk_tuned_models[model]["fit_time"])
            mlflow.sklearn.log_model(
                sk_model=self.sk_tuned_models[model]["best_model"],
                artifact_path='melbourne/sklearn',
                registered_model_name=f"{self.name}-{model}"
                )
            
            for param in self.sk_tuned_models[model]["best_parameters"].keys():
                mlflow.log_param(f"{model}-{param}", self.sk_tuned_models[model]["best_parameters"][param])

            for score in self.scores.keys():
                for set in range(len(self.dataset.X_sets)):
                    mlflow.log_metric(f"{model}-{self.scores[score]['name']}-{set}", self.sk_tuned_models[model][self.scores[score]['name']][set])

    def sk_flow(self):
        self.sk_tuned_models = {}
        if self.mlflow_tracking:
            mlflow.set_tracking_uri("http://localhost:5000")
            with mlflow.start_run():
                # Create experiment (artifact_location=./ml_runs by default)
                mlflow.set_experiment(self.name)
                if self.multiprocess:
                    parmap(lambda model: self.sk_experiment(model), list(self.sk_models.keys()))
                else:
                    for model in self.sk_models.keys():
                        self.sk_experiment(model)

                mlflow.sklearn.log_model(
                    sk_model=self.best_model["best_model"],
                    artifact_path='melbourne/sklearn',
                    registered_model_name=f"{self.name}-BestModel"
                    )
                mlflow.log_param("BestModel", self.best_model["name"])
        else:
            if self.multiprocess:
                parmap(lambda model: self.sk_experiment(model), list(self.sk_models.keys()))
            else:
                for model in self.sk_models.keys():
                    self.sk_experiment(model)
        
    def torch_train(self, config, model):
        """ Turns hyperparameters of the model. """
        net = self.torch_models[model]["model"](
            n_features=self.dataset.n_features,
            n_labels=self.dataset.n_labels,
            num_layers=config["num_layers"],
            neuron_incr=config["neuron_incr"],
            dropout=config["dropout"],
            batchnorm=config["batchnorm"]
            )
        net.fit(
            train_load=self.dataset.tensors[0].dataloader,
            test_load=self.dataset.tensors[1].dataloader,
            epochs=3, lr=config["learning_rate"]
            )
    
    def torch_tune(self, model, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        
        reporter = CLIReporter(
            parameter_columns=list(self.torch_models[model]["parameters"].keys()),
            metric_columns=["train_loss", "val_loss", "training_iteration"]
            )

        tuning_start = time.time()
        result = tune.run(
            partial(self.torch_train, data_dir=data_dir),
            resources_per_trial={"cpu": self.num_cpu, "gpu": gpus_per_trial},
            config=self.torch_models[model]["parameters"],
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            checkpoint_at_end=True)
        tuning_end = time.time()
        
        best_trial = result.get_best_trial("val_loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))

        self.torch_tuned_models[model] = {"best_model": self.torch_models[model]["model"](
            n_features=self.dataset.n_features,
            n_labels=self.dataset.n_labels,
            num_layers=best_trial.config["num_layers"],
            neuron_incr=best_trial.config["neuron_incr"],
            dropout=best_trial.config["dropout"],
            batchnorm=best_trial.config["batchnorm"]
            )}
        self.torch_tuned_models[model]["tune_time"] = tuning_end - tuning_start
        tuning_start = time.time()
        train_result =  self.torch_tuned_models[model]["model"].fit(
            train_load=self.dataset.tensors[0].dataloader,
            test_load=self.dataset.tensors[1].dataloader,
            epochs=3, lr=config["learning_rate"]
            )
        tuning_end = time.time()
        self.torch_tuned_models[model]["fit_time"] = tuning_end - tuning_start
        self.torch_tuned_models[model]["best_parameters"] = best_trial.config
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(
            best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

        test_acc = test_accuracy(best_trained_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))

    def sk_score(self, model):
        """ Returns the score of the tuned model for every set of the data. """
        y_pred_sets = {}
        for score in self.scores.keys():
            self.sk_tuned_models[model][self.scores[score]["name"]] = {}
            for set in range(len(self.dataset.X_sets)):
                y_pred_sets[set] = self.sk_tuned_models[model]["best_model"].predict(self.dataset.X_sets[set])

                if self.scores[score]["name"] == "Accuracy":
                    self.sk_tuned_models[model][self.scores[score]["name"]][set] = self.scores[score]["function"](
                        self.dataset.y_sets[set], y_pred_sets[set], average='weighted'
                    )
                else:
                    self.sk_tuned_models[model][self.scores[score]["name"]][set] = self.scores[score]["function"](
                        self.dataset.y_sets[set], y_pred_sets[set]
                    )
                print(f"{model} - dataset {set}: {self.scores[score]['name']} = {self.sk_tuned_models[model][self.scores[score]['name']][set]}")
        print(f"{model} - parameters: {self.sk_tuned_models[model]['best_parameters']}")

        if self.sk_tuned_models[model][self.scores['first']['name']][1] > self.max_score:
            self.best_model = self.sk_tuned_models[model]
            self.best_model["name"] = model