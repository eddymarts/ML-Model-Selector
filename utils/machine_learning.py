from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from skorch.regressor import NeuralNetRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from skorch.classifier import NeuralNetClassifier, NeuralNetBinaryClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import *
import matplotlib.pyplot as plt
import time
import numpy as np
import mlflow
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.multiproc import parmap
from utils.device import get_device
from utils.neural_networks import *



class SKLearnModelSelector:
    def __init__(self, name, models=None, X_sets=None, y_sets=None, n_features=None, n_labels=None,
                mode='regressor', mlflow_tracking_uri=None, multiproc=False) -> None:
        self.X_sets = X_sets
        self.y_sets = y_sets
        self.name = name
        self.n_features=n_features
        self.n_labels=n_labels
        self.multiprocess = multiproc
        self.sk_max_score = np.NINF
        self.mlflow_tracking = False
        self.device, self.num_cpu, self.num_gpu = get_device()
        self.sk_models = self.get_sk_models_and_metrics(mode, models)
        if self.device == "cuda:0":
            self.num_workers = self.num_gpu
        else:
            self.num_workers = self.num_cpu

        if mlflow_tracking_uri is not None:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(self.name)
            mlflow.start_run()
            self.mlflow_tracking = True
    
    def get_sk_models_and_metrics(self, mode, models):
        if mode == 'regressor':
            self.scores = {
                "first": {"name": "r2", "function": r2_score},
                "second": {"name": "mse", "function": mean_squared_error}
            }

            sk_total_models = {
                "linear_reg": {
                    "name": "linear_reg",
                    "model": LinearRegression(),
                    "parameters": {'fit_intercept': [True, False]}
                },
                "knn_reg": {
                    "name": "knn_reg",
                    "model": KNeighborsRegressor(),
                    "parameters": {
                        'n_neighbors': list(range(10, 30)), 
                        'weights': ['uniform', 'distance']
                        }
                },
                "svm_reg": {
                    "name": "svm_reg",
                    "model": SVR(),
                    "parameters": {
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 
                        'degree': list(range(2, 10)),
                        'gamma': ['scale', 'auto'],
                        'coef0': list(range(100)),
                        'C': [num/1 for num in range(10)],
                        'epsilon': [num/100 for num in range(10)],
                        'shrinking': [False, True]
                        }
                },
                "dt_reg": {
                    "name": "dt_reg",
                    "model": DecisionTreeRegressor(),
                    "parameters": {
                        'min_samples_split': list(range(2, 10)), 
                        'min_samples_leaf': list(range(1, 10)),
                        'min_weight_fraction_leaf': [num for num in range(5)]
                        }
                },
                "rf_reg": {
                    "name": "rf_reg",
                    "model": RandomForestRegressor(),
                    "parameters": {
                        'n_estimators': [num*10 for num in range(10, 11)],
                        'min_samples_split': list(range(2, 4)), 
                        'min_samples_leaf': list(range(1, 3)),
                        'min_weight_fraction_leaf': [num/10 for num in range(2)]
                        }
                },
                "mlp_reg": {
                    "name": "mlp_reg",
                    "model": MLPRegressor(),
                    "parameters": {
                        'hidden_layer_sizes': [(100, 100, 100, 100, 100)],
                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
                        'alpha': [num/10000 for num in range(1, 100)],
                        'max_iter': [1000]
                        }
                },
                "fnn_reg": {
                    "name": "fnn_reg",
                    "model": NeuralNetRegressor(
                        module=CustomBaseNetRegression,
                        optimizer=torch.optim.SGD,
                        max_epochs=3,
                        device=self.device
                        ),
                    "parameters": {
                        'batch_size': [2**batch for batch in range(4, 5)],
                        'lr': [0.0001 * 10**num for num in range(2)],
                        'optimizer__momentum': [num/10 for num in range(2)],
                        'optimizer__dampening': [num/10 for num in range(2)],
                        'optimizer__weight_decay': [0.0001 * 10**num for num in range(2)],
                        'optimizer__nesterov': [False, True],
                        'module__n_features': [self.n_features],
                        'module__n_labels': [self.n_labels],
                        'module__num_layers': [2**num for num in range(1, 3)],
                        'module__neuron_incr': [num for num in range(2)],
                        'module__dropout': [num/10 for num in range(2)],
                        'module__batchnorm': [False, True]
                        }
                }
            }

        elif mode == 'classifier':
            self.n_classes = len(np.unique(self.y_sets[0]))
            self.scores = {
                "first": {"name": "accuracy", "function": accuracy_score},
                "second": {"name": "f1", "function": f1_score}
            }

            sk_total_models = {
                "log_reg_clf": {
                    "name": "log_reg_clf",
                    "model": LogisticRegression(),
                    "parameters": {
                        'fit_intercept': [True, False],
                        # 'penalty': ["none"],
                        # 'l1_ratio': [0],
                        'max_iter': [1000]
                        }                            
                },
                "knn_clf": {
                    "name": "knn_clf",
                    "model": KNeighborsClassifier(),
                    "parameters": {
                        'n_neighbors': list(range(10, 30)), 
                        'weights': ['uniform', 'distance']
                        }
                },
                "svm_clf": {
                    "name": "svm_clf",
                    "model": SVC(),
                    "parameters": {
                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 
                        'degree': list(range(2, 10)),
                        'gamma': ['scale', 'auto'],
                        'coef0': list(range(100)),
                        'C': [num/1 for num in range(10)],
                        'epsilon': [num/100 for num in range(10)],
                        'shrinking': [False, True],
                        'class_weight': ["balanced", None]
                        }
                },
                "dt_clf": {
                    "name": "dt_clf",
                    "model": DecisionTreeClassifier(),
                    "parameters": {
                        'criterion': ["gini", "entropy"],
                        'min_samples_split': list(range(2, 10)), 
                        'min_samples_leaf': list(range(1, 10)),
                        'min_weight_fraction_leaf': [num for num in range(5)]
                        }
                },
                "rf_clf": {
                    "name": "rf_clf",
                    "model": RandomForestClassifier(),
                    "parameters": {
                        'n_estimators': [num*10 for num in range(10, 11)],
                        'criterion': ["gini", "entropy"],
                        'min_samples_split': list(range(2, 4)), 
                        'min_samples_leaf': list(range(1, 3)),
                        'min_weight_fraction_leaf': [num/10 for num in range(2)],
                        'class_weight': ["balanced", "balanced_subsample", None]
                        }
                },
                "mlp_clf": {
                    "name": "mlp_clf",
                    "model": MLPClassifier(),
                    "parameters": {
                        'hidden_layer_sizes': [(100, 100, 100, 100, 100)],
                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
                        'alpha': [num/10000 for num in range(1, 100)],
                        'max_iter': [1000]
                        }
                },
                "fnn_bi_clf": {
                    "name": "fnn_bi_clf",
                    "model": NeuralNetBinaryClassifier(
                        module=CustomBaseNetBinaryClassification,
                        optimizer=torch.optim.SGD,
                        max_epochs=3,
                        device=self.device
                        ),
                    "parameters": {
                        'batch_size': [2**batch for batch in range(4, 5)],
                        'lr': [0.0001 * 10**num for num in range(2)],
                        'optimizer__momentum': [num/10 for num in range(2)],
                        'optimizer__dampening': [num/10 for num in range(2)],
                        'optimizer__weight_decay': [0.0001 * 10**num for num in range(2)],
                        'optimizer__nesterov': [False, True],
                        'module__n_features': [self.n_features],
                        'module__n_labels': [self.n_labels],
                        'module__num_layers': [2**num for num in range(1, 3)],
                        'module__neuron_incr': [num for num in range(2)],
                        'module__dropout': [num/10 for num in range(2)],
                        'module__batchnorm': [False, True]
                        }
                },
                "fnn_clf": {
                    "name": "fnn_clf",
                    "model": NeuralNetClassifier(
                        module=CustomBaseNetClassification,
                        optimizer=torch.optim.SGD,
                        max_epochs=3,
                        device=self.device
                        ),
                    "parameters": {
                        'batch_size': [2**batch for batch in range(4, 5)],
                        'lr': [0.0001 * 10**num for num in range(2)],
                        'optimizer__momentum': [num/10 for num in range(2)],
                        'optimizer__dampening': [num/10 for num in range(2)],
                        'optimizer__weight_decay': [0.0001 * 10**num for num in range(2)],
                        'optimizer__nesterov': [False, True],
                        'module__n_features': [self.n_features],
                        'module__n_labels': [self.n_classes],
                        'module__num_layers': [2**num for num in range(1, 3)],
                        'module__neuron_incr': [num for num in range(2)],
                        'module__dropout': [num/10 for num in range(2)],
                        'module__batchnorm': [False, True]
                        }
                },
                "cnn_clf": {
                    "name": "cnn_clf",
                    "model": NeuralNetClassifier(
                        module=CNNClassifier,
                        optimizer=torch.optim.SGD,
                        max_epochs=3,
                        device=self.device
                        ),
                    "parameters": {
                        'batch_size': [2**batch for batch in range(4, 5)],
                        'lr': [0.0001 * 10**num for num in range(1)],
                        'optimizer__momentum': [num/10 for num in range(2)],
                        'optimizer__dampening': [num/10 for num in range(2)],
                        'optimizer__weight_decay': [0.0001 * 10**num for num in range(2)],
                        'optimizer__nesterov': [False, True]
                        }
                }
            }
        
        sk_models = []
        if models is not None:
            for model in models:
                sk_models.append(sk_total_models[model])
        else:
            for model in sk_total_models.keys():
                sk_models.append(sk_total_models[model])
        return sk_models

    def sk_flow(self):
        def sk_tune(name, model, scores, X_sets, y_sets, mlflow_tracking, model_type="sklearn"):
            """ Turns hyperparameters of the model. """
            def sk_score(model, model_name, scores, X_sets, y_sets, mlflow_tracking):
                """ Returns the score of the tuned model for every set of the data. """
                for score in scores.keys():
                    for set in range(len(X_sets)):
                        y_pred = model.predict(X_sets[set])

                        if scores[score]["name"] == "f1":
                            metric = scores[score]["function"](
                                y_sets[set], y_pred, average='weighted'
                            )
                        else:
                            metric = scores[score]["function"](
                                y_sets[set], y_pred
                            )
                        print(f"{model_name} - dataset {set}: {scores[score]['name']} = {metric}")

                        if mlflow_tracking:
                            mlflow.log_metric(f"{model_name}-{scores[score]['name']}-{set}", metric)

            def log_train_curve(model, model_name, scores, X_sets, y_sets):
                train_sizes, train_scores, validation_scores, fit_times, score_times = learning_curve(
                    estimator=model,
                    X=X_sets[0],
                    y=y_sets[0],
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring=scores["first"]["name"],
                    return_times=True
                )

                mean_train_scores = np.mean(train_scores, axis=1)
                mean_validation_scores = np.mean(validation_scores, axis=1)

                for idx, size in enumerate(train_sizes):
                    mlflow.log_metric(
                        key=f"{model_name}-{scores['first']['name']}-train",
                        value=mean_train_scores[idx],
                        step=size
                        )
                    mlflow.log_metric(
                        key=f"{model_name}-{scores['first']['name']}-validation",
                        value=mean_validation_scores[idx],
                        step=size)
            tuned_model =  GridSearchCV(
                estimator=model["model"], param_grid=model["parameters"]
                )
            tuning_start = time.time()
            tuned_model.fit(X=X_sets[0], y=y_sets[0])
            tuning_end = time.time()
            tune_time = tuning_end - tuning_start
            print(f"{model['name']} - parameters: {tuned_model.best_params_}")

            sk_score(tuned_model.best_estimator_, model['name'], scores, X_sets, y_sets, mlflow_tracking)

            if mlflow_tracking:  
                mlflow.log_metric(f"{model['name']}-tune_time", tune_time)
                mlflow.log_metric(f"{model['name']}-fit_time", tuned_model.refit_time_)
                mlflow.sklearn.log_model(
                    sk_model=tuned_model.best_estimator_,
                    artifact_path=f'{name}/{model_type}',
                    registered_model_name=f"{name}-{model['name']}"
                    )
                
                for param in tuned_model.best_params_.keys():
                    mlflow.log_param(f"{model['name']}-{param}", tuned_model.best_params_[param])
                
                log_train_curve(tuned_model.best_estimator_, model['name'], scores, X_sets, y_sets)

        if self.multiprocess:
            mlflow_tracking = self.mlflow_tracking
            name = self.name
            models = self.sk_models
            scores = self.scores
            X_sets = self.X_sets
            y_sets = self.y_sets
            parmap(lambda model: sk_tune(name, model, scores, X_sets, y_sets, mlflow_tracking), models)
        else:
            for model in self.sk_models:
                sk_tune(self.name, model, self.scores, self.X_sets, self.y_sets, self.mlflow_tracking)
        if self.mlflow_tracking:
            mlflow.end_run()

class TorchModelSelector:
    def __init__(self, name, models=None, tensors=None, n_features=None, n_labels=None,
                mode='regressor', mlflow_tracking_uri=None, multiproc=False) -> None:
        self.tensors = tensors
        self.name = name
        self.n_features=n_features
        self.n_labels=n_labels
        self.multiprocess = multiproc
        self.torch_max_score = np.NINF
        self.mlflow_tracking = False
        self.device, self.num_cpu, self.num_gpu = get_device()
        self.get_torch_models_and_metrics(mode, models)
        if self.device == "cuda:0":
            self.num_workers = self.num_gpu
        else:
            self.num_workers = self.num_cpu

        if type(mlflow_tracking_uri) != type(None):
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(self.name)
            mlflow.start_run()
            self.mlflow_tracking = True
    
    def get_torch_models_and_metrics(self, mode, models):
        if mode == 'regressor':
            self.scores = {
                "first": {"name": "mse", "function": mean_squared_error},
                "second": {"name": "r2", "function": r2_score}
            }

            self.torch_total_models = {
                CustomNetRegression.__name__: {
                    "model": CustomNetRegression,
                    "parameters": {
                        "num_layers": tune.grid_search([x for x in range(1, 3)]),
                        "neuron_incr": tune.grid_search([x for x in range(1, 3)]),
                        "dropout": tune.grid_search([x/10 for x in range(1, 3)]),
                        "batchnorm": tune.grid_search([False, True]),
                        "learning_rate": tune.grid_search([x/100 for x in range(1, 3)])
                    }
                }
            }

        elif mode == 'classifier':
            self.n_classes = len(np.unique(self.y_sets[0]))
            self.scores = {
                "first": {"name": "accuracy", "function": accuracy_score},
                "second": {"name": "f1", "function": f1_score}
            }

            self.torch_total_models = {
                CustomNetRegression.__name__: {
                    "model": CustomNetRegression,
                    "parameters": {
                        "num_layers": tune.grid_search([x for x in range(1, 3)]),
                        "neuron_incr": tune.grid_search([x for x in range(1, 3)]),
                        "dropout": tune.grid_search([x/10 for x in range(1, 3)]),
                        "batchnorm": tune.grid_search([False, True]),
                        "learning_rate": tune.grid_search([x/100 for x in range(1, 3)])
                    }
                }
            }
        
        if models is not None:
            self.torch_models = {}

            for model in models:
                self.torch_models[model] = self.torch_total_models[model]
        else:
            self.torch_models = self.torch_total_models
        
    def torch_train(self, config, model):
        """ Turns hyperparameters of the model. """
        net = self.torch_models[model]["model"](
            n_features=self.tensors[0].n_features,
            n_labels=self.tensors[0].n_labels,
            num_layers=config["num_layers"],
            neuron_incr=config["neuron_incr"],
            dropout=config["dropout"],
            batchnorm=config["batchnorm"],
            lr=config["learning_rate"]
            )
        net.fit(
            train_load=self.tensors[0].dataloader,
            test_load=self.tensors[1].dataloader,
            epochs=3
            )
    
    def torch_tune(self, model, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        
        reporter = CLIReporter(
            parameter_columns=list(self.torch_models[model]["parameters"].keys()),
            metric_columns=["train_loss", "val_loss", "training_iteration"]
            )

        tuning_start = time.time()
        print("CPU num:", self.num_cpu)
        result = tune.run(
            partial(self.torch_train, model=model),
            resources_per_trial={"cpu": 1},
            config=self.torch_models[model]["parameters"],
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            checkpoint_at_end=True)
        tuning_end = time.time()
        
        best_trial = result.get_best_trial("train_loss", "val_loss", "last", mode="min")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final train loss: {}".format(
            best_trial.last_result["train_loss"]))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["val_loss"]))

        self.torch_tuned_models[model] = {"best_model": self.torch_models[model]["model"](
            n_features=self.tensors[0].n_features,
            n_labels=self.tensors[0].n_labels,
            num_layers=best_trial.config["num_layers"],
            neuron_incr=best_trial.config["neuron_incr"],
            dropout=best_trial.config["dropout"],
            batchnorm=best_trial.config["batchnorm"]
            )}
        self.torch_tuned_models[model]["tune_time"] = tuning_end - tuning_start
        tuning_start = time.time()
        loss =  self.torch_tuned_models[model]["best_model"].fit(
            train_load=self.tensors[0].dataloader,
            test_load=self.tensors[1].dataloader,
            epochs=3, lr=best_trial.config["learning_rate"]
            )
        tuning_end = time.time()
        self.torch_tuned_models[model]["fit_time"] = tuning_end - tuning_start
        self.torch_tuned_models[model]["best_parameters"] = best_trial.config

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(
            best_checkpoint_dir, "checkpoint"))
        self.torch_tuned_models[model]["best_model"].load_state_dict(model_state)

    def torch_score(self, model):
        """ Returns the score of the tuned model for every set of the data. """
        y_sets = {}
        y_pred_sets = {}
        for score in self.scores.keys():
            self.torch_tuned_models[model][self.scores[score]["name"]] = {}
            for set in range(len(self.X_sets)):
                y_sets[set], y_pred_sets[set] = self.torch_tuned_models[model]["best_model"].predict(
                    self.tensors[set], return_y=True
                    )

                if self.scores[score]["name"] == "Accuracy":
                    self.torch_tuned_models[model][self.scores[score]["name"]][set] = self.scores[score]["function"](
                        y_sets[set].detach().numpy(), y_pred_sets[set].detach().numpy(), average='weighted'
                    )
                else:
                    self.torch_tuned_models[model][self.scores[score]["name"]][set] = self.scores[score]["function"](
                        y_sets[set].detach().numpy(), y_pred_sets[set].detach().numpy()
                    )
                print(f"{model} - dataset {set}: {self.scores[score]['name']} = {self.torch_tuned_models[model][self.scores[score]['name']][set]}")
        print(f"{model} - parameters: {self.torch_tuned_models[model]['best_parameters']}")

        if self.torch_tuned_models[model][self.scores['first']['name']][1] > self.torch_max_score:
            self.torch_best_model = self.torch_tuned_models[model]
            self.torch_best_model["name"] = model
    
    def torch_experiment(self, model, model_type="torch"):
        self.torch_tune(model)
        self.torch_score(model)
        tuned_model = self.torch_tuned_models[model]

        if self.mlflow_tracking:
            mlflow.log_metric(f"{model}-tune_time", tuned_model["tune_time"])
            mlflow.log_metric(f"{model}-fit_time", tuned_model["fit_time"])
            mlflow.sklearn.log_model(
                sk_model=tuned_model["best_model"],
                artifact_path=f'{self.name}/{model_type}',
                registered_model_name=f"{self.name}-{model}"
                )
            
            for param in tuned_model["best_parameters"].keys():
                mlflow.log_param(f"{model}-{param}", tuned_model["best_parameters"][param])

            for score in self.scores.keys():
                for set in range(len(self.X_sets)):
                    mlflow.log_metric(f"{model}-{self.scores[score]['name']}-{set}", tuned_model[self.scores[score]['name']][set])
            
            for idx, size in enumerate(self.torch_tuned_models[model]["train_sizes"]):
                mlflow.log_metric(
                    key=f"{model}-{self.scores['first']['name']}-train",
                    value=self.torch_tuned_models[model]["train_scores"][idx],
                    step=size
                    )
                mlflow.log_metric(
                    key=f"{model}-{self.scores['first']['name']}-validation",
                    value=self.torch_tuned_models[model]["validation_scores"][idx],
                    step=size)

    def flow(self):
        self.torch_tuned_models = {}
        if self.multiprocess:
            parmap(lambda model: self.torch_experiment(model), list(self.torch_models.keys()))
        else:
            for model in self.torch_models.keys():
                self.torch_experiment(model)
        
        self.best_model = self.torch_best_model
        self.best_model["name"] = self.torch_best_model["name"]


        if self.mlflow_tracking:
            mlflow.log_param("BestModel", self.best_model["name"])
            mlflow.log_param("Torch_BestModel", self.torch_best_model["name"])
            mlflow.end_run()