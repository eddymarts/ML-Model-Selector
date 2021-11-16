from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import time
import numpy as np
import mlflow
from utils.multiproc import parmap

class ModelSelector:
    def __init__(self, name, X_sets, y_sets, sk_models, mode='regressor', mlflow_tracking=False, multiproc=False) -> None:
        self.X_sets = X_sets
        self.y_sets = y_sets
        self.sk_models = sk_models
        self.name = name
        self.mlflow_tracking = mlflow_tracking
        self.multiprocess = multiproc
        self.max_score = np.NINF
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
        self.sk_tuned_models[model]["model"].fit(X=self.X_sets[0], y=self.y_sets[0])
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
            for set in range(len(self.X_sets)):
                y_pred_sets[set] = self.sk_tuned_models[model]["best_model"].predict(self.X_sets[set])

                if self.scores[score]["name"] == "Accuracy":
                    self.sk_tuned_models[model][self.scores[score]["name"]][set] = self.scores[score]["function"](
                        self.y_sets[set], y_pred_sets[set], average='weighted'
                    )
                else:
                    self.sk_tuned_models[model][self.scores[score]["name"]][set] = self.scores[score]["function"](
                        self.y_sets[set], y_pred_sets[set]
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
                for set in range(len(self.X_sets)):
                    mlflow.log_metric(f"{model}-{self.scores[score]['name']}-{set}", self.sk_tuned_models[model][self.scores[score]['name']][set])

    def sk_flow(self):
        self.sk_tuned_models = {}
        if self.mlflow_tracking:
            # mlflow.set_tracking_uri("http://localhost:5000")
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

            
            

