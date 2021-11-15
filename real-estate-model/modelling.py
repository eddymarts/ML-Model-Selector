from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from get_data import MelbourneData, TorchDataSet
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from sklearn.metrics import *

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.machine_learning import ModelSelector
from utils.neural_networks import NeuralNetworkRegression

if __name__ == '__main__':
    melbourne = MelbourneData(split=True, normalize=True, shuffle=True, seed=None)

    models = {
        LinearRegression.__name__: {
            "model": LinearRegression(),
            "parameters": {'fit_intercept': [True, False]}
        },
        KNeighborsRegressor.__name__: {
            "model": KNeighborsRegressor(),
            "parameters": {
                'n_neighbors': list(range(15, 30)), 
                'weights': ['uniform', 'distance']
                }
        },
        # SVR.__name__: {
        #     "model": SVR(),
        #     "parameters": {
        #         'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 
        #         'degree': list(range(2, 10)),
        #         'gamma': ['scale', 'auto'],
        #         'coef0': list(range(100)),
        #         'C': [num/1 for num in range(10)],
        #         'epsilon': [num/100 for num in range(10)],
        #         'shrinking': [False, True]
        #         }
        # },
        DecisionTreeRegressor.__name__: {
            "model": DecisionTreeRegressor(),
            "parameters": {
                'min_samples_split': list(range(2, 10)), 
                'min_samples_leaf': list(range(1, 10)),
                'min_weight_fraction_leaf': [num for num in range(5)]
                }
        }
        # RandomForestRegressor.__name__: {
        #     "model": RandomForestRegressor(),
        #     "parameters": {
        #         'n_estimators': [num*10 for num in range(10, 11)],
        #         'min_samples_split': list(range(2, 4)), 
        #         'min_samples_leaf': list(range(1, 3)),
        #         'min_weight_fraction_leaf': [num/10 for num in range(2)]
        #         }
        # },
        # MLPRegressor.__name__: {
        #     "model": MLPRegressor(),
        #     "parameters": {
        #         'hidden_layer_sizes': [(100, 100, 100, 100, 100)],
        #         'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #         'alpha': [num/10000 for num in range(1, 100)],
        #         'max_iter': [1000]
        #         }
        # }
    }

    # model_selector = ModelSelector(
    #     name="Melbourne",
    #     X_sets=melbourne.X_sets,
    #     y_sets=melbourne.y_sets,
    #     sk_models=models,
    #     mode='regressor',
    #     mlflow_tracking=True,
    #     multiproc=False
    #     )
    # model_selector.sk_flow()

    
    nn_regressor = NeuralNetworkRegression(melbourne.n_features, melbourne.n_labels)
    nn_regressor.fit(train_load=melbourne.dataloaders[0], test_load=melbourne.dataloaders[1], epochs=2)
    X_test = TorchDataSet(X=melbourne.X_sets[1])
    X_test_loader = DataLoader(X_test, batch_size=16,
                shuffle=False, num_workers=cpu_count()-1)
    y_pred = nn_regressor.predict(X_test_loader)
    print(y_pred)
    print(type(y_pred))

    print("R2 score:", r2_score(y_pred.detach().numpy(), melbourne.y_sets[1]))



