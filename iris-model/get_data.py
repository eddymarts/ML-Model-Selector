import pandas as pd
pd.options.display.max_columns = None
from sklearn.preprocessing import OrdinalEncoder
from sklearn import datasets
import plotly.express as px
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.dataset import NumpyDataset, TorchDataSet

class IrisData(NumpyDataset):
    def __init__(self, split=False, normalize=False, shuffle=True, seed=None):
        X, y = datasets.load_iris(return_X_y=True)
        super().__init__(X=X, y=y, split=split, normalize=normalize, shuffle=shuffle, seed=seed, label_type='categoric')
        # self.get_tensors()
    
    def get_tensors(self):
        self.tensors = {}
        self.dataloaders = {}

        for set in self.X_sets.keys():
            if set == 0:
                shuffle = True
            else:
                shuffle = False

            self.tensors[set] = TorchDataSet(X=self.X_sets[set], y=self.y_sets[set], dataloader_shuffle=shuffle)

    


if __name__ == "__main__":
    iris = IrisData(split=True, normalize=True)
    y = iris.y_sets[0]
    print(y.shape)
    print(y.dtype)
    import numpy as np
    print(np.unique(iris.y))