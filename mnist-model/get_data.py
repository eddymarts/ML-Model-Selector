import pandas as pd
pd.options.display.max_columns = None
from sklearn.preprocessing import OrdinalEncoder
from torchvision import datasets, transforms
import torch
import plotly.express as px
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.dataset import NumpyDataset, TorchDataSet

class MNISTData(TorchDataSet):
    def __init__(self, split=False, normalize=False, shuffle=True, seed=None):
        X, y = self.get_X_y()
        super().__init__(X=X, y=y, one_hot_target=False, normalize=normalize, split=split, dataloader_shuffle=shuffle, seed=seed, label_type='categoric')
        # self.get_tensors()

    def get_X_y(self):
        mnist_train = datasets.MNIST(root="./mnist-model/datasets/mnist_train",
                             download=True, train=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))]))

        mnist_test = datasets.MNIST(root="./mnist-model/datasets/mnist_test",
                                    download=True, train=False,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))]))
        
        X = mnist_train.data
        X = torch.cat((X, mnist_test.data), dim=0).reshape(-1, 1, 28, 28)
        y = mnist_train.targets
        y = torch.cat((y, mnist_test.targets), dim=0)
        
        return X.detach().numpy(), y.detach().numpy()

    


if __name__ == "__main__":
    mnist = MNISTData(split=True, normalize=True)
    X = mnist.X
    print(X.shape)
    print(X.dtype)
    print(torch.unique(mnist.y))
    print(mnist.y_sets[0].shape)