import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import *

class NumpyDataset:
  def __init__(self, X, y, split=False, normalize=False, shuffle=True, seed=None):
    if len(X.shape) > 1:
      self.n_features = X.shape[1]
      self.X = X
    else:
      self.n_features = 1
      self.X = X.reshape(-1, self.n_features)

    if len(y.shape) > 1:
      self.n_labels = y.shape[1]
      self.y = y
    else:
      self.n_labels = 1
      self.y = y.reshape(-1, self.n_labels)

    if split:
      self.split(seed=seed)
    
    if normalize:
      self.normalize(split=split)

  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
      return len(self.X)

  def normalize(self, data=None, split=False):
    if data == None:
      if split:
        self.sc = StandardScaler().fit(self.X_sets[0])

        for set in self.X_sets.keys():
          self.X_sets[set] = self.sc.transform(self.X_sets[set])
      
      else:
        self.sc = StandardScaler().fit(self.X)
        self.X_raw = self.X
        self.X = self.sc.transform(self.X)
        return self.X
    else:
      return self.sc.transform(data)

  def split(self, X=None, y=None, test_size=0.25, sets=1, shuffle=True, seed=None):
    if X is None:
      X_train = self.X
    else:
      X_train = X

    if y is None:
      y_train = self.y
    else:
      y_train = y

    X_sets = {}
    y_sets = {}
    for set in range(sets):
      X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, 
                                      shuffle=shuffle, random_state=seed)
      X_sets[0] = X_train
      X_sets[set+1] = X_test
      y_sets[0] = np.array(y_train).reshape(-1,)
      y_sets[set+1] = np.array(y_test).reshape(-1,)
      print(y_sets[0].shape, y_sets[set+1].shape)
    
    if X is None and y is None:
      self.X_sets = X_sets
      self.y_sets = y_sets
    
    return X_sets, y_sets

class TorchDataSet(Dataset):
  """
  Class that implements torch.utils.data.Dataset
  """
  def __init__(self, X, y=None, one_hot_target=False,
              normalize=False, split=False, seed=None):
    super().__init__()

    if len(X.shape) > 1:
      self.n_features = X.shape[1]
      self.X = torch.Tensor(X).float()
    else:
      self.n_features = 1
      self.X = torch.Tensor(X.reshape(-1, self.n_features)).float()

    if type(y) != type(None):
      if len(y.shape) > 1:
        self.n_labels = y.shape[1]
        self.y = torch.Tensor(y).float()
      else:
        self.n_labels = 1
        self.y = torch.Tensor(y.reshape(-1, self.n_labels)).float()
    else:
      self.y = None

    self.mean = torch.mean(self.X, axis=0)
    self.std = torch.std(self.X, axis=0)

    if one_hot_target:
      self.y_oh = self.one_hot(self.y)

    if normalize:
      self.normalize()

    if split:
      self.split(seed)

  def __getitem__(self, idx):
    if type(self.y) != type(None):
      return self.X[idx], self.y[idx]
    else:
      return self.X[idx]

  def __len__(self):
      return len(self.X)
  
  def one_hot(self, y):
    """ For a categorical array y, returns a matrix of the one-hot encoded data. """
    m = y.shape[0]
    y = y.long()
    onehot = torch.zeros((m, int(torch.max(y)+1)))
    onehot[range(m), y] = 1
    return onehot

  def normalize(self, data=None):
    if data == None:
      self.X_raw = self.X
      self.X = (self.X - self.mean)/self.std
      return self.X
    else:
      return (data - self.mean)/self.std

  def split(self, seed, sizes=[0.7, 0.15, 0.15], shuffle=True):
    lengths = [round(len(self)*size) for size in sizes]
    lengths[-1] = len(self) - sum(lengths[:-1])

    if seed == None:
      self.splits = random_split(self, lengths)
    else:
      self.splits = random_split(self, lengths,
        generator=torch.Generator().manual_seed(seed))