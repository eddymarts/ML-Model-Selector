import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import *
from torchvision import transforms
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.device import get_device

class NumpyDataset:
  def __init__(self, X, y, split=False, normalize=False, shuffle=True, seed=None, label_type='continuous'):
    if len(X.shape) > 1:
      self.n_features = int(X.shape[1])
      self.X = X
    else:
      self.n_features = int(1)
      self.X = X.reshape(-1, self.n_features)
    
    if y is not None:
      if len(y.shape) > 1:
        self.n_labels = int(y.shape[1])
      else:
        self.n_labels = int(1)

      if label_type == 'continuous':
        self.label_type = np.float32
        self.label_shape = (-1, self.n_labels)
      else:
        if self.n_labels > 1:
          self.label_shape = (-1, self.n_labels)
        else:
          self.label_shape = (-1,)

        if label_type == 'binary':
          self.label_type = np.float32
        else:
          self.label_type = np.longlong

      self.y = y.astype(self.label_type).reshape(*self.label_shape)
    else:
      self.y = None

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
      X_sets[0] = X_train.astype(np.float32)
      X_sets[set+1] = X_test.astype(np.float32)
      y_sets[0] = np.array(y_train).astype(self.label_type).reshape(*self.label_shape)
      y_sets[set+1] = np.array(y_test).astype(self.label_type).reshape(*self.label_shape)
      print(y_sets[0].shape, y_sets[set+1].shape)
    
    if X is None and y is None:
      self.X_sets = X_sets
      self.y_sets = y_sets
    
    return X_sets, y_sets

class TorchDataSet(Dataset):
  """
  Class that implements torch.utils.data.Dataset
  """
  def __init__(self, X, y=None, one_hot_target=False, normalize=False,
              split=False, dataloader_shuffle=False, seed=None, label_type='continuous'):
    super().__init__()
    self.device, self.num_cpu, self.num_gpu = get_device()

    if self.device == "cuda:0":
        self.num_workers = self.num_gpu
    else:
        self.num_workers = self.num_cpu

    if len(X.shape) > 1:
      self.n_features = int(X.shape[1])
      self.X = torch.Tensor(X).float()
    else:
      self.n_features = int(1)
      self.X = torch.Tensor(X.reshape(-1, self.n_features)).float()

    if y is not None:
      if len(y.shape) > 1:
        self.n_labels = int(y.shape[1])
      else:
        self.n_labels = int(1)
      if label_type == 'continuous':
        self.label_type = torch.float32
        self.label_shape = (-1, self.n_labels)
      else:
        if self.n_labels > 1:
          self.label_shape = (-1, self.n_labels)
        else:
          self.label_shape = (-1,)

        if label_type == 'binary':
          self.label_type = torch.float32
        else:
          self.label_type = torch.long

      self.y = torch.Tensor(y).to(self.label_type).reshape(*self.label_shape)
    else:
      self.y = None

    if one_hot_target:
      self.y_oh = self.one_hot(self.y)

    self.split=split
    if split:
      self.get_split(seed=seed)

    if normalize:
      self.normalize()
    
    self.get_dataloaders(shuffle=dataloader_shuffle)

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
    if self.split:
      self.std, self.mean = torch.std_mean(self.X_sets[0], unbiased=False)
    else:
      self.std, self.mean = torch.std_mean(self.X, unbiased=False)

    self.normalize_tensor = transforms.Normalize((self.mean,), (self.std,))
    if data == None:
      self.X_raw = self.X
      self.X = self.normalize_tensor(self.X)
      
      if self.split:
        for idx in range(len(self.splits)):
          self.X_sets[idx] = self.normalize_tensor(self.X_sets[idx])
      return self.X
    else:
      return self.normalize_tensor(data)

  def get_split(self, seed, sizes=[0.7, 0.15, 0.15], shuffle=True):
    lengths = [round(len(self)*size) for size in sizes]
    lengths[-1] = len(self) - sum(lengths[:-1])

    if seed == None:
      self.splits = random_split(self, lengths)
    else:
      self.splits = random_split(self, lengths,
        generator=torch.Generator().manual_seed(seed))
    
    self.X_sets = {}
    self.y_sets = {}
    for idx in range(len(self.splits)):
      self.X_sets[idx] = self.X[self.splits[idx].indices]
      self.y_sets[idx] = self.y[self.splits[idx].indices]
  
  def get_dataloaders(self, shuffle=False):
    if self.split:
      self.dataloaders = []
      for idx, split in enumerate(self.splits):
        if idx == 0:
          shuffle = False
        else:
          shuffle = True
        self.dataloaders.append(DataLoader(split, batch_size=16, shuffle=shuffle, num_workers=self.num_workers))
    else:
      self.dataloader = DataLoader(self, batch_size=16, shuffle=shuffle, num_workers=self.num_workers)