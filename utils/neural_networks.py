import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import torch
import torch.nn.functional as F
from ray import tune
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils.device import get_device

class NeuralNetwork(torch.nn.Module):
    """
    Abstract class for Neural Network model.
    implemented from torch.nn.Module.
    Only accepts numerical features.

    Must implement:
    - layers attribute
    - get_loss method
    
    Methods created to match the ones used by Sci-kit Learn models.
    """
    def __init__(self, lr = 0.001):
        super().__init__()
        self.lr = lr
        self.device, self.num_cpu, self.num_gpu = get_device()

        if self.device == "cuda:0":
            self.num_workers = self.num_gpu
            if self.num_gpu > 1:
                self = torch.nn.DataParallel(self)
            
            self.to(self.device)
        else:
            self.num_workers = self.num_cpu
                
    def forward(self, X):
        """
        Predicts the value of an output for each row of X
        using the model.
        """
        return self.layers(X)

    def fit(self, train_load, test_load=None, optimiser=None, epochs=1000,
            acceptable_error=0.001, return_loss=False, checkpoint_dir=None):
        """
        Optimises the model parameters for the given data.

        INPUTS: train_load -> torch.utils.data.DataLoader object with the data.
                lr -> Learning Rate of Mini-batch Gradient Descent.
                        default = 0.001.
                epochs -> Number of iterationns of Mini-Batch Gradient Descent.
                        default = 100
        """

        if optimiser==None:
            self.optimiser = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            self.optimiser = optimiser

        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            self.load_state_dict(model_state)
            self.optimiser.load_state_dict(optimizer_state)

        writer = SummaryWriter()

        mean_train_loss = []
        mean_validation_loss = []

        for epoch in range(epochs):
            print(f"{type(self).__name__}: Train epoch {epoch}")
            training_loss = []
            self.train()
            for X_train, y_train in train_load:
                X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                self.optimiser.zero_grad()
                y_hat = self.forward(X_train)
                train_loss = self.get_loss(y_hat, y_train)
                if train_loss > 1e+38:
                    print(f"{type(self).__name__}: Train epoch {epoch}, Error overload")
                    return -1
                training_loss.append(train_loss.item())
                train_loss.backward()
                self.optimiser.step()
            
            mean_train_loss.append(np.mean(training_loss))
            writer.add_scalar("./loss/train", mean_train_loss[-1], epoch)

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((self.state_dict(), self.optimiser.state_dict()), path)
            
            if test_load:
                validation_loss = []
                self.eval() # set model in inference mode (need this because of dropout)
                for X_val, y_val in test_load:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    y_hat_val = self.forward(X_val)
                    val_loss = self.get_loss(y_hat_val, y_val)
                    validation_loss.append(val_loss.item())
                mean_validation_loss.append(np.mean(validation_loss))
                writer.add_scalar("./loss/validation", mean_validation_loss[-1], epoch)
                tune.report(train_loss=mean_train_loss[-1], val_loss=mean_validation_loss[-1])
        
                # if epoch > 2 and (
                #     (abs(mean_validation_loss[-2]- mean_validation_loss[-1])/mean_validation_loss[-1] < acceptable_error)
                #     or (mean_validation_loss[-1] > mean_validation_loss[-2])):
                #     print(f"Validation train_loss for epoch {epoch} is {mean_validation_loss[-1]}")
                #     break
        
        writer.close()
        print(f"{type(self).__name__}: Finished Training")
        if return_loss:
            return {'training': mean_train_loss,
                    'validation': mean_validation_loss}
        
    def predict(self, data_load, return_y=False):
        """
        Predicts the value of an output for each row of X
        using the fitted model.

        X is the data from data_load (DataLoader object).

        Returns the predictions.
        """
        self.eval()
        for idx, data in enumerate(data_load):
            if return_y:
                X, y = data
                X, y = X.to(self.device), y.to(self.device)
                if idx == 0:
                    y_hat = self(X)
                    y_label = y
                else:
                    y_hat = torch.cat((y_hat, self(X)), dim=0)
                    y_label = torch.cat((y_label, y), dim=0)
            else:
                X = data.to(self.device)
                if idx == 0:
                    y_hat = self(X)
                else:
                    y_hat = torch.cat((y_hat, self(X)), dim=0)

        if return_y:
            return y_label, y_hat
        return y_hat

class NeuralNetworkRegression(NeuralNetwork):
    def __init__(self, n_features, n_labels):
        super().__init__()
        self.layers = torch.nn.Linear(n_features, n_labels)
    
    def get_loss(self, y_hat, y):
        """
        Gets Mean Squared Error between predictions of X and actual value (y).
        
        INPUT:  y_hat -> Tensor with predicted values.
                y -> Tensor with labels.
        
        OUTPUT: loss -> Mean Squared Error between predictions and actual labels.
        """
        return F.mse_loss(y_hat, y)


        # def train(self, model, train_dataloader, test_dataloader, val_dataloader, epochs=25, lr=0.50, weight_decay_L2= 0, threshold= 0.5, print_losses=False):
        #     # optimiser = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay_L2) # create optimiser
        #     optimiser = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
        #     losses = []
        #     batch_idx = 0
        #     for epoch in tqdm(range(epochs)):
        #         for step, (X, y) in  enumerate(train_dataloader):
        # https://playlai-container-service-2.8ljng1cpf8ma0.eu-west-2.cs.amazonlightsail.com/#/

class CustomNetRegression(NeuralNetworkRegression):
    def __init__(self, n_features=11, n_labels=1, num_layers=10, neuron_incr=10, 
                dropout=0.5, batchnorm=False):
        super().__init__(n_features, n_labels)
        self.layers = torch.nn.ModuleList(self.get_layers(n_features, n_labels, num_layers,
                                        neuron_incr, dropout, batchnorm))
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def get_layers(self, n_features, n_labels, num_layers, neuron_incr, dropout, batchnorm):
        current_neurons = n_features
        layers = []

        for layer in range(num_layers):
            if layer <= round(num_layers/2):
                next_neurons = current_neurons + neuron_incr
            else:
                next_neurons = current_neurons - neuron_incr

            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(current_neurons))

            layers.append(torch.nn.Linear(current_neurons, next_neurons))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            current_neurons = next_neurons
        
        # print(current_neurons)

        if batchnorm:
            layers.append(torch.nn.BatchNorm1d(current_neurons))
        layers.append(torch.nn.Linear(current_neurons, n_labels))

        return layers

class CustomBaseNetRegression(torch.nn.Module):
    def __init__(self, n_features=11, n_labels=1, num_layers=10, neuron_incr=10, 
                dropout=0.5, batchnorm=False):
        super().__init__()
        self.layers = torch.nn.ModuleList(self.get_layers(n_features, n_labels, num_layers,
                                        neuron_incr, dropout, batchnorm))
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def get_layers(self, n_features, n_labels, num_layers, neuron_incr, dropout, batchnorm):
        current_neurons = n_features
        layers = []

        for layer in range(num_layers):
            if layer <= round(num_layers/2):
                next_neurons = current_neurons + neuron_incr
            else:
                next_neurons = current_neurons - neuron_incr

            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(current_neurons))

            layers.append(torch.nn.Linear(current_neurons, next_neurons))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            current_neurons = next_neurons
        
        # print(current_neurons)

        if batchnorm:
            layers.append(torch.nn.BatchNorm1d(current_neurons))
        layers.append(torch.nn.Linear(current_neurons, n_labels))

        return layers

class CustomBaseNetBiClassification(CustomBaseNetRegression):
    def __init__(self, n_features, n_labels, num_layers=10, neuron_incr=10,
                dropout=0.5, batchnorm=False):
        super().__init__(n_features, n_labels, num_layers=num_layers,
                neuron_incr=neuron_incr, dropout=dropout, batchnorm=batchnorm)
        self.layers = torch.nn.ModuleList(self.get_layers(n_features, n_labels, num_layers,
                                        neuron_incr, dropout, batchnorm) + [torch.nn.Sigmoid()])

class CustomBaseNetClassification(CustomBaseNetRegression):
    def __init__(self, n_features, n_labels, num_layers=10, neuron_incr=10,
                dropout=0.5, batchnorm=False):
        super().__init__(n_features, n_labels, num_layers=num_layers,
                neuron_incr=neuron_incr, dropout=dropout, batchnorm=batchnorm)
        self.layers = torch.nn.ModuleList(self.get_layers(n_features, n_labels, num_layers,
                                        neuron_incr, dropout, batchnorm) + [torch.nn.Softmax(1)])