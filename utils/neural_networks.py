import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        """
        Predicts the value of an output for each row of X
        using the model.
        """
        return self.layers(X)

    def fit(self, train_load, test_load=None, optimiser=None, lr = 0.001, epochs=1000,
            acceptable_error=0.001, return_loss=False):
        """
        Optimises the model parameters for the given data.

        INPUTS: train_load -> torch.utils.data.DataLoader object with the data.
                lr -> Learning Rate of Mini-batch Gradient Descent.
                        default = 0.001.
                epochs -> Number of iterationns of Mini-Batch Gradient Descent.
                        default = 100
        """

        if optimiser==None:
            optimiser = torch.optim.SGD(self.parameters(), lr=lr)

        writer = SummaryWriter()

        mean_train_loss = []
        mean_validation_loss = []

        for epoch in range(epochs):
            print(epoch)
            training_loss = []
            self.train()
            for X_train, y_train in train_load:
                optimiser.zero_grad()
                y_hat = self.forward(X_train)
                train_loss = self.get_loss(y_hat, y_train)
                print(train_loss)
                training_loss.append(train_loss.item())
                train_loss.backward()
                optimiser.step()
            
            mean_train_loss.append(np.mean(training_loss))
            writer.add_scalar("./loss/train", mean_train_loss[-1], epoch)
            
            if test_load:
                validation_loss = []
                self.eval() # set model in inference mode (need this because of dropout)
                for X_val, y_val in test_load:
                    y_hat_val = self.forward(X_val)
                    val_loss = self.get_loss(y_hat_val, y_val)
                    validation_loss.append(val_loss.item())
                mean_validation_loss.append(np.mean(validation_loss))
                writer.add_scalar("./loss/validation", mean_validation_loss[-1], epoch)

                # if epoch > 2 and (
                #     (abs(mean_validation_loss[-2]- mean_validation_loss[-1])/mean_validation_loss[-1] < acceptable_error)
                #     or (mean_validation_loss[-1] > mean_validation_loss[-2])):
                #     print(f"Validation train_loss for epoch {epoch} is {mean_validation_loss[-1]}")
                #     break
        
        writer.close()
        if return_loss:
            return {'training': mean_train_loss,
                    'validation': mean_validation_loss}
        
    def predict(self, data_load):
        """
        Predicts the value of an output for each row of X
        using the fitted model.

        X is the data from data_load (DataLoader object).

        Returns the predictions.
        """
        self.eval()
        for idx, X in enumerate(data_load):
            if idx == 0:
                y_hat = self(X)
            else:
                y_hat = torch.cat((y_hat, self(X)), dim=0)

        return y_hat

class NeuralNetworkRegression(NeuralNetwork):
    def __init__(self, n_features, n_labels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, 9),
            torch.nn.ReLU(),
            torch.nn.Linear(9, n_labels)
        )
    
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