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

class BostonData(NumpyDataset):
    def __init__(self, split=False, normalize=False, shuffle=True, seed=None):
        X, y = datasets.load_boston(return_X_y=True)
        super().__init__(X=X, y=y, split=split, normalize=normalize, shuffle=shuffle, seed=seed)
        self.get_tensors()
    
    def _feature_engineer(self, dataset):
        dataset["Date"] = pd.to_datetime(dataset["Date"], format='%d/%m/%Y')
        min_date = dataset["Date"].min()
        dataset["DateDelta"] = dataset["Date"].apply(lambda x: x - min_date)
        dataset["Days"] = dataset["DateDelta"].dt.days
        dataset["Year"] = pd.DatetimeIndex(dataset["Date"]).year
        dataset["Price"] = dataset["Price"].apply(lambda x: x/1000)
        dataset.drop(axis=1, columns=["Address", "Date"], inplace=True)
        dataset.dropna(axis=0, inplace=True)
        self.dataset = dataset[[
            "Type", "Rooms", "Method", "SellerG", "Days", "Postcode", "CouncilArea",
            "Regionname", "Suburb", "Distance", "Propertycount", "Price"
            ]]
    
    def _visualize(self):
        # histograms = []
        # histograms.append(px.histogram(self.dataset, "Type", labels={"value": "Type"}))
        # histograms.append(px.histogram(self.dataset, "Rooms", labels={"value": "Rooms"}, nbins=31))
        # histograms.append(px.histogram(self.dataset, "Method", labels={"value": "Method"}))
        # histograms.append(px.histogram(self.dataset, "SellerG", labels={"value": "SellerG"}))
        # histograms.append(px.histogram(self.dataset, "Year", labels={"value": "Year"}))
        
        # for fig in histograms:
        #     fig.show()

        price_method = self.dataset[["Method", "Price"]]
        price_method_groups = price_method.groupby(["Method", "Price"]).size().unstack()
        print(price_method_groups)
        sfsf
        fig = px.bar(price_method_groups, title="Count of Price per Method", labels={"title": "Price", "value": "Count"})
        fig.update_layout(legend_title_text='Method')
        # fig.update_layout(barmode="group")
        fig.show()

    
    def _encode(self):
        self.ord_enc = OrdinalEncoder()
        self.ord_enc.fit(self.dataset[["Type", "Method", "SellerG", "Postcode", "CouncilArea", "Regionname", "Suburb"]])
        self.enc_dataset = self.dataset
        self.enc_dataset[[
            "Type", "Method", "SellerG", "Postcode", "CouncilArea", "Regionname", "Suburb"
            ]] = self.ord_enc.transform(self.dataset[[
                "Type", "Method", "SellerG", "Postcode", "CouncilArea", "Regionname", "Suburb"
                ]])
        

        X = self.enc_dataset[[column for column in self.dataset if column != "Price"]]
        y = self.enc_dataset[["Price"]]

        X.drop(axis=1, columns=["SellerG", "CouncilArea", "Days"], inplace=True)

        return X.to_numpy(), y.to_numpy()
    
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
    boston = BostonData(split=True, normalize=True)
    for X, y in boston.tensors[0].dataloader:
        print(X, y)
        sfsd