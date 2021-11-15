from sklearn.metrics import *
from mlflow.tracking import MlflowClient
import mlflow
from get_data import MelbourneData

client = MlflowClient()
melbourne = MelbourneData(split=True, normalize=True, shuffle=True, seed=None)
model_name = "Melbourne-BestModel"

max_version = 0
for model in client.search_model_versions(f"name='{model_name}'"):
    model = dict(model)
    version = int(model['version'])
    
    if version > max_version:
        max_version = version

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{max_version}")
y_pred= model.predict(melbourne.X_sets[1])

score = r2_score(y_pred=y_pred, y_true=melbourne.y_sets[1])
print("Best R2:", score) 