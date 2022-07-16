import catboost
from ml.predict import PytorchModel
from ml.predict import CatboostModel


def get_model(model_path=None):
    model = CatboostModel(model_path)
    return model


def get_nn_model(device='cpu'):
    return PytorchModel(device=device)