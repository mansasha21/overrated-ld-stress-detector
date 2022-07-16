import catboost
from ml.predict import PytorchModel
from ml.predict import CatboostModel


def load_weights():
    return ""


def get_model(model_path=None,
              model_type='fastest'):
    if model_path is None:
        model_path = load_weights()
    model = CatboostModel(model_path)
    return model
