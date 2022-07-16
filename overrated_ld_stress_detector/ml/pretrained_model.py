import catboost
from overrated_ld_stress_detector.ml.predict import PytorchModel
from overrated_ld_stress_detector.ml.predict import CatboostModel


def get_model(model_path=None):
    model = CatboostModel(model_path)
    return model
