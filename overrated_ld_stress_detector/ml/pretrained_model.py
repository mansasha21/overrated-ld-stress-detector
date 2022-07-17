from overrated_ld_stress_detector.ml.predict import PytorchModel
from overrated_ld_stress_detector.ml.predict import CatboostModel


def get_model(model_type='catboost',
              model_path=None,
              device='cpu'):
    """
    Returns a model object: convolutional neural network or gradient boosting catboost.
    :param model_type: str, one of ['cnn', 'catboost']
    :param model_path: str, path to model file
    :param device: str, device to use ['cpu', 'cuda']
    """
    model = None
    if model_type == 'catboost':
        model = CatboostModel(model_path)
    elif model_type == 'cnn':
        model = PytorchModel(device=device)
    return model
