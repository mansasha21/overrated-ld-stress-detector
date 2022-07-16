import catboost
import torch


class PytorchModel:
    def __init__(self, model_path, device: str = "cpu") -> None:
        self.model = (model_path).to(device)
        self.device = device
        self.variance = [0.1, 0.2]

    def load_state_dict(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)

    def eval(self) -> None:
        self.model.eval()

    def predict(data):
        pass


class CatboostModel:
    def __init__(self,
                 model_path=None,
                 model_type="") -> None:
        self.model = catboost.CatBoost(model_path)

    def eval(self) -> None:
        self.model.eval()

    def predict(data):
        pass
