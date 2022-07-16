import pickle

from catboost import CatBoostClassifier
import overrated_ld_stress_detector.preprocessing as preprocessing
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
        self.model = pickle.load(open(model_path,'rb'))

    def predict(self, df):
        '''
        :param df: pandas.DataFrame
        '''
        modified_data = preprocessing.process_data(df)
        modified_data.to_csv("modified_data.csv")
        print(modified_data.columns)
        return self.model.predict(modified_data)
