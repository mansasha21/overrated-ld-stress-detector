import pickle

import numpy as np
import overrated_ld_stress_detector.preprocessing as preprocessing
from scipy import stats


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
                 model_count=5) -> None:
        self.models = [pickle.load(open(model_path + "/model_" + str(i) + ".pckl", 'rb')) for i in range(model_count)]
        self.model_count = model_count
    def predict(self, df):
        '''
        :param df: pandas.DataFrame
        '''
        modified_data = preprocessing.process_data(df)
        result = np.array([self.models[i].predict(modified_data).flatten() for i in range(0, self.model_count)])
        return stats.mode(result)[0][0]
