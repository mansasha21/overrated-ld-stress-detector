import pickle
import numpy as np
import torch
import torch.nn as nn
import overrated_ld_stress_detector.preprocessing as preprocessing
from scipy import stats


class SignalModel(nn.Module):
    """
    Define architecture of the used model
    """
    def __init__(self):
        """
        Initialize the model with Convolution 1d layers
        """
        super().__init__()
        self.conv_net1 = nn.Sequential(
            nn.Conv1d(1, 64, 40, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 40, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, 40, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(240),
            nn.Flatten(),
        )

        self.conv_net2 = nn.Sequential(
            nn.Conv1d(1, 64, 40, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 40, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, 40, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(240),
            nn.Flatten()
        )

        self.proc_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        """
        Forward pass of the model
        """
        x_1 = x[..., 0:240]
        x_2 = x[..., 240:]
        conv_out1 = self.conv_net1(x_1)
        conv_out2 = self.conv_net2(x_2)

        out_c = torch.concat((conv_out1, conv_out2), dim=-1)
        return self.proc_net(out_c)


class PytorchModel:
    """
    Convoluted neural network model for predicting the stress level of a signal.
    """
    def __init__(self,
                 model_path='models/nn_full.pth',
                 device: str = "cpu") -> None:
        """
        :param model_path: str, path to model file
        :param device: str, device to use ['cpu', 'cuda']
        """
        if model_path is None:
            model_path = 'models/nn_full.pth'
        self.model = SignalModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.device = device

    def predict(self, data):
        """
        Predict results by blending predictions from all models.
        :param data: pandas.DataFrame - raw dataset to analyze
        :return: pandas.DataFrame with predicted results
        """
        self.model.eval()
        prepared_data = torch.tensor(preprocessing.process_data_nn(data), dtype=torch.float32).to(self.device).unsqueeze(1)
        return self.model(prepared_data).cpu().detach().argmax(1).numpy().flatten()


class CatboostModel:
    """
    Gradient boosting catboost model
    """
    def __init__(self,
                 model_path='models',
                 model_count=5) -> None:
        """
        Load and initialize models from files.
        :param model_path: str, path to folder with models. Models name should be 'model_<number>.pkl'
        :param model_count: int, number of models to load
        """
        if model_path is None:
            model_path = 'models'
        self.models = [pickle.load(open(model_path + '/model_' + str(i) + ".pckl", 'rb')) for i in range(model_count)]
        self.model_count = model_count

    def predict(self, df):
        """
        Predict results by blending predictions from all models.
        :param df: pandas.DataFrame - raw dataset to analyze
        :return: pandas.DataFrame with predicted results
        """
        modified_data = preprocessing.process_data(df)
        result = np.array([self.models[i].predict(modified_data).flatten() for i in range(0, self.model_count)])
        return stats.mode(result)[0][0]
