import catboost
# from overrated_ld_stress_detector.ml.predict import PytorchModel
# from overrated_ld_stress_detector.ml.predict import CatboostModel

def load_weights():
    return ""


def get_model(model_type='fastest',
              model_path=None,
              max_size=2048):
    # load weights from gdrive to local path
    if model_path is None:
        model_path = load_weights()
    # load catboost model from path
    # model = CatboostModel(model_path)
    # local weights path
    # return model