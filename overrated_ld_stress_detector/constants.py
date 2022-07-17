import pathlib
from overrated_ld_stress_detector.visualization import utils

PROJECT_ROOT = pathlib.Path(utils.__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"