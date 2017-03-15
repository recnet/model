import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_PATH = os.path.join(ROOT_DIR, "resources/datasets")

CHECKPOINTS_PATH = os.path.join(ROOT_DIR, "checkpoints")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

TENSOR_DIR_TRAIN = 'tensorDir/train'
TENSOR_DIR_VALID = 'tensorDir/valid'

