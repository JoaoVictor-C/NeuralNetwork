import yaml
from sklearn.model_selection import KFold
import numpy as np

def load_config(config_path = 'config/mnist_config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_kfold(n_splits=5):
    config = load_config()
    return KFold(n_splits=n_splits, shuffle=config['data']['shuffle'], random_state=config['data']['seed'])

def calculate_average_score(scores):
    return np.mean(scores)
