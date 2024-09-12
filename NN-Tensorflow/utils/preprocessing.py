import yaml
from sklearn.model_selection import KFold
import numpy as np

def load_config():
    with open('config/mnist_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def create_kfold(n_splits=5):
    config = load_config()
    return KFold(n_splits=n_splits, shuffle=config['data']['shuffle'], random_state=config['data']['seed'])

def calculate_average_score(scores):
    return np.mean(scores)
