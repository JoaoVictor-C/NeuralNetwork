import numpy as np
from src.neural_components.ensemble.ensemble import Ensemble
from src.utils.data_handling import DataHandler
from src.utils import fancy_print
import json
from colorama import init, Fore, Style
import time
from src.core.neural_network import NeuralNetwork

init(autoreset=True)  # Initialize colorama

DATASET = 'mnist-fashion'

def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    fancy_print("Starting Neural Network Ensemble Training", Fore.CYAN, Style.BRIGHT)
    fancy_print("Dataset: " + DATASET, Fore.CYAN, Style.BRIGHT)
    
    fancy_print("Loading configuration...", Fore.YELLOW)
    config = load_config('src/config/base_parameters.json')[DATASET]
    fancy_print("Configuration loaded successfully!", Fore.GREEN)

    data_handler = DataHandler(config)

    X_train, y_train, X_test, y_test = data_handler.load_data(config['train_test_split_ratio'])

    ensemble = Ensemble(DATASET)
    fancy_print("Training ensemble...", Fore.YELLOW)
    ensemble.train(X_train, y_train, X_test, y_test)

    fancy_print("Evaluating ensemble on test data...", Fore.YELLOW)
    accuracy = ensemble.evaluate(X_test, y_test)
    fancy_print(f"Test accuracy: {accuracy*100:.2f}%", Fore.GREEN, Style.BRIGHT)

if __name__ == "__main__":
    main()
