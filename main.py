import numpy as np
from src.neural_components.ensemble import Ensemble
from src.utils.data_handling import DataHandler
from src.utils import fancy_print
import json
from colorama import init, Fore, Style
import time
from src.core.neural_network import NeuralNetwork
import os

init(autoreset=True)  # Initialize colorama

DATASET = 'mnist'

def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    fancy_print("Starting Neural Network Training", Fore.CYAN, Style.BRIGHT)
    fancy_print("Dataset: " + DATASET, Fore.CYAN, Style.BRIGHT)
    
    fancy_print("Loading configuration...", Fore.YELLOW)
    config = load_config('src/config/base_parameters.json')[DATASET]
    nn_config = load_config('src/config/nn_parameters.json')[DATASET]
    fancy_print("Configuration loaded successfully!", Fore.GREEN)

    data_handler = DataHandler(config)

    # Load data without augmentation
    X_train, y_train, X_test, y_test = data_handler.load_data(config['train_test_split_ratio'], augmentation=True, num_augmented=1, show_samples=True)

    fancy_print("Training single neural network...", Fore.YELLOW)
    model = NeuralNetwork(DATASET, model_index=0, verbose=True)
    model.train(X_train, y_train, X_test, y_test)

    fancy_print("Evaluating model on test data...", Fore.YELLOW)
    accuracy = model.evaluate(X_test, y_test)
    fancy_print(f"Test accuracy: {accuracy*100:.2f}%", Fore.GREEN, Style.BRIGHT)

    # Save the model
    model_path = 'src/models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save_model(0)
    fancy_print("Model saved successfully!", Fore.GREEN)

if __name__ == "__main__":
    main()
