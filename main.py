import numpy as np
from src.core import NeuralNetwork
from src.data_handling import DataHandler
from src.utils import fancy_print
import json
from colorama import init, Fore, Style
import time

init(autoreset=True)  # Initialize colorama

DATASET = 'mnist-fashion'

def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def fancy_print(message, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{message}{Style.RESET_ALL}")

def main():
    fancy_print("Starting Neural Network Training", Fore.CYAN, Style.BRIGHT)
    fancy_print("Dataset: " + DATASET, Fore.CYAN, Style.BRIGHT)
    
    fancy_print("Loading configuration...", Fore.YELLOW)
    config = load_config('src/config/base_parameters.json')[DATASET]
    fancy_print("Configuration loaded successfully!", Fore.GREEN)

    data_handler = DataHandler(config)

    start_time = time.time()
    X_train, y_train, X_test, y_test = data_handler.load_data(config['train_test_split_ratio'])

    nn = NeuralNetwork(DATASET)

    start_time = time.time()
    nn.train(X_train, y_train)

    fancy_print("Evaluating model on test data...", Fore.YELLOW)
    accuracy = nn.evaluate(X_test, y_test)
    fancy_print(f"Test accuracy: {accuracy*100:.2f}%", Fore.GREEN, Style.BRIGHT)


if __name__ == "__main__":
    main()
