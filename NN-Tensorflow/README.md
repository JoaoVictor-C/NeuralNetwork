# NN-Tensorflow: Neural Network Project with TensorFlow

Welcome to the NN-Tensorflow project! This repository contains a flexible and customizable neural network implementation using TensorFlow. Whether you're a beginner or an experienced machine learning practitioner, this project provides a solid foundation for building and experimenting with various neural network architectures.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Contributing](#contributing)
7. [License](#license)

## Project Overview

NN-Tensorflow is designed to simplify the process of creating, training, and evaluating neural networks using TensorFlow. It offers a configuration-based approach, allowing you to easily customize your model architecture, training parameters, and data preprocessing without modifying the core code.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. This project requires the following main dependencies:

tensorflow==2.6.0
numpy==1.19.5
matplotlib==3.4.3
pygame==2.1.0

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/JoaoVictor-C/Neural-Network.git
   cd Neuralnetwork/NN-Tensorflow
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The project uses YAML configuration files to define model architecture, training parameters, and data handling. For a detailed guide on configuration options, please refer to the Configuration Guide:

## Usage

1. Create or modify a configuration file in the `config` directory (e.g., `custom_config.yaml`).
2. Update the config file path in `utils/preprocessing.py` to point to your configuration file.
3. Run the main script to train and evaluate your model:
   ```bash
   python main.py
   ```

## Project Structure

- `config/`: Contains configuration files and the configuration guide.
- `utils/`: Utility scripts for data preprocessing and other helper functions.
- `main.py`: The main script to run the neural network training and evaluation.
- `requirements.txt`: List of project dependencies.
- `digit_recognizer.py`: A simple digit recognizer using a custom neural network.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open-source and available under the MIT License.