# Neural Network Configuration Guide

This README provides an overview of the configuration options available for your neural network models using TensorFlow. The configuration is typically stored in a YAML file (e.g., `config.yaml`) and allows you to customize various aspects of your model architecture, training process, and data handling.

## Configuration Sections

The configuration file is divided into three main sections:

1. `model`: Define the architecture of your neural network.
2. `training`: Set up training parameters and optimization strategies.
3. `data`: Configure data preprocessing and augmentation.

### Model Configuration

In the `model` section, you can define the layers of your neural network. Available layer types include:

- Convolutional (conv2d)
- Pooling (maxpooling2d)
- Dense (fully connected)
- Flatten
- Dropout
- BatchNormalization
- Reshape
- Permute
- Resizing
- RandomZoom
- RandomTranslation
- RandomRotation
- RandomFlip


For each layer, you can specify various parameters such as filters, kernel size, activation function, and more.

#### Available Activation Functions

You can choose from a wide range of activation functions for your layers, including:

elu, exponential, gelu, hard_sigmoid, hard_silu, hard_swish, leaky_relu, linear, log_softmax, mish, relu, relu6, selu, sigmoid, silu, softmax, softplus, softsign, swish, tanh

For more information on activation functions, visit: https://www.tensorflow.org/api_docs/python/tf/keras/activations

### Training Configuration

The `training` section allows you to set up various training parameters:

- Optimizer
- Loss function
- Metrics
- Number of epochs
- Validation split
- Checkpointing
- Batch size
- Early stopping
- Learning rate scheduling
- Tensorboard logging
- Regularization (L1 and L2)

#### Available Optimizers

Choose from the following optimizers:

Adadelta, Adafactor, Adagrad, Adam, AdamW, Adamax, Ftrl, Lion, Nadam, RMSprop, SGD

For more information on optimizers, visit: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

#### Available Loss Functions

Select a loss function from the following options:

BinaryCrossentropy, BinaryFocalCrossentropy, CategoricalCrossentropy, CategoricalFocalCrossentropy, CategoricalHinge, CosineSimilarity, Hinge, Huber, KLDivergence, LogCosh, MeanAbsoluteError, MeanSquaredError, Poisson, SparseCategoricalCrossentropy, SquaredHinge

For more information on loss functions, visit: https://www.tensorflow.org/api_docs/python/tf/keras/losses

#### Available Metrics

Choose from a variety of metrics to evaluate your model:

AUC, Accuracy, BinaryAccuracy, CategoricalAccuracy, F1Score, Precision, Recall, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, TopKCategoricalAccuracy, and more.

For a complete list of available metrics, visit: https://www.tensorflow.org/api_docs/python/tf/keras/metrics

### Data Configuration

In the `data` section, you can configure:

- Data normalization
- Dataset selection
- Input shape
- Data augmentation options (if enabled)

Datasets available (from tf.keras.datasets):

- MNIST
- CIFAR10
- CIFAR100
- FashionMNIST
- IMAGENET

## Example Configuration

For an example of how to structure your configuration file, please refer to the `config.yaml` file in this directory.

## Creating Your Own Configuration

To create a configuration for your specific dataset and model architecture:

1. Copy the example `config.yaml` file.
2. Modify the sections according to your requirements.
3. Ensure all required fields are filled out.
4. Save the file with a descriptive name, e.g., `mnist_config.yaml` for an MNIST dataset configuration.

Remember to adjust the model architecture, training parameters, and data handling options to best suit your specific machine learning task and dataset.

To use the config file, you can go to the utils/preprocessing.py file and set the config file path to the path of the config file you created.