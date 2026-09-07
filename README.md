# NeuralNetwork

Two tracks, deliberately separated: a neural network **written from scratch in NumPy with no ML framework**, and a second track using TensorFlow/Keras for reinforcement learning. Keeping both is the point — the same kind of problem solved once by hand and once with a framework.

---

## `NN-From-Scratch/` — no framework, NumPy only

Forward and backward propagation, the optimisers, the activations and their derivatives, batch normalisation, regularisation and the learning-rate schedules are all implemented by hand. NumPy does array arithmetic and nothing else.

### What is implemented

**10 optimisers** — SGD · Momentum · Nesterov · Adagrad · Adadelta · RMSprop · Adam · Adamax · Nadam · PSO (particle swarm)

**13 activations**, each with its analytic derivative — ReLU · LeakyReLU · ELU · GELU · Swish · Mish · Sigmoid · Tanh · Softmax · Softplus · Softsign · Linear · Argmax

**5 cost functions** — cross-entropy · MSE · LogCosh · exponential

**6 learning-rate schedulers** — step · exponential · cyclical · cosine annealing · warm-up · constant

**Regularisation** — L1, L2 and L1+L2 · dropout · batch normalisation

**Training** — mini-batches, early stopping on patience, best-model checkpointing to `.pkl`, data augmentation, and a coloured progress log.

### Configuration

The network is described by JSON rather than code, so an experiment is a config diff:

```json
{
  "layers": [784, 256, 128, 64, 10],
  "epochs": 50,
  "batch_size": 128,
  "patience": 7,
  "dropout_rate": 0.2,
  "activation_function": "relu",
  "output_activation": "softmax",
  "cost_function": "cross_entropy",
  "optimizer": "Adam",
  "learning_rate": { "learning_rate": 0.001, "lr_scheduler": "exponential", "lr_decay": 0.98 },
  "regularization": { "type": "l1_l2", "lambda": 0.0001 }
}
```

### Datasets

MNIST and MNIST-Fashion, both included as raw IDX files under `data/` — decoded by `utils/data_handling/`, not by a library helper.

### Running it

```bash
cd NN-From-Scratch
pip install numpy matplotlib colorama tqdm opencv-python pygame
python main.py
```

`DATASET` at the top of `main.py` selects `mnist` or `mnist-fashion`.

### `digit_recognizer.py`

A pygame canvas: draw a digit with the mouse and the trained network classifies it live, with per-class confidence drawn beside it. This is the part that makes the training loop worth having — it is one thing for test accuracy to be a number, another to draw a bad 7 and watch where the network hesitates.

```bash
python digit_recognizer.py
```

## `NN-Tensorflow/` — framework track

TensorFlow/Keras, used for the reinforcement-learning work that would be impractical to hand-roll:

- **DQN** (`models/dqn.py`) with experience replay (`utils/replay.py`)
- **Actor–critic** (`models/actor.py`, `models/critic.py`)
- **Environments** — OpenAI Gym `continuous_mountain_car`, plus a self-contained `snake_game.py`
- A digit recogniser built the framework way, for direct comparison with the from-scratch one

```bash
cd NN-Tensorflow
pip install tensorflow gym pygame numpy matplotlib
python main.py
```

## Layout

```
NN-From-Scratch/
├─ models/
│  ├─ neural_network.py            training loop, forward/backward, checkpointing
│  ├─ layer.py                     dense layer, weights and gradients
│  └─ neural_components/
│     ├─ activation/               13 activations + derivatives
│     ├─ optimizers/               10 optimisers
│     ├─ cost/                     5 cost functions
│     ├─ learning_rate/            6 schedulers
│     ├─ normalization/            batch normalisation
│     └─ regularization/           L1 / L2 / L1L2
├─ utils/data_handling/            IDX decoding, augmentation, batching
├─ config/                         JSON experiment configs
├─ data/                           MNIST, MNIST-Fashion (raw IDX)
├─ main.py                         train
└─ digit_recognizer.py             draw-a-digit demo

NN-Tensorflow/                     DQN, actor–critic, Gym, snake
```

## Why

To understand backpropagation by having to make it work. Reading the chain rule and implementing ten optimisers that all have to converge on the same dataset are different activities, and the second one is where the misunderstandings surface.

## License

See [LICENSE](LICENSE).
