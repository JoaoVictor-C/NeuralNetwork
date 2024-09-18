import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras import layers, models, optimizers
import os

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        total = sum(self.priorities)
        probabilities = [p / total for p in self.priorities]
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = [(len(self.buffer) * probabilities[i]) ** -self.beta for i in indices]
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        weights = np.array(weights).reshape((batch_size, 1))
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for i, e in zip(indices, errors):
            self.priorities[i] = (abs(e) + self.epsilon) ** self.alpha

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Hyperparameters
        self.gamma = config['training']['gamma']
        self.epsilon = config['training']['epsilon_start']
        self.epsilon_min = config['training']['epsilon_min']
        self.epsilon_decay = config['training']['epsilon_decay']
        self.learning_rate = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        self.train_start = config['training']['train_start']

        # Experience replay buffer
        self.memory = PrioritizedReplayBuffer(config['training']['buffer_size'])

        # Create primary and target networks
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='huber_loss')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = state.reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        state = state.reshape(1, -1)
        next_state = next_state.reshape(1, -1)
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        if self.memory.size() < self.train_start:
            return

        minibatch, indices, weights = self.memory.sample(self.batch_size)

        states = np.vstack([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.vstack([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Double DQN
        q_next = self.model.predict(next_states, verbose=0)
        q_target_next = self.target_model.predict(next_states, verbose=0)
        
        targets = self.model.predict(states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(q_next[i])
                targets[i][actions[i]] = rewards[i] + self.gamma * q_target_next[i][a]

        # Train the model
        loss = self.model.fit(states, targets, sample_weight=weights, batch_size=self.batch_size, epochs=1, verbose=0)
        
        # Update priorities
        errors = np.abs(targets[np.arange(self.batch_size), actions] - self.model.predict(states, verbose=0)[np.arange(self.batch_size), actions])
        self.memory.update_priorities(indices, errors)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.history['loss'][0]

    def save_models(self, path='checkpoints/dqn/'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, 'dqn_actor.keras'))
        self.target_model.save(os.path.join(path, 'dqn_target.keras'))

    def load_models(self, path='checkpoints/dqn/'):
        self.model = models.load_model(os.path.join(path, 'dqn_actor.keras'))
        self.target_model = models.load_model(os.path.join(path, 'dqn_target.keras'))