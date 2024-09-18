import os
import random
import time
from collections import deque
from queue import Queue
from multiprocessing import Process, Queue
import multiprocessing
import threading

import gym
import keyboard
import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.callbacks import Callback
from models.actor import create_actor
from models.critic import create_critic
from tqdm import tqdm
from utils.preprocessing import load_config

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Print GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load configuration
config = load_config('config/continuous_mountain-car_config.yaml')

# Extract training constants
SEED = config['training']['seed']
GAMMA = config['training']['gamma']
TAU = config['training']['tau']
LEARNING_RATE_ACTOR = config['training']['learning_rate_actor']
BUFFER_SIZE = config['training']['buffer_size']
BATCH_SIZE = config['training']['batch_size']
NUM_EPISODES = config['training']['num_episodes']
INPUT_SHAPE = config['data']['input_shape']

# Configure TensorFlow for GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')


class BestRewardLogger(Callback):
    """Keras callback to log the best and average rewards."""

    def __init__(self):
        super(BestRewardLogger, self).__init__()
        self.best_reward = -float('inf')

    def on_episode_end(self, episode, logs=None):
        reward = logs.get('episode_reward', 0)
        if reward > self.best_reward:
            self.best_reward = reward
            tf.summary.scalar('Best Reward', data=self.best_reward, step=episode)

        average_reward = logs.get('average_reward', 0)
        epsilon = logs.get('epsilon', 0)
        tf.summary.scalar('Average Reward', data=average_reward, step=episode)
        tf.summary.scalar('Epsilon', data=epsilon, step=episode)

    def on_train_batch_end(self, batch, logs=None):
        tf.summary.scalar('Actor Loss', data=logs.get('actor_loss', 0), step=batch)
        tf.summary.scalar('Critic Loss', data=logs.get('critic_loss', 0), step=batch)


class InputManager:
    """Manages user input for toggling simulation state and simulating the best model."""

    BUTTON_CTRL_SHIFT_SPACE = 0b0001
    BUTTON_CTRL_SHIFT_ENTER = 0b0010  # New button flag for Ctrl+Shift+Enter

    def __init__(self):
        self.input_state = 0
        self.running = True
        self.lock = threading.Lock()
        self.listener_thread = threading.Thread(target=self._input_listener)
        self.listener_thread.start()
        # Simulation is active by default
        self.current_input_state = 1
        self.best_episode = []  # To store the best episode steps
        self.queue = None

    def toggle_input_state(self, key_event):
        if key_event == 'ctrl+shift+space':
            self.current_input_state ^= self.BUTTON_CTRL_SHIFT_SPACE  # Toggle bit
        elif key_event == 'ctrl+shift+enter':
            self.simulate_best_model()  # Trigger simulation of the best model

    def _input_listener(self):
        while self.running:
            if keyboard.is_pressed('ctrl+shift+space'):
                self.toggle_input_state('ctrl+shift+space')
                # Debounce to prevent multiple triggers
                while keyboard.is_pressed('ctrl+shift+space'):
                    time.sleep(0.1)
            if keyboard.is_pressed('ctrl+shift+enter'):
                self.toggle_input_state('ctrl+shift+enter')
                while keyboard.is_pressed('ctrl+shift+enter'):
                    time.sleep(0.1)
            time.sleep(0.1)

    def is_simulation_active(self):
        return bool(self.current_input_state & self.BUTTON_CTRL_SHIFT_SPACE)

    def simulate_best_model(self):
        """Simulate the best model by replaying the best episode steps."""
        if self.best_episode:
            simulate_game(self.queue, self.best_episode)
            tqdm.write("Simulating the best model using Ctrl+Shift+Enter...")
        else:
            tqdm.write("No best episode recorded yet.")

    def stop(self):
        self.running = False
        self.listener_thread.join()


class DDPGAgent:
    """Deep Deterministic Policy Gradient (DDPG) Agent."""

    def __init__(self, config, action_size):
        self.state_size = INPUT_SHAPE
        self.action_size = action_size
        self.gamma = GAMMA
        self.tau = TAU

        # Initialize Actor Network
        self.actor = create_actor(config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ACTOR)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        self.actor.compile(optimizer=optimizer)

        # Initialize Target Actor Network
        self.target_actor = create_actor(config)
        self.target_actor.set_weights(self.actor.get_weights())

        # Initialize Critic Network
        self.critic = create_critic(config)

        # Initialize Target Critic Network
        self.target_critic = create_critic(config)
        self.target_critic.set_weights(self.critic.get_weights())

        # Replay Buffer
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.batch_size = BATCH_SIZE

        # Exploration Noise
        self.noise = OUNoise(self.action_size)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Select action based on current state."""

        state = np.array(state).reshape(1, 2)  
        action = self.actor(state, training=False).numpy()[0] 
        action = np.clip(action, -1, 1)
        noise = self.noise.sample()
        return np.clip(action + noise, -1, 1)

    def replay(self):
        """Train the agent using experiences from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Update Critic Network
        target_actions = self.target_actor.predict(next_states, verbose=0)
        target_q = self.target_critic.predict([next_states, target_actions], verbose=0)
        y = rewards + self.gamma * target_q.squeeze() * (1 - dones)
        self.critic.train_on_batch([states, actions], y)

        # Update Actor Network
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states, training=True)
            critic_value = self.critic([states, actions_pred], training=True)
            actor_loss = -tf.reduce_mean(critic_value)
            scaled_loss = self.actor.optimizer.get_scaled_loss(actor_loss)
        scaled_grads = tape.gradient(scaled_loss, self.actor.trainable_variables)
        grads = self.actor.optimizer.get_unscaled_gradients(scaled_grads)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        # Soft update Target Networks
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target, source):
        """Perform soft update of target network parameters."""
        target_weights = target.get_weights()
        source_weights = source.get_weights()
        updated_weights = [
            self.tau * src + (1 - self.tau) * tgt
            for src, tgt in zip(source_weights, target_weights)
        ]
        target.set_weights(updated_weights)


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(self, size, mu=0.0, theta=0.2, sigma=0.3):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed = SEED
        self.state = self.mu.copy()

    def reset(self):
        """Reset the internal state to mean."""
        self.state = self.mu.copy()

    def sample(self):
        """Generate a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


def simulation_worker(simulation_queue, result_queue):
    """Worker process to handle simulation playback."""
    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    env.reset(seed=SEED)
    
    while True:
        steps = simulation_queue.get()
        if steps is None:
            break  # Graceful shutdown
        
        env.reset(seed=SEED)
        episode_reward = 0
        for action in steps:
            env.render(mode="human")
            _, reward, _, _ = env.step(action)
            episode_reward += reward

        env.render()

        result_queue.put(episode_reward)
    
    env.close()


def simulate_game(simulation_queue, steps):
    """Queue a game simulation with given steps."""
    simulation_queue.put(steps)


def shutdown_simulation(simulation_queue, sim_process, input_manager):
    """Gracefully shut down the simulation worker."""
    simulation_queue.put(None)  # Signal shutdown
    sim_process.join()
    input_manager.stop()


def main():
    """Main training loop for the DDPG agent."""
    # Set random seeds for reproducibility
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # Initialize environment
    env = gym.make('MountainCarContinuous-v0')
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    state = env.reset(seed=SEED)
    action_size = env.action_space.shape[0]

    agent = DDPGAgent(config, action_size)

    best_reward = -np.inf
    replay_all_steps = {}
    total_steps = 0
    total_episodes = NUM_EPISODES + 1

    # Initialize Simulation Queues
    simulation_queue = Queue()
    result_queue = Queue()

    # Initialize Input Manager
    input_manager = InputManager()
    input_manager.queue = simulation_queue

    # Start simulation worker process
    sim_process = Process(target=simulation_worker, args=(simulation_queue, result_queue))
    sim_process.start()

    try:
        for episode in tqdm(range(1, total_episodes), desc="Training Episodes", smoothing=0.3):
            state = env.reset(seed=SEED)
            total_reward = 0
            agent.noise.reset()
            max_velocity = -np.inf
            visited_positions = set()
            episode_steps = []

            for step in tqdm(range(1, 501), desc=f"Episode {episode}", leave=False, smoothing=0.3):
                action = agent.act(state)
                next_state, _, done, _ = env.step(action)

                if input_manager.is_simulation_active():
                    episode_steps.append(action)

                reward = -1  # Default reward

                # Reward for reaching the goal
                if done and state[0] >= 0.45:
                    reward += 100

                # Reward for maximum velocity improvement
                if next_state[1] > max_velocity:
                    max_velocity = next_state[1]
                    reward += max_velocity * 100

                # Reward for visiting new positions
                position_normalized = round(next_state[0], 2)
                if position_normalized not in visited_positions:
                    visited_positions.add(position_normalized)
                    reward += 1

                agent.remember(state, action, reward, next_state, done)
                agent.replay()

                state = next_state
                total_reward += reward
                total_steps = step

                if done:
                    break

            # Save the best model
            if total_reward > best_reward:
                best_reward = total_reward
                agent.target_actor.save('checkpoints/best_actor.keras')
                agent.target_critic.save('checkpoints/best_critic.keras')

            tqdm.write(
                f"Episode {episode}/{NUM_EPISODES} - Reward: {total_reward:.2f} - Total steps: {total_steps}"
            )

            # Record steps for simulation if active
            if input_manager.is_simulation_active():
                replay_all_steps[episode] = {
                    'episode_steps': episode_steps,
                    'total_reward': total_reward
                }
                best_episode_data = max(replay_all_steps.values(), key=lambda x: x['total_reward'], default=None)
                if best_episode_data:
                    best_episode_steps = best_episode_data['episode_steps']
                    input_manager.best_episode = best_episode_steps  # Store in InputManager
                simulate_game(simulation_queue, episode_steps)

        # Optionally, retrieve simulation results
        while not result_queue.empty():
            simulation_reward = result_queue.get()
            print(f"Simulation Reward: {simulation_reward}")

    finally:
        # Ensure that the simulation process is shut down gracefully
        shutdown_simulation(simulation_queue, sim_process, input_manager)
        
        # Save the trained models
        agent.actor.save('checkpoints/mountain_car_actor.keras')
        agent.critic.save('checkpoints/mountain_car_critic.keras')
        tqdm.write("Models saved to checkpoints/")
    
        # Close the environment
        env.close()


def test():
    """Test the trained agent."""
    # Load configuration
    config = load_config('config/continuous_mountain-car_config.yaml')

    # Initialize environment
    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    state = env.reset(seed=SEED)

    action_size = env.action_space.shape[0]

if __name__ == "__main__":
    #test()
    main()