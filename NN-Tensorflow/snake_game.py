import os
import random
import threading
import queue
import time
import gym
import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.callbacks import Callback
from models.dqn import DQNAgent  # Updated import
from tqdm import tqdm
from utils.preprocessing import load_config
import pickle
from utils.replay import replay_episode

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Print GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load configuration
config = load_config('config/snake_game_config.yaml')

# Extract training constants
SEED = config['training']['seed']
GAMMA = config['training']['gamma']
BUFFER_SIZE = config['training']['buffer_size']
BATCH_SIZE = config['training']['batch_size']
NUM_EPISODES = config['training']['num_episodes']
INPUT_SHAPE = config['data']['input_shape']

ACTION_SIZE = config['data']['action_size']
ENV_NAME = config['data']['env_name']

# Configure TensorFlow for GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
    except RuntimeError as e:
        tqdm.write(e)

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# Initialize a queue to store episode states
replay_queue = queue.Queue()

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


def replay_worker():
    while True:
        episode_index = replay_queue.get()
        if episode_index is None:
            break
        replay_episode(episode_index)
        replay_queue.task_done()

replay_thread = threading.Thread(target=replay_worker)
replay_thread.start()

def main():
    """Main training loop for the DQN agent."""
    # Set random seeds for reproducibility
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    # Initialize environment
    env = gym.make(ENV_NAME)
    env.reset()
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    state_dim = INPUT_SHAPE
    action_size = ACTION_SIZE
    agent = DQNAgent(INPUT_SHAPE, action_size, config)


    def custom_reward(state: dict) -> float:
        import numpy as np

        reward = 0.0

        if state['ate_food']:
            reward += 20.0  # Increased reward for eating food

        if state['done']:
            reward -= 20.0  # Increased penalty for dying

        # Optional Distance-Based Reward
        head_pos = np.array(state['new_head'])
        food_pos = np.array(state['food'])
        distance = np.linalg.norm(head_pos - food_pos)
        reward -= distance * 0.1  # Adjusted penalty based on distance

        reward -= 0.05  # Step penalty to encourage efficiency

        return reward

    env.env.set_reward_fn(custom_reward)
    tqdm.write("Reward function updated.")

    # Initialize TensorBoard Logging
    log_dir = "logs/dqn/" + time.strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Initialize Best Reward
    best_reward = -np.inf

    # Initialize Replay Steps Record
    all_episode_states = []

    for episode in tqdm(range(1, NUM_EPISODES + 1), desc="Training Episodes"):
        state, _ = env.reset()
        state = np.array(state).flatten()
        episode_reward = 0
        done = False
        step = 0
        episode_states = [env.get_full_state()]

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state).flatten()

            episode_states.append(env.get_full_state())

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step += 1

            if done:
                all_episode_states.append(episode_states)
                agent.update_target_model()
                break

        for _ in tqdm(range(10), desc="Replaying", leave=False):
            agent.replay()
        

        # Log episode reward and other metrics
        with summary_writer.as_default():
            tf.summary.scalar('Episode Reward', episode_reward, step=episode)
            tf.summary.scalar('Epsilon', agent.epsilon, step=episode)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_models()
            tqdm.write(f"New best reward {best_reward} at episode {episode}!")

        if not os.path.exists('states'):
            os.makedirs('states')
        with open('states/all_episode_states.pkl', 'wb') as f:
            pickle.dump(all_episode_states, f)

        # If the first initial training process (10% of population)
        if episode >= 190:
            replay_queue.put(episode-1)

        tqdm.write(f"Episode {episode} - Reward {episode_reward} - Steps {step} - Epsilon {agent.epsilon:.4f}")

    # Save the trained models after training
    agent.save_models()

    # Close the environment
    env.close()

    # Close TensorBoard Summary Writer
    summary_writer.close()


def test():
    """Test the trained agent."""
    # Load configuration
    config = load_config('config/snake_game_config.yaml')

    # Initialize environment
    env = gym.make(ENV_NAME, render_mode="human")
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    state, _ = env.reset()
    state = np.reshape(state, [1, INPUT_SHAPE])
    action_size = ACTION_SIZE

    # Load trained models
    agent = DQNAgent(INPUT_SHAPE, action_size, config)
    agent.load_models()
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = agent.act(state)
        one_hot_action = np.zeros(action_size)
        one_hot_action[action] = 1
        next_state, reward, done, _ = env.step(one_hot_action)
        next_state = np.reshape(next_state, [1, INPUT_SHAPE])
        state = next_state
        total_reward += reward
    tqdm.write(f"Test Run Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tqdm.write("Training interrupted by user.")
    finally:
        replay_queue.put(None)
        replay_thread.join()