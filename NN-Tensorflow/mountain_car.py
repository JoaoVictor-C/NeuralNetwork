import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #Disable oneDNN
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2' #Enable XLA
os.environ['PRINT_STEP'] = '1'
os.environ['PRINT_EPISODE'] = '1'

import gymnasium as gym
import numpy as np
import tensorflow as tf
from models.model import create_model
from utils.preprocessing import load_config
from collections import deque
import random

def train_model(config):
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = create_model(config)
    target_model = create_model(config)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=config['data']['buffer_size'])
    epsilon = 1.0
    epsilon_decay = 0.9
    epsilon_min = 0.01
    batch_size = config['data']['batch_size']
    update_target_every = 5

    for episode in range(config['training']['num_episodes']):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        time_step = 0
        total_reward = 0

        while not done:
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(state, verbose=0)[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])

            # Custom reward
            reward = abs(next_state[0][1])  # Reward based on velocity
            if next_state[0][0] >= 0.5:
                reward += 10

            total_reward += reward

            replay_memory.append((state, action, reward, next_state, done))
            state = next_state
            time_step += 1

            if len(replay_memory) > batch_size:
                minibatch = random.sample(replay_memory, batch_size)
                train_on_batch(model, target_model, minibatch, config['training']['gamma'])

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % update_target_every == 0:
            target_model.set_weights(model.get_weights())

        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return model

def train_on_batch(model, target_model, minibatch, gamma):
    states = np.array([transition[0][0] for transition in minibatch])
    actions = np.array([transition[1] for transition in minibatch])
    rewards = np.array([transition[2] for transition in minibatch])
    next_states = np.array([transition[3][0] for transition in minibatch])
    dones = np.array([transition[4] for transition in minibatch])

    targets = model.predict(states, verbose=0)
    target_vals = target_model.predict(next_states, verbose=0)
    
    for i in range(len(minibatch)):
        if dones[i]:
            targets[i][actions[i]] = rewards[i]
        else:
            targets[i][actions[i]] = rewards[i] + gamma * np.amax(target_vals[i])

    model.fit(states, targets, epochs=1, verbose=0)

def test_model(model, config):
    env = gym.make('MountainCar-v0')
    num_test_episodes = config['testing']['num_episodes']
    max_steps = config['testing']['max_steps']
    
    total_rewards = []
    
    for _ in range(num_test_episodes):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = np.argmax(model.predict(state.reshape(1, -1), verbose=0))
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_test_episodes} episodes: {avg_reward}")

def main():
    config = load_config("config/mountain-car_config.yaml")
    model = train_model(config)
    test_model(model, config)

if __name__ == "__main__":
    main()
