import pickle
import time
import gym
import pygame
import os

def replay_episode(episode_index: int, states_file: str = 'states/all_episode_states.pkl', speed: float = 0.25):
    """Replay a specific episode using saved states."""
    # Load the saved states
    with open(states_file, 'rb') as f:
        all_episode_states = pickle.load(f)

    if episode_index >= len(all_episode_states):
        print(f"Episode index {episode_index} out of range. Total episodes: {len(all_episode_states)}")
        return
    
    episode_states = all_episode_states[episode_index]
    
    # Initialize the environment in human render mode
    env = gym.make('SnakeGame-v0', render_mode="human")
    
    # Reset the environment to the first state
    env.reset()

    # Initialize font for rendering text
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)

    for step, state in enumerate(episode_states):
        # Set the environment to the next state
        env.set_full_state(state)
        
        # Render the game state
        env.render(options={'episode_index': episode_index, 'step': step})
        
        # Control the replay speed
        time.sleep(1 / env.metadata["render_fps"] * speed)

    env.close()

if __name__ == "__main__":
    # Example: Replay the first episode
    replay_episode(0, '../states/all_episode_states.pkl')