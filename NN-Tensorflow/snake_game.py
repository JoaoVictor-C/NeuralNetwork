import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pygame
import random
import numpy as np
import tensorflow as tf
from collections import deque
from utils.preprocessing import load_config
from models.model import create_model
from utils.callbacks import create_callbacks
from tqdm import tqdm

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Game dimensions
WIDTH = 400
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]

        # Initializes with 3 cells
        for _ in range(3):
            self.snake.append((self.snake[0][0] - 1, self.snake[0][1]))

        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.moves = 0

    def generate_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def move(self, action):
        self.moves += 1

        reward = 0

        # 0: straight, 1: right, 2: left
        if action == 1:
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:
            self.direction = (-self.direction[1], self.direction[0])

        new_head = ((self.snake[0][0] + self.direction[0]) % GRID_WIDTH,
                    (self.snake[0][1] + self.direction[1]) % GRID_HEIGHT)

        # Calculate distance to food before and after move
        old_distance = ((self.snake[0][0] - self.food[0])**2 + (self.snake[0][1] - self.food[1])**2)**0.5
        new_distance = ((new_head[0] - self.food[0])**2 + (new_head[1] - self.food[1])**2)**0.5

        if new_head in self.snake[1:]:
            self.game_over = True
        
        if new_head[0] == 0 or new_head[0] == GRID_WIDTH - 1 or new_head[1] == 0 or new_head[1] == GRID_HEIGHT - 1:
            self.game_over = True

        self.snake.insert(0, new_head)
        self.snake.pop()

        # If the snake ate the food
        if new_head == self.food:
            reward = 10
            self.score += 1
            self.snake.append(self.snake[-1])
            self.food = self.generate_food()
            self.moves = 0  # Reset moves counter when food is eaten
        elif self.game_over:
            reward = -10
        else:
            reward = -0.1  # Small negative reward for each move
        
        # Reward for moving closer to food
        old_distance = ((self.snake[1][0] - self.food[0])**2 + (self.snake[1][1] - self.food[1])**2)**0.5
        new_distance = ((self.snake[0][0] - self.food[0])**2 + (self.snake[0][1] - self.food[1])**2)**0.5
        if new_distance < old_distance:
            reward += 0.1
        
        reward -= np.log(self.moves+1) # To prevent the snake from moving in circles

        return reward

    def get_state(self):
        head = self.snake[0]
        point_l = ((head[0] - 1) % GRID_WIDTH, head[1])
        point_r = ((head[0] + 1) % GRID_WIDTH, head[1])
        point_u = (head[0], (head[1] - 1) % GRID_HEIGHT)
        point_d = (head[0], (head[1] + 1) % GRID_HEIGHT)
        
        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food[0] < self.snake[0][0],  # food left
            self.food[0] > self.snake[0][0],  # food right
            self.food[1] < self.snake[0][1],  # food up
            self.food[1] > self.snake[0][1]   # food down
            ]

        return np.array(state, dtype=int)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        return pt in self.snake[1:]

    def draw(self, game_number=None, epsilon=None):
        screen.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pt[0]*GRID_SIZE, pt[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED, pygame.Rect(self.food[0]*GRID_SIZE, self.food[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
        if game_number is not None and epsilon is not None:
            self.draw_info(game_number, epsilon)
        pygame.display.flip()

    def draw_info(self, game_number, epsilon):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Game: {game_number} | Score: {self.score} | Epsilon: {epsilon:.2f}", True, WHITE)
        screen.blit(text, (10, 10))

def train_snake():
    # Load configuration
    config = load_config('config/snake_config.yaml')
    
    # Initialize game
    game = SnakeGame()
    
    # Create model
    model = create_model(config)
    
    # Create target model
    target_model = create_model(config)
    target_model.set_weights(model.get_weights())
    
    # Initialize replay memory
    memory = deque(maxlen=config['training-snake']['max_memory'])
    
    # Training parameters
    n_games = config['training-snake']['n_games']
    batch_size = config['training-snake']['batch_size']
    epsilon = config['training-snake']['epsilon']
    epsilon_min = config['training-snake']['epsilon_min']
    epsilon_decay = config['training-snake']['epsilon_decay']
    gamma = config['training-snake']['gamma']
    
    # Callbacks for training
    callbacks = create_callbacks(config)

    # Training loop
    for game_num in tqdm(range(n_games)):
        game.reset()
        done = False
        
        while not done:
            state = game.get_state()
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                q_values = model.predict(np.array([state]), verbose=0)
                action = np.argmax(q_values[0])
            
            # Perform action
            reward = game.move(action)
            next_state = game.get_state()
            done = game.game_over

            # Store experience in memory
            memory.append((state, action, reward, next_state, done))
            
            # Train on a batch of experiences
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states = np.array([experience[0] for experience in batch])
                actions = np.array([experience[1] for experience in batch])
                rewards = np.array([experience[2] for experience in batch])
                next_states = np.array([experience[3] for experience in batch])
                dones = np.array([experience[4] for experience in batch])
                
                # Compute target Q-values
                target_q_values = target_model.predict(next_states, verbose=0)
                max_target_q_values = np.max(target_q_values, axis=1)
                target_q_values = rewards + gamma * max_target_q_values * (1 - dones)
                
                # Compute current Q-values and update
                current_q_values = model.predict(states, verbose=0)
                current_q_values[np.arange(batch_size), actions] = target_q_values
                
                # Train the model
                model.fit(states, current_q_values, verbose=0, callbacks=callbacks)
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Print game statistics
        if game_num % 100 == 0:
            print(f"\nGame {game_num}, Score: {game.score}, Epsilon: {epsilon:.2f}")
    
    # Save the trained model
    model.save('snake_model.h5')
    print("Training completed. Model saved as 'snake_model.h5'")

def play_game():
    game = SnakeGame()
    model = tf.keras.models.load_model('snake_model.h5')

    while not game.game_over:
        state = game.get_state()
        prediction = model(tf.convert_to_tensor(state.reshape((1, -1)), dtype=tf.float32))
        action = np.argmax(prediction[0])

        game.move(action)
        game.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.game_over = True

        pygame.time.delay(75)

    print(f"Game Over. Final Score: {game.score}")

if __name__ == "__main__":
    train_snake()
    play_game()
