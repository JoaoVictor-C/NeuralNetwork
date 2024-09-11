import pygame
import numpy as np
from src.core.neural_network import NeuralNetwork
import json
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, zoom
from src.utils import fancy_print
import os
from colorama import Fore, Style
import cv2
# Initialize Pygame
pygame.init()

# Set up the drawing window
width, height = 280, 280
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw a digit")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)



def load_model():
    model_path = 'src/models/model_0.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open('src/config/nn_parameters.json', 'r') as f:
        nn_config = json.load(f)['mnist']
    model = NeuralNetwork('mnist', model_index=0, verbose=False)

    model = model.load_model(0)
    
    # Verify the model structure
    if len(model.layers) != len(nn_config['layers']) - 1:
        raise ValueError("Loaded model structure does not match the configuration")
    
    return model

def preprocess_image(surface):
    # Convert surface to numpy array
    arr = pygame.surfarray.array3d(surface)
    
    # Convert to grayscale
    arr = np.mean(arr, axis=2).astype(np.uint8)

    # Rotates 270 degrees
    arr = np.rot90(arr, k=3)
    
    # Flips horizontally
    arr = np.fliplr(arr)

    # Find bounding box of digit
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Crop to bounding box
    cropped = arr[ymin:ymax+1, xmin:xmax+1]

    # Resize to 28x28
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to 0-1
    normalized = resized / 255.0
    
    return normalized.reshape(1, 784), normalized

def main():
    try:
        model = load_model()
        fancy_print("Model loaded successfully!", Fore.GREEN)
    except Exception as e:
        fancy_print(f"Error loading model: {e}", Fore.RED)
        return

    # Drawing variables
    drawing = False
    last_pos = None
    brush_size = 15
    
    running = True
    draw_surface = pygame.Surface((width, height))
    draw_surface.fill(WHITE)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                last_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                current_pos = pygame.mouse.get_pos()
                pygame.draw.line(draw_surface, BLACK, last_pos, current_pos, brush_size)
                last_pos = current_pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    img_array, img_28x28 = preprocess_image(draw_surface)
                    
                    # Set all layers to evaluation mode
                    for layer in model.layers:
                        if hasattr(layer, 'use_batch_norm') and layer.use_batch_norm:
                            layer.batch_norm.training = False
                    
                    # Make prediction
                    try:
                        prediction = model.predict(img_array, training=False)
                        probabilities = prediction[0]
                        predicted_digit = np.argmax(probabilities)
                        
                        print(f"Predicted digit: {predicted_digit}")
                        print("Digit probabilities:")
                        for i, prob in enumerate(probabilities):
                            print(f"Digit {i}: {prob*100:.2f}%")
                        
                        # Display the preprocessed image
                        plt.figure(figsize=(5, 5))
                        plt.imshow(img_28x28, cmap='gray', interpolation='nearest')
                        plt.title(f"Preprocessed Image - Predicted: {predicted_digit}")
                        plt.axis('off')
                        plt.show()
                    
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                    
                    print("\nPress 'C' to clear the screen and draw again.")
                elif event.key == pygame.K_c:
                    draw_surface.fill(WHITE)

        screen.fill(WHITE)
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
