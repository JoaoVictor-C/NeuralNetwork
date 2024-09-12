import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
# Disable OneDNN options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pygame
import numpy as np
import cv2
import tensorflow as tf
from models.neural_network import create_model
import matplotlib.pyplot as plt
from colorama import Fore

# Initialize Pygame
pygame.init()

# Set up the drawing window
width, height = 280, 280
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw a digit")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def fancy_print(text, color):
    print(color + text + Fore.RESET)

def load_model():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Check if there are saved weights
    checkpoints_dir = 'checkpoints/model_8_fold_1'
    if os.path.exists(checkpoints_dir):
        # Reads the 4th to the 7th character of the last file, this is the epoch number
        # Get the latest checkpoint file
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.weights.h5')]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x[3:7]))
            epoch = int(latest_checkpoint[3:7])
        else:
            epoch = 0
        print(epoch)
        latest_checkpoint = os.path.join(checkpoints_dir, f'cp-{epoch:04d}.weights.h5')
        if latest_checkpoint:
            model.load_weights(latest_checkpoint)
            fancy_print(f"Model weights loaded successfully! {latest_checkpoint}", Fore.GREEN)
        else:
            fancy_print("No saved weights found. Using untrained model.", Fore.YELLOW)
    else:
        fancy_print("No checkpoints directory found. Using untrained model.", Fore.YELLOW)
    
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

    # put the resized image in the center of a 28x28 black image
    final = np.zeros((28, 28), dtype=np.uint8)
    x_center = (28 - resized.shape[1]) // 2
    y_center = (28 - resized.shape[0]) // 2
    final[y_center:y_center+resized.shape[0], x_center:x_center+resized.shape[1]] = resized

    # Normalize to 0-1
    normalized = final / 255.0

    # invert the values
    normalized = 1 - normalized
    
    return normalized.reshape(1, 28, 28), normalized

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
                    
                    # Set model to evaluation mode
                    model.trainable = False
                    
                    # Make prediction
                    try:
                        prediction = model.predict(img_array, verbose=0)
                        probabilities = prediction[0]
                        predicted_digit = np.argmax(probabilities)
                        
                        print(f"Predicted digit: {predicted_digit}")
                        print("Digit probabilities:")
                        for i, prob in enumerate(probabilities):
                            print(f"Digit {i}: {prob*100:.2f}%")
                        
                        # Display the preprocessed image
                        plt.figure(figsize=(5, 5))
                        plt.imshow(img_28x28, cmap='binary', interpolation='nearest')
                        plt.title(f"Preprocessed Image - Predicted: {predicted_digit}")
                        plt.axis('off')
                        plt.show()
                    
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                    
                    # Add this line to reset the model to training mode
                    model.trainable = True
                    
                    print("\nPress 'C' to clear the screen and draw again.")
                elif event.key == pygame.K_c:
                    draw_surface.fill(WHITE)

        screen.fill(WHITE)
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
