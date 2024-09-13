import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
# Disable OneDNN options
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pygame
import numpy as np
import cv2
import matplotlib.pyplot as plt
from colorama import Fore
import pygame.gfxdraw
import models.neural_network as nn


pygame.init()

# Increase window size
width, height = 800, 600
screen = pygame.display.set_mode((width * 2, height))
pygame.display.set_caption("Digit Recognizer")

# Colors
BACKGROUND_COLOR = (240, 240, 245)  # Light grayish blue
DRAWING_COLOR = (50, 50, 50)  # Dark gray
BORDER_COLOR = (180, 180, 200)  # Light grayish purple
TEXT_COLOR = (60, 60, 80)  # Dark grayish blue

def fancy_print(text, color):
    print(color + text + Fore.RESET)

def load_model():
    model = nn.NeuralNetwork('mnist', 0, 0)
    model.load_model(0)
    return model

def preprocess_image(surface):
    # Convert surface to numpy array
    arr = pygame.surfarray.array3d(surface)
    
    # Convert to grayscale
    arr = np.mean(arr, axis=2).astype(np.uint8)

    # Find bounding box of digit
    rows = np.any(arr < 240, axis=1)
    cols = np.any(arr < 240, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]] if len(np.where(rows)[0]) > 0 else (0, arr.shape[0])
    xmin, xmax = np.where(cols)[0][[0, -1]] if len(np.where(cols)[0]) > 0 else (0, arr.shape[1])
    
    # Add padding
    padding = max((ymax - ymin), (xmax - xmin)) // 10
    ymin, ymax = max(0, ymin - padding), min(arr.shape[0], ymax + padding)
    xmin, xmax = max(0, xmin - padding), min(arr.shape[1], xmax + padding)
    
    # Crop to bounding box
    cropped = arr[ymin:ymax+1, xmin:xmax+1]

    # Resize to 28x28
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to 0-1
    normalized = resized / 255.0

    # Invert the values
    normalized = 1 - normalized

    # Rotate 270 degrees
    normalized = np.rot90(normalized, k=3)

    # Flip horizontally
    normalized = np.fliplr(normalized)

    # Reshape to (784,1) to match the input shape expected by the model
    normalized = normalized.reshape(1, 784)

    return normalized

def draw_smooth_line(surface, color, start, end, width):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.gfxdraw.filled_circle(surface, x, y, width // 2, color)
        pygame.gfxdraw.aacircle(surface, x, y, width // 2, color)

def draw_prediction(surface, digit, probability, probabilities, indices):
    font = pygame.font.Font(None, 120)
    text = font.render(str(digit), True, TEXT_COLOR)
    
    # Calculate the position on the right side of the surface
    right_center_x = int(surface.get_width() * 0.75)  # 3/4 of the width
    right_center_y = surface.get_height() // 2
    
    rect = text.get_rect(center=(right_center_x, right_center_y))
    
    # Draw a colored circle behind the digit
    color = (int(255 * (1 - probability)), int(255 * probability), 100)
    pygame.draw.circle(surface, color, rect.center, 70)
    
    surface.blit(text, rect)

    bar_height = 30
    bar_margin = 15
    total_bar_height = (bar_height + bar_margin) * 10
    start_y = (surface.get_height() - total_bar_height) // 2
    
    for idx, i in enumerate(indices):
        prob = probabilities[i]
        bar_width = int(prob * 300)  # Scale the bar width
        bar_color = (0, 255, 0) if i == digit else (200, 200, 200)
        pygame.draw.rect(surface, bar_color, (10, start_y + idx * (bar_height + bar_margin), bar_width, bar_height))
        
        # Draw probability percentage
        prob_text = f"{i}: {prob*100:.1f}%"
        font = pygame.font.Font(None, 50)
        text = font.render(prob_text, True, TEXT_COLOR)
        surface.blit(text, (320, start_y + idx * (bar_height + bar_margin)))

def draw_rounded_rect(surface, color, rect, radius):
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def draw_active_area(surface):
    active_area_size = min(width, height) * 0.8
    x = (width - active_area_size) // 2
    y = (height - active_area_size) // 2
    pygame.draw.rect(surface, (200, 200, 220), (x, y, active_area_size, active_area_size), 2)

def draw_brush_size(surface, brush_size):
    font = pygame.font.Font(None, 24)
    text = font.render(f"Brush Size: {brush_size}", True, TEXT_COLOR)
    surface.blit(text, (width + 50, height - 50))

def is_in_active_area(pos):
    active_area_size = min(width, height) * 0.8
    x = (width - active_area_size) // 2
    y = (height - active_area_size) // 2
    return (x <= pos[0] <= x + active_area_size) and (y <= pos[1] <= y + active_area_size)

def main():
    model = load_model()
    fancy_print("Model loaded successfully!", Fore.GREEN)

    drawing = False
    erasing = False
    last_pos = None
    brush_size = 23
    last_image = None
    running = True
    draw_surface = pygame.Surface((width, height))
    draw_surface.fill(BACKGROUND_COLOR)
    draw_active_area(draw_surface)

    prediction_surface = pygame.Surface((width, height))
    prediction_surface.fill(BACKGROUND_COLOR)

    probabilities = [0] * 10

    fig, ax = plt.subplots(figsize=(5, 4))
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
                    erasing = False
                if event.button == 3:
                    erasing = True
                    drawing = False
                last_pos = pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                erasing = False
                img_array = preprocess_image(draw_surface)
                last_image = img_array
                prediction = model.predict(img_array)
                probabilities = prediction[0]
                predicted_digit = np.argmax(probabilities)

                # Update prediction surface
                prediction_surface.fill(BACKGROUND_COLOR)
                # Sort the probabilities in descending order
                sorted_indices = np.argsort(probabilities)[::-1]
                draw_prediction(prediction_surface, predicted_digit, probabilities[predicted_digit], probabilities, sorted_indices)

            elif event.type == pygame.MOUSEMOTION and drawing:
                current_pos = pygame.mouse.get_pos()
                if current_pos[0] < width and is_in_active_area(current_pos):
                    draw_smooth_line(draw_surface, DRAWING_COLOR, last_pos, current_pos, brush_size)
                last_pos = current_pos
            elif event.type == pygame.MOUSEMOTION and erasing:
                current_pos = pygame.mouse.get_pos()
                if current_pos[0] < width:  # Only erase within the left half
                    pygame.draw.line(draw_surface, BACKGROUND_COLOR, last_pos, current_pos, brush_size)
                    pygame.gfxdraw.aacircle(draw_surface, *current_pos, brush_size // 2 + 10, BACKGROUND_COLOR)
                    pygame.gfxdraw.filled_circle(draw_surface, *current_pos, brush_size // 2 + 10, BACKGROUND_COLOR)
                last_pos = current_pos

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    draw_surface.fill(BACKGROUND_COLOR)
                elif event.key == pygame.K_UP:
                    brush_size = min(50, brush_size + 1)
                elif event.key == pygame.K_DOWN:
                    brush_size = max(1, brush_size - 1)
                elif event.key == pygame.K_s:
                    # Show the last image
                    if last_image is not None:
                        plt.imshow(last_image.reshape(28, 28), cmap='gray')
                        plt.show()
        
        # Draw the drawable area
        screen.blit(draw_surface, (0, 0))
        # Draw the prediction area
        screen.blit(prediction_surface, (width, 0))

        # Draw the border around the prediction area
        pygame.draw.rect(screen, BORDER_COLOR, (width, 0, width, height), 2)

        # Draw the brush size
        draw_brush_size(screen, brush_size)

        # Draw the border around the drawable area
        pygame.draw.rect(screen, BORDER_COLOR, (0, 0, width, height), 2)

        # Draw the active area
        draw_active_area(screen)

        # Add instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "Draw a digit (0-9) in the left box",
            "Right-click to erase",
            "Press 'C' to clear",
            "Use Up/Down arrows to adjust brush size",
            "For best results, draw in the active area"
        ]
        for i, text in enumerate(instructions):
            rendered_text = font.render(text, True, TEXT_COLOR)
            screen.blit(rendered_text, (20, height - 100 + i * 25))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
