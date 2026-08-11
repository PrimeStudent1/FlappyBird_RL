"""
Manual Human Gameplay Script for FlappyBird Gymnasium Environment using PyGame.
"""

import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Initialize environment with PyGame rendering
env = gym.make("FlappyBird-v0", render_mode="human")
state, info = env.reset()
done = False

# Initialize PyGame window and event loop
pygame.init()
screen = pygame.display.get_surface()

while not done:
    action = 0  # Default action: 0 = idle/no flap, 1 = flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1  # Trigger flap action on spacebar

    state, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
pygame.quit()