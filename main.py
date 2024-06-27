import copy
import os
import pickle
import pygame
import time
from math import sqrt, floor
import numpy as np
import random

from food import Food
from model import game_state_to_data_sample, prepare_probability
from snake import Snake, Direction


random.seed(1)

def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    # agent = HumanAgent(block_size, bounds)  # Once your agent is good to go, change this line
    agent = BehavioralCloningAgent(block_size, bounds)
    scores = []
    run = True
    pygame.time.delay(1000)
    while len(scores) < 100:
    # for i in range(100):
        pygame.time.delay(1)  # Adjust game speed, decrease to test your agent and model quickly

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state)
        snake.turn(direction)

        snake.move()
        snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            # pygame.time.delay(300)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    print('Średni wynik: ', np.mean(scores))
    pygame.quit()


class BehavioralCloningAgent:
    
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

        self.p, self.probs = prepare_probability()


    def predict_proba(self, x):
        # Predykcja
        p = np.ones(4)
        for i in range(4):
            p[i] = self.p[i]
            for j in range(8):
                p[i] *= self.probs[i, j, x[j]]
        return p / p.sum()

    def predict(self, x):
        return np.argmax(self.predict_proba(x))

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        data_sample = game_state_to_data_sample(game_state, self.bounds, self.block_size)
        self.atrributes = data_sample

        action = Direction(self.predict(data_sample))

        if action == None:
            action = game_state['snake_direction']

        self.data.append((copy.deepcopy(game_state), action))
        
        return action


if __name__ == "__main__":
    main()

