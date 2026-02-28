from drone_prototype import Ai_Drone
from scoring_prototype import hover_scorer
from mutation_prototype import Innovations, add_connection
from genome_prototype import Genome
from breeding_prototype import breed
import pygame as pg
import numpy as np
import cProfile

config = {
    "population": 1000,
    "width": 800,
    "height": 600,
    "meters_to_pixels": 30
}

# pygame setup
pg.init()
screen = pg.display.set_mode((config["width"], config["height"]))
pg.display.set_caption("Visual Training")
clock = pg.time.Clock()

# create innovation tracker
innovations = Innovations()

# generate blank population
current_gen: list[Genome] = [Genome.new() for i in range(config['population'])]
# add one randome connection
for g in current_gen:
    add_connection(g, innovations)

drones: list[Ai_Drone] = [Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], g) for g in current_gen]
# print('starting')
# cProfile.run('hover_scorer(drones, config["width"], config["height"], config["meters_to_pixels"], screen, clock, visualize=False)')

gen = 0
threshold = 1.5
limit = 5
while True:
    drones: list[Ai_Drone] = [Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], g) for g in current_gen]

    return_code, scores, iterations = hover_scorer(
        drones,
        config["width"],
        config["height"],
        config["meters_to_pixels"],
        screen,
        clock,
        visualize=True,
        limit=5
    )

    if return_code:
        break

    target_score = (limit * iterations / 2) * .9
    average_score = np.average(scores)
    max_score = max(scores)
    if max_score > target_score:
        limit += 5

    next_gen, species = breed(current_gen, scores, innovations, config["population"], threshold=threshold)
    current_gen = next_gen
    gen += 1

    print(f'gen: {gen} | score: {average_score*100/target_score: .2f}% | max score: {max_score*100/target_score: .2f}% |', 
          f'target score: {target_score: .2f} | species count: {len(species)} | threshold: {threshold: .2f}')
    if len(species) < 10:
        threshold *= .95
    elif len(species) > 15:
        threshold *= 1.05
