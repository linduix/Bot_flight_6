from drone_prototype import Ai_Drone
from scoring_prototype import hover_scorer
from mutation_prototype import Innovations, add_connection
from genome_prototype import Genome
from breeding_prototype import breed
import util_prototype as utils
import multiprocessing as mp
import pygame as pg
import numpy as np
import cProfile

config = {
    "population": 1000,
    "width": 800,
    "height": 600,
    "meters_to_pixels": 15
}

if __name__ == '__main__':
    # pygame setup
    pg.init()
    screen = pg.display.set_mode((config["width"], config["height"]))
    pg.display.set_caption("Visual Training")
    clock = pg.time.Clock()

    # node graph vis
    viz_queue = mp.Queue()
    viz = mp.Process(target=utils.viz_process, args=(viz_queue,))
    viz.start()

    # load saved state
    if utils.save_path.exists():
        state = utils.load()
    else:
        print('created raw gen')
        # create the training state
        state = {
            'gen': 0,
            'current_gen': [],
            'innovations': Innovations(),
            'threshold': 1.5,
            'best_drone': None
        }
        # add one randome connection
        for g in state['current_gen']:
            add_connection(g, state['innovations'])
        # populate current gen
        state['current_gen'] = [Genome.new()for _ in range(config['population'])]
        for g in state['current_gen']:
            g.base_connections(state['innovations'])

    # drones: list[Ai_Drone] = [Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], g) for g in state['current_gen']]
    # print('starting')
    # cProfile.run('hover_scorer(drones, config["width"], config["height"], config["meters_to_pixels"], screen, clock, visualize=False)')

    limit = 5
    done = False
    while not done:
        drones: list[Ai_Drone] = [Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], g) for g in state['current_gen']]

        # run scorer
        return_code, scores, iterations = hover_scorer(
            drones,
            config["width"],
            config["height"],
            config["meters_to_pixels"],
            screen,
            clock,
            limit=limit
        )

        if return_code:
            break

        # calculate performance
        target_score = (limit * iterations / 2) * .9
        average_score = np.average(scores)
        average_connections = np.average([len(g.connections) for g in state['current_gen']])
        max_score = max(scores)
        if max_score > target_score:
            # finish if getting 98% of final target score
            if limit >= 60 and max_score/target_score > .98:
                done = True
            limit += 5

        # record best drone
        ix = np.argsort(scores)[-1]
        state['best_drone'] = state['current_gen'][ix]

        # visualize genome
        viz_queue.put(state['best_drone'])

        # breed next generation
        species = []
        if not return_code:
            next_gen, species = breed(state['current_gen'], scores, state['innovations'], config["population"], threshold=state["threshold"])
            state['current_gen'] = next_gen
            state['gen'] += 1

        # log training stats
        print(f'gen: {state["gen"]} | score: {average_score*100/target_score: .2f}% | max score: {max_score*100/target_score: .2f}% |', 
            f'target score: {target_score: .2f} | species count: {len(species)} | threshold: {state["threshold"]: .2f} | limit: {limit} |',
            f'bloat: {average_connections/average_score: .2f}')
        
        # adjust species thresholds
        if len(species) < 10:
            state["threshold"] *= .95
        elif len(species) > 15:
            state["threshold"] *= 1.05

    utils.save(state)
    viz_queue.put(None)
    viz.join()
