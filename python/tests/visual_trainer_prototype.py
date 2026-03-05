from drone_prototype import Ai_Drone
from scoring_prototype import stage1_viz
from mutation_prototype import Innovations, add_connection
from genome_prototype import Genome
from breeding_prototype import breed
import util_prototype as utils
import multiprocessing as mp
import pygame as pg
import numpy as np
import cProfile

config = {
    "population": 200,
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
            'threshold': 0.5,
            'best_drone': None,
            'historical_score': []
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

    limit = 10
    done = False
    while not done:
        drones: list[Ai_Drone] = [Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], g) for g in state['current_gen']]

        # run scorer
        return_code, scores, completions = stage1_viz(
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
        max_score = max(scores)

        # log score history
        state.setdefault('historical_score', [])
        state['historical_score'].append(max_score)
        # get past 10 rolling average
        rolling_average = np.average(state['historical_score'][-10:])
        # calculate improvement from roling average change
        improvement = rolling_average - np.average(state['historical_score'][-20:-10]) if len(state['historical_score']) > 10 else 0

        # get average connections
        connections = []
        for g in state['current_gen']:
            sum = 0
            for c in g.connections:
                sum += 1 if c.enabled else 0
            connections.append(sum)
        average_connections = np.average(connections)

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
        print(f'gen: {state["gen"]} | score: {rolling_average: .2f} | max score: {max_score: .1f} | completions: {completions} |',
            f'improvement: {improvement: .1f} | species count: {len(species)} | threshold: {state["threshold"]: .2f} | limit: {limit} |',
            f'bloat: {average_connections/rolling_average: .2f}')
        
        # adjust species thresholds
        if len(species) < 10:
            diff = abs(len(species) - (10+15)/2)
            state["threshold"] *= max(1 - (diff * 0.016), 0.1)
        elif len(species) > 15:
            diff = abs(len(species) - (10+15)/2)
            state["threshold"] *= 1 + (diff * 0.016)

    utils.save(state)
    viz_queue.put(None)
    viz.join()
