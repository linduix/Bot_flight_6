from drone_prototype import Ai_Drone
from scoring_prototype import hover_scorer_headless, stage1
from mutation_prototype import Innovations, add_connection
from genome_prototype import Genome, NodeType
from breeding_prototype import breed
from dotenv import load_dotenv
import util_prototype as utils
import numpy as np
import cProfile
import time
import pstats
import io
import os
import requests

config = {
    "population": 1000,
    "width": 800,
    "height": 600,
    "meters_to_pixels": 15
}

if __name__ == '__main__':
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
            'historical_score': [],
            'stage': 0,
            'difficulty': 15,
            'species': []
        }

        # populate current gen and add base connections
        state['current_gen'] = [Genome.new()for _ in range(config['population'])]
        for g in state['current_gen']:
            g.base_connections(state['innovations'])

    # drones: list[Ai_Drone] = [Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], g) for g in state['current_gen']]
    # print('starting')
    # cProfile.run('hover_scorer_headless(drones, config["width"], config["height"], config["meters_to_pixels"], limit=5)')

    # setup discord logger
    load_dotenv()
    WEBHOOK = os.environ["DISCORD_WEBHOOK"]
    NAME = os.environ['NAME']
    logging = os.environ['LOGGING'] == 'ON'
    discord_logger = utils.DiscordLogger(WEBHOOK, interval=5)

    # webhook test
    if logging:
        r = requests.post(WEBHOOK, json={"content": f"{NAME}>> TRAINING INIT"}, timeout=5)
        print("discord test:", r.status_code, r.text[:200])

    print('training starting...')
    try:
        stage = state['stage']
        if stage == 0:
            limit = 5
        else:
            limit = 7
        done = False
        profile = False
        first = True
        while not done:
            #create drones
            drones: list[Ai_Drone] = [Ai_Drone((0, 0), config['meters_to_pixels'], config["height"], g) for g in state['current_gen']]

            # time start
            start = time.time()

            # Difficulty
            state.setdefault('difficulty', 15)
            difficulty = state['difficulty']
            adj_diff = difficulty

            return_code = 1
            if stage == 0:
                # run scorer
                completions = []
                return_code, scores, iterations = hover_scorer_headless(
                    drones,
                    config["width"],
                    config["height"],
                    config["meters_to_pixels"],
                    limit=limit
                )
            else:
                iterations = 0
                if np.random.rand() < .15:
                    # ocassionally easier difficulty so it dont forget earlier training
                    adj_diff *= np.random.rand() * 0.7
                    adj_diff = max(adj_diff, 10)
                return_code, scores, completions = stage1 (
                    drones,
                    config["width"],
                    config["height"],
                    config["meters_to_pixels"],
                    limit=limit,
                    diff=adj_diff
                )

            # time end
            elapsed = time.time() - start

            if return_code:
                break

            # calculate performance
            if stage == 0:
                target_score = iterations * .85
            else:
                target_score = 0
            max_score = max(scores)

            # log score history
            state.setdefault('historical_score', [])
            state['historical_score'].append(max_score)
            # get past 10 rolling average
            rolling_average = np.average(state['historical_score'][-10:])
            # calculate improvement from roling average change
            improvement = rolling_average - np.average(state['historical_score'][-20:-10]) if len(state['historical_score']) > 20 else 0
            
            # get average connections
            connections = []
            for g in state['current_gen']:
                enabled_sum = 0
                for c in g.connections:
                    enabled_sum += 1 if c.enabled else 0
                connections.append(enabled_sum)
            average_connections = np.average(connections)

            # record best drone
            ix = np.argsort(scores)[-1]
            state['best_drone'] = state['current_gen'][ix]

            # breed profiling
            if profile:
                print('profiling')
                pr = cProfile.Profile()
                pr.enable()

            # breed next generation
            state.setdefault('species', [])
            next_gen, species_pop, spec, deaths = breed(state['current_gen'], scores, state['innovations'], config["population"], state['species'], threshold=state["threshold"])
            state['current_gen'] = next_gen
            state['gen'] += 1
            state['species'] = spec

            # breed profiling
            if profile:
                pr.disable() # type: ignore
                pr.dump_stats("breed0.prof")  # type: ignore
                print("wrote breed0.prof")
                profile = False

            # log training stats to terminal
            if stage == 0:
                print(f'stage: {stage} | gen: {state["gen"]} | rolling max: {rolling_average: .2f} | max score: {max_score: .1f} |',
                    f'target score: {target_score : .0f} | improvement: {improvement: .1f} | species count: {len(species_pop)} | threshold: {state["threshold"]: .2f} | limit: {limit}s |',
                    f'time: {elapsed: .2f}s')
            else:
                assert isinstance(completions, list)
                c_time = np.average(completions) if completions else float("nan")
                print(f'stage: {stage} | gen: {state["gen"]} | rolling max: {rolling_average: .2f} | max score: {max_score: .2f} | complete: {len(completions)} |',
                    f'c time: {c_time: .2f}s | improved: {improvement: .1f} | species: {len(species_pop)} |',
                    f'threshold: {state["threshold"]: .2f} | diff: {adj_diff: .2f}m | time: {elapsed: .2f}s')

            # log to discord
            if (state['gen'] % 50 == 0 or first) and logging:
                print('logging...')
                if stage == 0:
                    log = (
                        f"{NAME}>> stage: {stage} | gen: {state['gen']} | rolling max: {rolling_average:.2f} | "
                        f"max score: {max_score:.1f} | improvement: {improvement:.1f} | "
                        f"limit: {limit}\n"
                        f"{NAME}>> species distribution: {[len(s) for s in species_pop]} | deaths: {deaths}"
                    )
                else:
                    assert isinstance(completions, list)
                    log = (
                        f"{NAME}>> stage: {stage} | gen: {state['gen']} | rolling max: {rolling_average:.2f} | "
                        f"max score: {max_score:.2f} | improvement: {improvement:.1f} |"
                        f"completions: {len(completions)} | c time: {c_time: .2f}s | diff: {adj_diff: .2f}m\n" # type: ignore
                        f"{NAME}>> species distribution: {[len(s) for s in species_pop]} | deaths: {deaths}"
                    )
                discord_logger.log(log)
                first = False

            # adjust species thresholds
            if len(species_pop) < 10:
                diff = abs(len(species_pop) - (10+15)/2)
                state["threshold"] *= max(1 - (diff * 0.016), 0.1)
            elif len(species_pop) > 15:
                diff = abs(len(species_pop) - (10+15)/2)
                state["threshold"] *= 1 + (diff * 0.016)

            # adjust difficulty
            target = 0.1
            if stage == 1:
                assert isinstance(completions, list)
                error = len(completions) / config['population'] - target
                if abs(error) > 0.02:
                    difficulty *= np.sqrt(error + 1)
                    difficulty = max(difficulty, 10)
                    state['difficulty'] = difficulty

            # progress hover stage
            if stage == 0:
                # finish if getting 90% of final target score
                if limit >= 30 and max_score/target_score > .95:
                    stage = 1
                    state['stage'] = 1
                    limit = 7
                    state['historical_score'] = []
                    state['species'] = []
                    utils.save(state)
                if max_score / target_score > .9:
                    limit += 5

            # save progress every 500 gens
            if state['gen'] % 100 == 0:
                utils.save(state)

    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    finally:
        print('---------------------------')
        if logging:
            print('closing logger...')
            discord_logger.log(f'{NAME}>> TRAINING TERM')
        discord_logger.close()
        utils.save(state)
