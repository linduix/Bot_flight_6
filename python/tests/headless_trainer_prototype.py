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
        best_ever = max(state['historical_score']) if state.get('historical_score') else 0
        training_start = time.time()
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
            if not return_code:
                next_gen, species_pop, spec, deaths, cull_stats = breed(state['current_gen'], scores, state['innovations'], config["population"], state['species'], threshold=state["threshold"])
                state['current_gen'] = next_gen
                state['gen'] += 1
                state['species'] = spec
            else:
                species_pop = []
                deaths = 0
                cull_stats = {'stagnant_killed': 0, 'killed_genomes': 0}

            # breed profiling
            if profile:
                pr.disable() # type: ignore
                pr.dump_stats("breed0.prof")  # type: ignore
                print("wrote breed0.prof")
                profile = False

            # compute stats for logging
            best_ever = max(best_ever, max_score)
            avg_score = np.average(scores)
            pop_size = len(scores)
            prev_max = state['historical_score'][-2] if len(state['historical_score']) >= 2 else max_score
            score_delta = max_score - prev_max
            total_elapsed = time.time() - training_start
            gen_rate = 60 / elapsed if elapsed > 0 else 0
            # format total elapsed
            t_hrs, t_rem = divmod(total_elapsed, 3600)
            t_min, t_sec = divmod(t_rem, 60)
            if t_hrs > 0:
                elapsed_fmt = f"{int(t_hrs)}h {int(t_min)}m"
            else:
                elapsed_fmt = f"{int(t_min)}m {int(t_sec)}s"

            # species line
            species_info = f"count: {len(species_pop)}"
            if cull_stats['stagnant_killed'] > 0:
                species_info += f" | stagnant killed: {cull_stats['stagnant_killed']} ({cull_stats['killed_genomes']} genomes removed)"

            # log training stats to terminal
            delta_sign = "+" if score_delta >= 0 else ""
            print(f"── S{stage} Gen {state['gen']} {'─' * 50}")
            print(f"  score    max: {max_score:.2f} | avg: {avg_score:.2f} | rolling: {rolling_average:.2f} | best ever: {best_ever:.2f} | Δ: {delta_sign}{score_delta:.1f}")
            if stage == 0:
                pct = max_score / target_score * 100 if target_score > 0 else 0
                print(f"  progress target: {target_score:.0f} ({pct:.1f}%) | improvement: {improvement:.1f} | limit: {limit}s")
            else:
                assert isinstance(completions, list)
                c_time = np.average(completions) if completions else float("nan")
                comp_pct = len(completions) / pop_size * 100
                print(f"  progress complete: {len(completions)}/{pop_size} ({comp_pct:.1f}%) | avg c_time: {c_time:.2f}s | difficulty: {adj_diff:.2f}m | improvement: {improvement:.1f}")
            print(f"  species  {species_info}")
            print(f"  genome   avg connections: {average_connections:.1f} | pop: {pop_size}")
            print(f"  timing   gen: {elapsed:.2f}s | rate: {gen_rate:.1f} gen/min | elapsed: {elapsed_fmt}")

            # log to discord
            if (state['gen'] % 50 == 0 or first) and logging:
                print('logging...')
                delta_str = f"{delta_sign}{score_delta:.1f}"
                lines = [
                    f"**{NAME} | S{stage} Gen {state['gen']}**",
                    f"```",
                    f"Score    max: {max_score:.2f}  avg: {avg_score:.2f}  rolling: {rolling_average:.2f}  best: {best_ever:.2f}  Δ{delta_str}",
                ]
                if stage == 0:
                    lines.append(f"Progress target: {target_score:.0f} ({pct:.1f}%)  improvement: {improvement:.1f}  limit: {limit}s")
                else:
                    assert isinstance(completions, list)
                    lines.append(f"Progress complete: {len(completions)}/{pop_size} ({comp_pct:.1f}%)  c_time: {c_time:.2f}s  diff: {adj_diff:.2f}m")
                lines += [
                    f"Species  {species_info}",
                    f"Genome   avg conn: {average_connections:.1f}  pop: {pop_size}",
                    f"Timing   gen: {elapsed:.2f}s  rate: {gen_rate:.1f}/min  elapsed: {elapsed_fmt}",
                    f"```",
                ]
                discord_logger.log("\n".join(lines))
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
                    first = True
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
