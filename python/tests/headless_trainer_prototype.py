import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from scoring_prototype import hover_scorer_headless
from prototype_stage1 import stage1_vmax_test
from prototype_stage2 import stage2_vmax_test, POOL_REFRESH_GENS, NUM_WAYPOINTS
from mutation_prototype import Innovations, add_connection
from genome_prototype import Genome, NodeType
from breeding_prototype import breed, breed_pareto, STAGNATION_CHANCES, STAGNATION_LIMIT
from dotenv import load_dotenv
import util_prototype as utils
import numpy as np
import multiprocessing as mp
import cProfile
import time
import pstats
import math
import io
import requests

BREED_MODE = "pareto"  # "pareto" or "neat"

config = {
    "population": 300,
    "width": 800,
    "height": 600,
    "meters_to_pixels": 15
}

import signal
def _pool_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # workers ignore Ctrl+C

if __name__ == '__main__':
    # load saved state
    if utils.save_path.exists():
        state = utils.load()
        # patch old Species objects missing new fields
        for i, s in enumerate(state.get('species', [])):
            if not hasattr(s, 'best_history'):
                s.best_history = []
            if not hasattr(s, 'chances'):
                s.chances = STAGNATION_CHANCES
            if not hasattr(s, 'age'):
                s.age = len(s.best_history)  # best guess from existing data
            if not hasattr(s, 'id'):
                s.id = i  # backfill legacy species with unique index
        if state.get('species'):
            from breeding_prototype import Species
            Species._next_id = max(s.id for s in state['species']) + 1
        # patch old Genome objects missing new fields
        for g in state.get('current_gen', []):
            if not hasattr(g, 'mutation_power'):
                g.mutation_power = 0.3
            if not hasattr(g, '_species_id'):
                g._species_id = None
            if not hasattr(g, '_conn_cache'):
                g._conn_cache = None
        if state.get('best_drone') and not hasattr(state['best_drone'], 'mutation_power'):
            state['best_drone'].mutation_power = 0.3
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
    logging = os.environ.get('LOGGING', 'OFF') == 'ON'

    # webhook test
    if logging:
        NAME = os.environ['NAME']
        WEBHOOK = os.environ["DISCORD_WEBHOOK"]
        discord_logger = utils.DiscordLogger(WEBHOOK, interval=5)
        r = requests.post(WEBHOOK, json={"content": f"{NAME}>> TRAINING INIT"}, timeout=5)
        print("discord test:", r.status_code, r.text[:200])

    # multiprocessing pool
    num_workers = max(1, os.cpu_count())
    use_mp = num_workers > 1
    if use_mp:
        pool = mp.Pool(processes=num_workers, initializer=_pool_init)
        print(f'started pool with {num_workers} workers')
    else:
        pool = None
        print('single CPU detected, running without multiprocessing')

    print('training starting...')
    try:
        stage = state['stage']
        FORCE_STAGE = None  # set to 2 to force stage 2 for testing
        if FORCE_STAGE is not None:
            stage = FORCE_STAGE
            state['stage'] = FORCE_STAGE
        if stage == 0:
            limit = 5
        elif stage == 1:
            limit = 7
        else:
            limit = 15
        done = False
        profile = False
        first = True
        best_ever = max(state['historical_score']) if state.get('historical_score') else 0
        plateau_counter = 0  # gens since global best was improved
        training_start = time.time()

        # ── 50-gen stats buffer for discord ──
        LOG_INTERVAL = 50
        log_buf = utils.pareto_log_buf() if BREED_MODE == "pareto" else utils.neat_log_buf()

        # threshold stuff
        spec_target_max = 0.3
        spec_target_min = 0.05

        while not done:
            # time start
            start = time.time()

            # Difficulty
            state.setdefault('difficulty', 15)

            return_code = 1
            genomes = state['current_gen']

            if stage == 0:
                if use_mp:
                    # chunked parallel scoring
                    N = len(genomes)
                    chunk_size = math.ceil(N / num_workers)
                    chunks = [genomes[i:i+chunk_size] for i in range(0, N, chunk_size)]
                    completions = []
                    results = pool.starmap(hover_scorer_headless, [
                        (chunk, config["width"], config["height"], config["meters_to_pixels"], limit)
                        for chunk in chunks
                    ])
                    return_code = max(r[0] for r in results)
                    scores = np.concatenate([r[1] for r in results])
                    iterations = results[0][2]
                else:
                    completions = []
                    return_code, scores, iterations = hover_scorer_headless(
                        genomes, config["width"], config["height"],
                        config["meters_to_pixels"], limit=limit
                    )
            elif stage == 1:
                iterations = 0
                if use_mp:
                    N = len(genomes)
                    chunk_size = math.ceil(N / num_workers)
                    chunks = [genomes[i:i+chunk_size] for i in range(0, N, chunk_size)]
                    results = pool.starmap(stage1_vmax_test, [
                        (chunk, config["width"], config["height"], config["meters_to_pixels"], limit, state['difficulty'])
                        for chunk in chunks
                    ])
                    return_code = max(r[0] for r in results)
                    scores = np.concatenate([r[1] for r in results])
                    completions = []
                    for r in results:
                        completions.extend(r[2])
                    avg_completions = sum(r[3] for r in results)
                else:
                    return_code, scores, completions, avg_completions = stage1_vmax_test(
                        genomes, config["width"], config["height"],
                        config["meters_to_pixels"], limit=limit, diff=state['difficulty'],
                    )
            elif stage == 2:
                iterations = 0
                # Pool refresh: same seed for POOL_REFRESH_GENS, then rotate
                if 'pool_seed' not in state or state.get('pool_gen', 0) >= POOL_REFRESH_GENS:
                    val_seeds = set(range(11))  # validation uses seeds 0-10
                    new_seed = int(np.random.randint(0, 2**31))
                    while new_seed in val_seeds:
                        new_seed = int(np.random.randint(0, 2**31))
                    state['pool_seed'] = new_seed
                    state['pool_gen'] = 0
                    state['pool_baseline_pending'] = True
                seed = state['pool_seed']
                state['pool_gen'] = state.get('pool_gen', 0) + 1
                if use_mp:
                    N = len(genomes)
                    chunk_size = math.ceil(N / num_workers)
                    chunks = [genomes[i:i+chunk_size] for i in range(0, N, chunk_size)]
                    results = pool.starmap(stage2_vmax_test, [
                        (chunk, config["width"], config["height"], config["meters_to_pixels"], limit, state['difficulty'], seed)
                        for chunk in chunks
                    ])
                    return_code = max(r[0] for r in results)
                    scores = np.concatenate([r[1] for r in results])
                    completions = []
                    for r in results:
                        completions.extend(r[2])
                    avg_completions = sum(r[3] for r in results)
                    wp_stats = {
                        'min': min(r[4]['min'] for r in results),
                        'q1': np.mean([r[4]['q1'] for r in results]),
                        'q3': np.mean([r[4]['q3'] for r in results]),
                        'max': max(r[4]['max'] for r in results),
                    }
                    avg_leg_dist = results[0][5]  # same chains, same value across chunks
                else:
                    return_code, scores, completions, avg_completions, wp_stats, avg_leg_dist = stage2_vmax_test(
                        genomes, config["width"], config["height"],
                        config["meters_to_pixels"], limit=limit, diff=state['difficulty'], seed=seed,
                    )

            # time end
            sim_time = time.time() - start

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

            # genome stats
            gs = utils.compute_genome_stats(state['current_gen'])

            # record best drone
            prev_best_drone = state.get('best_drone')
            ix = np.argsort(scores)[-1]
            state['best_drone'] = state['current_gen'][ix]

            # breed profiling
            if profile:
                print('profiling')
                pr = cProfile.Profile()
                pr.enable()

            # Pool baseline: recalibrate species best_score on first gen of new pool
            # so the new pool's score scale doesn't cause false stagnation.
            # Stagnation counters reset to 0 naturally (best_score = -inf → improved).
            # Pool lasts 100 gens vs 25 gen stagnation limit, so bad species still die.
            breed_start = time.time()
            if BREED_MODE == "pareto":
                best_genome = state['best_drone']
                next_gen, pareto_stats = breed_pareto(state['current_gen'], scores, state['innovations'], config["population"], best_genome)
                state['current_gen'] = next_gen
                state['gen'] += 1
            else:
                if stage == 2 and state.get('pool_baseline_pending', False):
                    for s in state.get('species', []):
                        s.best_score = -np.inf
                state.setdefault('species', [])
                if not return_code:
                    next_gen, species_pop, spec, deaths, cull_stats = breed(state['current_gen'], scores, state['innovations'], config["population"], state['species'], threshold=state["threshold"])
                    state['current_gen'] = next_gen
                    state['gen'] += 1
                    state['species'] = spec
                    stagnant_count = sum(1 for s in spec if s.stagnation > STAGNATION_LIMIT // 2)
                else:
                    species_pop = []
                    deaths = 0
                    cull_stats = {'stagnant_killed': 0, 'killed_genomes': 0}
                    stagnant_count = 0

            breed_time = time.time() - breed_start
            elapsed = sim_time + breed_time

            # breed profiling
            if profile:
                pr.disable() # type: ignore
                pr.dump_stats("breed0.prof")  # type: ignore
                print("wrote breed0.prof")
                profile = False

            # track best & plateau
            if stage == 2 and state.get('pool_baseline_pending', False):
                state['pool_baseline_pending'] = False
            if stage == 2:
                state.setdefault('best_validation_score', 0.0)
                ix_best = int(np.argsort(scores)[-1])
                best_genome = state['current_gen'][ix_best]
                val_total = 0.0
                for vs in range(11):  # validate across seeds 0-10
                    _, vs_scores, *_ = stage2_vmax_test(
                        [best_genome], config["width"], config["height"],
                        config["meters_to_pixels"], limit=limit, diff=state['difficulty'],
                        seed=vs,
                    )
                    val_total += float(vs_scores[0])
                val_score = val_total / 11
                if val_score > state['best_validation_score']:
                    state['best_validation_score'] = val_score
                    plateau_counter = 0
                    utils.save(state, "prototype_best.pkl")
                else:
                    plateau_counter += 1
                best_ever = state['best_validation_score']
            elif stage == 1:
                if max_score > best_ever:
                    # Score beaten — pit candidate against current best
                    candidate = state['best_drone']
                    if prev_best_drone is not None:
                        _, val_scores, *_ = stage1_vmax_test(
                            [candidate, prev_best_drone],
                            config["width"], config["height"],
                            config["meters_to_pixels"], limit=limit, diff=state['difficulty'],
                        )
                        if val_scores[0] > val_scores[1]:
                            best_ever = max_score
                            plateau_counter = 0
                            utils.save(state, "prototype_best.pkl")
                            print(f'  VAL PASS  candidate {val_scores[0]:.4f} > best {val_scores[1]:.4f}')
                        else:
                            plateau_counter += 1
                            print(f'  VAL FAIL  candidate {val_scores[0]:.4f} <= best {val_scores[1]:.4f}')
                    else:
                        best_ever = max_score
                        plateau_counter = 0
                        utils.save(state, "prototype_best.pkl")
                else:
                    plateau_counter += 1
            elif max_score > best_ever:  # stage 0
                best_ever = max_score
                plateau_counter = 0
                utils.save(state, "prototype_best.pkl")
            else:
                plateau_counter += 1

            avg_score = np.average(scores)
            pop_size = len(scores)
            elapsed_fmt = utils.format_elapsed(time.time() - training_start)

            # stage-specific kwargs shared by terminal, accumulate, and discord
            stage_kw = {}
            if stage == 0:
                stage_kw['target_score'] = target_score
                stage_kw['limit'] = limit
            elif stage == 1:
                stage_kw.update(completions=completions, avg_completions=avg_completions,
                               difficulty=state['difficulty'], limit=limit)
            elif stage == 2:
                stage_kw.update(completions=completions, avg_completions=avg_completions,
                               difficulty=state['difficulty'], limit=limit,
                               wp_stats=wp_stats, pool_gen=state.get('pool_gen', 0),
                               pool_refresh=POOL_REFRESH_GENS,
                               avg_leg_dist=avg_leg_dist, num_wp=NUM_WAYPOINTS)

            if BREED_MODE == "pareto":
                # terminal log
                utils.pareto_log_terminal(
                    state['gen'], stage, max_score, avg_score, best_ever, plateau_counter,
                    pareto_stats, gs, scores, sim_time, breed_time,
                    elapsed_fmt, pop_size,
                    val_score=val_score if stage == 2 else None,
                    **stage_kw,
                )

                # accumulate into 50-gen buffer
                utils.pareto_accumulate_buf(
                    log_buf, max_score, avg_score, gs, scores, sim_time, breed_time,
                    pareto_stats, pop_size,
                    stage=stage,
                    completions=completions if stage >= 1 else None,
                    avg_completions=avg_completions if stage >= 1 else None,
                    wp_stats=wp_stats if stage == 2 else None,
                )

                # discord log on interval
                if (state['gen'] % LOG_INTERVAL == 0 or first) and logging:
                    print('logging...')
                    msg = utils.pareto_log_discord(
                        NAME, state['gen'], stage, log_buf, best_ever, plateau_counter,
                        pop_size, elapsed_fmt,
                        val_score=val_score if stage == 2 else None,
                        difficulty=state.get('difficulty'),
                        limit=limit,
                        wp_stats=wp_stats if stage == 2 else None,
                        pool_gen=state.get('pool_gen', 0) if stage == 2 else None,
                        pool_refresh=POOL_REFRESH_GENS if stage == 2 else None,
                        avg_leg_dist=avg_leg_dist if stage == 2 else None,
                        num_wp=NUM_WAYPOINTS if stage == 2 else None,
                    )
                    discord_logger.log(msg)
                    first = False
                    log_buf = utils.pareto_log_buf()
            else:
                ss = utils.compute_species_stats(state['species'], species_pop, STAGNATION_LIMIT)
                species_count = len(species_pop)

                # terminal log
                utils.neat_log_terminal(
                    state['gen'], stage, max_score, avg_score, best_ever, plateau_counter,
                    gs, ss, species_count, cull_stats, scores, sim_time, breed_time,
                    elapsed_fmt, pop_size,
                    val_score=val_score if stage == 2 else None,
                    **stage_kw,
                )

                # accumulate into 50-gen buffer
                utils.neat_accumulate_buf(
                    log_buf, max_score, avg_score, gs, scores, sim_time, breed_time,
                    cull_stats, species_count, pop_size, ss,
                    stage=stage,
                    completions=completions if stage >= 1 else None,
                    avg_completions=avg_completions if stage >= 1 else None,
                    wp_stats=wp_stats if stage == 2 else None,
                )

                # discord log on interval
                if (state['gen'] % LOG_INTERVAL == 0 or first) and logging:
                    print('logging...')
                    msg = utils.neat_log_discord(
                        NAME, state['gen'], stage, log_buf, best_ever, plateau_counter,
                        pop_size, elapsed_fmt, species_count, ss,
                        val_score=val_score if stage == 2 else None,
                        difficulty=state.get('difficulty'),
                        limit=limit,
                        wp_stats=wp_stats if stage == 2 else None,
                        pool_gen=state.get('pool_gen', 0) if stage == 2 else None,
                        pool_refresh=POOL_REFRESH_GENS if stage == 2 else None,
                        avg_leg_dist=avg_leg_dist if stage == 2 else None,
                        num_wp=NUM_WAYPOINTS if stage == 2 else None,
                    )
                    discord_logger.log(msg)
                    first = False
                    log_buf = utils.neat_log_buf()

                # adjust species thresholds
                diversity_ratio = len(species_pop)/config['population']
                avg_target = (spec_target_min+spec_target_max)/2
                if diversity_ratio < avg_target:
                    diff = abs(diversity_ratio - avg_target)
                    state["threshold"] *= max(1 - (diff * 0.5), 0.1)
                elif diversity_ratio > avg_target:
                    diff = abs(diversity_ratio - avg_target)
                    state["threshold"] *= 1 + (diff * 0.5)

            # adjust unified difficulty
            if stage == 1:
                assert isinstance(completions, list)
                target_rate = 0.1
                rate = avg_completions / config['population']
                error = rate - target_rate
                if error > 0.02:
                    state['difficulty'] *= error + 1
                    state['difficulty'] = max(state['difficulty'], 10)

            # progress hover stage
            if stage == 0:
                # finish if getting 90% of final target score
                if limit >= 30 and max_score/target_score > .95:
                    stage = 1
                    state['stage'] = 1
                    limit = 7
                    state['historical_score'] = []
                    best_ever = 0
                    plateau_counter = 0
                    state['species'] = []
                    first = True
                    utils.save(state)
                elif max_score / target_score > .9:
                    limit = min(limit + 5, 30)

            # progress stage 1 → stage 2
            if stage == 1 and state['difficulty'] >= 40:
                stage = 2
                state['stage'] = 2
                limit = 15
                state['historical_score'] = []
                best_ever = 0
                plateau_counter = 0
                state['species'] = []
                first = True
                utils.save(state)

            # save progress every 500 gens
            if state['gen'] % 100 == 0:
                utils.save(state)

    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f'CRASH: {e}\n{tb}')
        if logging:
            discord_logger.log(f'{NAME}>> CRASH: {type(e).__name__}: {e}\n```{tb[-1500:]}```')
            discord_logger.close()
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        print('---------------------------')
        if logging:
            print('closing logger...')
            discord_logger.log(f'{NAME}>> TRAINING TERM')
            discord_logger.close()
        utils.save(state)
