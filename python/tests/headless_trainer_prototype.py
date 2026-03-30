import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from scoring_prototype import hover_scorer_headless
from prototype_stage1 import stage1_vmax_test
from mutation_prototype import Innovations, add_connection
from genome_prototype import Genome, NodeType
from breeding_prototype import breed, STAGNATION_CHANCES
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

config = {
    "population": 500,
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
        for s in state.get('species', []):
            if not hasattr(s, 'best_history'):
                s.best_history = []
            if not hasattr(s, 'chances'):
                s.chances = STAGNATION_CHANCES
            if not hasattr(s, 'age'):
                s.age = len(s.best_history)  # best guess from existing data
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
        if stage == 0:
            limit = 5
        else:
            limit = 7
        done = False
        profile = False
        first = True
        best_ever = max(state['historical_score']) if state.get('historical_score') else 0
        plateau_counter = 0  # gens since global best was improved
        training_start = time.time()

        # ── 50-gen stats buffer for discord ──
        LOG_INTERVAL = 50
        SCORE_BINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        log_buf = {
            'max_scores': [],
            'avg_scores': [],
            'comp_counts': [],      # number of completions each gen
            'comp_times': [],       # individual completion times (flat)
            'stagnant_killed': 0,
            'killed_genomes': 0,
            'connections': [],      # avg connections each gen
            'nodes': [],            # avg nodes each gen
            'gen_times': [],        # seconds per gen
            'score_hist': np.zeros(len(SCORE_BINS) - 1, dtype=int),  # cumulative histogram
            'elite_ratios': [],     # best/mean fitness each gen
            'density_ratios': [],   # mean_nodes/mean_connections each gen
            'species_densities': [], # species_count/pop_size each gen
            'stagnant_ratios': [],  # stagnant_species/total_species each gen
            'disabled_ratios': [],  # disabled_genes/total_genes each gen
            'score_deltas': [],     # Δ best_fitness each gen
        }

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
            else:
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

            # get average connections, nodes, and disabled-gene ratio
            connections = []
            nodes_counts = []
            disabled_counts = []
            total_gene_counts = []
            for g in state['current_gen']:
                enabled_sum = 0
                disabled_sum = 0
                for c in g.connections:
                    if c.enabled:
                        enabled_sum += 1
                    else:
                        disabled_sum += 1
                connections.append(enabled_sum)
                nodes_counts.append(len(g.nodes))
                disabled_counts.append(disabled_sum)
                total_gene_counts.append(enabled_sum + disabled_sum)
            average_connections = np.average(connections)
            average_nodes = np.average(nodes_counts)
            avg_total_genes = np.average(total_gene_counts)
            avg_disabled = np.average(disabled_counts)
            disabled_ratio = avg_disabled / avg_total_genes if avg_total_genes > 0 else 0.0

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
                stagnant_count = sum(1 for s in spec if s.stagnation > 0)
            else:
                species_pop = []
                deaths = 0
                cull_stats = {'stagnant_killed': 0, 'killed_genomes': 0}
                stagnant_count = 0

            # breed profiling
            if profile:
                pr.disable() # type: ignore
                pr.dump_stats("breed0.prof")  # type: ignore
                print("wrote breed0.prof")
                profile = False

            # compute stats for logging
            if max_score > best_ever:
                best_ever = max_score
                plateau_counter = 0
                utils.save(state, "prototype_best.pkl")
            else:
                plateau_counter += 1
            avg_score = np.average(scores)
            pop_size = len(scores)
            prev_max = state['historical_score'][-2] if len(state['historical_score']) >= 2 else max_score
            score_delta = max_score - prev_max

            # ratios
            elite_ratio = max_score / avg_score if avg_score > 0 else float('nan')
            density_ratio = average_nodes / average_connections if average_connections > 0 else float('nan')
            species_density = len(species_pop) / pop_size if pop_size > 0 else 0.0
            stagnant_ratio = stagnant_count / len(species_pop) if species_pop else 0.0
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
                species_info += f" | stagnant killed: {cull_stats['stagnant_killed']}"

            # per-species summary
            if species_pop:
                largest_species = max(len(sp) for sp in species_pop)
                top_species_fit = max(s.best_history[-1] for s in state['species'] if s.best_history) if state['species'] else 0
                oldest_species = max(s.age for s in state['species']) if state['species'] else 0
                most_stagnant = max(s.stagnation for s in state['species']) if state['species'] else 0
            else:
                largest_species = top_species_fit = oldest_species = most_stagnant = 0

            # log training stats to terminal
            delta_sign = "+" if score_delta >= 0 else ""
            print(f"── S{stage} Gen {state['gen']} {'─' * 50}")
            print(f"  score     max: {max_score:.4f} | avg: {avg_score:.4f} | best ever: {best_ever:.4f} | Δ: {delta_sign}{score_delta:.4f} | plateau: {plateau_counter}")
            if stage == 0:
                pct = max_score / target_score * 100 if target_score > 0 else 0
                print(f"  progress  target: {target_score:.0f} ({pct:.1f}%) | limit: {limit}s")
            else:
                assert isinstance(completions, list)
                c_time = np.average(completions) if completions else float("nan")
                comp_pct = avg_completions / pop_size * 100
                print(f"  progress  complete: {avg_completions:.1f}/{pop_size} ({comp_pct:.1f}%) | avg c_time: {c_time:.2f}s | difficulty: {state['difficulty']:.2f}m | limit: {limit}")
            pop = config['population']
            print(f"  species   {species_info} | largest: {largest_species} | top_fit: {top_species_fit:.1f} | oldest: {oldest_species} gens | most_stagnant: {most_stagnant} gens | target: {pop * spec_target_min:.0f} - {pop * spec_target_max:.0f}")
            print(f"  genome    avg connections: {average_connections:.1f} | avg nodes: {average_nodes:.1f} | disabled: {disabled_ratio:.2%} | pop: {pop_size}")
            print(f"  ratios    elite: {elite_ratio:.2f} | density: {density_ratio:.2f} | spec_den: {species_density:.3f} | stagnant: {stagnant_ratio:.2f}")
            print(f"  timing    gen: {elapsed:.2f}s | rate: {gen_rate:.1f} gen/min | elapsed: {elapsed_fmt}")

            # ── accumulate stats into 50-gen buffer ──
            log_buf['max_scores'].append(max_score)
            log_buf['avg_scores'].append(avg_score)
            log_buf['connections'].append(average_connections)
            log_buf['nodes'].append(average_nodes)
            log_buf['gen_times'].append(elapsed)
            log_buf['elite_ratios'].append(elite_ratio)
            log_buf['density_ratios'].append(density_ratio)
            log_buf['species_densities'].append(species_density)
            log_buf['stagnant_ratios'].append(stagnant_ratio)
            log_buf['disabled_ratios'].append(disabled_ratio)
            log_buf['score_deltas'].append(score_delta)
            log_buf['stagnant_killed'] += cull_stats['stagnant_killed']
            log_buf['killed_genomes'] += cull_stats['killed_genomes']
            # score distribution histogram
            log_buf['score_hist'] += np.histogram(scores, bins=SCORE_BINS)[0]
            if stage == 1:
                assert isinstance(completions, list)
                log_buf['comp_counts'].append(avg_completions)
                log_buf['comp_times'].extend(completions)

            # log to discord
            if (state['gen'] % LOG_INTERVAL == 0 or first) and logging:
                print('logging...')
                n = len(log_buf['max_scores'])
                buf_max = max(log_buf['max_scores'])
                buf_min = min(log_buf['max_scores'])
                buf_avg_max = np.average(log_buf['max_scores'])
                buf_avg_avg = np.average(log_buf['avg_scores'])

                lines = [
                    f"**{NAME} | S{stage} Gen {state['gen']}** ({n} gens)",
                    f"```",
                    f"Score    peak: {buf_max:.4f}  low: {buf_min:.4f}  avg_best: {buf_avg_max:.4f}  avg_pop: {buf_avg_avg:.4f}",
                    f"         best_ever: {best_ever:.4f}  plateau: {plateau_counter}",
                ]
                if stage == 0:
                    lines.append(f"Progress target: {target_score:.0f} ({pct:.1f}%)  improvement: {improvement:.1f}  limit: {limit}s")
                else:
                    # completion stats over window (avg_completions = per-direction avg)
                    avg_comp = np.average(log_buf['comp_counts']) if log_buf['comp_counts'] else 0
                    max_comp = max(log_buf['comp_counts']) if log_buf['comp_counts'] else 0
                    avg_ct = np.average(log_buf['comp_times']) if log_buf['comp_times'] else float('nan')
                    comp_rate = avg_comp / pop_size * 100

                    lines.append(f"Complet  avg: {avg_comp:.1f}/{pop_size} ({comp_rate:.1f}%)  peak: {max_comp:.1f}  avg_time: {avg_ct:.2f}s  diff: {state['difficulty']:.1f}m")

                # species & stagnation over window
                stag_info = f"now: {len(species_pop)}"
                if log_buf['stagnant_killed'] > 0:
                    stag_info += f"  stag_killed: {log_buf['stagnant_killed']} ({log_buf['killed_genomes']} genomes)"
                lines.append(f"Species  {stag_info}")
                lines.append(f"         largest: {largest_species}  top_fit: {top_species_fit:.1f}  oldest: {oldest_species}  most_stag: {most_stagnant}")

                # score distribution
                hist = log_buf['score_hist']
                bin_labels = [f"{SCORE_BINS[i]}-{SCORE_BINS[i+1]}" for i in range(len(SCORE_BINS)-1)]
                dist_parts = [f"{lbl}: {cnt}" for lbl, cnt in zip(bin_labels, hist)]
                lines.append(f"ScoreDis {' | '.join(dist_parts)}")

                avg_conn = np.average(log_buf['connections'])
                conn_delta = log_buf['connections'][-1] - log_buf['connections'][0] if n > 1 else 0
                avg_nodes_buf = np.average(log_buf['nodes'])
                avg_gt = np.average(log_buf['gen_times'])

                # ratio aggregates over window
                buf_elite    = np.nanmean(log_buf['elite_ratios'])
                buf_density  = np.nanmean(log_buf['density_ratios'])
                buf_spec_den = np.mean(log_buf['species_densities'])
                buf_stagnant = np.mean(log_buf['stagnant_ratios'])
                buf_disabled = np.mean(log_buf['disabled_ratios'])
                buf_delta_gen = np.mean(log_buf['score_deltas'])

                lines += [
                    f"Genome   avg_conn: {avg_conn:.1f} (Δ{conn_delta:+.1f})  avg_nodes: {avg_nodes_buf:.1f}  pop: {pop_size}",
                    f"Ratios   elite: {buf_elite:.2f}  density: {buf_density:.2f}  spec_den: {buf_spec_den:.3f}  stagnant: {buf_stagnant:.2f}  disabled: {buf_disabled:.2%}  Δ/gen: {buf_delta_gen:+.1f}",
                    f"Timing   avg: {avg_gt:.2f}s/gen  rate: {60/avg_gt:.1f}/min  elapsed: {elapsed_fmt}",
                    f"```",
                ]

                discord_logger.log("\n".join(lines))
                first = False

                # reset buffer
                log_buf = {
                    'max_scores': [],
                    'avg_scores': [],
                    'comp_counts': [],
                    'comp_times': [],
                    'stagnant_killed': 0,
                    'killed_genomes': 0,
                    'connections': [],
                    'nodes': [],
                    'gen_times': [],
                    'score_hist': np.zeros(len(SCORE_BINS) - 1, dtype=int),
                    'elite_ratios': [],
                    'density_ratios': [],
                    'species_densities': [],
                    'stagnant_ratios': [],
                    'disabled_ratios': [],
                    'score_deltas': [],
                }

            # adjust species thresholds
            diversity_ratio = len(species_pop)/config['population']
            if diversity_ratio < spec_target_min:
                diff = abs(diversity_ratio - (spec_target_min+spec_target_max)/2)
                state["threshold"] *= max(1 - (diff * 0.5), 0.1)
            elif diversity_ratio > spec_target_max:
                diff = abs(diversity_ratio - (spec_target_min+spec_target_max)/2)
                state["threshold"] *= 1 + (diff * 0.5)

            # adjust unified difficulty
            if stage == 1:
                assert isinstance(completions, list)
                target_rate = 0.1
                rate = avg_completions / config['population']
                error = rate - target_rate
                if abs(error) > 0.02:
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

            # save progress every 500 gens
            if state['gen'] % 100 == 0:
                utils.save(state)

    except KeyboardInterrupt:
        print('Keyboard Interrupt')
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
