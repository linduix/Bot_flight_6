from drone_prototype import Ai_Drone
from scoring_prototype import hover_scorer_headless
from prototype_stage1 import stage1, pick_direction, adjust_dir_difficulty, format_dir_rates, make_dir_stats, DIR_NAMES, stage1_vmax_test
from mutation_prototype import Innovations, add_connection
from genome_prototype import Genome, NodeType
from breeding_prototype import breed, STAGNATION_CHANCES
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
    "population": 500,
    "width": 800,
    "height": 600,
    "meters_to_pixels": 15
}

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
    # load_dotenv()
    # logging = os.environ['LOGGING'] == 'ON'
    logging = False

    # webhook test
    if logging:
        NAME = os.environ['NAME']
        WEBHOOK = os.environ["DISCORD_WEBHOOK"]
        discord_logger = utils.DiscordLogger(WEBHOOK, interval=5)
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

        # ── 50-gen stats buffer for discord ──
        LOG_INTERVAL = 50
        SCORE_BINS = [0, 10, 50, 100, 200, 400, 600, 1000]
        log_buf = {
            'max_scores': [],
            'avg_scores': [],
            'comp_counts': [],      # number of completions each gen
            'comp_times': [],       # individual completion times (flat)
            'dir_counts': {},       # direction -> number of gens tested
            'dir_comps': {},        # direction -> total completions
            'stagnant_killed': 0,
            'killed_genomes': 0,
            'connections': [],      # avg connections each gen
            'nodes': [],            # avg nodes each gen
            'gen_times': [],        # seconds per gen
            'score_hist': np.zeros(len(SCORE_BINS) - 1, dtype=int),  # cumulative histogram
            'completer_conn': [],   # avg enabled connections for completers each gen
            'non_completer_conn': [],  # avg enabled connections for non-completers each gen
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

                # Directional balance — pick direction, get per-direction difficulty
                state.setdefault('dir_stats', make_dir_stats(state.get('difficulty', 15)))
                dir_name, dir_theta, adj_diff = pick_direction(state['dir_stats'])

                # # Occasionally easier difficulty so it doesn't forget earlier training
                # if np.random.rand() < .15:
                #     adj_diff *= np.random.rand() * 0.7
                #     adj_diff = max(adj_diff, 10)

                return_code, scores, completions, completed = stage1_vmax_test(
                    drones,
                    config["width"],
                    config["height"],
                    config["meters_to_pixels"],
                    limit=limit,
                    diff=adj_diff,
                    theta=dir_theta,
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
            best_ever = max(best_ever, max_score)
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
                species_info += f" | stagnant killed: {cull_stats['stagnant_killed']} ({cull_stats['killed_genomes']} genomes removed)"

            # log training stats to terminal
            delta_sign = "+" if score_delta >= 0 else ""
            print(f"── S{stage} Gen {state['gen']} {'─' * 50}")
            print(f"  score     max: {max_score:.2f} | avg: {avg_score:.2f} | rolling: {rolling_average:.2f} | best ever: {best_ever:.2f} | Δ: {delta_sign}{score_delta:.1f}")
            if stage == 0:
                pct = max_score / target_score * 100 if target_score > 0 else 0
                print(f"  progress  target: {target_score:.0f} ({pct:.1f}%) | improvement: {improvement:.1f} | limit: {limit}s")
            else:
                assert isinstance(completions, list)
                c_time = np.average(completions) if completions else float("nan")
                comp_pct = len(completions) / pop_size * 100
                print(f"  progress  complete: {len(completions)}/{pop_size} ({comp_pct:.1f}%) | avg c_time: {c_time:.2f}s | difficulty: {adj_diff:.2f}m | improvement: {improvement:.1f} | limit: {limit}")
                print(f"  direction {dir_name} | diffs: {format_dir_rates(state['dir_stats'])}")
            print(f"  species   {species_info} | target species: {config["population"] * spec_target_min: .0f} -{config["population"] * spec_target_max: .0f}")
            print(f"  genome    avg connections: {average_connections:.1f} | avg nodes: {average_nodes:.1f} | disabled: {disabled_ratio:.2%} | pop: {pop_size}")
            print(f"  ratios    elite: {elite_ratio:.2f} | density: {density_ratio:.2f} | spec_den: {species_density:.3f} | stagnant: {stagnant_ratio:.2f} | Δ/gen: {delta_sign}{score_delta:.1f}")
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
            # completer vs non-completer complexity
            if stage == 1:
                assert isinstance(completions, list)
                log_buf['comp_counts'].append(len(completions))
                log_buf['comp_times'].extend(completions)
                log_buf['dir_counts'][dir_name] = log_buf['dir_counts'].get(dir_name, 0) + 1
                log_buf['dir_comps'][dir_name] = log_buf['dir_comps'].get(dir_name, 0) + len(completions)
                non_completed = [i for i in range(len(drones)) if i not in completed]
                if completed:
                    log_buf['completer_conn'].append(np.mean([len([c for c in drones[i].brain.genome.connections if c.enabled]) for i in completed]))
                if non_completed:
                    log_buf['non_completer_conn'].append(np.mean([len([c for c in drones[i].brain.genome.connections if c.enabled]) for i in non_completed]))

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
                    f"Score    peak: {buf_max:.0f}  low: {buf_min:.0f}  avg_best: {buf_avg_max:.0f}  avg_pop: {buf_avg_avg:.1f}",
                    f"         rolling: {rolling_average:.2f}  best_ever: {best_ever:.2f}",
                ]
                if stage == 0:
                    lines.append(f"Progress target: {target_score:.0f} ({pct:.1f}%)  improvement: {improvement:.1f}  limit: {limit}s")
                else:
                    # completion stats over window
                    avg_comp = np.average(log_buf['comp_counts']) if log_buf['comp_counts'] else 0
                    max_comp = max(log_buf['comp_counts']) if log_buf['comp_counts'] else 0
                    avg_ct = np.average(log_buf['comp_times']) if log_buf['comp_times'] else float('nan')
                    comp_rate = avg_comp / pop_size * 100

                    lines.append(f"Complet  avg: {avg_comp:.0f}/{pop_size} ({comp_rate:.1f}%)  peak: {max_comp}  avg_time: {avg_ct:.2f}s")

                    # per-direction summary
                    dir_parts = []
                    for d in DIR_NAMES:
                        cnt = log_buf['dir_counts'].get(d, 0)
                        if cnt > 0:
                            avg_c = log_buf['dir_comps'][d] / cnt
                            dir_parts.append(f"{d}:{cnt}x")
                    lines.append(f"Dirs     {' '.join(dir_parts)}")
                    lines.append(f"Diffs    {format_dir_rates(state['dir_stats'])}")

                # species & stagnation over window
                stag_info = f"now: {len(species_pop)}"
                if log_buf['stagnant_killed'] > 0:
                    stag_info += f"  stag_killed: {log_buf['stagnant_killed']} ({log_buf['killed_genomes']} genomes)"
                lines.append(f"Species  {stag_info}")

                # score distribution
                hist = log_buf['score_hist']
                bin_labels = [f"{SCORE_BINS[i]}-{SCORE_BINS[i+1]}" for i in range(len(SCORE_BINS)-1)]
                dist_parts = [f"{lbl}: {cnt}" for lbl, cnt in zip(bin_labels, hist)]
                lines.append(f"ScoreDis {' | '.join(dist_parts)}")

                # completer vs non-completer complexity
                if stage == 1:
                    avg_cc = np.mean(log_buf['completer_conn']) if log_buf['completer_conn'] else 0
                    avg_nc = np.mean(log_buf['non_completer_conn']) if log_buf['non_completer_conn'] else 0
                    lines.append(f"Complex  completers: {avg_cc:.1f}  non-completers: {avg_nc:.1f}")

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
                    'dir_counts': {},
                    'dir_comps': {},
                    'stagnant_killed': 0,
                    'killed_genomes': 0,
                    'connections': [],
                    'nodes': [],
                    'gen_times': [],
                    'score_hist': np.zeros(len(SCORE_BINS) - 1, dtype=int),
                    'completer_conn': [],
                    'non_completer_conn': [],
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

            # adjust per-direction difficulty
            if stage == 1:
                assert isinstance(completions, list)
                adjust_dir_difficulty(state['dir_stats'], dir_name, len(completions), config['population'])

            # progress hover stage
            if stage == 0:
                # finish if getting 90% of final target score
                if limit >= 30 and max_score/target_score > .95:
                    stage = 1
                    state['stage'] = 1
                    limit = 7
                    state['historical_score'] = []
                    best_ever = 0
                    state['species'] = []
                    first = True
                    utils.save(state)
                elif max_score / target_score > .9:
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
