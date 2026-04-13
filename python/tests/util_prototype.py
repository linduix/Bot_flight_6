from pathlib import Path
from genome_prototype import NodeType
import matplotlib.pyplot as plt
import multiprocessing as mp
import networkx as nx
import numpy as np
import dataclasses
import threading
import requests
import pickle
import queue
import time

checkpoint_dir = Path(__file__).parent.parent.parent / "data" / "checkpoints"
save_path = checkpoint_dir / "prototype_save.pkl"
def save(state: dict, filename: str = "prototype_save.pkl"):
    path = checkpoint_dir / filename
    try:
        with open(path, 'wb') as f:
            print(f'saving {filename}')
            pickle.dump(state, f)
    except Exception as e:
        import traceback
        print(f'ERROR saving {filename}: {type(e).__name__}: {e}')
        print(f'  path: {path}')
        print(f'  path exists: {path.exists()}')
        print(f'  dir exists: {path.parent.exists()}')
        print(f'  state keys: {list(state.keys())}')
        traceback.print_exc()

def load(path=None):
    p = path or save_path
    with open(p, 'rb') as f:
        print(f'loading {p}')
        state = pickle.load(f)
    return state

# ─── Shared helpers ───────────────────────────────────────────────────────────

def compute_genome_stats(genomes):
    """Aggregate genome topology & mutation stats for a population."""
    enabled = []
    disabled = []
    node_counts = []
    for g in genomes:
        e = sum(1 for c in g.connections if c.enabled)
        d = sum(1 for c in g.connections if not c.enabled)
        enabled.append(e)
        disabled.append(d)
        node_counts.append(len(g.nodes))
    avg_conn = np.mean(enabled)
    avg_nodes = np.mean(node_counts)
    avg_total = np.mean([e + d for e, d in zip(enabled, disabled)])
    avg_dis = np.mean(disabled)
    mut_powers = [g.mutation_power for g in genomes]
    return {
        'avg_connections': float(avg_conn),
        'avg_nodes': float(avg_nodes),
        'disabled_ratio': float(avg_dis / avg_total) if avg_total > 0 else 0.0,
        'mut_power_mean': float(np.mean(mut_powers)),
        'mut_power_std': float(np.std(mut_powers)),
    }


def compute_species_stats(species, species_pop, stagnation_limit):
    """Compute species-level stats for NEAT logging."""
    if species_pop:
        largest = max(len(sp) for sp in species_pop)
        top_sp = max((s for s in species if s.best_history),
                     key=lambda s: s.best_history[-1], default=None)
        top_fit = top_sp.best_history[-1] if top_sp else 0
        top_id = top_sp.id if top_sp else 0
        oldest_id = max(species, key=lambda s: s.age).id if species else 0
        stagnant_count = sum(1 for s in species if s.stagnation > stagnation_limit // 2)
        sp_muts = [np.mean([g.mutation_power for g in sp]) for sp in species_pop if sp]
        sp_mut_min = min(sp_muts) if sp_muts else 0.0
        sp_mut_max = max(sp_muts) if sp_muts else 0.0
    else:
        largest = top_fit = top_id = oldest_id = stagnant_count = 0
        sp_mut_min = sp_mut_max = 0.0
    return {
        'largest': largest,
        'top_fit': top_fit,
        'top_id': top_id,
        'oldest_id': oldest_id,
        'stagnant_count': stagnant_count,
        'sp_mut_min': sp_mut_min,
        'sp_mut_max': sp_mut_max,
    }


def format_elapsed(seconds):
    """Format seconds into a human-readable elapsed string."""
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{int(hrs)}h {int(mins)}m"
    return f"{int(mins)}m {int(secs)}s"


# ─── NEAT logging ─────────────────────────────────────────────────────────────

def neat_log_buf():
    """Empty log buffer for NEAT trainer (50-gen window)."""
    return {
        'max_scores': [],
        'avg_scores': [],
        'comp_counts': [],
        'comp_times': [],
        'stagnant_killed': 0,
        'killed_genomes': 0,
        'connections': [],
        'nodes': [],
        'sim_times': [],
        'breed_times': [],
        'all_scores': [],
        'elite_ratios': [],
        'density_ratios': [],
        'species_densities': [],
        'high_stag_ratios': [],
        'disabled_ratios': [],
        'mut_power_means': [],
        'mut_power_stds': [],
    }


def neat_accumulate_buf(log_buf, max_score, avg_score, gs, scores, sim_time, breed_time,
                        cull_stats, species_count, pop_size, ss,
                        stage=0, completions=None, avg_completions=None, wp_stats=None):
    """Append one generation's stats into the rolling NEAT buffer."""
    elite_ratio = max_score / avg_score if avg_score > 0 else float('nan')
    density_ratio = gs['avg_nodes'] / gs['avg_connections'] if gs['avg_connections'] > 0 else float('nan')
    species_density = species_count / pop_size if pop_size > 0 else 0.0
    high_stag_ratio = ss['stagnant_count'] / species_count if species_count > 0 else 0.0

    log_buf['max_scores'].append(max_score)
    log_buf['avg_scores'].append(avg_score)
    log_buf['connections'].append(gs['avg_connections'])
    log_buf['nodes'].append(gs['avg_nodes'])
    log_buf['sim_times'].append(sim_time)
    log_buf['breed_times'].append(breed_time)
    log_buf['all_scores'].extend(scores.tolist() if hasattr(scores, 'tolist') else list(scores))
    log_buf['elite_ratios'].append(elite_ratio)
    log_buf['density_ratios'].append(density_ratio)
    log_buf['species_densities'].append(species_density)
    log_buf['high_stag_ratios'].append(high_stag_ratio)
    log_buf['disabled_ratios'].append(gs['disabled_ratio'])
    log_buf['mut_power_means'].append(gs['mut_power_mean'])
    log_buf['mut_power_stds'].append(gs['mut_power_std'])
    log_buf['stagnant_killed'] += cull_stats['stagnant_killed']
    log_buf['killed_genomes'] += cull_stats['killed_genomes']
    if stage >= 1 and completions is not None:
        log_buf['comp_counts'].append(avg_completions)
        log_buf['comp_times'].extend(completions)
    if stage == 2 and wp_stats is not None:
        log_buf.setdefault('wp_mins', []).append(wp_stats['min'])
        log_buf.setdefault('wp_q1s', []).append(wp_stats['q1'])
        log_buf.setdefault('wp_q3s', []).append(wp_stats['q3'])
        log_buf.setdefault('wp_maxs', []).append(wp_stats['max'])


def neat_log_terminal(gen, stage, max_score, avg_score, best_ever, plateau_counter,
                      gs, ss, species_count, cull_stats, scores,
                      sim_time, breed_time, elapsed_fmt, pop_size,
                      val_score=None, completions=None, avg_completions=None,
                      difficulty=None, limit=None, target_score=None,
                      wp_stats=None, pool_gen=None, pool_refresh=None,
                      avg_leg_dist=None, num_wp=None):
    """Print per-gen terminal stats for NEAT trainer."""
    print(f"── S{stage} Gen {gen} {'─' * 50}")

    score_line = f"  score      max: {max_score:.4f} | avg: {avg_score:.4f} | best ever: {best_ever:.4f} | plateau: {plateau_counter}"
    if val_score is not None:
        score_line += f" | val: {val_score:.4f}"
    print(score_line)

    if stage == 0 and target_score is not None:
        pct = max_score / target_score * 100 if target_score > 0 else 0
        print(f"  progress   target: {target_score:.0f} ({pct:.1f}%) | limit: {limit}s")
    elif stage == 1 and completions is not None:
        c_time = np.average(completions) if completions else float("nan")
        comp_pct = avg_completions / pop_size * 100
        print(f"  progress   complete: {avg_completions:.1f}/{pop_size} ({comp_pct:.1f}%) | avg c_time: {c_time:.2f}s | difficulty: {difficulty:.2f}m | limit: {limit}")
    elif stage == 2 and completions is not None:
        c_time = np.average(completions) if completions else float("nan")
        comp_pct = avg_completions / pop_size * 100
        pool_fresh = " [NEW POOL]" if pool_gen == 1 else ""
        print(f"  progress   chains: {avg_completions:.1f}/{pop_size} ({comp_pct:.1f}%) | avg c_time: {c_time:.2f}s | limit: {limit}")
        if wp_stats:
            print(f"  waypoints  min: {wp_stats['min']:.1f} | Q1: {wp_stats['q1']:.1f} | Q3: {wp_stats['q3']:.1f} | max: {wp_stats['max']:.1f}")
        print(f"  pool       gen {pool_gen}/{pool_refresh} | avg_leg: {avg_leg_dist:.1f}m | avg_total: {avg_leg_dist * num_wp:.1f}m{pool_fresh}")

    sp_info = f"count: {species_count}"
    if cull_stats['stagnant_killed'] > 0:
        sp_info += f" | stagnant killed: {cull_stats['stagnant_killed']}"
    print(f"  species    {sp_info} | largest: {ss['largest']} | top_fit: {ss['top_fit']:.1f} (sp{ss['top_id']}) | oldest: sp{ss['oldest_id']}")

    print(f"  genome     avg connections: {gs['avg_connections']:.1f} | avg nodes: {gs['avg_nodes']:.1f} | disabled: {gs['disabled_ratio']:.2%} | pop: {pop_size}")

    elite_ratio = max_score / avg_score if avg_score > 0 else float('nan')
    density_ratio = gs['avg_nodes'] / gs['avg_connections'] if gs['avg_connections'] > 0 else float('nan')
    species_density = species_count / pop_size if pop_size > 0 else 0.0
    high_stag_ratio = ss['stagnant_count'] / species_count if species_count > 0 else 0.0
    print(f"  ratios     elite: {elite_ratio:.2f} | density: {density_ratio:.2f} | spec_den: {species_density:.3f} | high_stag: {high_stag_ratio:.2f}")

    mp_pct = gs['mut_power_std'] * 100 / gs['mut_power_mean'] if gs['mut_power_mean'] > 0 else 0
    print(f"  mut_power  mean: {gs['mut_power_mean']:.3f} | std: {mp_pct:.2f}% | species range: [{ss['sp_mut_min']:.2f}, {ss['sp_mut_max']:.2f}]")

    p10, p25, p50, p75, p90 = np.percentile(scores, [10, 25, 50, 75, 90])
    print(f"  scores     p10: {p10:.3f} | p25: {p25:.3f} | p50: {p50:.3f} | p75: {p75:.3f} | p90: {p90:.3f}")

    gen_rate = 60 / (sim_time + breed_time) if (sim_time + breed_time) > 0 else 0
    print(f"  timing     sim: {sim_time:.2f}s | breed: {breed_time:.2f}s | rate: {gen_rate:.1f} gen/min | elapsed: {elapsed_fmt}")


def neat_log_discord(name, gen, stage, log_buf, best_ever, plateau_counter,
                     pop_size, elapsed_fmt, species_count, ss,
                     val_score=None, difficulty=None, limit=None,
                     wp_stats=None, pool_gen=None, pool_refresh=None,
                     avg_leg_dist=None, num_wp=None):
    """Build discord message string for NEAT trainer 50-gen window."""
    n = len(log_buf['max_scores'])
    buf_max = max(log_buf['max_scores'])
    buf_min = min(log_buf['max_scores'])
    buf_avg_max = np.average(log_buf['max_scores'])
    buf_avg_avg = np.average(log_buf['avg_scores'])

    lines = [
        f"**{name} | S{stage} Gen {gen}** ({n} gens)",
        f"```",
        f"Score    peak: {buf_max:.4f}  low: {buf_min:.4f}  avg_best: {buf_avg_max:.4f}  avg_pop: {buf_avg_avg:.4f}",
        f"         best_ever: {best_ever:.4f}  plateau: {plateau_counter}" + (f"  val: {val_score:.4f}" if val_score is not None else ""),
    ]

    if stage == 0:
        lines.append(f"Progress limit: {limit}s")
    elif stage == 1 and log_buf['comp_counts']:
        avg_comp = np.average(log_buf['comp_counts'])
        max_comp = max(log_buf['comp_counts'])
        avg_ct = np.average(log_buf['comp_times']) if log_buf['comp_times'] else float('nan')
        comp_rate = avg_comp / pop_size * 100
        lines.append(f"Complet  avg: {avg_comp:.1f}/{pop_size} ({comp_rate:.1f}%)  peak: {max_comp:.1f}  avg_time: {avg_ct:.2f}s  diff: {difficulty:.1f}m")
    elif stage == 2 and log_buf['comp_counts']:
        avg_comp = np.average(log_buf['comp_counts'])
        max_comp = max(log_buf['comp_counts'])
        avg_ct = np.average(log_buf['comp_times']) if log_buf['comp_times'] else float('nan')
        comp_rate = avg_comp / pop_size * 100
        lines.append(f"Chains   avg: {avg_comp:.1f}/{pop_size} ({comp_rate:.1f}%)  peak: {max_comp:.1f}  avg_time: {avg_ct:.2f}s")
        if log_buf.get('wp_mins'):
            bwmin = np.mean(log_buf['wp_mins'])
            bwq1 = np.mean(log_buf['wp_q1s'])
            bwq3 = np.mean(log_buf['wp_q3s'])
            bwmax = np.mean(log_buf['wp_maxs'])
            lines.append(f"Waypnts  min: {bwmin:.1f}  Q1: {bwq1:.1f}  Q3: {bwq3:.1f}  max: {bwmax:.1f}")
        if pool_gen is not None:
            lines.append(f"Pool     gen {pool_gen}/{pool_refresh}  avg_leg: {avg_leg_dist:.1f}m  avg_total: {avg_leg_dist * num_wp:.1f}m")

    # species
    stag_info = f"now: {species_count}"
    if log_buf['stagnant_killed'] > 0:
        stag_info += f"  stag_killed: {log_buf['stagnant_killed']} ({log_buf['killed_genomes']} genomes)"
    lines.append(f"Species  {stag_info}")
    lines.append(f"         largest: {ss['largest']}  top_fit: {ss['top_fit']:.1f} (sp{ss['top_id']})  oldest: sp{ss['oldest_id']}")

    avg_conn = np.average(log_buf['connections'])
    conn_delta = log_buf['connections'][-1] - log_buf['connections'][0] if n > 1 else 0
    avg_nodes_buf = np.average(log_buf['nodes'])
    avg_sim = np.average(log_buf['sim_times'])
    avg_breed = np.average(log_buf['breed_times'])

    buf_elite = np.nanmean(log_buf['elite_ratios'])
    buf_density = np.nanmean(log_buf['density_ratios'])
    buf_spec_den = np.mean(log_buf['species_densities'])
    buf_stagnant = np.mean(log_buf['high_stag_ratios'])
    buf_disabled = np.mean(log_buf['disabled_ratios'])
    buf_mut_mean = np.mean(log_buf['mut_power_means'])
    buf_mut_std = np.mean(log_buf['mut_power_stds'])

    all_scores_buf = np.array(log_buf['all_scores'])
    bp10, bp25, bp50, bp75, bp90 = np.percentile(all_scores_buf, [10, 25, 50, 75, 90])

    lines += [
        f"ScoreDis p10: {bp10:.3f}  p25: {bp25:.3f}  p50: {bp50:.3f}  p75: {bp75:.3f}  p90: {bp90:.3f}",
        f"Genome   avg_conn: {avg_conn:.1f} (Δ{conn_delta:+.1f})  avg_nodes: {avg_nodes_buf:.1f}  pop: {pop_size}",
        f"MutPower mean: {buf_mut_mean:.3f}  std: {buf_mut_std*100/max(buf_mut_mean,1e-9):.2f}%",
        f"Ratios   elite: {buf_elite:.2f}  density: {buf_density:.2f}  spec_den: {buf_spec_den:.3f}  high_stag: {buf_stagnant:.2f}  disabled: {buf_disabled:.2%}",
        f"Timing   sim: {avg_sim:.2f}s  breed: {avg_breed:.2f}s  rate: {60/(avg_sim+avg_breed):.1f}/min  elapsed: {elapsed_fmt}",
        f"```",
    ]

    return "\n".join(lines)


def viz_process(q: mp.Queue):
    plt.ion()
    while True:
        try:
            genome = q.get(timeout=0.1)
        except:
            plt.pause(0.001)
            continue

        if genome is None:
            break

        # build incoming connection map
        incoming = {n.id: [] for n in genome.nodes}
        for c in genome.connections:
            if c.enabled:
                incoming[c.output].append(c.input)

        # calculate depth of each node
        depth = {}

        # input nodes at depth 0
        for node in genome.nodes:
            if node.node_type == NodeType.INPUT:
                depth[node.id] = 0

        # build topo order for hidden nodes
        in_degree = {n.id: 0 for n in genome.nodes if n.node_type == NodeType.HIDDEN}
        outgoing = {n.id: [] for n in genome.nodes if n.node_type == NodeType.HIDDEN}
        for c in genome.connections:
            if c.enabled and c.output in in_degree and c.input in in_degree:
                in_degree[c.output] += 1
                outgoing[c.input].append(c.output)

        bfs_queue = [n for n in in_degree if in_degree[n] == 0]
        topo_order = []
        while bfs_queue:
            node = bfs_queue.pop(0)
            topo_order.append(node)
            for out in outgoing[node]:
                in_degree[out] -= 1
                if in_degree[out] == 0:
                    bfs_queue.append(out)
        topo_order += [n for n in in_degree if n not in topo_order]

        # assign depth in topo order
        for node_id in topo_order:
            depends_depths = [0]
            for dep_id in incoming[node_id]:
                if dep_id in depth:
                    depends_depths.append(depth[dep_id])
            depth[node_id] = max(depends_depths) + 1

        # output nodes at max depth + 1
        max_depth = max(depth.values()) if depth else 0
        for node in genome.nodes:
            if node.node_type == NodeType.OUTPUT:
                depth[node.id] = max_depth + 1

        G = nx.DiGraph()
        for node in genome.nodes:
            G.add_node(node.id, node_type=node.node_type, layer=depth[node.id])

        for c in genome.connections:
            if c.enabled:
                G.add_edge(c.input, c.output, weight=c.weight)

        pos = nx.multipartite_layout(G, subset_key='layer')

        colors = []
        for node in G.nodes:
            node_type = G.nodes[node]['node_type']
            if node_type == NodeType.INPUT:
                colors.append('steelblue')
            elif node_type == NodeType.OUTPUT:
                colors.append('tomato')
            else:
                colors.append('mediumseagreen')

        edge_colors = ['green' if G[u][v]['weight'] > 0 else 'red' for u, v in G.edges]

        plt.clf()
        nx.draw_networkx(G, pos, node_color=colors, edge_color=edge_colors,
                node_size=500, arrows=True)
        plt.title(f'Genome — {len(genome.nodes)} nodes, {len(genome.connections)} connections')
        plt.pause(0.001)

# ─── Pareto logging ───────────────────────────────────────────────────────────

def pareto_log_buf():
    """Empty log buffer for Pareto trainer (50-gen window)."""
    return {
        'max_scores': [],
        'avg_scores': [],
        'comp_counts': [],
        'comp_times': [],
        'connections': [],
        'nodes': [],
        'sim_times': [],
        'breed_times': [],
        'all_scores': [],
        'elite_ratios': [],
        'disabled_ratios': [],
        'mut_power_means': [],
        'mut_power_stds': [],
        # pareto-specific
        'num_fronts': [],
        'f1_sizes': [],
        'hypervolumes': [],
    }


def pareto_accumulate_buf(log_buf, max_score, avg_score, gs, scores, sim_time, breed_time,
                          pareto_stats, pop_size,
                          stage=0, completions=None, avg_completions=None, wp_stats=None):
    """Append one generation's stats into the rolling Pareto buffer."""
    elite_ratio = max_score / avg_score if avg_score > 0 else float('nan')

    log_buf['max_scores'].append(max_score)
    log_buf['avg_scores'].append(avg_score)
    log_buf['connections'].append(gs['avg_connections'])
    log_buf['nodes'].append(gs['avg_nodes'])
    log_buf['sim_times'].append(sim_time)
    log_buf['breed_times'].append(breed_time)
    log_buf['all_scores'].extend(scores.tolist() if hasattr(scores, 'tolist') else list(scores))
    log_buf['elite_ratios'].append(elite_ratio)
    log_buf['disabled_ratios'].append(gs['disabled_ratio'])
    log_buf['mut_power_means'].append(gs['mut_power_mean'])
    log_buf['mut_power_stds'].append(gs['mut_power_std'])
    # pareto-specific
    log_buf['num_fronts'].append(pareto_stats['num_fronts'])
    log_buf['f1_sizes'].append(pareto_stats['f1_size'])
    log_buf['hypervolumes'].append(pareto_stats['hypervolume'])
    if stage >= 1 and completions is not None:
        log_buf['comp_counts'].append(avg_completions)
        log_buf['comp_times'].extend(completions)
    if stage == 2 and wp_stats is not None:
        log_buf.setdefault('wp_mins', []).append(wp_stats['min'])
        log_buf.setdefault('wp_q1s', []).append(wp_stats['q1'])
        log_buf.setdefault('wp_q3s', []).append(wp_stats['q3'])
        log_buf.setdefault('wp_maxs', []).append(wp_stats['max'])


def pareto_log_terminal(gen, stage, max_score, avg_score, best_ever, plateau_counter,
                        pareto_stats, gs, scores, sim_time, breed_time,
                        elapsed_fmt, pop_size, val_score=None,
                        completions=None, avg_completions=None, difficulty=None, limit=None,
                        target_score=None,
                        wp_stats=None, pool_gen=None, pool_refresh=None, avg_leg_dist=None, num_wp=None):
    """Print per-gen terminal stats for Pareto trainer.

    Accepts gs dict from compute_genome_stats() for genome/mutation stats.
    Legacy callers passing individual params can build gs manually.
    """
    ps = pareto_stats
    print(f"── S{stage} Gen {gen} {'─' * 50}")
    score_line = f"  score      max: {max_score:.4f} | avg: {avg_score:.4f} | best ever: {best_ever:.4f} | plateau: {plateau_counter}"
    if val_score is not None:
        score_line += f" | val: {val_score:.4f}"
    print(score_line)

    if stage == 0 and target_score is not None:
        pct = max_score / target_score * 100 if target_score > 0 else 0
        print(f"  progress   target: {target_score:.0f} ({pct:.1f}%) | limit: {limit}s")
    elif stage == 1 and completions is not None:
        c_time = np.average(completions) if completions else float("nan")
        comp_pct = avg_completions / pop_size * 100
        print(f"  progress   complete: {avg_completions:.1f}/{pop_size} ({comp_pct:.1f}%) | avg c_time: {c_time:.2f}s | difficulty: {difficulty:.2f}m | limit: {limit}")
    elif stage == 2 and completions is not None:
        c_time = np.average(completions) if completions else float("nan")
        comp_pct = avg_completions / pop_size * 100
        pool_fresh = " [NEW POOL]" if pool_gen == 1 else ""
        print(f"  progress   chains: {avg_completions:.1f}/{pop_size} ({comp_pct:.1f}%) | avg c_time: {c_time:.2f}s | limit: {limit}")
        if wp_stats:
            print(f"  waypoints  min: {wp_stats['min']:.1f} | Q1: {wp_stats['q1']:.1f} | Q3: {wp_stats['q3']:.1f} | max: {wp_stats['max']:.1f}")
        print(f"  pool       gen {pool_gen}/{pool_refresh} | avg_leg: {avg_leg_dist:.1f}m | avg_total: {avg_leg_dist * num_wp:.1f}m{pool_fresh}")

    print(f"  pareto     fronts: {ps['num_fronts']} | F1: {ps['f1_size']} | hypervolume: {ps['hypervolume']:.2f}")
    print(f"  genome     avg connections: {gs['avg_connections']:.1f} | avg nodes: {gs['avg_nodes']:.1f} | disabled: {gs['disabled_ratio']:.2%} | pop: {pop_size}")
    mp_pct = gs['mut_power_std'] * 100 / gs['mut_power_mean'] if gs['mut_power_mean'] > 0 else 0
    print(f"  mut_power  mean: {gs['mut_power_mean']:.3f} | std: {mp_pct:.2f}%")
    p10, p25, p50, p75, p90 = np.percentile(scores, [10, 25, 50, 75, 90])
    print(f"  scores     p10: {p10:.3f} | p25: {p25:.3f} | p50: {p50:.3f} | p75: {p75:.3f} | p90: {p90:.3f}")
    gen_rate = 60 / (sim_time + breed_time) if (sim_time + breed_time) > 0 else 0
    print(f"  timing     sim: {sim_time:.2f}s | breed: {breed_time:.2f}s | rate: {gen_rate:.1f} gen/min | elapsed: {elapsed_fmt}")


def pareto_log_discord(name, gen, stage, log_buf, best_ever, plateau_counter,
                       pop_size, elapsed_fmt, val_score=None,
                       difficulty=None, limit=None,
                       wp_stats=None, pool_gen=None, pool_refresh=None,
                       avg_leg_dist=None, num_wp=None):
    """Build discord message string for Pareto trainer 50-gen window."""
    n = len(log_buf['max_scores'])
    buf_max = max(log_buf['max_scores'])
    buf_min = min(log_buf['max_scores'])
    buf_avg_max = np.average(log_buf['max_scores'])
    buf_avg_avg = np.average(log_buf['avg_scores'])

    lines = [
        f"**{name} | S{stage} Gen {gen}** ({n} gens)",
        f"```",
        f"Score    peak: {buf_max:.4f}  low: {buf_min:.4f}  avg_best: {buf_avg_max:.4f}  avg_pop: {buf_avg_avg:.4f}",
        f"         best_ever: {best_ever:.4f}  plateau: {plateau_counter}" + (f"  val: {val_score:.4f}" if val_score is not None else ""),
    ]

    if stage == 0:
        lines.append(f"Progress limit: {limit}s")
    elif stage == 1 and log_buf['comp_counts']:
        avg_comp = np.average(log_buf['comp_counts'])
        max_comp = max(log_buf['comp_counts'])
        avg_ct = np.average(log_buf['comp_times']) if log_buf['comp_times'] else float('nan')
        comp_rate = avg_comp / pop_size * 100
        lines.append(f"Complet  avg: {avg_comp:.1f}/{pop_size} ({comp_rate:.1f}%)  peak: {max_comp:.1f}  avg_time: {avg_ct:.2f}s  diff: {difficulty:.1f}m")
    elif stage == 2 and log_buf['comp_counts']:
        avg_comp = np.average(log_buf['comp_counts'])
        max_comp = max(log_buf['comp_counts'])
        avg_ct = np.average(log_buf['comp_times']) if log_buf['comp_times'] else float('nan')
        comp_rate = avg_comp / pop_size * 100
        lines.append(f"Chains   avg: {avg_comp:.1f}/{pop_size} ({comp_rate:.1f}%)  peak: {max_comp:.1f}  avg_time: {avg_ct:.2f}s")
        if log_buf.get('wp_mins'):
            bwmin = np.mean(log_buf['wp_mins'])
            bwq1 = np.mean(log_buf['wp_q1s'])
            bwq3 = np.mean(log_buf['wp_q3s'])
            bwmax = np.mean(log_buf['wp_maxs'])
            lines.append(f"Waypnts  min: {bwmin:.1f}  Q1: {bwq1:.1f}  Q3: {bwq3:.1f}  max: {bwmax:.1f}")
        if pool_gen is not None:
            lines.append(f"Pool     gen {pool_gen}/{pool_refresh}  avg_leg: {avg_leg_dist:.1f}m  avg_total: {avg_leg_dist * num_wp:.1f}m")

    # pareto stats over window
    avg_fronts = np.mean(log_buf['num_fronts'])
    avg_f1 = np.mean(log_buf['f1_sizes'])
    avg_hv = np.mean(log_buf['hypervolumes'])
    hv_delta = log_buf['hypervolumes'][-1] - log_buf['hypervolumes'][0] if n > 1 else 0
    lines.append(f"Pareto   fronts: {avg_fronts:.1f}  F1: {avg_f1:.1f}  hv: {avg_hv:.2f} (Δ{hv_delta:+.2f})")

    avg_conn = np.average(log_buf['connections'])
    conn_delta = log_buf['connections'][-1] - log_buf['connections'][0] if n > 1 else 0
    avg_nodes_buf = np.average(log_buf['nodes'])
    avg_sim = np.average(log_buf['sim_times'])
    avg_breed = np.average(log_buf['breed_times'])
    buf_disabled = np.mean(log_buf['disabled_ratios'])
    buf_mut_mean = np.mean(log_buf['mut_power_means'])
    buf_mut_std = np.mean(log_buf['mut_power_stds'])

    all_scores_buf = np.array(log_buf['all_scores'])
    bp10, bp25, bp50, bp75, bp90 = np.percentile(all_scores_buf, [10, 25, 50, 75, 90])

    lines += [
        f"ScoreDis p10: {bp10:.3f}  p25: {bp25:.3f}  p50: {bp50:.3f}  p75: {bp75:.3f}  p90: {bp90:.3f}",
        f"Genome   avg_conn: {avg_conn:.1f} (Δ{conn_delta:+.1f})  avg_nodes: {avg_nodes_buf:.1f}  disabled: {buf_disabled:.2%}  pop: {pop_size}",
        f"MutPower mean: {buf_mut_mean:.3f}  std: {buf_mut_std*100/max(buf_mut_mean,1e-9):.2f}%",
        f"Timing   sim: {avg_sim:.2f}s  breed: {avg_breed:.2f}s  rate: {60/(avg_sim+avg_breed):.1f}/min  elapsed: {elapsed_fmt}",
        f"```",
    ]

    return "\n".join(lines)


class DiscordLogger:
    def __init__(self, webhook_url, interval=60):
        self.webhook = webhook_url
        self.interval = interval
        self.q = queue.Queue()
        self.stop_event = threading.Event()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def log(self, msg):
        if not self.stop_event.is_set():
            self.q.put(msg)

    def close(self):
        self.stop_event.set()
        self.q.put(None)          # sentinel to unblock queue
        self.thread.join()

    def _worker(self):
        last_send = 0

        while not self.stop_event.is_set():
            msg = self.q.get()

            if msg is None:      # shutdown signal
                break

            now = time.time()
            if now - last_send < self.interval:
                time.sleep(self.interval - (now - last_send))

            try:
                requests.post(self.webhook, json={"content": msg}, timeout=5)
                last_send = time.time()
            except Exception as e:
                print('thread error:', repr(e))
