from mutation_prototype import Innovations, mutate
from genome_prototype import Genome, ConnectionGene, NodeGene, NodeType
import numpy as np
from copy import copy, deepcopy
import random
STAGNATION_LIMIT = 25
STAGNATION_CHANCES = 2          # lives before death (0 = instant kill at limit)
PROTECTION_WINDOW = 10          # avg of last N best-genome scores for protection
IMPROVEMENT_WINDOW = 10         # generations to look back for improvement rate
IMPROVEMENT_ALPHA = 0.3         # exponent on improvement_rate multiplier

class Species:
    _next_id: int = 0  # class-level counter, auto-increments on each new Species

    def __init__(self, rep) -> None:
        self.id = Species._next_id
        Species._next_id += 1
        self.rep: Genome = rep
        self.stagnation = 0
        self.best_score = -np.inf
        self.chances = STAGNATION_CHANCES
        self.best_history: list[float] = []   # best genome score per gen (sliding window)
        self.age: int = 0                      # gens this species has survived

def crossover(genome1: Genome, genome2: Genome, score1: float, score2: float) -> Genome:
    # set parents
    if score1 >= score2:
        parent1, parent2 = genome1, genome2
    else:
        parent1, parent2 = genome2, genome1

    # use cached connection dicts — no rebuild cost
    connections1: dict[int, ConnectionGene] = parent1.conn_dict
    connections2: dict[int, ConnectionGene] = parent2.conn_dict

    # build baby genome
    baby = Genome.new()
    # inheret connections from parents
    for innovation, connectionGene in connections1.items():
        # for matching connections in both parents
        if innovation in connections2:

            # inherit from parent 1 if connection activation mismatched
            if connectionGene.enabled != connections2[innovation].enabled:
                baby.connections.append(copy(connectionGene))
            else:
                # randomly inherit from either
                p = np.random.rand()
                if p < .5:
                    baby.connections.append(copy(connectionGene))
                else:
                    baby.connections.append(copy(connections2[innovation]))

        # only copy disjoint genes from parent 1
        else:
            baby.connections.append(copy(connectionGene))

    # inherit mutation_power as average of parents
    p1_power = getattr(parent1, 'mutation_power', 0.3)
    p2_power = getattr(parent2, 'mutation_power', 0.3)
    baby.mutation_power = (p1_power + p2_power) / 2

    # inherit species ID from fitter parent — gives sticky speciation a good first guess
    baby._species_id = parent1._species_id

    # baby connections are now final — build cache once here
    baby.invalidate_cache()

    # collect all required nodes
    nodes_queue = set()
    for c in baby.connections:
        nodes_queue.add(c.input)
        nodes_queue.add(c.output)

    # fabricate each hidden node
    for id in nodes_queue:
        # skip if reserved node (input/output)
        if id < 13:
            continue

        node = NodeGene(id=id, node_type=NodeType.HIDDEN)
        baby.nodes.append(node)

    return baby

def distance(genome1: Genome, genome2: Genome, c1=1, c2=1, c3=0.4) -> float:
    if genome1.connections == [] or genome2.connections == []:
        raise ValueError("Both genomes need atleast 1 connection")

    # use cached dicts — rebuilt only when connections list structurally changes
    g1_connections = genome1.conn_dict
    g2_connections = genome2.conn_dict

    # max innovation vals for both
    g1_max, g2_max = max(g1_connections.keys()), max(g2_connections.keys())
    lower_bound = min(g1_max, g2_max)

    # number of genes in bigger genome, larger than max innovation of smaller genome
    bigger = g1_connections if g1_max >= g2_max else g2_connections
    Excess = sum(1 for innovation in bigger.keys() if innovation > lower_bound)

    # number of genes less than the lowerbound in xor genes
    different = g1_connections.keys() ^ g2_connections.keys()
    Disjoint = sum(1 for innovation in different if innovation <= lower_bound)

    # average weight difference of matching genes
    Weight = 0
    matching = g1_connections.keys() & g2_connections.keys()
    if len(matching) > 0:
        for node in matching:
            weight1 = g1_connections[node].weight
            weight2 = g2_connections[node].weight

            Weight += abs(weight1 - weight2)

        # average the weight differences
        Weight /= len(matching)

    # distance = (c1 * E + c2 * D) / N + c3 * W | where N is number of connections in bigger genome
    distance = (c1 * Excess + c2 * Disjoint) / max(len(g1_connections), len(g2_connections)) + c3 * Weight

    return distance

def speciate(species, threshold, genomes: list[Genome]):
    s: list[Species] = species    # species class array representing the species
    species_pop = [[] for _ in s] # the population grouped into species, index matched to species array

    # build O(1) lookup: species_id -> index in s
    species_id_to_idx = {spec.id: i for i, spec in enumerate(s)}

    # loop through each genome
    for genome in genomes:
        # if reps empty add first genome as rep
        if s == []:
            s.append(Species(genome))
            species_pop.append([genome])
            genome._species_id = s[0].id
            continue

        match = False

        # sticky: check last gen's species first (O(1) lookup, one distance call)
        if genome._species_id is not None:
            cached_idx = species_id_to_idx.get(genome._species_id)
            if cached_idx is not None and distance(genome, s[cached_idx].rep) < threshold:
                species_pop[cached_idx].append(genome)
                match = True

        if not match:
            # fall back to full scan
            for i, spec in enumerate(s):
                if distance(genome, spec.rep) < threshold:
                    species_pop[i].append(genome)
                    genome._species_id = spec.id
                    match = True
                    break

        # if no matches found, turn it into a rep
        if not match:
            s.append(Species(genome))
            species_pop.append([genome])
            genome._species_id = s[-1].id

    return s, species_pop

def breed(current_gen: list[Genome], scores: list[float] | np.ndarray, innovations: Innovations, poputlation_size: int, prev_species, threshold=3.0):
    # map genome to scores
    raw_scores = list(scores)
    adjusted_scores = [0.0] * len(raw_scores)

    # penalize score for too much complexity (disabled)
    ix = 0
    for score, g in zip(raw_scores, current_gen):
        # edges = sum([1 for c in g.connections if c.enabled])
        # nodes = sum([1 for n in g.nodes if n.node_type == NodeType.HIDDEN])
        # excess = max(0, edges - 50) + max(0, nodes - 13) * 2
        # adjusted_scores[ix] = score / (1 + 0.005 * excess)
        adjusted_scores[ix] = score
        ix += 1

    min_score = min(adjusted_scores)
    shifted_scores = [s - min_score + 1e-6 for s in adjusted_scores]

    raw_genome_scores = {genome: score for genome, score in zip(current_gen, raw_scores)}
    unshifted_scores = {genome: score for genome, score in zip(current_gen, adjusted_scores)}
    genome_scores = {genome: score for genome, score in zip(current_gen, shifted_scores)}

    # speciation sorted by score
    species, species_pop = speciate(prev_species, threshold, current_gen)
    # filter out emtpty lists
    species = [s for s, pop in zip(species, species_pop) if pop]
    species_pop = [pop for pop in species_pop if pop]
    # sort and cull species
    for i, s in enumerate(species_pop):
        s.sort(key=lambda g: genome_scores[g], reverse=True)
        # update representative to random member
        species[i].rep = random.choice(s)
        # cull the worst half of the species
        species_pop[i] = s[:max(1, len(s)//2)]

    # stagnation counter
    survivors = []
    stagnant_killed = 0
    killed_genomes = 0
    for i, s in enumerate(species):
        improved = False
        current_best = max(raw_genome_scores[g] for g in species_pop[i])
        s.best_history.append(current_best)
        if len(s.best_history) > PROTECTION_WINDOW:
            s.best_history.pop(0)
        s.age += 1

        for g in species_pop[i]:
            score = raw_genome_scores[g]
            if score > s.best_score:
                s.best_score = score
                improved = True
        if improved:
            s.stagnation = 0
        else:
            s.stagnation += 1

        if s.stagnation < STAGNATION_LIMIT:
            survivors.append(i)
        elif s.chances > 0:
            # burn a chance, reset stagnation, keep alive
            s.chances -= 1
            s.stagnation = 0
            survivors.append(i)
        else:
            stagnant_killed += 1
            killed_genomes += len(species_pop[i])

    # protection: immune species cannot be stagnation-killed
    def _protect(idx):
        if idx not in survivors:
            survivors.append(idx)
            nonlocal stagnant_killed, killed_genomes
            stagnant_killed -= 1
            killed_genomes -= len(species_pop[idx])

    # 1) best average recent performer
    def protection_score(s):
        return np.mean(s.best_history) if s.best_history else -np.inf

    best_prot = max(protection_score(s) for s in species)
    for i, s in enumerate(species):
        if protection_score(s) >= best_prot:
            _protect(i)

    # 2) historical best scoring species (highest all-time best_score)
    best_historical = max(range(len(species)), key=lambda i: species[i].best_score)
    _protect(best_historical)

    # 3) global best species (contains the top genome this generation)
    best_global = max(range(len(species)), key=lambda i: max(raw_genome_scores[g] for g in species_pop[i]))
    _protect(best_global)

    # 4) most structurally isolated species (furthest avg distance from all other reps)
    # Build shared S×S rep distance matrix — reused by novelty shortfall below
    S = len(species)
    rep_dist = [[0.0] * S for _ in range(S)]
    if S >= 2:
        reps = [s.rep for s in species]
        for i in range(S):
            for j in range(i + 1, S):
                if reps[i].connections and reps[j].connections:
                    d = distance(reps[i], reps[j])
                else:
                    d = 0.0
                rep_dist[i][j] = d
                rep_dist[j][i] = d

    if S >= 3:
        mean_dists = [np.mean([rep_dist[i][j] for j in range(S) if j != i]) for i in range(S)]
        most_isolated = max(range(S), key=lambda i: mean_dists[i])
        _protect(most_isolated)

    deaths = len(species) - len(survivors)

    # cull stagnated species — also reindex rep_dist to match survivors
    temp_species = []
    temp_species_pop  = []
    for i in survivors:
        temp_species.append(species[i])
        temp_species_pop.append(species_pop[i])
    species = temp_species
    species_pop = temp_species_pop
    # reindex rep_dist rows/cols to survivors only so novelty loop indices are correct
    rep_dist = [[rep_dist[i][j] for j in survivors] for i in survivors]

    # fitness sharing and average fitness per species
    species_fitness = []
    for s in species_pop:
        fitness = 0
        for genome in s:
            # reduce score by size of species
            genome_scores[genome] /= len(s)
            # add adjusted score to species fitness
            fitness += genome_scores[genome]
        species_fitness.append(fitness)

    # dyNEAT-style improvement rate boost
    for i, s in enumerate(species):
        window = min(IMPROVEMENT_WINDOW, len(s.best_history))
        old = s.best_history[-window]
        if old > 0:
            species_fitness[i] *= max(1.0, s.best_history[-1] / old) ** IMPROVEMENT_ALPHA

    # species quota calculation
    quotas = []
    total_fitness = sum(species_fitness)
    for fitness in species_fitness:
        # calculate quota as proportion of total fitness
        quota = int(fitness * poputlation_size / total_fitness)
        quotas.append(quota)
    # print(f'size:      {[len(s) for s in species_pop]}')
    # print(f'survivors: {survivors}')
    # print(f'quotas:    {quotas}')
    # print(f'scores:    {[round(float(s.best_history[-1]), 2) for s in species]}')

    # top up leftover slots using distance-from-mean as diversity pressure
    shortfall = poputlation_size - sum(quotas)
    if shortfall > 0 and len(species) > 1:
        # reuse the rep_dist matrix already computed above — no extra distance() calls
        S_now = len(species)
        novelty_scores = []
        for i in range(S_now):
            if not species[i].rep.connections:
                novelty_scores.append(0.0)
                continue
            dists = [rep_dist[i][j] for j in range(S_now) if j != i and species[j].rep.connections]
            novelty_scores.append(np.mean(dists) if dists else 0.0)
        total_novelty = sum(novelty_scores)
        if total_novelty > 0:
            # distribute proportionally to novelty, floored
            bonus = [int(n / total_novelty * shortfall) for n in novelty_scores]
            # sprinkle any remaining 1-by-1 to highest novelty species
            remainder = shortfall - sum(bonus)
            ranked = np.argsort(novelty_scores)[::-1]
            for k in range(remainder):
                bonus[ranked[k % len(ranked)]] += 1
            for i in range(len(quotas)):
                quotas[i] += bonus[i]

    # cull stats returned to caller
    cull_stats = {
        'stagnant_killed': stagnant_killed,
        'killed_genomes': killed_genomes,
    }

    # breeding
    next_gen: list[Genome] = []
    # elietism
    for i, s in enumerate(species_pop):
        if quotas[i] == 0:
            continue
        next_gen.append(deepcopy(s[0]))

    # breed rest of population
    for i, s in enumerate(species_pop):
        species_scores = [genome_scores[g] for g in s]
        for _ in range(quotas[i] - 1):
            # choose parents
            parent1, parent2 = random.choices(s, weights=species_scores, k=2)
            if np.random.rand() < 0.01:
                parent2 = random.choices(current_gen, weights=shifted_scores, k=1)[0]
            # breed baby
            baby = crossover(parent1, parent2, raw_genome_scores[parent1], raw_genome_scores[parent2])
            # mutate baby
            mutate(baby, innovations)
            # add baby
            next_gen.append(baby)


    return next_gen, species_pop, species, deaths, cull_stats


def _pareto_fronts(x_vals, y_vals):
    """Peel successive Pareto fronts (maximise both objectives)."""
    n = len(x_vals)
    remaining = np.ones(n, dtype=bool)
    fronts = []
    while remaining.any():
        rx, ry = x_vals[remaining], y_vals[remaining]
        local_mask = np.zeros(len(rx), dtype=bool)
        order = np.lexsort((-ry, -rx))
        best_y = -np.inf
        for idx in order:
            if ry[idx] >= best_y:
                local_mask[idx] = True
                best_y = ry[idx]
        global_mask = np.zeros(n, dtype=bool)
        rem_indices = np.where(remaining)[0]
        global_mask[rem_indices[local_mask]] = True
        fronts.append(global_mask)
        remaining &= ~global_mask
    return fronts


def breed_pareto(pool: list[Genome], scores: list[float] | np.ndarray,
                 innovations: Innovations, population_size: int,
                 best_genome: Genome, best_ever: float = 0.0):
    """
    (μ+λ) NSGA-II-style breed for NEAT.

    Input:  combined pool (parents + offspring from last round), all freshly scored.
            First call: just the initial population.
    Output: N genomes: top N/2 survivors via Pareto selection + N/2 offspring.
            Caller scores all N together, then feeds them back next round.
    """
    raw_scores = np.array(scores, dtype=float)

    # ── Score floor: cull genomes below 10% of historical best ──
    floor = best_ever * 0.10
    if floor > 0:
        keep = [i for i in range(len(pool)) if raw_scores[i] >= floor]
        if len(keep) < population_size:
            # not enough above floor — keep top population_size by score
            keep = list(np.argsort(raw_scores)[-population_size:])
        pool = [pool[i] for i in keep]
        raw_scores = raw_scores[keep]

    n = len(pool)
    half = population_size // 2

    # ── Complexity-adjusted scores (disabled) ──
    adj_scores = np.zeros(n)
    for ix, g in enumerate(pool):
        # edges = sum(1 for c in g.connections if c.enabled)
        # nodes = sum(1 for nd in g.nodes if nd.node_type == NodeType.HIDDEN)
        # excess = max(0, edges - 50) + max(0, nodes - 13) * 2
        # adj_scores[ix] = raw_scores[ix] / (1 + 0.005 * excess)
        adj_scores[ix] = raw_scores[ix]

    # ── KNN genetic diversity (avg distance to sqrt(n) nearest neighbors) ──
    # Approximate via random sampling instead of full O(n²) pairwise matrix.
    # Sample 4×k_neighbors candidates per genome — good enough for ranking diversity.
    k_neighbors = max(1, int(n ** 0.5))
    sample_size = min(n - 1, max(k_neighbors * 4, 30))
    obj_dists = np.zeros(n)
    for i in range(n):
        candidates = random.sample([j for j in range(n) if j != i], sample_size)
        dists = [distance(pool[i], pool[j]) for j in candidates]
        dists.sort()
        obj_dists[i] = float(np.mean(dists[:k_neighbors]))

    # ── Pareto rank the pool ──
    # Scale fitness axis so it dominates diversity in Pareto ranking.
    # A genome needs FITNESS_WEIGHT× more diversity to compensate for 1 unit of fitness loss.
    FITNESS_WEIGHT = 10.0
    fronts = _pareto_fronts(obj_dists, adj_scores * FITNESS_WEIGHT)

    # normalize objectives to [0,1] for HV contribution (used by selection + tournament)
    d_min, d_max = float(obj_dists.min()), float(obj_dists.max())
    s_min, s_max = float(adj_scores.min()), float(adj_scores.max())
    d_span = d_max - d_min if d_max > d_min else 1.0
    s_span = s_max - s_min if s_max > s_min else 1.0

    # ── Select top half (survivors) front by front ──
    survivors: list[Genome] = []
    for mask in fronts:
        indices = list(np.where(mask)[0])
        if len(survivors) + len(indices) <= half:
            for i in indices:
                survivors.append(pool[i])
        else:
            # partial front — iteratively remove lowest HV contribution
            need = half - len(survivors)
            remaining = list(indices)
            while len(remaining) > need:
                if len(remaining) <= 2:
                    break
                d_norm = [(obj_dists[i] - d_min) / d_span for i in remaining]
                s_norm = [(adj_scores[i] - s_min) / s_span for i in remaining]
                order = sorted(range(len(remaining)), key=lambda k: d_norm[k])
                hv = {}
                k_last = len(order) - 1
                for pos in range(len(order)):
                    idx = remaining[order[pos]]
                    if pos == 0 or pos == k_last:
                        hv[idx] = float('inf')
                    else:
                        hv[idx] = (d_norm[order[pos + 1]] - d_norm[order[pos]]) * (s_norm[order[pos]] - s_norm[order[pos + 1]])
                worst = min(remaining, key=lambda i: hv[i])
                remaining.remove(worst)
            for i in remaining:
                survivors.append(pool[i])
            break

    # ── Build lookups for tournament on survivors ──
    genome_front = {}
    genome_hv = {}   # hypervolume contribution per genome
    for i, g in enumerate(pool):
        genome_front[g] = len(fronts)
        genome_hv[g] = 0.0

    for rank, mask in enumerate(fronts):
        indices = list(np.where(mask)[0])
        for i in indices:
            genome_front[pool[i]] = rank
        # compute HV contribution for each genome on this front
        if len(indices) <= 1:
            for i in indices:
                genome_hv[pool[i]] = float('inf')
            continue
        # sort by normalized diversity axis ascending (score descending on Pareto front)
        d_norm = [(obj_dists[i] - d_min) / d_span for i in indices]
        s_norm = [(adj_scores[i] - s_min) / s_span for i in indices]
        order = sorted(range(len(indices)), key=lambda k: d_norm[k])
        k_last = len(order) - 1
        for pos in range(len(order)):
            idx = indices[order[pos]]
            if pos == 0 or pos == k_last:
                # boundary points always preserved — anchor the front extremes
                hv_c = float('inf')
            else:
                # interior: exclusive area = (d_{i+1} - d_i) * (s_i - s_{i+1})
                hv_c = (d_norm[order[pos + 1]] - d_norm[order[pos]]) * (s_norm[order[pos]] - s_norm[order[pos + 1]])
            genome_hv[pool[idx]] = max(hv_c, 0.0)

    raw_genome_scores = {g: s for g, s in zip(pool, raw_scores)}

    def tournament(k=3):
        candidates = random.sample(survivors, min(k, len(survivors)))
        # lower front wins; tiebreak: higher hypervolume contribution
        candidates.sort(key=lambda g: (genome_front[g], -genome_hv[g]))
        return candidates[0]

    # ── Breed offspring to fill second half ──
    offspring: list[Genome] = []
    num_offspring = population_size - len(survivors)
    while len(offspring) < num_offspring:
        p1 = tournament()
        p2 = tournament()
        baby = crossover(p1, p2, raw_genome_scores[p1], raw_genome_scores[p2])
        mutate(baby, innovations)
        offspring.append(baby)

    # ── Pareto stats ──
    front_sizes = [int(mask.sum()) for mask in fronts]
    best_idx = int(np.argmax(raw_scores))
    survivor_indices = [i for i, g in enumerate(pool) if g in set(survivors)]
    survivor_scores = raw_scores[survivor_indices] if survivor_indices else np.array([0.0])

    # ── Front 1 metrics ──
    f1_indices = list(np.where(fronts[0])[0]) if fronts else []
    f1_dists = obj_dists[f1_indices] if f1_indices else np.array([0.0])
    f1_scores = adj_scores[f1_indices] if f1_indices else np.array([0.0])

    # Hypervolume: area dominated by front 1 (normalized to [0,1] on both axes)
    hypervolume = 0.0
    if len(f1_indices) >= 1:
        d_min, d_max = float(obj_dists.min()), float(obj_dists.max())
        s_min, s_max = float(adj_scores.min()), float(adj_scores.max())
        d_span = d_max - d_min if d_max > d_min else 1.0
        s_span = s_max - s_min if s_max > s_min else 1.0
        f1_d_norm = (f1_dists - d_min) / d_span
        f1_s_norm = (f1_scores - s_min) / s_span
        order = np.argsort(f1_d_norm)
        f1_d_sorted = f1_d_norm[order]
        f1_s_sorted = f1_s_norm[order]
        prev_d = 0.0
        for i in range(len(order)):
            width = f1_d_sorted[i] - prev_d
            hypervolume += width * f1_s_sorted[i]
            prev_d = f1_d_sorted[i]

    # Front spread: range of each objective on front 1
    f1_dist_spread = float(f1_dists.max() - f1_dists.min()) if len(f1_indices) >= 2 else 0.0
    f1_score_spread = float(f1_scores.max() - f1_scores.min()) if len(f1_indices) >= 2 else 0.0

    # Spacing: std of nearest-neighbor distances along front 1 (normalized 2D)
    spacing = 0.0
    if len(f1_indices) >= 3:
        d_span = f1_dists.max() - f1_dists.min()
        s_span = f1_scores.max() - f1_scores.min()
        f1_dn = (f1_dists - f1_dists.min()) / d_span if d_span > 0 else np.zeros(len(f1_indices))
        f1_sn = (f1_scores - f1_scores.min()) / s_span if s_span > 0 else np.zeros(len(f1_indices))
        coords = np.column_stack((f1_dn, f1_sn))
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        pdists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(pdists, np.inf)
        nn_dists = pdists.min(axis=1)
        spacing = float(nn_dists.std())

    pareto_stats = {
        'num_fronts': len(fronts),
        'f1_size': front_sizes[0] if front_sizes else 0,
        'hypervolume': hypervolume,
        'f1_dist_spread': f1_dist_spread,
        'f1_score_spread': f1_score_spread,
        'spacing': spacing,
        'dist_mean': float(np.mean(obj_dists)),
        'dist_std': float(np.std(obj_dists)),
        'dist_max': float(np.max(obj_dists)),
        'dist_min': float(np.min(obj_dists[obj_dists > 0])) if (obj_dists > 0).any() else 0.0,
        'best_front': genome_front[pool[best_idx]],
        'best_dist': float(obj_dists[best_idx]),
        'survivor_mean': float(np.mean(survivor_scores)),
        'survivor_max': float(np.max(survivor_scores)),
        'penalty_mean': float(np.mean(raw_scores - adj_scores)),
    }

    # combined output: survivors + offspring, all need scoring next round
    return survivors + offspring, pareto_stats
