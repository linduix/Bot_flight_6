from mutation_prototype import Innovations, mutate
from genome_prototype import Genome, ConnectionGene, NodeGene, NodeType
import numpy as np
from copy import copy, deepcopy
import random
STAGNATION_LIMIT = 25
STAGNATION_CHANCES = 2          # lives before death (0 = instant kill at limit)
PROTECTION_WINDOW = 10          # avg of last N best-genome scores for protection

class Species:
    def __init__(self, rep) -> None:
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

    # index connectienos using their innovation number per parent
    connections1: dict[int, ConnectionGene] = {c.innovation: c for c in parent1.connections}
    connections2: dict[int, ConnectionGene] = {c.innovation: c for c in parent2.connections}

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

    # innvoations in genomes
    g1_connections = {c.innovation: c for c in genome1.connections}
    g2_connections = {c.innovation: c for c in genome2.connections}

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

    # loop through each genome
    for genome in genomes:
        # if reps empty add first genome as rep
        if s == []:
            s.append(Species(genome))
            species_pop.append([genome])
            continue

        match = False
        # loop through and add to rep's species if distance < threshold
        for i, spec in enumerate(s):
            if distance(genome, spec.rep) < threshold:
                species_pop[i].append(genome)
                match = True
                break

        # if no matches found, turn it into a rep
        if not match:
            s.append(Species(genome))
            species_pop.append([genome])

    return s, species_pop

def breed(current_gen: list[Genome], scores: list[float] | np.ndarray, innovations: Innovations, poputlation_size: int, prev_species, threshold=3.0):
    # map genome to scores
    raw_scores = list(scores)
    adjusted_scores = [0.0] * len(raw_scores)

    # penalize score for too much complexity
    ix = 0
    for score, g in zip(raw_scores, current_gen):
        edges = sum([1 for c in g.connections if c.enabled])
        nodes = sum([1 for n in g.nodes if n.node_type == NodeType.HIDDEN])
        excess = max(0, edges - 50) + max(0, nodes - 13) * 2
        adjusted_scores[ix] = score / (1 + 0.005 * excess)
        ix += 1

    min_score = min(adjusted_scores)
    shifted_scores = [s - min_score + 1e-6 for s in adjusted_scores]

    unshifted_scores = {genome: score for genome, score in zip(current_gen, adjusted_scores)}
    genome_scores = {genome: score for genome, score in zip(current_gen, shifted_scores)}
    raw_genome_scores = genome_scores.copy()

    # speciation sorted by score
    species, species_pop = speciate(prev_species, threshold, current_gen)
    # filter out emtpty lists
    species = [s for s, pop in zip(species, species_pop) if pop]
    species_pop = [pop for pop in species_pop if pop]
    # sort and cull species
    for i, s in enumerate(species_pop):
        s.sort(key=lambda g: genome_scores[g], reverse=True)
        # update representative to best genome in species
        species[i].rep = s[0]
        # cull the worst half of the species
        species_pop[i] = s[:max(1, len(s)//2)]

    # stagnation counter
    survivors = []
    stagnant_killed = 0
    killed_genomes = 0
    for i, s in enumerate(species):
        improved = False
        current_best = max(unshifted_scores[g] for g in species_pop[i])
        s.best_history.append(current_best)
        if len(s.best_history) > PROTECTION_WINDOW:
            s.best_history.pop(0)
        s.age += 1

        for g in species_pop[i]:
            score = unshifted_scores[g]
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
    best_global = max(range(len(species)), key=lambda i: max(unshifted_scores[g] for g in species_pop[i]))
    _protect(best_global)
    deaths = len(species) - len(survivors)

    # cull stagnated species
    temp_species = []
    temp_species_pop  = []
    for i in survivors:
        temp_species.append(species[i])
        temp_species_pop.append(species_pop[i])
    species = temp_species
    species_pop = temp_species_pop

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

    # cap quotas
    # max_quota = int(poputlation_size * max(0.35, 1.1 / len(species_pop)))
    # # excess = 0
    # for ix, quota in enumerate(quotas):
    #     # excess += max(quota - max_quota, 0)
    #     quotas[ix] = min(quota, max_quota)

    # i = 0
    # while sum(quotas) < poputlation_size:
    #     ix = i % len(quotas)
    #     if quotas[ix] < max_quota:
    #         quotas[ix] += 1
    #     i += 1

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
