from mutation_prototype import Innovations, mutate
from genome_prototype import Genome, ConnectionGene, NodeGene, NodeType
import numpy as np
from copy import copy, deepcopy
import random

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

def speciate(threshold, genomes: list[Genome]) -> list[list[Genome]]:
    representatives = []
    species = []

    # loop through each genome
    for genome in genomes:
        # if reps empty add first genome as rep
        if representatives == []:
            representatives.append(genome)
            species.append([genome])
            continue

        match = False
        # loop through and add to rep's species if distance < threshold
        for i, representative in enumerate(representatives):
            if distance(genome, representative) < threshold:
                species[i].append(genome)
                match = True
                break
        
        # if no matches found, turn it into a rep
        if not match:
            representatives.append(genome)
            species.append([genome])

    return species

def breed(current_gen: list[Genome], scores: list[float], innovations: Innovations, poputlation=100, threshold=3.0) -> list[Genome]:
    # map genome to scores
    genome_scores = {genome: score for genome, score in zip(current_gen, scores)}
    
    # speciation sorted by score    
    species: list[list[Genome]] = speciate(threshold, current_gen)
    for s in species:
        s.sort(key=lambda g: genome_scores[g], reverse=True)

    # fitness sharing and average fitness per species
    species_fitness = []
    for s in species:
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
        quota = int(fitness * poputlation / total_fitness)
        quotas.append(quota)

    # breeding
    next_gen = []
    # elietism
    for s in species:
        next_gen.append(deepcopy(s[0]))

    # breed rest of population
    for i, s in enumerate(species):
        species_scores = [genome_scores[g] for g in s]
        for _ in range(quotas[i] - 1):
            # choose parents
            parent1, parent2 = random.choices(s, weights=species_scores, k=2)
            # breed baby
            baby = crossover(parent1, parent2, genome_scores[parent1], genome_scores[parent2])
            # mutate baby
            mutate(baby, innovations)
            # add baby
            next_gen.append(baby)

    return next_gen