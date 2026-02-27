from genome_prototype import Genome, ConnectionGene, NodeGene, NodeType
import numpy as np
from copy import copy

def crossover(genome1: Genome, genome2: Genome):
    # set parents
    parent1 = genome1
    parent2 = genome2

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

def distance(genome1: Genome, genome2: Genome, c1=1, c2=1, c3=0.4):
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

def speciate(threshold):
    pass