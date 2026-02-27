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
