from genome_prototype import NodeGene, ConnectionGene, create_connection, NodeType, Genome
import numpy as np

class Innovations:
    def __init__(self) -> None:
        # 0-12 reserverd fron input/output nodes
        self.counter = 13
        self.dict = {}

    def resolve(self, connectedNodes: tuple):
        # split tuple into input and output id
        a, b = connectedNodes
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("Wrong input type, need (int, int)")

        # get or create innovation number for this pair
        value = self.dict.get(connectedNodes)
        if value is None:
            value = self.counter
            self.counter += 1
            self.dict[connectedNodes] = value

        return value

def mutate_weights(connections: list[ConnectionGene], mutation_rate: float=.6, mutation_strength: float=.3):
    for connection in connections:
        roll = np.random.rand()
        # 10% chance for nothing, 72% nudge, 8% reset
        if roll < mutation_rate * .1:
            connection.weight = np.random.uniform(-1, 1)
        elif roll < mutation_rate:
            connection.weight += np.random.normal(0, mutation_strength)

# self-adaptive mutation power bounds
MUTATION_POWER_MIN = 0.05
MUTATION_POWER_MAX = 3.0
MUTATION_POWER_TAU = 0.1  # controls how fast mutation_power itself evolves

MUTATION_POWER_DEFAULT = 0.3
MUTATION_POWER_RESET_RATE = 0.05

def mutate_mutation_power(genome):
    """Mutate the genome's mutation_power using log-normal self-adaptation."""
    if np.random.rand() < MUTATION_POWER_RESET_RATE:
        genome.mutation_power = MUTATION_POWER_DEFAULT
    else:
        genome.mutation_power *= np.exp(MUTATION_POWER_TAU * np.random.randn() - 0.5 * MUTATION_POWER_TAU**2)
    genome.mutation_power = np.clip(genome.mutation_power, MUTATION_POWER_MIN, MUTATION_POWER_MAX)

def add_connection(genome: Genome, innovations: Innovations, weight=None):
    # get all node ids and existing connections
    pairs = [(c.input, c.output) for c in genome.connections]

    # generate candidates for input / output
    input_candidates = [n.id for n in genome.nodes if n.node_type != NodeType.OUTPUT]
    output_candidates = [n.id for n in genome.nodes if n.node_type != NodeType.INPUT]

    # try 10 times to find a valid connection
    for _ in range(10):
        a = int(np.random.choice(input_candidates))
        b = int(np.random.choice(output_candidates))
        # skip if connections exists / connecting to itself
        if (a, b) in pairs or a == b:
            continue

        innovation = innovations.resolve((a, b))
        genome.connections.append(create_connection(innovation, (a, b), weight=weight))
        break

def add_node(genome: Genome, innovations: Innovations):
    if not genome.connections:
        return

    # choose random connection to split
    selected_ix = np.random.randint(len(genome.connections))
    selected = genome.connections[selected_ix]

    # bail if connection is disabled
    if not selected.enabled:
        return

    # disable connection and create new node
    # connection is disabled to preserve historical lineage for speciation
    selected.enabled = False
    genome.nodes.append(NodeGene(id=selected.innovation, node_type=NodeType.HIDDEN))

    # get the innovation numbers for each connection
    innovation1 = innovations.resolve((selected.input, selected.innovation))
    innovation2 = innovations.resolve((selected.innovation, selected.output))

    # append the new connections to genome
    genome.connections.append(create_connection(innovation=innovation1, pair=(selected.input, selected.innovation), weight=selected.weight))
    genome.connections.append(create_connection(innovation=innovation2, pair=(selected.innovation, selected.output), weight=selected.weight))

def mutate(genome, innovations: Innovations):
    # probabilities
    add_connection_rate = .09
    add_node_rate = .03

    # self-adapt mutation power first, then use it for weight mutations
    mutate_mutation_power(genome)
    mutate_weights(genome.connections, mutation_strength=genome.mutation_power)

    # add connection
    if np.random.rand() < add_connection_rate:
        add_connection(genome, innovations)

    # add node
    if np.random.rand() < add_node_rate:
        add_node(genome, innovations)
