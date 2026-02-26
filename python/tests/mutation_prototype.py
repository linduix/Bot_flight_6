from genome_prototype import NodeGene, ConnectionGene, create_connection, NodeType
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

def mutate_weights(connections: list[ConnectionGene], mutation_rate: float=.8, mutation_strength: float=.1):
    for connection in connections:
        roll = np.random.rand()
        # 10% chance for nothing, 72% nudge, 8% reset
        if roll < mutation_rate * .1:
            connection.weight = np.random.uniform(-1, 1)
        elif roll < mutation_rate:
            connection.weight += np.random.normal(0, mutation_strength)

def add_connection(nodes: list[NodeGene], connections: list[ConnectionGene], innovations: Innovations):
    # get all node ids and existing connections
    pairs = [(c.input, c.output) for c in connections]
    
    # generate candidates for input / output
    input_candidates = [n.id for n in nodes if n.node_type != NodeType.OUTPUT]
    output_candidates = [n.id for n in nodes if n.node_type != NodeType.INPUT]

    # try 10 times to find a valid connection
    for _ in range(10):
        a = int(np.random.choice(input_candidates))
        b = int(np.random.choice(output_candidates))
        if (a, b) in pairs:
            continue
        innovation = innovations.resolve((a, b))
        connections.append(create_connection(innovation, (a, b)))
        break

def add_node(nodes: list[NodeGene], connections: list[ConnectionGene], innovations: Innovations):
    if not connections:
        return

    # choose random connection to split
    selected_ix = np.random.randint(len(connections))
    selected = connections[selected_ix]

    # bail if connection is disabled
    if not selected.enabled:
        return

    # disable connection and create new node
    # connection is disabled to preserve historical lineage for speciation
    selected.enabled = False
    nodes.append(NodeGene(id=selected.innovation, node_type=NodeType.HIDDEN))

    # get the innovation numbers for each connection
    innovation1 = innovations.resolve((selected.input, selected.innovation))
    innovation2 = innovations.resolve((selected.innovation, selected.output))

    # append the new connections to genome
    connections.append(create_connection(innovation=innovation1, pair=(selected.input, selected.innovation)))
    connections.append(create_connection(innovation=innovation2, pair=(selected.innovation, selected.output)))
