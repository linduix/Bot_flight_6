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
        genome.invalidate_cache()
        break

MAX_HIDDEN_NODES = 32

def add_node(genome: Genome, innovations: Innovations):
    if not genome.connections:
        return

    # cap hidden node count
    hidden_count = sum(1 for n in genome.nodes if n.node_type == NodeType.HIDDEN)
    if hidden_count >= MAX_HIDDEN_NODES:
        return

    # choose random connection to split
    selected_ix = np.random.randint(len(genome.connections))
    selected = genome.connections[selected_ix]

    # bail if connection is disabled
    if not selected.enabled:
        return

    # bail if this connection was already split (node still exists from prior split)
    if any(n.id == selected.innovation for n in genome.nodes):
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
    genome.invalidate_cache()

def _hidden_nodes_connected(genome: Genome, skip_conn: ConnectionGene) -> bool:
    """Check that every hidden node is still reachable from an input AND can reach an output
    when `skip_conn` is treated as disabled."""
    node_types = {n.id: n.node_type for n in genome.nodes}
    hidden_ids = {nid for nid, nt in node_types.items() if nt == NodeType.HIDDEN}

    if not hidden_ids:
        return True

    # build adjacency from enabled connections, excluding skip_conn
    forward  = {}  # node -> set of nodes it feeds into
    backward = {}  # node -> set of nodes that feed into it
    for c in genome.connections:
        if not c.enabled or c is skip_conn:
            continue
        forward.setdefault(c.input, set()).add(c.output)
        backward.setdefault(c.output, set()).add(c.input)

    # forward flood from inputs: which nodes can be reached?
    inputs = {nid for nid, nt in node_types.items() if nt == NodeType.INPUT}
    reachable_from_input = set()
    stack = list(inputs)
    while stack:
        n = stack.pop()
        for nb in forward.get(n, ()):
            if nb not in reachable_from_input:
                reachable_from_input.add(nb)
                stack.append(nb)

    # backward flood from outputs: which nodes can reach an output?
    outputs = {nid for nid, nt in node_types.items() if nt == NodeType.OUTPUT}
    reaches_output = set()
    stack = list(outputs)
    while stack:
        n = stack.pop()
        for nb in backward.get(n, ()):
            if nb not in reaches_output:
                reaches_output.add(nb)
                stack.append(nb)

    # every hidden node must appear in both sets
    return hidden_ids <= (reachable_from_input & reaches_output)


def delete_node(genome: Genome):
    """Delete a random hidden node and disable its connections, only if no other hidden node is orphaned."""
    hidden = [n for n in genome.nodes if n.node_type == NodeType.HIDDEN]
    if not hidden:
        return

    target = hidden[np.random.randint(len(hidden))]
    tid = target.id

    # build graph without this node's connections
    remaining_hidden = {n.id for n in genome.nodes if n.node_type == NodeType.HIDDEN and n.id != tid}
    if remaining_hidden:
        node_types = {n.id: n.node_type for n in genome.nodes if n.id != tid}
        forward, backward = {}, {}
        for c in genome.connections:
            if not c.enabled or c.input == tid or c.output == tid:
                continue
            forward.setdefault(c.input, set()).add(c.output)
            backward.setdefault(c.output, set()).add(c.input)

        # forward flood from inputs
        reachable = set()
        stack = [nid for nid, nt in node_types.items() if nt == NodeType.INPUT]
        while stack:
            n = stack.pop()
            for nb in forward.get(n, ()):
                if nb not in reachable:
                    reachable.add(nb)
                    stack.append(nb)

        # backward flood from outputs
        reaches_output = set()
        stack = [nid for nid, nt in node_types.items() if nt == NodeType.OUTPUT]
        while stack:
            n = stack.pop()
            for nb in backward.get(n, ()):
                if nb not in reaches_output:
                    reaches_output.add(nb)
                    stack.append(nb)

        if not (remaining_hidden <= (reachable & reaches_output)):
            return

    # safe — remove node and purge its connections entirely
    # (disabling would leave ghosts that conflict if the ancestor connection re-splits)
    genome.nodes = [n for n in genome.nodes if n.id != tid]
    genome.connections = [c for c in genome.connections if c.input != tid and c.output != tid]
    genome.invalidate_cache()


def reenable_connection(genome: Genome):
    """Re-enable a random disabled connection, only if both endpoint nodes still exist."""
    node_ids = {n.id for n in genome.nodes}
    candidates = [c for c in genome.connections if not c.enabled and c.input in node_ids and c.output in node_ids]
    if not candidates:
        return
    pick = candidates[np.random.randint(len(candidates))]
    pick.enabled = True
    genome.invalidate_cache()


def disable_connection(genome: Genome):
    """Disable a random enabled connection, only if it won't orphan or strand any hidden node."""
    enabled = [c for c in genome.connections if c.enabled]
    if not enabled:
        return

    for _ in range(10):
        candidate = enabled[np.random.randint(len(enabled))]
        if _hidden_nodes_connected(genome, candidate):
            candidate.enabled = False
            genome.invalidate_cache()
            return


def mutate(genome, innovations: Innovations):
    # probabilities
    add_connection_rate = .15
    add_node_rate = .05
    delete_node_rate = 0
    reenable_connection_rate = .08
    disable_connection_rate = .06

    # self-adapt mutation power first, then use it for weight mutations
    mutate_mutation_power(genome)
    mutate_weights(genome.connections, mutation_strength=genome.mutation_power)

    # add connection
    if np.random.rand() < add_connection_rate:
        add_connection(genome, innovations)

    # add node
    if np.random.rand() < add_node_rate:
        add_node(genome, innovations)

    # delete node
    if np.random.rand() < delete_node_rate:
        delete_node(genome)

    # reenable connection
    if np.random.rand() < reenable_connection_rate:
        reenable_connection(genome)

    # disable connection
    if np.random.rand() < disable_connection_rate:
        disable_connection(genome)

    if delete_node_rate > 0:
        _dedup_check(genome)

def _dedup_check(genome):
    """Check and fix duplicate nodes/connections. Call periodically or enable via DEDUP_CHECK."""
    seen_nodes = set()
    duped_nodes = []
    for n in genome.nodes:
        if n.id in seen_nodes:
            duped_nodes.append(n.id)
        seen_nodes.add(n.id)
    if duped_nodes:
        print(f'  !! DUPLICATE NODES: {duped_nodes}')
        genome.nodes = list({n.id: n for n in genome.nodes}.values())

    seen_conns = set()
    duped_conns = []
    for c in genome.connections:
        if c.innovation in seen_conns:
            duped_conns.append(c.innovation)
        seen_conns.add(c.innovation)
    if duped_conns:
        print(f'  !! DUPLICATE CONNECTIONS: {duped_conns}')
        genome.connections = list({c.innovation: c for c in genome.connections}.values())

    if duped_nodes or duped_conns:
        genome.invalidate_cache()
