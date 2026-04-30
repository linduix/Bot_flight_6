from genome import Genome, NodeType, ConnectionGene
import numpy as np
import math
from numba import njit

# ── Activation config ───────────────────────────────────────────────────────
# Options: "tanh", "sigmoid", "leaky_relu", "relu"
HIDDEN_ACTIVATION = "softsign"
TURN_ACTIVATION   = "softsign"   # outputs 9, 10 (thruster turns)
THRUST_ACTIVATION = "softsign"   # outputs 11, 12 (throttle)

_ACT_IDS = {"tanh": 0, "sigmoid": 1, "leaky_relu": 2, "relu": 3, "softsign": 4}
_HIDDEN_ACT = _ACT_IDS[HIDDEN_ACTIVATION]
_TURN_ACT   = _ACT_IDS[TURN_ACTIVATION]
_THRUST_ACT = _ACT_IDS[THRUST_ACTIVATION]

class NeatNN:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.node_order = self.topo_sort()
        N = len(self.node_order)

        # map node id to its position in the array
        self.nodeix = {nid: ix for ix, nid in enumerate(self.node_order)}

        # precalculate all incomming connections for nodes
        # contains list of incoming connections as (input node ix, weight, recurrent)
        self.incoming_connections: list[list[tuple[int, float, bool]]] = [[] for _ in range(N)]
        for c in genome.connections:
            if c.enabled:
                in_ix, weight = self.nodeix[c.input], c.weight       # input ix and weight
                recur = self.nodeix[c.output] < self.nodeix[c.input] # recursive if input ix > output ix
                out_ix = self.nodeix[c.output]                       # output ix
                self.incoming_connections[out_ix].append((in_ix, weight, recur))

        # node activation values
        self.previous_value = [0.0] * N # {node: previous activation}
        self.current_value  = [0.0] * N  # {node: current activation}

    def topo_sort(self):
        nodes = [n.id for n in self.genome.nodes if n.node_type != NodeType.OUTPUT]
        outputs = set(n.id for n in self.genome.nodes if n.node_type == NodeType.OUTPUT)
        connections = [(c.input, c.output) for c in self.genome.connections if c.enabled]

        # kahn's algorithm
        # remove nodes with no inputs recursively untill empty
        # if all nodes are not removed, there is cycle and they are added at the end
        in_degree = {n: 0 for n in nodes}
        for _, output in connections:
            if output not in outputs:
                in_degree[output] += 1

        queue = [n for n in nodes if in_degree[n] == 0]
        sorted_nodes = []

        # pre map input nodes to their outputs
        outgoing = {n: [] for n in nodes}
        for input, output in connections:
            outgoing[input].append(output)

        while True:
            # once out of targets
            if not queue:
                # add remaining nodes (cycle)
                sorted_nodes += [n for n in nodes if n not in sorted_nodes]
                break

            # move node from queue to sorted
            node = queue.pop(0)
            sorted_nodes.append(node)

            # go through node outputs and decrement their in degree
            for output in outgoing[node]:
                # skip network outputs
                if output in outputs:
                    continue

                in_degree[output] -= 1
                # if new 0 degree nodes discovered, add to queue
                if in_degree[output] == 0:
                    queue.append(output)


        sorted_nodes += outputs
        return sorted_nodes

    def forward(self, delta_x, delta_y, angle, vel_x, vel_y, angular_vel, t1_angle, t2_angle):
        # buffer swap current to previous
        self.previous_value, self.current_value = self.current_value, self.previous_value

        # bind to local var to reduce lookup in
        prev, cur = self.previous_value, self.current_value
        incoming = self.incoming_connections
        nodeix = self.nodeix
        tanh = math.tanh

        # set input node activations
        cur[nodeix[0]] = 1.0 # bias
        cur[nodeix[1]] = delta_x
        cur[nodeix[2]] = delta_y
        cur[nodeix[3]] = angle
        cur[nodeix[4]] = vel_x
        cur[nodeix[5]] = vel_y
        cur[nodeix[6]] = angular_vel
        cur[nodeix[7]] = t1_angle
        cur[nodeix[8]] = t2_angle

        # forward pass
        for ix, node in enumerate(self.node_order):
            # skip input nodes
            if node < 9:
                continue

            # calculate value for this node
            value = 0.0
            # loop through all incoming connections of this node
            for inputix, weight, recur in incoming[ix]:
                # weight * node value per connection
                # if the connection is recurrent use prev val
                value += weight * (prev[inputix] if recur else cur[inputix])

            # go through activation function
            if node == 11 or node == 12:
                act_id = _THRUST_ACT
            elif node == 9 or node == 10:
                act_id = _TURN_ACT
            else:
                act_id = _HIDDEN_ACT
            activation = _activate(value, act_id)

            # write to current value
            cur[ix] = activation

        return cur[nodeix[9]], cur[nodeix[10]], cur[nodeix[11]], cur[nodeix[12]]

@njit(cache=True)
def _activate(value, act_id):
    if act_id == 0:    # tanh
        return math.tanh(value)
    elif act_id == 1:  # sigmoid
        return 1.0 / (1.0 + math.exp(-value))
    elif act_id == 2:  # leaky_relu
        return value if value > 0 else 0.01 * value
    elif act_id == 3:  # relu
        return max(0.0, value)
    else:              # softsign
        return value / (1.0 + abs(value))

@njit(cache=True)
def _forward_loop(
    node_order, connection_start, connection_end,
    src, weight, recur,
    prev, cur,
    hidden_act, turn_act, thrust_act
):
    for node_ix, node in enumerate(node_order):
        # skip input nodes
        if node < 9:
            continue

        # calculate value for this node
        value = 0.0
        # loop through all incoming connections of this node
        for conn_ix in range(connection_start[node_ix], connection_end[node_ix]):
            # value += conn weight * (previous input node value if recurrent else current input node value)
            node_value = (prev[src[conn_ix]] if recur[conn_ix] else cur[src[conn_ix]])
            value += weight[conn_ix] * node_value

        # go through activation function
        if node == 11 or node == 12:
            activation = _activate(value, thrust_act)
        elif node == 9 or node == 10:
            activation = _activate(value, turn_act)
        else:
            activation = _activate(value, hidden_act)

        # write to current value
        cur[node_ix] = activation

class NeatNN_fast:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.node_order = np.array(self.topo_sort())
        N = len(self.node_order)

        # map node id to its position in the array
        self.nodeix = {nid: ix for ix, nid in enumerate(self.node_order)}

        # precalculate all incomming connections for nodes
        # contains list of incoming connections as (input node ix, weight, recurrent)
        self.incoming_connections: list[list[tuple[int, float, bool]]] = [[] for _ in range(N)]
        for c in genome.connections:
            if c.enabled:
                _in_ix, _weight = self.nodeix[c.input], c.weight       # input ix and weight
                _recur = self.nodeix[c.output] < self.nodeix[c.input] # recursive if input ix > output ix
                _out_ix = self.nodeix[c.output]                       # output ix
                self.incoming_connections[_out_ix].append((_in_ix, _weight, _recur))

        # CSR mapping
        Nconnections = sum(1 for c in genome.connections if c.enabled)
        # connection data - basically all connections in a 2d array grouped by output node's index
        self.src = np.zeros(Nconnections, dtype=np.int32)
        self.weight = np.zeros(Nconnections, dtype=np.float64)
        self.recur = np.zeros(Nconnections, dtype=np.int32)
        # node connection slice indecies - gives the start and end position of the node's incoming connection group
        self.connection_start = np.zeros(N, dtype=np.int32)
        self.connection_end   = np.zeros(N, dtype=np.int32)

        # point to next free connection position
        pointer = 0
        for ix, node_conns in enumerate(self.incoming_connections):
            # set node start position to the pointer
            self.connection_start[ix] = pointer
            for conn in node_conns:
                _in_ix, _weight, _recur = conn
                self.src[pointer]    = _in_ix
                self.weight[pointer] = _weight
                self.recur[pointer]  = _recur
                # point to next free connection slot
                pointer += 1
            # the node end position is exlusive so this end slice pointer will still be accurate
            self.connection_end[ix] = pointer

        # node activation values
        self.previous_value = np.zeros(N, dtype=np.float64)  # {node: previous activation}
        self.current_value  = np.zeros(N, dtype=np.float64)  # {node: current activation}

    def topo_sort(self):
        nodes = [n.id for n in self.genome.nodes if n.node_type != NodeType.OUTPUT]
        outputs = set(n.id for n in self.genome.nodes if n.node_type == NodeType.OUTPUT)
        connections = [(c.input, c.output) for c in self.genome.connections if c.enabled]

        # kahn's algorithm
        # remove nodes with no inputs recursively untill empty
        # if all nodes are not removed, there is cycle and they are added at the end
        in_degree = {n: 0 for n in nodes}
        for _, output in connections:
            if output not in outputs:
                in_degree[output] += 1

        queue = [n for n in nodes if in_degree[n] == 0]
        sorted_nodes = []

        # pre map input nodes to their outputs
        outgoing = {n: [] for n in nodes}
        for input, output in connections:
            outgoing[input].append(output)

        while True:
            # once out of targets
            if not queue:
                # add remaining nodes (cycle)
                sorted_nodes += [n for n in nodes if n not in sorted_nodes]
                break

            # move node from queue to sorted
            node = queue.pop(0)
            sorted_nodes.append(node)

            # go through node outputs and decrement their in degree
            for output in outgoing[node]:
                # skip network outputs
                if output in outputs:
                    continue

                in_degree[output] -= 1
                # if new 0 degree nodes discovered, add to queue
                if in_degree[output] == 0:
                    queue.append(output)


        sorted_nodes += outputs
        return sorted_nodes

    def forward(self, delta_x, delta_y, angle, vel_x, vel_y, angular_vel, t1_angle, t2_angle):
        # buffer swap current to previous
        self.previous_value, self.current_value = self.current_value, self.previous_value  

        # bind to local var to reduce lookup in
        prev, cur = self.previous_value, self.current_value  
        nodeix = self.nodeix  

        # set input node activations
        cur[nodeix[0]] = 1.0 # bias  
        cur[nodeix[1]] = delta_x     
        cur[nodeix[2]] = delta_y     
        cur[nodeix[3]] = angle       
        cur[nodeix[4]] = vel_x       
        cur[nodeix[5]] = vel_y       
        cur[nodeix[6]] = angular_vel 
        cur[nodeix[7]] = t1_angle    
        cur[nodeix[8]] = t2_angle    

        # compiled hotloop
        _forward_loop(
            self.node_order, self.connection_start, self.connection_end,
            self.src, self.weight, self.recur,
            prev, cur,
            _HIDDEN_ACT, _TURN_ACT, _THRUST_ACT
        )

        return cur[nodeix[9]], cur[nodeix[10]], cur[nodeix[11]], cur[nodeix[12]]