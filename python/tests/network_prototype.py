from genome_prototype import Genome, NodeType, ConnectionGene
import numpy as np
import math

# class NeatNN:
#     def __init__(self, genome: Genome):
#         self.genome = genome
#         self.node_order = self.topo_sort()

#         # precalculate all incomming connections for nodes
#         self.incoming_connections: dict[int, list[ConnectionGene]] = {n.id: [] for n in genome.nodes}
#         for c in genome.connections:
#             if c.enabled:
#                 self.incoming_connections[c.output].append(c)

#         # precalculate recurrent connections (loops)
#         self.recurrent = set()
#         index = {n: ix for ix, n in enumerate(self.node_order)}
#         for c in genome.connections:
#             # add connection id to recurrent if output comes before input in node order
#             if c.enabled and index[c.output] < index[c.input]:
#                 self.recurrent.add(c.innovation)

#         # node activation values
#         self.previous_value = {n.id: 0.0 for n in genome.nodes} # {node: previous activation}
#         self.current_value = {n.id: 0.0 for n in genome.nodes}  # {node: current activation}

#     def topo_sort(self):
#         nodes = [n.id for n in self.genome.nodes if n.node_type != NodeType.OUTPUT]
#         outputs = set(n.id for n in self.genome.nodes if n.node_type == NodeType.OUTPUT)
#         connections = [(c.input, c.output) for c in self.genome.connections if c.enabled]

#         # kahn's algorithm
#         # remove nodes with no inputs recursively untill empty
#         # if all nodes are not removed, there is cycle and they are added at the end
#         in_degree = {n: 0 for n in nodes}
#         for _, output in connections:
#             if output not in outputs:
#                 in_degree[output] += 1

#         queue = [n for n in nodes if in_degree[n] == 0]
#         sorted_nodes = []

#         # pre map input nodes to their outputs
#         outgoing = {n: [] for n in nodes}
#         for input, output in connections:
#             outgoing[input].append(output)

#         while True:
#             # once out of targets
#             if not queue:
#                 # add remaining nodes (cycle)
#                 sorted_nodes += [n for n in nodes if n not in sorted_nodes]
#                 break

#             # move node from queue to sorted
#             node = queue.pop(0)
#             sorted_nodes.append(node)

#             # go through node outputs and decrement their in degree
#             for output in outgoing[node]:
#                 # skip network outputs
#                 if output in outputs:
#                     continue

#                 in_degree[output] -= 1
#                 # if new 0 degree nodes discovered, add to queue
#                 if in_degree[output] == 0:
#                     queue.append(output)


#         sorted_nodes += outputs
#         return sorted_nodes

#     def forward(self, delta_x, delta_y, angle, vel_x, vel_y, angular_vel, t1_angle, t2_angle):
#         # buffer swap current to previous
#         self.previous_value, self.current_value = self.current_value, self.previous_value

#         # set input node activations
#         self.current_value[0] = 1.0 # bias
#         self.current_value[1] = delta_x
#         self.current_value[2] = delta_y
#         self.current_value[3] = angle
#         self.current_value[4] = vel_x
#         self.current_value[5] = vel_y
#         self.current_value[6] = angular_vel
#         self.current_value[7] = t1_angle
#         self.current_value[8] = t2_angle

#         # forward pass
#         for node in self.node_order:
#             # skip input nodes
#             if node < 9:
#                 continue

#             # calculate value for this node
#             value = 0.0
#             # loop through all incoming connections of this node
#             for c in self.incoming_connections[node]:
#                 # if the connection is recurrent use prev val
#                 if c.innovation in self.recurrent:
#                     value += self.previous_value[c.input] * c.weight
#                 # else use current
#                 else:
#                     value += self.current_value[c.input] * c.weight

#             # go through activation function
#             activation = math.tanh(value)

#             # write to current value
#             self.current_value[node] = activation

#         return self.current_value[9], self.current_value[10], self.current_value[11], self.current_value[12]


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
            activation = tanh(value)

            # write to current value
            cur[ix] = activation

        return cur[nodeix[9]], cur[nodeix[10]], cur[nodeix[11]], cur[nodeix[12]]


