from genome_prototype import Genome, NodeType
from mutation_prototype import Innovations
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing as mp
import pickle

save_path = Path(__file__).parent.parent.parent / "data" / "checkpoints" / "prototype_save.pkl"
def save(state: dict):
    with open(save_path, 'wb') as f:
        print('saving file')
        pickle.dump(state, f)

def load():
    with open(save_path, 'rb') as f:
        print('loading file')
        state = pickle.load(f)
    return state

def viz_process(q: mp.Queue):
    plt.ion()
    while True:
        try:
            genome = q.get(timeout=0.1)
        except:
            plt.pause(0.001)
            continue

        if genome is None:
            break

        # build incoming connection map
        incoming = {n.id: [] for n in genome.nodes}
        for c in genome.connections:
            if c.enabled:
                incoming[c.output].append(c.input)

        # calculate depth of each node
        depth = {}

        # input nodes at depth 0
        for node in genome.nodes:
            if node.node_type == NodeType.INPUT:
                depth[node.id] = 0

        # build topo order for hidden nodes
        in_degree = {n.id: 0 for n in genome.nodes if n.node_type == NodeType.HIDDEN}
        outgoing = {n.id: [] for n in genome.nodes if n.node_type == NodeType.HIDDEN}
        for c in genome.connections:
            if c.enabled and c.output in in_degree and c.input in in_degree:
                in_degree[c.output] += 1
                outgoing[c.input].append(c.output)

        bfs_queue = [n for n in in_degree if in_degree[n] == 0]
        topo_order = []
        while bfs_queue:
            node = bfs_queue.pop(0)
            topo_order.append(node)
            for out in outgoing[node]:
                in_degree[out] -= 1
                if in_degree[out] == 0:
                    bfs_queue.append(out)
        topo_order += [n for n in in_degree if n not in topo_order]

        # assign depth in topo order
        for node_id in topo_order:
            depends_depths = [0]
            for dep_id in incoming[node_id]:
                if dep_id in depth:
                    depends_depths.append(depth[dep_id])
            depth[node_id] = max(depends_depths) + 1

        # output nodes at max depth + 1
        max_depth = max(depth.values()) if depth else 0
        for node in genome.nodes:
            if node.node_type == NodeType.OUTPUT:
                depth[node.id] = max_depth + 1

        G = nx.DiGraph()
        for node in genome.nodes:
            G.add_node(node.id, node_type=node.node_type, layer=depth[node.id])

        for c in genome.connections:
            if c.enabled:
                G.add_edge(c.input, c.output, weight=c.weight)

        pos = nx.multipartite_layout(G, subset_key='layer')

        colors = []
        for node in G.nodes:
            node_type = G.nodes[node]['node_type']
            if node_type == NodeType.INPUT:
                colors.append('steelblue')
            elif node_type == NodeType.OUTPUT:
                colors.append('tomato')
            else:
                colors.append('mediumseagreen')

        edge_colors = ['green' if G[u][v]['weight'] > 0 else 'red' for u, v in G.edges]

        plt.clf()
        nx.draw_networkx(G, pos, node_color=colors, edge_color=edge_colors,
                node_size=500, arrows=True)
        plt.title(f'Genome — {len(genome.nodes)} nodes, {len(genome.connections)} connections')
        plt.pause(0.001)
