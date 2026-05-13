import sys
import matplotlib.pyplot as plt
import networkx as nx
import util as utils
from genome import NodeType

def draw_genome(ax, genome, title):
    incoming = {n.id: [] for n in genome.nodes}
    for c in genome.connections:
        if c.enabled:
            incoming[c.output].append(c.input)

    depth = {}
    for node in genome.nodes:
        if node.node_type == NodeType.INPUT:
            depth[node.id] = 0

    in_degree = {n.id: 0 for n in genome.nodes if n.node_type == NodeType.HIDDEN}
    outgoing  = {n.id: [] for n in genome.nodes if n.node_type == NodeType.HIDDEN}
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

    for node_id in topo_order:
        depends_depths = [0] + [depth[d] for d in incoming[node_id] if d in depth]
        depth[node_id] = max(depends_depths) + 1

    max_depth = max(depth.values()) if depth else 0
    for node in genome.nodes:
        if node.node_type == NodeType.OUTPUT:
            depth[node.id] = max_depth + 1

    G = nx.DiGraph()
    for node in genome.nodes:
        G.add_node(node.id, node_type=node.node_type, layer=depth.get(node.id, 0))
    for c in genome.connections:
        if c.enabled:
            G.add_edge(c.input, c.output, weight=c.weight)

    pos = nx.multipartite_layout(G, subset_key='layer')

    colors = []
    for node in G.nodes:
        nt = G.nodes[node]['node_type']
        if nt == NodeType.INPUT:
            colors.append('steelblue')
        elif nt == NodeType.OUTPUT:
            colors.append('tomato')
        else:
            colors.append('mediumseagreen')

    edge_colors = ['green' if G[u][v]['weight'] > 0 else 'red' for u, v in G.edges]

    nx.draw_networkx(G, pos, ax=ax, node_color=colors, edge_color=edge_colors,
                     node_size=200, arrows=True, with_labels=False, width=0.8)
    enabled = sum(1 for c in genome.connections if c.enabled)
    ax.set_title(f"{title}\n{len(genome.nodes)}n  {enabled}c", fontsize=8)
    ax.axis('off')


if __name__ == '__main__':
    if not utils.save_path.exists():
        print("No checkpoint found at", utils.save_path)
        sys.exit(1)

    state = utils.load()
    genomes = state['current_gen'][:10]
    gen = state.get('gen', '?')
    print(f"Loaded gen {gen}, showing top {len(genomes)} F1 genomes")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"F1 Top-10 Neural Nets — Gen {gen}", fontsize=13)

    for i, (ax, genome) in enumerate(zip(axes.flat, genomes)):
        draw_genome(ax, genome, f"F1[{i}]")

    # blank out unused axes if fewer than 10 genomes
    for ax in axes.flat[len(genomes):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
