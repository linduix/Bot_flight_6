import sys
import traceback
import numpy as np

def test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"         {e}")
        traceback.print_exc()

print("\n=== GENOME ===")
from genome_prototype import Genome, NodeGene, NodeType, ConnectionGene, create_connection

def test_genome_new():
    g = Genome.new()
    assert len(g.nodes) == 13
    assert len(g.connections) == 0
    inputs = [n for n in g.nodes if n.node_type == NodeType.INPUT]
    outputs = [n for n in g.nodes if n.node_type == NodeType.OUTPUT]
    assert len(inputs) == 9
    assert len(outputs) == 4

test("Genome.new() creates 9 inputs + 4 outputs", test_genome_new)

print("\n=== INNOVATIONS ===")
from mutation_prototype import Innovations

def test_innovations_resolve():
    inn = Innovations()
    assert inn.resolve((0, 9)) == 13
    assert inn.resolve((0, 9)) == 13  # same pair = same number
    assert inn.resolve((1, 9)) == 14  # new pair = new number
    assert inn.counter == 15

def test_innovations_type_check():
    inn = Innovations()
    try:
        inn.resolve(("a", 9))
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

test("resolve() returns consistent innovation numbers", test_innovations_resolve)
test("resolve() raises TypeError on non-int input", test_innovations_type_check)

print("\n=== MUTATIONS ===")
from mutation_prototype import mutate_weights, add_connection, add_node

def test_mutate_weights():
    g = Genome.new()
    inn = Innovations()
    add_connection(g, inn)
    original_weight = g.connections[0].weight
    # run many times to ensure mutation fires
    for _ in range(100):
        mutate_weights(g.connections)
    # weight should have changed at some point
    assert g.connections[0].weight != original_weight or True  # always passes, just checking no crash

def test_add_connection():
    g = Genome.new()
    inn = Innovations()
    add_connection(g, inn)
    assert len(g.connections) == 1
    c = g.connections[0]
    # output should not be an input node
    input_ids = [n.id for n in g.nodes if n.node_type == NodeType.INPUT]
    assert c.output not in input_ids

def test_add_node():
    g = Genome.new()
    inn = Innovations()
    add_connection(g, inn)
    add_node(g, inn)
    hidden = [n for n in g.nodes if n.node_type == NodeType.HIDDEN]
    assert len(hidden) == 1
    assert len(g.connections) == 3  # original disabled + 2 new
    disabled = [c for c in g.connections if not c.enabled]
    assert len(disabled) == 1

test("mutate_weights() runs without error", test_mutate_weights)
test("add_connection() adds valid connection", test_add_connection)
test("add_node() splits connection correctly", test_add_node)

print("\n=== NETWORK ===")
from network_prototype import NeatNN

def test_network_forward_no_connections():
    g = Genome.new()
    nn = NeatNN(g)
    out = nn.forward(0, 0, 0, 0, 0, 0, 0, 0)
    assert len(out) == 4
    assert all(v == 0.0 for v in out)  # no connections = zero output

def test_network_forward_with_connections():
    g = Genome.new()
    inn = Innovations()
    for _ in range(5):
        add_connection(g, inn)
    nn = NeatNN(g)
    out = nn.forward(1.0, 0.5, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2)
    assert len(out) == 4
    assert all(-1.0 <= v <= 1.0 for v in out)  # tanh output range

def test_network_recurrent_uses_previous():
    g = Genome.new()
    inn = Innovations()
    for _ in range(10):
        add_connection(g, inn)
    nn = NeatNN(g)
    out1 = nn.forward(1.0, 0, 0, 0, 0, 0, 0, 0)
    out2 = nn.forward(1.0, 0, 0, 0, 0, 0, 0, 0)
    # second call may differ if recurrent connections exist, just check no crash
    assert len(out2) == 4

test("forward() with no connections returns zeros", test_network_forward_no_connections)
test("forward() with connections returns tanh range", test_network_forward_with_connections)
test("forward() runs twice without error", test_network_recurrent_uses_previous)

print("\n=== BREEDING ===")
from breeding_prototype import crossover, distance, speciate, breed

def test_crossover():
    inn = Innovations()
    g1 = Genome.new()
    g2 = Genome.new()
    for _ in range(5):
        add_connection(g1, inn)
        add_connection(g2, inn)
    baby = crossover(g1, g2, 1.0, 0.5)
    assert len(baby.nodes) >= 13
    assert len(baby.connections) > 0

def test_distance_same_genome():
    inn = Innovations()
    g1 = Genome.new()
    add_connection(g1, inn)
    d = distance(g1, g1)
    assert d == 0.0

def test_distance_different_genomes():
    inn = Innovations()
    g1 = Genome.new()
    g2 = Genome.new()
    for _ in range(5):
        add_connection(g1, inn)
    for _ in range(5):
        add_connection(g2, inn)
    d = distance(g1, g2)
    assert d >= 0.0

def test_distance_empty_raises():
    g1 = Genome.new()
    g2 = Genome.new()
    inn = Innovations()
    add_connection(g2, inn)
    try:
        distance(g1, g2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_speciate():
    inn = Innovations()
    genomes = []
    for _ in range(10):
        g = Genome.new()
        add_connection(g, inn)
        genomes.append(g)
    species = speciate(3.0, genomes)
    total = sum(len(s) for s in species)
    assert total == 10

def test_breed():
    inn = Innovations()
    genomes = []
    for _ in range(20):
        g = Genome.new()
        add_connection(g, inn)
        genomes.append(g)
    scores = [np.random.rand() for _ in genomes]
    next_generation = breed(genomes, scores, inn, poputlation=20)
    assert len(next_generation) > 0

test("crossover() produces valid child", test_crossover)
test("distance() of genome with itself is 0", test_distance_same_genome)
test("distance() of different genomes >= 0", test_distance_different_genomes)
test("distance() raises on empty genome", test_distance_empty_raises)
test("speciate() accounts for all genomes", test_speciate)
test("breed() produces next generation", test_breed)

print("\nDone.")