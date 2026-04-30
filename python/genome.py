'''
This is the genome stuff for the Neat NN
'''

from dataclasses import dataclass
from enum import Enum
import numpy as np

class NodeType(Enum):
    INPUT  = 0
    HIDDEN = 1
    OUTPUT = 2

@dataclass
class NodeGene:
    id: int
    node_type: NodeType

@dataclass
class ConnectionGene:
    innovation: int
    weight: float
    input: int
    output: int
    enabled: bool

# 9 input nodes:  bias node, delta x, delta y, drone angle, drone vel x, drone vel y, dronee angular velocity, thruster1 angle, thruster 2 angle
# 4 output nodes: thruster 1 turn signal, thruster 2 turn signale, thruster 1 throttle, thruster 2 throttle
# innovation counter init at 13

@dataclass(eq=False)
class Genome:
    connections: list[ConnectionGene]
    nodes: list[NodeGene]
    mutation_power: float = 0.3  # self-adaptive weight mutation magnitude
    _conn_cache: dict | None = None  # lazily built {innovation: ConnectionGene}, invalidated on structural change
    _species_id: int | None = None   # ID of species this genome was assigned to last gen

    @property
    def conn_dict(self) -> dict:
        """Cached {innovation: ConnectionGene}. Rebuilt only when connections list changes."""
        if self._conn_cache is None:
            self._conn_cache = {c.innovation: c for c in self.connections}
        return self._conn_cache

    def invalidate_cache(self):
        self._conn_cache = None

    @classmethod
    def new(cls):
        # pre populate input and output nodes
        nodes = [NodeGene(i, NodeType.INPUT) for i in range(9)]
        nodes += [NodeGene(i, NodeType.OUTPUT) for i in range(9, 13)]
        return cls(connections=[], nodes=nodes)

    def base_connections(self, innovations):
        # for a in range(9):
        #     for b in range(9, 13):
        #         innovation = innovations.resolve((a, b))
        #         self.connections.append(create_connection(innovation, (a, b)))
        a = int(np.random.randint(9))
        b = int(np.random.randint(9, 13))
        innovation = innovations.resolve((a, b))
        self.connections.append(create_connection(innovation, (a, b)))

def create_connection(innovation, pair, weight=None):
    return ConnectionGene(innovation=innovation,
                          weight=weight if weight else np.random.uniform(-1, 1),
                          input=pair[0], output=pair[1],
                          enabled=True)
