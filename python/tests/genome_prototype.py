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

    @classmethod
    def new(cls):
        # pre populate input and output nodes
        nodes = [NodeGene(i, NodeType.INPUT) for i in range(9)]
        nodes += [NodeGene(i, NodeType.OUTPUT) for i in range(9, 13)]
        return cls(connections=[], nodes=nodes)

def create_connection(innovation, pair):
    return ConnectionGene(innovation=innovation,
                          weight=np.random.uniform(-1, 1),
                          input=pair[0], output=pair[1],
                          enabled=True)
