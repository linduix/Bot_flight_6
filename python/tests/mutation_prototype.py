class Innovations:
    def __init__(self) -> None:
        # 0-12 reserverd fron input/output nodes
        self.counter = 13
        self.dict = {}

    def resolve(self, connectedNodes: tuple):
        a, b = connectedNodes
        if not isinstance(a, int) or not isinstance(b, int):
            raise TypeError("Wrong input type, need (int, int)")

        value = self.dict.get(connectedNodes)
        if value is None:
            value = self.counter
            self.counter += 1
            self.dict[connectedNodes] = value

        return value
