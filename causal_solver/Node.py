class Node:
    children    : list[int]
    parents     : list[int]
    latentParent: int

    def __init__(self, children, parents, latentParent):
        self.children = children
        self.parents = parents
        self.latentParent = latentParent