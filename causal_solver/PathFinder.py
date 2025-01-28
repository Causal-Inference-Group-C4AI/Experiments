from partition_methods.relaxed_problem.python.graph import Graph

def inCcomponent(graph : Graph, node : int):
     for cComponent in graph.dagComponents:
          if node in cComponent:
               return graph.dagComponents.index(cComponent)
     return None


def findPath_it(nodes : list[int], graph : Graph, possiblePath : set[tuple] = set()):
     """
     Find the c-components path between nodes
     """
    # dest | do(src) 
     src : int = nodes[0]
     dest : int = nodes[1]

     if not(dest in graph.currNodes):
          graph.currNodes.append(dest)

          path : set[tuple] = set()
          possiblePath.add((inCcomponent(node=dest, graph= graph),dest))
          if dest == src :
               return  possiblePath

          if not(graph.parents[dest]):
               return {}

          for parent in graph.parents[dest]:

               path.update(findPath_it(nodes= [src,parent], graph= graph, possiblePath= possiblePath.copy()))

          return path
     return possiblePath

def findPath(nodes : list[str], graph : Graph):
     src = graph.labelToIndex[nodes[0]]
     dest = graph.labelToIndex[nodes[1]]
     graph.currNodes.clear()
     path = findPath_it(nodes=[src, dest], graph=graph)
     graph.currNodes.clear()
     componentsPath : list[int] = []
     nodesPath : list[int] = []

     for pair in path:
        if not(pair[0] in componentsPath):
          componentsPath.append(pair[0])

        if not(pair[1] in nodesPath): 
          nodesPath.append(graph.indexToLabel[pair[1]])
     

     return componentsPath, nodesPath

if __name__ == "__main__":
    graph: Graph = Graph.parse()
    graph.find_cComponents()
    nodes= ["Y", "Y"]
    componentsPath, nodesPath = findPath(nodes= nodes, graph= graph)
    
    print(f"Path of c-components: {componentsPath} \n")
    print(f"Path of nodes: {nodesPath}")