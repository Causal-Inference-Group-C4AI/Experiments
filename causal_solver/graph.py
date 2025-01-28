import networkx as nx
from causal_solver.SupertailFinder import Node

class Graph:
    def __init__(self, numberOfNodes: int, currNodes: list[int], visited: list[bool], cardinalities: dict[int, int],
                 parents: list[list[int]], adj: list[list[int]], labelToIndex: dict[str, int], indexToLabel: dict[int, str],
                 dagComponents: list[list[int]], exogenous : list[int], endogenous : list[int], topologicalOrder: list[int],
                 DAG: nx.digraph, cComponentToUnob: dict[int, int], graphNodes: list[Node]):#, interventions: list[(str, int)]):

        self.numberOfNodes = numberOfNodes
        self.currNodes = currNodes
        self.visited = visited
        self.cardinalities = cardinalities
        self.parents = parents
        self.adj = adj
        self.labelToIndex = labelToIndex
        self.indexToLabel = indexToLabel
        self.dagComponents = dagComponents
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.topologicalOrder = topologicalOrder
        self.DAG = DAG
        self.cComponentToUnob = cComponentToUnob
        self.graphNodes = graphNodes
        # self.interventions = interventions

    def parseTerminal():
        numberOfNodes = int(input())
        numberOfEdges = int(input())

        labelToIndex: dict[str, int] = {}; indexToLabel: dict[int, str] = {}
        adj: list[list[int]] = [[] for _ in range(numberOfNodes)]
        cardinalities: dict[int, int] = {}
        parents: list[list[int]] = [[] for _ in range(numberOfNodes)]

        for i in range(numberOfNodes):
            label, cardinality = input().split()
            cardinality = int(cardinality)
            labelToIndex[label] = i
            indexToLabel[i] = label
            cardinalities[i] = cardinality

        for _ in range(numberOfEdges):
            u, v = input().split()
            uIndex = labelToIndex[u]
            vIndex = labelToIndex[v]
            adj[uIndex].append(vIndex)
            parents[vIndex].append(uIndex)

        return numberOfNodes, labelToIndex, indexToLabel, adj, cardinalities, parents

    def parseInterface(nodesString: str, edgesString: str):
        nodesAndCardinalitiesList: list[str] = nodesString.split(',')
        numberOfNodes = len(nodesAndCardinalitiesList)

        cardinalities: dict[int, int] = {}; labelToIndex: dict[str, int] = {}; indexToLabel: dict[int, str] = {}
        adj    : list[list[int]] = [[] for _ in range(numberOfNodes)]
        parents: list[list[int]] = [[] for _ in range(numberOfNodes)]

        for i, element in enumerate(nodesAndCardinalitiesList):
            auxPair = element.split('=')
            cardinalities[i] = auxPair[1]
            labelToIndex[auxPair[0]] = i
            indexToLabel[i] = auxPair[0]
            cardinalities[i] = int(auxPair[1])

        for element in edgesString.split(','):            
            elAux = element.split('->')            
            fatherIndex = labelToIndex[elAux[0]] 
            sonIndex = labelToIndex[elAux[1]]
            adj[fatherIndex].append(sonIndex)
            parents[sonIndex].append(fatherIndex)

        return numberOfNodes, labelToIndex, indexToLabel, adj, cardinalities, parents

    def parseFile(file_path: str):
        with open(file_path, 'r') as file:

            numberOfNodes = file.readline().strip()
            numberOfEdges = file.readline().strip()
            # numberOfInterventions = file.readline().strip()

            numberOfNodes = int(numberOfNodes)
            numberOfEdges = int(numberOfEdges)

            labelToIndex: dict[str, int] = {}; indexToLabel: dict[int, str] = {}
            adj: list[list[int]] = [[] for _ in range(numberOfNodes)]
            cardinalities: dict[int, int] = {}
            parents: list[list[int]] = [[] for _ in range(numberOfNodes)]

            for i in range(numberOfNodes):
                label, cardinality = file.readline().strip().split()
                cardinality = int(cardinality)
                labelToIndex[label] = i
                indexToLabel[i] = label
                cardinalities[i] = cardinality

            for _ in range(numberOfEdges):
                u, v = file.readline().strip().split()
                uIndex = labelToIndex[u]
                vIndex = labelToIndex[v]
                adj[uIndex].append(vIndex)
                parents[vIndex].append(uIndex)

            # interventions = []
            # for _ in range(numberOfInterventions):
            #     intervene, value = file.readline().strip().split()
            #     interventions.append((intervene, value))

            return numberOfNodes, labelToIndex, indexToLabel, adj, cardinalities, parents#, interventions

    def parse(fromInterface=False, file_path="/home/lawand/Canonical-Partition/test_cases/inputs/balke_pearl.txt", nodesString="", edgesString=""):
        # if fromInterface:            
        #     auxTuple = Graph.parseInterface(nodesString, edgesString)
        # else:
        #     auxTuple = Graph.parseTerminal()

        auxTuple = Graph.parseFile(file_path)

        numberOfNodes, labelToIndex, indexToLabel, adj, cardinalities, parents = auxTuple

        inpDAG: nx.DiGraph = nx.DiGraph()
        for i in range(numberOfNodes):
            inpDAG.add_node(i)
    
        for parent, edge in enumerate(adj):
            if bool(edge):
               for ch in edge:
                   inpDAG.add_edge(parent, ch)
        
        order = list(nx.topological_sort(inpDAG))
        
        for i in range(numberOfNodes) :
        
             name_node = indexToLabel[i] 

             nx.relabel_nodes(inpDAG, {i : name_node}, copy=False)

        endogenIndex : list[int] = []; exogenIndex : list[int] = []
        for i in range(numberOfNodes):
           if not (bool(parents[i])):
               exogenIndex.append(i)
           else:
               endogenIndex.append(i)                

        graphNodes: list[Node] = [Node(latentParent=-1, parents=[], children=[]) for _ in range(numberOfNodes)]
        for node in range(numberOfNodes):
            if cardinalities[node] == 0:
                graphNodes[node] = Node(children=adj[node],parents=[],latentParent=None)
            else:
                latentParent = -1
                for nodeParent in parents[node]:
                    if cardinalities[nodeParent] == 0:
                        latentParent = nodeParent
                        break
            
                if latentParent == -1:
                    print(f"PARSE ERROR: ALL OBSERVABLE VARIABLES SHOULD HAVE A LATENT PARENT, BUT {node} DOES NOT.")
                
                graphNodes[node] = Node(children=adj[node],parents=parents[node],latentParent=latentParent)
            pass

        return Graph(numberOfNodes=numberOfNodes,currNodes=[], visited=[False] * (numberOfNodes), cardinalities=cardinalities, parents=parents,
                    adj=adj, indexToLabel=indexToLabel, labelToIndex=labelToIndex, dagComponents=[], exogenous= exogenIndex,endogenous = endogenIndex, topologicalOrder= order, DAG= inpDAG,
                    cComponentToUnob = {}, graphNodes=graphNodes)
    
    def dfs(self, node: int):        
        self.visited[node] = True
        self.currNodes.append(node)
        is_observable = self.cardinalities[node] > 1

        if not is_observable:
            for adj_node in self.adj[node]:
                if not self.visited[adj_node]:
                   self.dfs(adj_node)
        else:
            for parent_node in self.parents[node]:
                if not self.visited[parent_node] and self.cardinalities[parent_node] < 1:
                    self.dfs(parent_node)
    
    def find_cComponents(self):
        for i in range(1, self.numberOfNodes + 1):
            if not self.visited[i] and self.cardinalities[i] < 1:            
                self.currNodes.clear()
                self.dfs(i)
                self.dagComponents.append(self.currNodes[:])
                self.cComponentToUnob[len(self.dagComponents) - 1] = i
if __name__ == "__main__":
    graph: Graph = Graph.parse()
