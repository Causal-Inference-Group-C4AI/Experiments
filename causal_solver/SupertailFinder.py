from causal_solver.Node import Node

class SupertailFinder:
    """
    Finds the supertail T given a DAG and an intervention (target and intervention variables).
    """
    def findSuperTail(interventionNode: int, targetNode: int, graphNodes: list[Node]):
        """
        Given a graph, the node under intervention X and the target node Y, it computes the supertail T and the set of latent
        variables U such that T is independent of U and X.
        """
        setS: set[int] = set([targetNode])
        setU: set[int] = set([graphNodes[targetNode].latentParent])
        setT: set[int] = set() 

        for node in setS:
            for parentNode in graphNodes[node].parents:
                if (parentNode not in setU) and (parentNode not in setS) and (parentNode != interventionNode):
                    setT.add(parentNode)

        independencyCondition = False
        while not independencyCondition:
            independencyCondition = True
            for supertailVar in setT:
                if not SupertailFinder.areDSeparated(node=supertailVar, targetNodes=(setU|set([interventionNode])), graphNodes=graphNodes):
                    independencyCondition = False
                    setT.remove(supertailVar)
                    setU.add(graphNodes[supertailVar].latentParent)
                    setS.add(supertailVar)
                    setT.update(set(graphNodes[supertailVar].parents) - (setS | setU | set([interventionNode])))    
                    break
                
        return setS, setT, setU
    
    def findSuperTailLatent(latentNode: int, graphNodes: list[Node], completeSetS: set[int]):
        """
        Given a graph, it determines the minimal U,S,T such that latentNode is in U and T is independent of U.
        latentNode: latentNode which should be included in U
        graphNodes: the graph - all nodes parents and children
        completeSetS: variables in the setS of the objective function plus the intervention variable
        """
        setU: set[int] = set([latentNode])
        setS: set[int] = set(graphNodes[latentNode].children) & completeSetS
        setT: set[int] = set()

        for node in setS:
            for parentNode in graphNodes[node].parents:
                if (parentNode not in setU) and (parentNode not in setS):
                    setT.add(parentNode)

        independencyCondition = False
        while not independencyCondition:
            independencyCondition = True
            for supertailVar in setT:
                if not SupertailFinder.areDSeparated(node=supertailVar, targetNodes=(setU), graphNodes=graphNodes):
                    independencyCondition = False
                    setT.remove(supertailVar)
                    setU.add(graphNodes[supertailVar].latentParent)
                    setS.add(supertailVar)
                    setT.update(set(graphNodes[supertailVar].parents) - (setS | setU))    
                    break
                
        return list(setS), list(setT), list(setU)

    def areDSeparated(node: int, targetNodes: set[int], graphNodes: list[Node]):
        """
        checks if a node is d-separated from a set of targetNodes in a given a DAG. Returns true if there is independency
        between then node and any in the set.            
        """
        visited: set[int] = set()
        SupertailFinder.customDfs(node, graphNodes, visited, 2)

        return 1 if len(visited.intersection(targetNodes)) == 0 else 0

    def customDfs(currNode: int, graphNodes: list[Node], visited: set[int], lastDirection: int):
        """
        lastDirection: 0 means it entered the node, 1 means it left the node
        """
        visited.add(currNode)
        
        for node in graphNodes[currNode].children:
            if node not in visited:
                SupertailFinder.customDfs(node, graphNodes, visited, 0)                

        if lastDirection != 0:
            for node in (graphNodes[currNode].parents):
                if node not in visited:
                    SupertailFinder.customDfs(node, graphNodes, visited, 1)

        return False

def testDseparation():
    graphNodes = [Node(children=[], latentParent=4, parents=[1, 2, 4]),
                  Node(children=[0, 2], latentParent=4, parents=[4]),
                  Node(children=[0], latentParent=5, parents=[1, 3, 5]),
                  Node(children=[2], latentParent=6, parents=[6]),
                  Node(children=[0, 1], latentParent=[], parents=[]),
                  Node(children=[2], latentParent=[], parents=[]),
                  Node(children=[3], latentParent=[], parents=[])
                  ]
    
    tests: list[bool] = [SupertailFinder.areDSeparated(node=1, targetNodes={2}, graphNodes=graphNodes),
                        SupertailFinder.areDSeparated(node=2, targetNodes={4}, graphNodes=graphNodes),
                        SupertailFinder.areDSeparated(node=5, targetNodes={4}, graphNodes=graphNodes),
                        SupertailFinder.areDSeparated(node=6, targetNodes={4}, graphNodes=graphNodes),
                        SupertailFinder.areDSeparated(node=3, targetNodes={4}, graphNodes=graphNodes),
                        SupertailFinder.areDSeparated(node=1, targetNodes={4}, graphNodes=graphNodes),
                        SupertailFinder.areDSeparated(node=6, targetNodes={2}, graphNodes=graphNodes)
                        ]
    expectedResults = [0, 0, 1, 1, 1, 0, 0]    

    for i in range(len(tests)):
        if tests[i] != expectedResults[i]:
            print(f"ASSERTION ERROR IN TEST {i + 1}")

def testSupertailFinder():
    graphNodes: list[Node] = [
        Node(children=[1],parents=[4],latentParent=4),
        Node(children=[2],parents=[0,4],latentParent=4),
        Node(children=[3],parents=[1,5],latentParent=5),
        Node(children=[],parents=[2,5],latentParent=5),
        Node(children=[0,1],parents=[],latentParent=None),
        Node(children=[2,3],parents=[],latentParent=None)
    ]
    
    setS, setT, setU = SupertailFinder.findSuperTail(1, 3, graphNodes)
    result = [setS, setT, setU]
    print(f"Obtained result: {result}")
    expectedResult = [{2,3}, set(), {5}]
    cases: str = "STU"

    for i in range(len(expectedResult)):
        check = True
        set1: set[int] = result[i].copy()
        for element in expectedResult[i]:
            if element not in set1:
                check = False
                break

            set1.remove(element)
        
        if (not check) or (len(set1) > 0):
            print(f"Error in set{cases[i]} ")


def main():
    testDseparation()
    testSupertailFinder()

if __name__ == "__main__":
    main()
