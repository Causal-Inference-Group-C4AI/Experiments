import pandas as pd
from collections import namedtuple
import itertools

from causal_solver.Node import Node
from causal_solver.Supersets import Supersets

dictAndIndex = namedtuple('dictAndIndex', ['mechanisms', 'index'])

class Helper:
    """
    Common methods used to create the optimization problem
    """
    def equationsSimplifier(triples: list[Supersets], latentList: list[int], latentSetsDict: dict[int, Supersets]):
        """
        Input:
        Triples: list of (U,S,T) to be merged.
        latentList: setU of the objective function
        latentSetsDict: dictionary that receives a latent in latentList and returns an object with the setS, setT obtained from
        the supertail finder algorithm when initialized with this latent variable.
        
        Output: 
        list of merged(U,S,T).
        """
        adjNodes: dict[int, list[int]] ={}
        for latent in latentList:
            adjNodes[latent] = []

        for triple in triples:
            if len(triple.listU) > 1:
                firstLatent: int = triple.listU[0]        
                for adjLatent in triple.listU:                    
                    if (adjLatent != firstLatent) and (adjLatent not in adjNodes[firstLatent]):
                        adjNodes[firstLatent].append(adjLatent)
                        adjNodes[adjLatent].append(firstLatent)
    
        maximalConnectedComponents: list[list[int]] = []
        visited: dict[int, bool] = {}
        for latentNode in latentList:
            visited[latentNode] = False

        for latentNode in latentList:
            if not visited[latentNode]:
                connectedSuperset = Supersets(listS=[],listT=[],listU=[])
                Helper.dfsHelper(currNode=latentNode, visited=visited,connectedSuperset=connectedSuperset, adjNodes=adjNodes,
                                 latentSetsDict=latentSetsDict)
                maximalConnectedComponents.append(connectedSuperset)
    
        return maximalConnectedComponents

    def dfsHelper(currNode: int, visited: dict[int, bool], connectedSuperset: Supersets, adjNodes: dict[int, list[int]],
                  latentSetsDict: dict[int, Supersets]):        
        visited[currNode] = True
        connectedSuperset.listU.append(currNode)
        connectedSuperset.listS = list(set(connectedSuperset.listS) | set(latentSetsDict[currNode].listS))
        connectedSuperset.listT = list((set(connectedSuperset.listT) | set(latentSetsDict[currNode].listT)) - set(connectedSuperset.listS))

        for node in adjNodes[currNode]:
            if not visited[node]:
                Helper.dfsHelper(node, visited, connectedSuperset, adjNodes, latentSetsDict)

    def findConditionalProbability(dataFrame, indexToLabel, targetRealization: dict[int, int], conditionRealization: dict[int, int], v=True):
        """
        dataFrame              : pandas dataFrama that contains the data from the csv
        indexToLabel           : dictionary that converts an endogenous variable index to its label
        targetRealization      : specifies the values assumed by the endogenous variables V
        conditionalRealization : specifies the values assumed by the c-component tail T

        Calculates: P(V|T) = P(V,T) / P(T)
        """
        conditionProbability = Helper.findProbability(dataFrame, indexToLabel, conditionRealization, False)

        if conditionProbability == 0:
            return 0

        targetAndConditionProbability = Helper.findProbability(dataFrame, indexToLabel, targetRealization | conditionRealization, False)
        return targetAndConditionProbability / conditionProbability

    def findProbability(dataFrame, indexToLabel, variableRealizations: dict[int, int], v=True):
        conditions = pd.Series([True] * len(dataFrame), index=dataFrame.index)
        for variable in variableRealizations:
            conditions &= (dataFrame[indexToLabel[variable]] == variableRealizations[variable])

        compatibleCasesCount = dataFrame[conditions].shape[0]
        if v:
            print(f"Count compatible cases: {compatibleCasesCount}")
            print(f"Total cases: {dataFrame.shape[0]}")

        return compatibleCasesCount / dataFrame.shape[0]

    def helperGenerateSpaces(nodes: list[int], cardinalities: dict[int, int]):
        spaces: list[list[int]] = []
        
        for node in nodes:
            spaces.append(range(0,cardinalities[node]))
        
        return spaces
    
    def fetchCsv(filepath="balke_pearl.csv"):        
        #prefix = "/home/c4ai-wsl/projects/Canonical-Partition/causal_solver/"
        prefix = "/home/lawand/Canonical-Partition/causal_solver/"
        return pd.read_csv(prefix + filepath)

    def generateCrossProducts(sets: list[list[int]]):
        crossProductsTuples = itertools.product(*sets)
        return [list(combination) for combination in crossProductsTuples]

    
    def mechanisms_generator(latentNode: int, endogenousNodes: list[int], cardinalities: dict[int, int], graphNodes: list[Node], v=True):
        """
        Generates an enumeration (list) of all mechanism a latent value can assume in its c-component. The c-component has to have
        exactly one latent variable.

        latentNode: an identifier for the latent node of the c-component
        endogenousNodes: list of endogenous node of the c-component
        cardinalities: dictionary with the cardinalities of the endogenous nodes. The key for each node is the number that represents
        it in the endogenousNode list
        parentsDict: dictionary that has the same key as the above argument, but instead returns a list with the parents of each endogenous
        node. PS: Note that some parents may not be in the c-component, but the ones in the tail are also necessary for this function.
        v (verbose): enable or disable the logs
        """

        auxSpaces: list[list[int]] = []
        headerArray: list[str] = []
        allCasesList: list[list[list[int]]] = []
        dictKeys: list[str] = []

        for var in endogenousNodes:
            auxSpaces.clear()
            header: str = f"determines variable: {var}"
            amount: int = 1
            orderedParents: list[int] = []
            for parent in graphNodes[var].parents:
                if parent != latentNode:
                    orderedParents.append(parent)
                    header = f"{parent}, " + header
                    auxSpaces.append(range(cardinalities[parent]))
                    amount *= cardinalities[parent]

            headerArray.append(header + f" (x {amount})")
            functionDomain: list[list[int]] = [list(auxTuple) for auxTuple in itertools.product(*auxSpaces)]
            if v:
                print(functionDomain)

            imageValues: list[int] = range(cardinalities[var])

            varResult = [[domainCase + [c] for c in imageValues] for domainCase in functionDomain]
            if v:
                print(f"For variable {var}:")          
                print(f"Function domain: {functionDomain}")
                print(f"VarResult: {varResult}")

            for domainCase in functionDomain:
                key: str = ""
                for index, el in enumerate(domainCase):
                    key = key + f"{orderedParents[index]}={el},"
                dictKeys.append(key[:-1])

            allCasesList  = allCasesList + varResult
        
        if v:
            print(headerArray)
            print(f"List all possible mechanism, placing in the same array those that determine the same function:\n{allCasesList}")
            print(f"List the keys of the dictionary (all combinations of the domains of the functions): {dictKeys}")        

        allPossibleMechanisms = list(itertools.product(*allCasesList))
        mechanismDicts: list[dict[str, int]] = []
        for index, mechanism in enumerate(allPossibleMechanisms):
            if v:
                print(f"{index}) {mechanism}")
            currDict: dict[str, int] = {} 
            for domainIndex, nodeFunction in enumerate(mechanism):
                if v:
                    print(f"The node function = {nodeFunction}")
                currDict[dictKeys[domainIndex]] = nodeFunction[-1]
            
            mechanismDicts.append(currDict)

        if v:
            print("Check if the mechanism dictionary is working as expected:")
            for mechanismDict in mechanismDicts:
                for key in mechanismDict:
                    print(f"key: {key} & val: {mechanismDict[key]}")
                print("------------")
        
        return allPossibleMechanisms, dictKeys, mechanismDicts

    def mechanismListGenerator(cardinalities: dict[int, int], listU: list[int], setS: set[int], graphNodes: list[Node]):
        mechanismDictsList: list[list[dictAndIndex]] = [] # Same order as in listU
        globalIndex: int = 0
        latentCardinalities: dict[int, int] = {}
        for latentVariable in listU:
            endogenousInS: list[int] = list(set(graphNodes[latentVariable].children) & setS)
            _, _, mechanismDicts = Helper.mechanisms_generator(latentNode=latentVariable, endogenousNodes=endogenousInS,
                                            cardinalities=cardinalities,graphNodes=graphNodes,v=False)

            mechanismIndexDict: list[dictAndIndex] = []
            initVal: int = globalIndex
            for mechanismDict in mechanismDicts:
                mechanismIndexDict.append(dictAndIndex(mechanismDict, globalIndex))
                globalIndex += 1

            latentCardinalities[latentVariable] = globalIndex - initVal
            mechanismDictsList.append(mechanismIndexDict)

        return mechanismDictsList, latentCardinalities