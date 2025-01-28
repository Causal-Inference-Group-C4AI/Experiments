from collections import namedtuple
from causal_solver.Helper import Helper
from causal_solver.SupertailFinder import Node

latentAndCcomp = namedtuple('latentAndCcomp', ['latent', 'nodes'])
dictAndIndex   = namedtuple('dictAndIndex', ['mechanisms', 'index'])
equationsObject = namedtuple('equationsObject', ['probability', 'dictionary'])

class NonLinearConstraints:
    """
    Determines the a "supertail" T of a set of endogenous variables V associated to a set of latent variables
    U such that T and U fully determine V and T and U are indepedent. We consider that the independency hypothesis
    are already inserted in the graph and assume d-separation to be its equivalent.
    The case1 is when both the variable under intervention as well as the target variable are in the same c-component
    """
    def equationsGenerator(mechanismDictsList: list[list[dictAndIndex]], listT: list[int], listS: list[int], listU: list[int],
                            cardinalities: dict[int,int], graphNodes: list[Node], topoOrder: list[int], indexToLabel: dict[int, str],
                            filepath: str):
        """
        Given the supertail (set T), the endogenous variables S (set S) and the latent variables U (set U), it generates the equations:
        p = P(S=s|T=t) = Sum{P(U=u)*1(U=u,T=t -> S=s)}. Hence, each combination of latents has a coefficient, which
        can be 0 or 1 only.

        - mechanismDictsList : a list in which each element corresponds to an enumeration of the mechanisms (states) of a latent variable.
        - setT               : supertail
        - setS               : set of endogenous variables determined by the supertail
        - cardinalities      : dict with the cardinalities of the variables in T and S.
        - parents            : equal to nodesIn, and contains the set of all parents of an endogenous variable
        - topoOrder          : a topological order to run through S
        - indexToLabel       : convert from index of a variable in S to the label used in a CSV file
        - filepath           : path for the csv file with data.

        Returns:
        - latentCardinalities: dictionary that returns the cardinality of the latent variables in setU.
        - equations          : list of equationObjects, each with the probability (LHS) and the coefficients (0 or 1) of each
                               term of the summation in the RHS.
        """
        df = Helper.fetchCsv(filepath)

        topoOrderAux: list[int] = topoOrder.copy()
        for node in topoOrder:
            if (node not in listS) and (node in topoOrderAux):
                topoOrderAux.remove(node)

        cardinalitiesTail: dict[int, int] = {}; cardinalitiesEndo: dict[int, int] = {}
        for key in cardinalities:
            if key in listT:
                cardinalitiesTail[key] = cardinalities[key]
            elif key in listS:
                cardinalitiesEndo[key] = cardinalities[key]

        tailSpace: list[list[int]] = Helper.helperGenerateSpaces(nodes=listT, cardinalities=cardinalitiesTail)
        tailCombinations = Helper.generateCrossProducts(tailSpace)
        endoSpace = Helper.helperGenerateSpaces(nodes=listS,  cardinalities=cardinalitiesEndo)
        endoCombinations = Helper.generateCrossProducts(endoSpace)
        latentCombinations = Helper.generateCrossProducts(mechanismDictsList)

        equations: list[equationsObject] = []

        for tailRealization in tailCombinations:
            tailRealizationDict = dict(zip(listT, tailRealization))
            for endoRealization in endoCombinations:
                endoRealizationDict = dict(zip(listS, endoRealization))
                probability: float = Helper.findConditionalProbability(dataFrame=df,indexToLabel=indexToLabel,targetRealization=endoRealizationDict,
                                                  conditionRealization=tailRealizationDict)
                
                coefficientsDict: dict[str,int] = {}
                for latentRealization in latentCombinations:
                    isCompatible: bool = NonLinearConstraints.checkRealization(mechanismDictsList=latentRealization,
                                                graphNodes=graphNodes,
                                                topoOrder=topoOrderAux,
                                                tailValues=tailRealizationDict,
                                                latentVariables=listU,
                                                expectedRealizations=endoRealizationDict,
                                                )
                    indexer: str = ""
                    for latentMechanism in latentRealization:
                        indexer += f"{latentMechanism.index},"
                    indexer = indexer.rstrip(',')

                    coefficientsDict[indexer] = int(isCompatible)

                equations.append(equationsObject(probability, coefficientsDict))

        # TODO: Union of U states must have probability one

        return equations

    def objectiveFunctionBuilder(mechanismDictsList: list[list[dictAndIndex]], graphNodes: list[Node],
                                cardinalities: dict[int, int], listS: list[int], listT: list[int], listU: list[int],
                                interventionVariable: int, interventionValue, targetVariable: int,
                                targetValue: int, topoOrder: list[int], filepath: str,
                                indexToLabel: dict[int, str], verbose: bool):
        """
        Tests all cases of the tail and of the mechanisms in order to build the objective function.

        latentIntervention: latent variable which is a parent of both the intervention variable in the target variable
        cComponentIntervention: complete cComponent in which the intervention variable and the target variable are
        children: from each node as a key it returns a list with all the nodes that can be accessed from it (arrow points out of the node)
        parents: from each node as a key it returns a list with all the nodes that are parents from it
        cardinalities: from each node return the number of states
        cCompDict: from each latent as a key get the complete cComponent (all its children)
        interventionVariable: the variable under intervention - do operator.
        interventionValue: the value that the variable under intervention is fixed on
        targetVariable: the variable we want to measure the probability
        targetValue: the value assumed by such a variable considering the intervention
        topoOrder: a topological order in which we can run through the endogenous nodes (from the whole graph)
        filepath: path to the CSV file with the experiment results.
        indexToLabel: a dictionary that translates a node index to its label (which is the same as in the CSV)
        """
        dataFrame = Helper.fetchCsv(filepath)

        # Remove nodes not in S from topoOrder AND the interventionNode:
        topoOrderAux: list[int] = topoOrder.copy()
        if interventionVariable in topoOrderAux:
            topoOrderAux.remove(interventionVariable)
                
        for node in topoOrder:
            if (node not in listS) and (node in topoOrderAux):
                topoOrderAux.remove(node)

        tailVarSpaces: list[list[int]] = Helper.helperGenerateSpaces(listT, cardinalities)
                
        tailCombinations = Helper.generateCrossProducts(tailVarSpaces) # the order is the same as in list T.
        latentCombinations = Helper.generateCrossProducts(mechanismDictsList)
    
        objectiveFunction: dict[str, float] = { }
        for latentCombination in latentCombinations:
            totalProbability: float = 0.0

            indexer: str = ""
            for latentMechanism in latentCombination:                
                indexer += f"{latentMechanism.index},"
            indexer = indexer.rstrip(',')

            for tailRealization in tailCombinations:                                
                tailValues: dict[int,int] = {}
                for j, tailValue in enumerate(tailRealization):
                    tailValues[listT[j]] = tailValue

                isCompatible = NonLinearConstraints.checkRealization(mechanismDictsList=latentCombination,
                                             graphNodes=graphNodes,
                                             topoOrder=topoOrderAux,
                                             tailValues=tailValues,                                            
                                             latentVariables=listU,
                                             expectedRealizations={targetVariable: targetValue},
                                             interventionVariable=interventionVariable,
                                             interventionValue=interventionValue,
                                             )

                if isCompatible:
                    empiricalProbability = Helper.findProbability(dataFrame, indexToLabel, tailValues, False)
                    totalProbability += empiricalProbability                
            
            objectiveFunction[indexer] = totalProbability
            if verbose:            
                print(f"Checking: [{indexer}] = {totalProbability}")

        return objectiveFunction
    
    def checkRealization(mechanismDictsList: list[dict[str, int]], graphNodes: list[Node], topoOrder: list[int],
                tailValues: dict[int, int], latentVariables: list[int], expectedRealizations: dict[int, int],
                interventionVariable=-1, interventionValue=-1):
        """
        For some mechanism in the discretization of a latent variable, as well as a tuple of values for the tail, check
        if the set of deterministic functions implies the expected values for the variables that belong to the c-component.

        mechanismDictsList : list with the realization of each latent variable, in the form of mechanisms
        parents            : dictionary that return a list of the parents of a variable (including the exogenous parent)
        topoOrder          : order in which we can run through the graph without any dependency problems in the deterministic
                             functions.
        tailValues         : values taken by the variables in the c-component's tail T.
        endogenousValues   : values that should be assumed by the endogenous variables V of the c-component
        """
        computedNodes: dict[int, int] = {}
        if interventionVariable != -1:
            computedNodes[interventionVariable] = interventionValue

        for key in tailValues:
            computedNodes[key] = tailValues[key]

        for node in topoOrder:
            dictKey: str = ""
            for parentOfNode in graphNodes[node].parents:
                if parentOfNode not in latentVariables:
                    dictKey += f"{parentOfNode}={computedNodes[parentOfNode]},"
                else:
                    nodeLatentParent: int = parentOfNode

            for i, latentVar in enumerate(latentVariables):
                if latentVar == nodeLatentParent:
                    nodeLatentParentIndex: int = i
                    break

            nodeValue = mechanismDictsList[nodeLatentParentIndex].mechanisms[dictKey[:-1]]
            computedNodes[node] = nodeValue
            if (node in expectedRealizations) and (computedNodes[node] != expectedRealizations[node]):
                return False

        return True
