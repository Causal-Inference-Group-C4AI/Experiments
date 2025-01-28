from partition_methods.relaxed_problem.python.graph import Graph
from scipy.optimize import linprog
import pandas as pd
from causal_solver.Helper import helper

class LinearSolver:    
    def checkValues(mechanismDict: dict[str, int], parents: dict[int, list[int]], topoOrder: list[int],
                 tailValues: dict[int, int], endogenousValues: dict[int, int], v=True):
        """
        For some mechanism in the discretization of a latent variable, as well as a tuple of values for the tail, check
        if the set of deterministic functions implies the expected values for the variables that belong to the c-component.

        mechanismDict    : dictionary that returns the value assumed by a node given (the key) the values of its parents
        parents          : dictionary that return a list of the parents of a variable
        topoOrder        : order in which we can run through the c-component without any dependency problems in the deterministic
                          functions.
        tailValues       : values taken by the variables in the c-component's tail T.
        endogenousValues : values that should be assumed by the endogenous variables V of the c-component
        """

        if v:
            print("Debug tailValues in dfs")        
            for key in tailValues:
                print(f"key: {key} - value: {tailValues[key]}")

        isValid: bool = True
        for node in topoOrder:
            dictKey: str = ""
            if v:
                print(f"Check node: {node}")
            for parentOfNode in parents[node]:                
                if v:
                    print(f"Parent node: {parentOfNode}")
                    print(f"Assumes value {tailValues[parentOfNode]}")
                dictKey += f"{parentOfNode}={tailValues[parentOfNode]},"
            
            if v:
                print(f"key: {dictKey}")
            nodeValue = mechanismDict[dictKey[:-1]] # exclude an extra comma
            tailValues[node] = nodeValue
            if nodeValue != endogenousValues[node]:
                isValid = False
                break
        
        if isValid:
            return 1
        else:
            return 0
    
    def equations_generator(mechanismDicts: list[dict[str, int]], tail: list[int], cardinalitiesTail: dict[int,int], endoVars: list[int],
                            cardinalitiesEndo: dict[int,int], endoParents: dict[int, list[int]], topoOrder: list[int], endoIndexToLabel: dict[int, str], filepath: str, 
                            precision: int, v: True):
        """
        Generate the system of equations that represents the constraints over a c-component, supposing that only the latent 
        variable states are used as variables in the optimization problem.
        1) Create an array with the spaces of each variable (belonging to the c-component or to the tail)
        2) Cross product on it to generate an enumeration of all cases
        3) Each element that results from the cross product represents one equation in the optimization system. In order
        to fill the line with the coefficients, which will all be 0 or 1, we need to check if each mechanism implies the 
        expected output given the tail.
        4) For the LHS of the linear system we need the empirical probabilities, which will come from the operations
        on the data frame. Use that P(X|Y) = P(X,Y) / P(Y) = #El(X=x,Y=y) / #El(Y=y)
        """
        
        
        # variablesOrder = tail + endoVars
        df = helper.fetchCsv(filepath)
        
        tailSpace = helper.helperGenerateSpaces(tail, cardinalitiesTail)
        endoSpace = helper.helperGenerateSpaces(endoVars, cardinalitiesEndo)
        combinationOfSpaces = helper.generateCrossProducts(tailSpace + endoSpace)

        if v:
            for i, case in enumerate(combinationOfSpaces):
                print(f"{i}) {case}")

        matrix: list[list[int]] = [] # 0s and 1s matrix        
        probabilities: list[float] = []
        
        for combination in combinationOfSpaces:
            print(f"Combination (case): {combination}")
            
            # Generate endoValues and tailValues based on combination + endoVars + tail

            # Tail values:
            tailValues: dict[int, int] = {}
            for index in range(len(tail)):
                tailValues[tail[index]] = combination[index]
            
            # endoValues:
            endoValues: dict[int, int] = {}
            for index in range(len(endoVars)):
                endoValues[endoVars[index]] = combination[index + len(tail)]
                        
            probability: float = helper.findConditionalProbability(df, endoIndexToLabel,
                                                                         endoValues, tailValues, False)            
            probability = round(probability * pow(10, precision)) / pow(10, precision)

            probabilities.append(probability)
            # combination order = tail and then the expected values.            
            systemCoefficients: list[int] = []
            conditionalVars: dict[int, int] = {}
            expectedValues:  dict[int, int] = {}
            
            for index in range(len(tail)):
                if v:
                    print(f"var = {tail[index]} has value {combination[index]} in the combination")
                conditionalVars[tail[index]] = combination[index]
            
            for index in range(len(tail), len(combination)):
                if v:
                    print(f"index = {index} and endoVar = {endoVars[index - len(tail)]} with value {combination[index]}")
                expectedValues[endoVars[index - len(tail)]] = combination[index]


            if v:
                print("ConditionalVars dbg:")
                for key in conditionalVars:
                    print(f"key: {key} & value: {conditionalVars[key]}")                            

                print("Expected values dbg:")
                for key in expectedValues:
                    print(f"key: {key} & value: {expectedValues[key]}")                            
            
            for mechanismDict in mechanismDicts:
                isValid: bool = LinearSolver.checkValues(mechanismDict=mechanismDict, parents=endoParents, topoOrder=topoOrder,
                                                             tailValues=conditionalVars, endogenousValues=expectedValues, v=False) 
                systemCoefficients.append(isValid)
            matrix.append(systemCoefficients)
        
        # Union of U states must have probability one
        probabilities.append(1.0)
        matrix.append([1] * len(mechanismDicts))

        if v:
            for index, eq in enumerate(matrix):
                print(f"{probabilities[index]} - Equation: {eq}")        

        return probabilities, matrix

    # Linear case: same c-component OR tail + Sachs condition
    def generateObjectiveFunction(df, indexToLabel: dict[int, str], mechanismDicts: list[dict[str,int]], targetVariable: int, targetValue: int, interventionVariable: int, interventionValue: int, topoOrder: list[int],
                                  tail: list[int], tailCardinalities: dict[int,int], v: True, endoParents: dict[int,list[int]],
                                  latent: int):
        """
        generate a linear objective function for the case in which the variable under intervention and the target are in 
        the same c-component (or at least on the tail).

        Algo: for each possible realization of the tail, consider all possile mechanisms. If it implies the expected target value, then it adds
        P(U|T)*P(T) = P(U,T) = P(U)*P(T) to the system.

        Important: this can be optimized by marginalizing on irrelevant variables. Hence, the argument tail does NOT need to contain all variables in the tail. In the
        same way, the topoOrder does NOT need to contain all endogenous variables of the c-component
        """        
        # Generate all tail spaces:        
        tailSpaces = helper.helperGenerateSpaces(tail, tailCardinalities)        
        tailRealizations = helper.generateCrossProducts(tailSpaces)
        
        if v:
            print("Debug - Check all tail realizations")
            for index, case in enumerate(tailRealizations):
                print(f"{index}) {case}")        

        topoOrder.remove(interventionVariable) # We do not need to run through this variable
        
        objectiveFunction: list[int] = [0] * len(mechanismDicts)
        for tailRealization in tailRealizations:
            computedNodes: dict[int, int] = {interventionVariable: interventionValue}
            if v:
                print(f"TailRealization = {tailRealization}")

            for index, tailNode in enumerate(tail):
                computedNodes[tailNode] = tailRealization[index] # Consider the values of the tail and of the intervention            
            for Uindex, mechanism in enumerate(mechanismDicts): # Consider each U for the given T + do(var=val)
                for node in topoOrder:
                    # --------- Build the key ---------
                    nodeParents: list[int] = endoParents[node]                    
                    if latent in nodeParents:
                        nodeParents.remove(latent)                    

                    dictKey: str = ""                    
                    for index, nodeParent in enumerate(nodeParents):
                        dictKey += f"{nodeParent}={computedNodes[nodeParent]},"                        
                        # TODO: PENSAR EM UMA CHAVE QUE INDEPENDE DA ORDEM!                        
                    # --------- ---------

                    print(f"dictKey = {dictKey[:-1]}")
                    nodeValue = mechanism[dictKey[:-1]]                    
                    computedNodes[node] = nodeValue
                    if node == targetVariable:
                        break
                
                if computedNodes[targetVariable] == targetValue:                    
                    tailDict: dict[int,int] = {}
                    for tailIndex, tailVar in enumerate(tail):
                        tailDict[tailVar] = tailRealization[tailIndex]
                    
                    tailProbability = helper.findProbability(df, indexToLabel, tailDict, False)                    
                    objectiveFunction[Uindex] += tailProbability
                    print(f"Sum to index {Uindex} P = {tailProbability} ") 
        
        print(f"Coefficients of the obj function: {objectiveFunction}")
        
        return objectiveFunction
                    

    def interventionalQuery(probabilities: list[int], empiricalEquations: list[list[int]], objectiveFunctionPos: list[int],
                             objectiveFunctionNeg: list[int]):
        PosLower, PosUpper = LinearSolver.createLinearProblem(probabilities, empiricalEquations, objectiveFunctionPos)
        NegLower, NegUpper = LinearSolver.createLinearProblem(probabilities, empiricalEquations, objectiveFunctionNeg)

        print(f"The positive query has bounds: [{PosLower},{PosUpper}]")
        print(f"The negative query has bounds: [{NegLower},{NegUpper}]")
        
        finalLower = PosLower - NegUpper
        finalUpper = PosUpper - NegLower
        print(f"The interventional bounds are: [{finalLower},{finalUpper}]")
        pass
    
    def createLinearProblem(probabilities: list[int], empiricalEquations: list[list[int]], objectiveFunction: list[int]):
        """
        Creates a linear programming problem with the following restrictions:
        - Equations from empirical probabilities (from args)
        - The sum of probabilities of a latent variable in its space must add to one (from args)
        - All the variables must be between 0 and 1 (not from args)
        - Objective function (from args)
        """        
        
        # Call the solver
        bounds = [(0, 1) for _ in range(len(objectiveFunction))]
        negObj = [-x for x in objectiveFunction]
        result = linprog(c=objectiveFunction, A_ub=None, b_ub=None, A_eq=empiricalEquations, b_eq=probabilities, method="highs", bounds=bounds)
        resultNeg = linprog(negObj, A_ub=None, b_ub=None, A_eq=empiricalEquations, b_eq=probabilities, method="highs", bounds=bounds)

        # Verify the minimum (lb)
        if result.success:
            # print("Solução ótima encontrada:", result.x)
            print("Valor da função objetivo:", result.fun)
        else:
            print("Solução não encontrada:", result.message)

        # Verify the maximum (ub)
        if result.success:
            # print("Solução ótima encontrada:", resultNeg.x)
            print("Valor da função objetivo:", -resultNeg.fun)
        else:
            print("Solução não encontrada:", resultNeg.message)        
        
        return  result.fun, -resultNeg.fun

def testMechanismGenerator():    
    print(f"Test case 1:")
    
    helper.mechanisms_generator(0, [3, 6], {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}, {3: [0, 1, 2], 6: [0, 4, 5] })
    print(f"Test case 2: Balke & Pearl")
    helper.mechanisms_generator(0, [1, 2], {0: 2, 1: 2, 2: 2, 3: 2}, {1: [0, 3], 2: [0, 1] })

def testBalkePearl():
    print("Teste for Balke & Pearl")
    #mechanismDicts: dict[str, int], tail: list[int], cardinalitiesTail: dict[int,int], endoVars: list[int],
                            # cardinalitiesEndo: dict[int,int]):        
    _, _, mechanismDicts = helper.mechanisms_generator(0, [1, 2], {1: 2, 2: 2, 3: 2}, {1: [0, 3], 2: [0, 1] }, False)
    print("Checking the dictionary")
    for element in mechanismDicts:
        for key in element:
            print(f"Key = {key} & value = {element[key]}")
        print ("--------")
        
    # Balke & Pearl graph in which: Z = 3, U = 0, X = 1, Y = 2
    print("\n=== Call equation generator ===\n")
    probabilities, equations = LinearSolver.equations_generator(mechanismDicts, [3], {3: 2}, [1, 2], {1: 2, 2: 2}, {1: [3], 2: [1]} ,
                                          [1, 2], {0: "U", 1: "X", 2: "Y", 3: "Z"}, "balke_pearl.csv", 3, False)    

    for i, eq in enumerate(equations):
        print(f"Equation {i}: {probabilities[i]} = {eq} * U^T")

    # mockObj = [0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0]    
    # mockPos = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]         
    # mockNeg = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]    
    # LinearSolver.createLinearProblem(probabilities, equations, mockObj)
    
    # print("Test the second approach:")
    # LinearSolver.interventionalQuery(probabilities, equations, mockPos, mockNeg)
        
    # tail: list[int], tailCardinalities: dict[int,int], v: True):
    # P(Y=1|do(X=0))
    df = helper.fetchCsv()
    LinearSolver.generateObjectiveFunction(df, {3: "Z", 2: "Y", 1: "X"},mechanismDicts=mechanismDicts, targetVariable=2, targetValue=1, interventionVariable=1, 
                                                interventionValue=0, topoOrder=[1, 2], tail=[3], tailCardinalities={3: 2}, v=True,
                                                endoParents={1: [0, 3], 2: [0, 1]}, latent=0)

def testItau():
    print("Teste grafo itau")

    _, _, mechanismDicts = helper.mechanisms_generator(0, [1, 2], {1: 2, 2: 2, 3: 2}, {1: [0, 3], 2: [0, 1, 3] }, False)
    print("Checking the dictionary")
    for element in mechanismDicts:
        for key in element:
            print(f"Key = {key} & value = {element[key]}")
        print ("--------")
        
    # Itau graph in which: U = 0, T = 1, Y = 2, D = 3
    print("\n=== Call equation generator ===\n")
    probabilities, equations = LinearSolver.equations_generator(mechanismDicts, [3], {3: 2}, [1, 2], {1: 2, 2: 2}, {1: [3], 2: [1, 3]} ,
                                          [1, 2], {0: "U", 1: "T", 2: "Y", 3: "D"}, "itau.csv", 3, False)  

    for i, eq in enumerate(equations):
        print(f"Equation {i}: {probabilities[i]} = {eq} * U^T")

if __name__ == "__main__":    
    testBalkePearl()
    #testItau()
