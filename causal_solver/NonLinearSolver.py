import pyomo.environ  as pyo
from collections import namedtuple
from pyomo.opt import *
from causal_solver.GurobiSolver import solveModelGurobi

equationsObject = namedtuple('equationsObject', ['probability', 'dictionary'])

def dictKey_to_index(key : str):
    unobCardinalities: list[int] = []
    commma_index: list[int] = [0]

    for index,num in enumerate(key):
        if num == ",":
            unobCardinalities.append(int(key[commma_index[-1]:index]))
            commma_index.append(index + 1)
        elif index == len(key) - 1:
            unobCardinalities.append(int(key[commma_index[-1]:]))
    return unobCardinalities


def createModel(objective : dict[str, float], constraints : list[list[equationsObject]],latentCardinalities: list[int], maximize : bool = False,
                 initVal: float = .5):
    numVar: int = 0
    latentsNums: list[int] = [0]

    for key in latentCardinalities:
        latentsNums.append(latentCardinalities[key] + numVar)
        numVar += latentCardinalities[key]

    model = pyo.ConcreteModel(name = "opt")
    model.q = pyo.Var(range(numVar), bounds=(0,1), within = pyo.Reals, initialize = initVal)
    model.eqConstrain = pyo.ConstraintList()
    def o_rule(model):
        expr = 0
        for key in objective:
            coef = objective[key]
            unobs = dictKey_to_index(key= key)

            if coef > 0:
                aux = coef
                for u in unobs:
                    aux *= model.q[u]
                expr += aux
        return expr
    if maximize :
        model.obj = pyo.Objective(rule = o_rule, sense = pyo.maximize)
    else:
        model.obj = pyo.Objective(rule = o_rule, sense = pyo.minimize)
    for group in constraints:
        for eqn in group:
            expr = 0
            for key in eqn.dictionary:
                coef = eqn.dictionary[key]
                unobs = dictKey_to_index(key = key)
                if coef > 0:
                        aux = coef
                        for u in unobs:
                            aux *= model.q[u]
                        expr += aux
            model.eqConstrain.add((expr == eqn.probability))

    for i in range(1, len(latentsNums)):
        expr = 0
        for var in range(latentsNums[i -1], latentsNums[i]):
            expr += model.q[var]
        model.eqConstrain.add(expr == 1.)
    return model

def solveModel(objective : dict[str, float], constraints : list[list[equationsObject]],latentCardinalities: list[int],verbose: bool = True,
                 initVal: float = .5, solver_name="ipopt"):

    if solver_name == "ipopt":
        return solveModelIpopt(objective, constraints, latentCardinalities, verbose, initVal)
    else:
        return solveModelGurobi(objective, constraints, latentCardinalities, verbose, initVal)

def solveModelIpopt(objective : dict[str, float], constraints : list[list[equationsObject]],latentCardinalities: list[int],verbose: bool = True, initVal: float = .5):
    
    opt = SolverFactory("ipopt")
    numVar: int = 0
    latentsNums: list[int] = [0]
    
    for key in latentCardinalities:
        latentsNums.append(latentCardinalities[key] + numVar)
        numVar += latentCardinalities[key]
    
    model = pyo.ConcreteModel(name = "opt")
    model.q = pyo.Var(range(numVar), bounds=(0,1), within = pyo.Reals, initialize = initVal)
    model.eqConstrain = pyo.ConstraintList()
    
    def o_rule(model):
        expr = 0
        for key in objective:
            coef = objective[key]
            unobs = dictKey_to_index(key= key)
                
            if coef > 0:
                aux = coef
                for u in unobs:
                    aux *= model.q[u]
                expr += aux
        return expr
    
    model.obj = pyo.Objective(rule = o_rule, sense = pyo.maximize)
        
    for group in constraints:
        for eqn in group:
            expr = 0
            for key in eqn.dictionary:
                coef = eqn.dictionary[key]
                unobs = dictKey_to_index(key = key)
                if coef > 0:
                        aux = coef
                        for u in unobs:
                            aux *= model.q[u]
                        expr += aux
            model.eqConstrain.add((expr == eqn.probability))

    for i in range(1, len(latentsNums)):
        expr = 0
        for var in range(latentsNums[i -1], latentsNums[i]):
            expr += model.q[var]
        model.eqConstrain.add(expr == 1.)
    
    if verbose:
        model.obj.pprint()
        model.eqConstrain.pprint()
    
    results = opt.solve(model)
    upper = pyo.value(model.obj)
    print(f"MAX Query:{upper}")
    model.del_component(model.obj)
    model.obj = pyo.Objective(rule = o_rule, sense = pyo.minimize) 
    results = opt.solve(model)
    lower = pyo.value(model.obj)
    print(f"MIN query: {lower}")

    return lower, upper
