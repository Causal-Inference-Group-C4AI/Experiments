from gurobipy import Model, GRB
import pandas as pd
from typing import Tuple


class MinMaxModels:
    def __init__(self, model_type: str, number_of_vars: int) -> None:
        feasability = 1e-4
        verbose = 0
        self.model_max = Model(model_type)
        # self.model_max.setParam("FeasibilityTol", feasability)
        self.model_max.setParam("OutputFlag", False)
        self.vars_model_max = self.model_max.addVars(number_of_vars, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="vars")
        self.model_min = Model(model_type)
        # self.model_min.setParam("FeasibilityTol", feasability)
        self.model_min.setParam("OutputFlag", False)
        self.vars_model_min = self.model_min.addVars(number_of_vars, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="vars")

class Experiment:
    def __init__(self, df: pd.DataFrame, doX0_groundtruth: float, doX1_groundtruth: float, u1_prob : Tuple[float, float] = None, u2_prob: Tuple[float, float] = None) -> None:
        self.df = df
        self.n_rows = len(df.index)
        self.doX0_result = ""
        self.doX1_result = ""
        self.doX0_groundtruth = doX0_groundtruth
        self.doX1_groundtruth = doX1_groundtruth
        self.u1_prob = str(u1_prob)
        self.u2_prob = str(u2_prob)        

    def set_models(self, model_type: str, number_of_vars: int) -> None:
        self.doX0_models=MinMaxModels(model_type, number_of_vars)
        self.doX1_models=MinMaxModels(model_type, number_of_vars)

    def set_time_to_maximize(self, time):
        self.time_to_maximize = time
    def set_time_to_minimize(self, time):
        self.time_to_minimize = time
