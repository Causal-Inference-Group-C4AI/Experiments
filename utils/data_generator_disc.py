import numpy as np
import pandas as pd
from collections import defaultdict, deque
from utils.funcoes import get_nodes
import random


def _sample_categorical(prob_matrix):
    """
    Given prob_matrix of shape (n, card), each row sums ~1,
    sample from that distribution row-wise. Return shape (n,).
    """
    n_, card = prob_matrix.shape
    cdf = np.cumsum(prob_matrix, axis=1)
    r = np.random.rand(n_,1)
    # category = number of cdf columns < r
    return (r > cdf).sum(axis=1)

def make_func_categorical_exog(dist, n):
    """
    For exogenous node with distribution dist (shape=card),
    return a param_func that yields shape (n, card) repeated dist for each row.
    """
    card = len(dist)
    def param_func(_parent_arrays):
        prob_matrix = np.tile(dist, (n,1))
        return prob_matrix
    return param_func

def make_func_categorical_softmax(parents, card, intercepts, coefs):
    """
    param_func that for each row:
     raw_scores[k] = intercepts[k] + sum( coefs[p][k] * parent_arrays[p][i] )
    then we take softmax across the 'card' categories.
    """
    def param_func(parent_arrays):
        # number of rows = length of parent's array
        n_ = 1
        if parents:
            # get the length from the first parent's array
            any_par = parents[0]
            n_ = len(parent_arrays[any_par])

        raw_scores = np.zeros((n_, card), dtype=float)
        # add intercept for each category
        for k in range(card):
            raw_scores[:,k] = intercepts[k]
        # add parent's effect
        for p_ in parents:
            parvals = parent_arrays[p_]
            # parvals shape is (n_,), add coefs for each category
            for k in range(card):
                raw_scores[:,k] += coefs[p_][k] * parvals

        # softmax row-wise
        max_in_row = np.max(raw_scores, axis=1, keepdims=True)
        stabilized = raw_scores - max_in_row
        exp_ = np.exp(stabilized)
        sum_exp = np.sum(exp_, axis=1, keepdims=True)
        prob_matrix = exp_/sum_exp
        return prob_matrix
    return param_func

class DataGenerator:
    """
    A class to generate discrete data for a user-specified causal graph,
    with each node's cardinality >= 2.

    node_specs or edges/edges_str can define the graph. If auto-building node_specs,
    user can optionally supply node_cardinalities[node]=card. If not present,
    we default to cardinality=2 for that node.
    """

    def __init__(
        self,
        node_specs=None,
        edges=None,
        edges_str=None,
        n=10000,
        seed=42,
        verbose=False,
        node_cardinalities=None
    ):
        """
        node_specs : list[dict] or None
          if not None, each dict has:
            - 'name': str
            - 'parents': list of str
            - 'cardinality': int
            - 'param_func': (dict_of_parent_arrays)->(n, card) prob matrix

        edges : list of (str,str) or None
          if node_specs=None, describes DAG edges

        edges_str : str or None
          if node_specs=None and edges=None, parse from this string

        n : int
          number of samples

        seed : int
          random seed

        verbose : bool
          prints param_func info if auto-generated

        node_cardinalities : dict[str,int] or None
          if auto-building node_specs, you can override cardinalities
          for each node. If not specified, default=2.
        """
        self.node_specs = node_specs
        self.edges = edges
        self.edges_str = edges_str
        self.n = n
        self.seed = seed
        self.verbose = verbose
        self.node_cardinalities = node_cardinalities if node_cardinalities else {}

        self._data = None

    def _parse_edges_if_needed(self):
        if self.edges_str and not self.edges:
            self.edges = get_nodes(self.edges_str)

    def _topological_sort(self, edges):
        nodes = set()
        for p,c in edges:
            nodes.add(p)
            nodes.add(c)
        nodes = sorted(nodes)

        adjacency = defaultdict(list)
        in_degree = defaultdict(int)
        for nd in nodes:
            in_degree[nd] = 0

        for p,c in edges:
            adjacency[p].append(c)
            in_degree[c]+=1

        queue = deque([nd for nd in nodes if in_degree[nd]==0])
        topo_order = []
        while queue:
            nd = queue.popleft()
            topo_order.append(nd)
            for child in adjacency[nd]:
                in_degree[child]-=1
                if in_degree[child]==0:
                    queue.append(child)

        parents_dict = defaultdict(list)
        for p,c in edges:
            parents_dict[c].append(p)

        return topo_order, parents_dict

    def _auto_build_specs(self):
        """
        If node_specs is None, parse edges/edges_str, topologically sort,
        create random param_funcs. Each node has cardinality from user or default=2..4.
        Actually let's default=2 if not in node_cardinalities (per request).
        """
        self._parse_edges_if_needed()
        edges = self.edges
        if not edges:
            raise ValueError("No edges found to auto-build specs.")
        topo_order, parents_dict = self._topological_sort(edges)

        new_specs = []
        random.seed(self.seed)
        np.random.seed(self.seed)

        for nd in topo_order:
            prnts = sorted(parents_dict[nd])
            # cardinality from user or default=2
            if nd in self.node_cardinalities:
                card = self.node_cardinalities[nd]
            else:
                # default is 2
                card = 2

            if len(prnts)==0:
                # exogenous => random distribution with 'card' categories
                dist = np.random.rand(card)
                dist /= dist.sum()
                if self.verbose:
                    print(f"Node {nd}: exogenous, card={card}, distribution={dist}")
                param_func = make_func_categorical_exog(dist, self.n)
            else:
                # we do a random softmax approach
                intercepts = np.random.uniform(0.05,0.2,size=card)
                coefs = {}
                for p_ in prnts:
                    coefs[p_] = np.random.uniform(0.1,0.3,size=card)
                if self.verbose:
                    param_str_parts = []
                    for k in range(card):
                        s = f"(cat{k}): {intercepts[k]:.3f}"
                        for p_ in prnts:
                            s+=f" + {coefs[p_][k]:.3f}*{p_}"
                        param_str_parts.append(s)
                    param_str = "\n".join(param_str_parts)
                    print(f"Node {nd}: cardinality={card}, param_func softmax\n{param_str}")

                param_func = make_func_categorical_softmax(prnts, card, intercepts, coefs)

            new_specs.append({
                'name': nd,
                'parents': prnts,
                'cardinality': card,
                'param_func': param_func
            })

        self.node_specs = new_specs

    def generate_data(self, save_csv=False, csv_path="synthetic_data.csv"):
        """
        Generate the data in topological order. Each node is shape (n,) in [0..card-1].
        Return DataFrame with columns node_name.
        """
        if self.node_specs is None:
            self._auto_build_specs()
        else:
            if self.verbose:
                print("Using user-provided node_specs. We assume param_func returns shape (n,card).")

        data = {}
        np.random.seed(self.seed)
        random.seed(self.seed)

        for spec in self.node_specs:
            name = spec['name']
            parents = spec['parents']
            card = spec['cardinality']
            param_func = spec['param_func']

            # gather parent's arrays
            parent_arrays = {p: data[p] for p in parents}
            prob_matrix = param_func(parent_arrays)  # shape (n, card)
            prob_matrix = np.clip(prob_matrix, 0, 1)

            # row sums
            row_sums = prob_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums==0] = 1.0  # avoid zero
            prob_matrix/=row_sums

            # sample
            samples = _sample_categorical(prob_matrix)
            data[name] = samples

        df = pd.DataFrame(data)
        self._data = df

        if save_csv:
            df.to_csv(csv_path, index=False)
            if self.verbose:
                print(f"Data saved to {csv_path}")

        return df
