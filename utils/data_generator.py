import numpy as np
import pandas as pd
from collections import defaultdict, deque
from utils.funcoes import get_nodes
import random

def make_func_exog(n):
    """
    Exogenous Bernoulli(0.5) function.
    """
    def param_func(_parent_arrays):
        return np.full(n, 0.5)
    return param_func

def make_func(parents, intercept, coefs):
    """
    Summation param_func: intercept + sum(coefs[p]*parent_arrays[p]).
    """
    def param_func(parent_arrays):
        prob = intercept
        for p_ in parents:
            prob += coefs[p_] * parent_arrays[p_]
        return prob
    return param_func

class DataGenerator:
    """
    A class to generate binary data for a user-specified causal graph.
    Also stores frequency information about each node's distribution 
    given its parents, and their conditional probabilities, 
    once the data is generated.
    """

    def __init__(self,
                 node_specs=None,
                 edges=None,
                 edges_str=None,
                 n=10000,
                 seed=42,
                 verbose=False,
                 store_global_joint=False):
        """
        Parameters
        ----------
        node_specs : list[dict] or None
            Each dict with 'name','parents','param_func'.
        edges : list of (str,str) or None
            If node_specs is None, specify edges describing the DAG.
        edges_str : str or None
            If node_specs is None and edges is None, parse from this string.
        n : int
            Number of samples.
        seed : int
            Random seed.
        verbose : bool
            Print param_func info if auto-generated.
        store_global_joint : bool
            If True, also store the global joint frequency distribution of all nodes.
            This might be large for many nodes.
        """
        self.node_specs = node_specs
        self.edges = edges
        self.edges_str = edges_str
        self.n = n
        self.seed = seed
        self.verbose = verbose
        self.store_global_joint = store_global_joint

        self._data = None
        self.local_freqs = {}
        self.local_cond_probs = {}
        self.global_counts = None

    def _auto_build_specs(self):
        """
        If self.node_specs is None, parse edges or edges_str and create node_specs
        with random logistic param_funcs.
        """
        if self.edges_str and not self.edges:
            self.edges = get_nodes(self.edges_str)

        nodes = set()
        for p, c in self.edges:
            nodes.add(p)
            nodes.add(c)
        nodes = sorted(nodes)

        from collections import defaultdict, deque
        adjacency = defaultdict(list)
        in_degree = defaultdict(int)
        for nd in nodes:
            in_degree[nd] = 0
        for p,c in self.edges:
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
        for p,c in self.edges:
            parents_dict[c].append(p)

        node_specs = []
        random.seed(self.seed)
        np.random.seed(self.seed)
        for nd in topo_order:
            prnts = sorted(parents_dict[nd])
            if len(prnts)==0:
                param_func = make_func_exog(self.n)
                if self.verbose:
                    print(f"Node {nd}: no parents -> Bernoulli(0.5).")
            else:
                intercept = random.uniform(0.05,0.2)
                coefs = {p_: random.uniform(0.1,0.3) for p_ in prnts}

                if self.verbose:
                    param_str = f"{intercept:.3f}" + "".join([
                        f" + {coefs[p_]:.3f}*{p_}" for p_ in prnts
                    ])
                    print(f"Node {nd}: param_func = {param_str}")

                param_func = make_func(prnts, intercept, coefs)

            node_specs.append({
                'name': nd,
                'parents': prnts,
                'param_func': param_func
            })

        self.node_specs = node_specs

    def generate_data(self, save_csv=False, csv_path="synthetic_data.csv"):
        """
        Generate the data, store freq info, and optionally save to CSV.
        Returns the DataFrame of shape (n, #nodes).
        """
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.node_specs is None:
            if not self.edges and not self.edges_str:
                raise ValueError("No node_specs or edges/edges_str provided.")
            self._auto_build_specs()
        else:
            if self.verbose:
                print("Using user-provided node_specs:")
                for spec in self.node_specs:
                    nd = spec['name']
                    prnts = spec['parents']
                    print(f"Node {nd}, parents={prnts}, custom param_func provided.")

        data = {}
        for spec in self.node_specs:
            node_name = spec['name']
            prnts = spec['parents']
            param_func = spec['param_func']

            parent_arrays = {p: data[p] for p in prnts}
            probs = param_func(parent_arrays)
            probs = np.clip(probs, 0, 1)

            node_vals = np.random.binomial(1, probs)
            data[node_name] = node_vals

        df = pd.DataFrame(data)
        self._data = df

        self._build_local_frequencies(df)

        if self.store_global_joint:
            self._build_global_counts(df)

        if save_csv:
            df.to_csv(csv_path, index=False)
            if self.verbose:
                print(f"Data saved to {csv_path}")

        return df

    def _build_local_frequencies(self, df):
        """
        For each node in self.node_specs, we group by its parents 
        to count how often the node=0 or node=1, 
        storing the results in local_freqs and local_cond_probs.
        Then optionally print them in the desired style: 
        "P(Y=1| U=0, X=0) = 0.093"
        """
        self.local_freqs = {}
        self.local_cond_probs = {}

        for spec in self.node_specs:
            node_name = spec['name']
            prnts = spec['parents']

            if len(prnts)==0:
                count_0 = (df[node_name]==0).sum()
                count_1 = (df[node_name]==1).sum()
                self.local_freqs[node_name] = {
                    '(exogenous)': {'count_0': count_0, 'count_1': count_1}
                }
                total = count_0 + count_1
                p1 = count_1 / total if total>0 else None
                self.local_cond_probs[node_name] = {
                    '(exogenous)': {'P(node=1)': p1}
                }
                if self.verbose:
                    print(f"P({node_name}=1|exogenous) = {p1:.3f}" if p1 is not None else "No data")
            else:
                grouped = df.groupby(prnts)[node_name]
                freq_dict = {}
                prob_dict = {}

                for parent_vals, sub in grouped:
                    if not isinstance(parent_vals, tuple):
                        parent_vals = (parent_vals,)
                    parent_str_list = []
                    for pval, parname in zip(parent_vals, prnts):
                        parent_str_list.append(f"{parname}={pval}")
                    parent_str = ", ".join(parent_str_list)

                    count_0 = (sub==0).sum()
                    count_1 = (sub==1).sum()
                    total = count_0 + count_1

                    freq_dict[parent_str] = {'count_0': count_0, 'count_1': count_1}
                    if total>0:
                        p1 = count_1/total
                    else:
                        p1 = None
                    prob_dict[parent_str] = {'P(node=1)': p1}

                    if self.verbose and p1 is not None:
                        print(f"P({node_name}=1 | {parent_str}) = {p1:.4f}")

                self.local_freqs[node_name] = freq_dict
                self.local_cond_probs[node_name] = prob_dict


    def _build_global_counts(self, df):
        """
        Build a global joint frequency distribution of all columns in df.
        We store the result in self.global_counts, which includes a 'count'
        column for how many times that combination appears, and a 'freq' column
        which is count / total_n.
        """
        joint_df = df.value_counts(sort=False).reset_index()
        
        node_cols = list(df.columns)
        joint_df.columns = node_cols + ['count']
        
        total_n = joint_df['count'].sum()
        joint_df['freq'] = joint_df['count'] / total_n
        
        self.global_counts = joint_df


    def get_local_freqs(self, node_name):
        """
        Return the local frequency dictionary for a given node,
        which includes how many times node=0 vs node=1 for each parent combination.
        """
        return self.local_freqs.get(node_name, {})

    def get_local_cond_probs(self, node_name):
        """
        Return the local conditional probability dictionary for a given node,
        i.e. for each parent combination, the probability node=1.
        """
        return self.local_cond_probs.get(node_name, {})

    def get_global_joint_counts(self):
        """
        If store_global_joint=True, returns the DataFrame enumerating 
        each combination of node values plus a 'count' column.
        """
        return self.global_counts

