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
    return (r > cdf).sum(axis=1)

class DataGeneratorDiscrete:
    """
    A data generator for discrete nodes (>=2 categories) using a softmax approach.

    Capabilities:
      1) If verbose=True, prints the node's param_func logit formula (intercepts, coefs) for each category
      2) Generates local frequency (local_freqs, local_cond_probs) after building data
      3) Generates global frequency distribution (global_counts) if store_global_joint=True
      4) Enumerates conditional probabilities for each node from the data
    """

    def __init__(
        self,
        node_specs=None,
        edges=None,
        edges_str=None,
        n=10000,
        seed=42,
        verbose=False,
        node_cardinalities=None,
        store_global_joint=True,
        max_enumeration=50
    ):
        """
        node_specs : list[dict] or None
          If not None, each dict must have:
            - 'name': node name
            - 'parents': list of parent names
            - 'cardinality': int
            - 'param_func': callable => shape(n, card) prob matrix
        edges : list of (str,str) or None
          If node_specs=None, we define DAG edges => auto-build param_funcs
        edges_str : str or None
          If node_specs=None and edges=None, parse from this string (not implemented here).
        n : int
          number of samples
        seed : int
          random seed
        verbose : bool
          prints param_func + enumerations of P(node=cat | parents=...)
        node_cardinalities : dict[node->int]
          If auto-building, read from here or default=2
        store_global_joint : bool
          If True, build global_counts with each combo => 'count','freq'
        max_enumeration : int
          If parent's cardinalities product>max_enumeration, skip enumerating combos 
          to avoid huge output
        """
        self.node_specs = node_specs
        self.edges = edges
        self.edges_str = edges_str
        self.n = n
        self.seed = seed
        self.verbose = verbose
        self.node_cardinalities = node_cardinalities if node_cardinalities else {}
        self.store_global_joint = store_global_joint
        self.max_enumeration = max_enumeration

        self._data = None
        self.local_freqs = {}
        self.local_cond_probs = {}
        self.global_counts = None

    def _make_func_categorical_exog(self, card):
        """
        For exogenous node => produce shape(n, card) 
        with a single random distribution repeated for each row
        so we get 'n' samples from that distribution.
        """
        dist = np.random.rand(card)
        dist/= dist.sum()

        def param_func(_parent_arrays):
            prob_matrix = np.tile(dist, (self.n,1))
            return prob_matrix
        return dist, param_func

    def _make_func_categorical_softmax(self, parents, card, intercepts, coefs):
        """
        param_func => shape(n, card). 
        raw_scores[i,c] = intercepts[c] + sum( coefs[p][c]* parent_arrays[p][i] ).
        Then softmax row-wise.
        """
        def param_func(parent_arrays):
            if len(parents)>0:
                n_ = len(parent_arrays[parents[0]])
            else:
                n_ = self.n
            raw_scores = np.zeros((n_,card), dtype=float)
            for c_ in range(card):
                raw_scores[:,c_] = intercepts[c_]
            for p_ in parents:
                pvals= parent_arrays[p_]
                for c_ in range(card):
                    raw_scores[:,c_]+= coefs[p_][c_]* pvals
            max_in_row= np.max(raw_scores, axis=1, keepdims=True)
            stabilized= raw_scores- max_in_row
            exp_ = np.exp(stabilized)
            sumexp= np.sum(exp_, axis=1, keepdims=True)
            prob_matrix= exp_/sumexp
            return prob_matrix
        return param_func

    def _parse_edges_if_needed(self):
        if self.edges_str and not self.edges:
            self.edges = get_nodes(self.edges_str)

    def _topological_sort(self, edges):
        nodes = set()
        for p,c in edges:
            nodes.add(p)
            nodes.add(c)
        nodes= sorted(nodes)
        adjacency= defaultdict(list)
        in_degree= defaultdict(int)
        for nd in nodes:
            in_degree[nd]=0
        for p,c in edges:
            adjacency[p].append(c)
            in_degree[c]+=1
        from collections import deque
        queue= deque([nd for nd in nodes if in_degree[nd]==0])
        topo_order=[]
        while queue:
            nd= queue.popleft()
            topo_order.append(nd)
            for child in adjacency[nd]:
                in_degree[child]-=1
                if in_degree[child]==0:
                    queue.append(child)
        parents_dict= defaultdict(list)
        for p,c in edges:
            parents_dict[c].append(p)
        return topo_order, parents_dict

    def _auto_build_specs(self):
        """
        If node_specs=None => parse edges => topological sort => 
        for each node => define random param_func in [-1.0..1.0] intercept, [-2.0..2.0] coefs.
        exogenous => _make_func_categorical_exog
        child => _make_func_categorical_softmax
        """
        self._parse_edges_if_needed()
        print(self.edges)
        if not self.edges:
            raise ValueError("No edges found to auto-build specs.")
        topo_order, parents_dict= self._topological_sort(self.edges)
        np.random.seed(self.seed)
        random.seed(self.seed)

        specs=[]
        for nd in topo_order:
            prnts= sorted(parents_dict[nd])
            card= self.node_cardinalities.get(nd,2)
            if len(prnts)==0:
                dist, param_func= self._make_func_categorical_exog(card)
                if self.verbose:
                    cat_str= ", ".join([f"(cat{k}) {dist[k]:.3f}" for k in range(card)])
                    print(f"Node {nd}: exogenous, card={card}, dist => {cat_str}")
            else:
                intercepts= np.random.uniform(-1.0,1.0, size=card)
                coefs={}
                for p_ in prnts:
                    coefs[p_]= np.random.uniform(-2.0,2.0, size=card)
                param_func= self._make_func_categorical_softmax(prnts, card, intercepts, coefs)
                if self.verbose:
                    lines=[]
                    for c_ in range(card):
                        s= f"(cat{c_}): {intercepts[c_]:+.3f}"
                        for p_ in prnts:
                            s+= f" + {coefs[p_][c_]:+.3f}*{p_}"
                        lines.append(s)
                    param_str= "\n".join(lines)
                    print(f"Node {nd}: card={card}, param_func softmax:\n{param_str}")

            specs.append({
                'name': nd,
                'parents': prnts,
                'cardinality': card,
                'param_func': param_func
            })

        self.node_specs= specs

    def generate_data(self, save_csv=False, csv_path="synthetic_data.csv"):
        """
        Build data in topological order => shape(n, #nodes).
        Then build local freq, global dist => store in local_freqs, local_cond_probs, global_counts
        """
        if self.node_specs is None:
            if not self.edges and not self.edges_str:
                raise ValueError("No node_specs or edges to build from.")
            self._auto_build_specs()

        data={}
        np.random.seed(self.seed)
        random.seed(self.seed)

        for spec in self.node_specs:
            name= spec['name']
            parents= spec['parents']
            card= spec['cardinality']
            pf= spec['param_func']

            if len(parents)==0:
                prob_matrix= pf({})
            else:
                parent_arrays={}
                for p_ in parents:
                    parent_arrays[p_]= data[p_]
                prob_matrix= pf(parent_arrays)

            prob_matrix= np.clip(prob_matrix,0,1)
            row_sums= prob_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums==0]=1
            prob_matrix/= row_sums

            cat_samples= _sample_categorical(prob_matrix)
            data[name]= cat_samples

        df= pd.DataFrame(data)
        self._data= df

        self._build_local_frequencies(df)
        if self.store_global_joint:
            self._build_global_counts(df)

        if save_csv:
            df.to_csv(csv_path,index=False)
            if self.verbose:
                print(f"Data saved to {csv_path}")

        return df

    def _build_local_frequencies(self, df):
        """
        For each node in node_specs => group by parents => freq of each cat => local_freqs, local_cond_probs
        """
        self.local_freqs={}
        self.local_cond_probs={}

        for spec in self.node_specs:
            node_name= spec['name']
            parents= spec['parents']
            card= spec['cardinality']

            if len(parents)==0:
                val_counts= df[node_name].value_counts(sort=False)
                freq_dict= {'(exogenous)': {}}
                prob_dict= {'(exogenous)': {}}
                total= val_counts.sum()
                for c_ in range(card):
                    ccount= val_counts.get(c_,0)
                    freq_dict['(exogenous)'][f'count_{c_}']= ccount
                    prob_dict['(exogenous)'][f'P(node={c_})']= ccount/total if total>0 else None
                self.local_freqs[node_name]= freq_dict
                self.local_cond_probs[node_name]= prob_dict
            else:
                group_cols= parents
                grouped= df.groupby(group_cols)[node_name]
                freq_dict={}
                prob_dict={}

                for pvals, sub in grouped:
                    if not isinstance(pvals, tuple):
                        pvals= (pvals,)
                    parent_str_list= [f"{col}={val}" for val,col in zip(pvals, group_cols)]
                    parent_str= ", ".join(parent_str_list)
                    val_counts= sub.value_counts(sort=False)
                    total= val_counts.sum()
                    freq_dict[parent_str]= {}
                    prob_dict[parent_str]= {}
                    for c_ in range(card):
                        ccount= val_counts.get(c_,0)
                        freq_dict[parent_str][f'count_{c_}']= ccount
                        prob_dict[parent_str][f'P(node={c_})']= ccount/total if total>0 else None
                self.local_freqs[node_name]= freq_dict
                self.local_cond_probs[node_name]= prob_dict

    def _build_global_counts(self, df):
        """
        Build global distribution => each combo => 'count','freq'
        """
        df_counts= df.value_counts(sort=False).reset_index()
        cols= list(df.columns)
        df_counts.columns= cols + ['count']
        total= df_counts['count'].sum()
        df_counts['freq']= df_counts['count']/ total if total>0 else None
        self.global_counts= df_counts

    def enumerate_conditional_probs(self):
        """
        Automatically calculates and prints the conditional probability of every node
        from the actual generated data. For each node:
        - If exogenous => single distribution of shape(n,)
        - Else group the dataset by parent's values => print P(node=cat | those parents)

        We skip enumerations if the product of parent's unique values > self.max_enumeration 
        to prevent huge printouts.
        """
        if self._data is None:
            raise ValueError("No data has been generated. Call generate_data() first.")

        if self.node_specs is None:
            raise ValueError("No node_specs found. We need node_specs to know each node’s parents/cardinality.")

        for spec in self.node_specs:
            node_name = spec['name']
            parents   = spec['parents']
            card      = spec['cardinality']

            if len(parents) == 0:
                counts = self._data[node_name].value_counts(sort=False)
                total  = counts.sum()
                print(f"\nNode {node_name} is exogenous => single distribution from data (card={card}):")
                for catv in range(card):
                    ccount = counts.get(catv,0)
                    pcat   = ccount/total if total>0 else 0
                    print(f"  P({node_name}={catv}|exogenous) = {pcat:.4f}")
                continue

            parent_uniques = [self._data[p].unique() for p in parents]
            unique_counts  = [len(uvals) for uvals in parent_uniques]
            prod = 1
            for uc in unique_counts:
                prod *= uc

            if prod > self.max_enumeration:
                print(f"\nNode {node_name}: skipping data combos => parent's unique combos product={prod} > max_enumeration={self.max_enumeration}")
                continue

            print(f"\n== Empirical conditional probs from data for Node {node_name} (card={card}), grouping by {parents} ==")
            grouped = self._data.groupby(parents)[node_name]

            for parent_vals, subdf in grouped:
                if not isinstance(parent_vals, tuple):
                    parent_vals = (parent_vals,)

                parent_str_parts = []
                for (val, p_) in zip(parent_vals, parents):
                    parent_str_parts.append(f"{p_}={val}")
                parent_combo_str = ", ".join(parent_str_parts)

                cat_counts = subdf.value_counts(sort=False)
                total = cat_counts.sum()
                if total == 0:
                    continue

                for catv in range(card):
                    ccount = cat_counts.get(catv,0)
                    pcat   = ccount/total
                    print(f"  P({node_name}={catv} | {parent_combo_str}) = {pcat:.4f}")


    def get_local_freqs(self, node_name):
        return self.local_freqs.get(node_name, {})

    def get_local_cond_probs(self, node_name):
        return self.local_cond_probs.get(node_name, {})

    def get_global_joint_counts(self):
        return self.global_counts
