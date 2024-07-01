import time
import math
from collections import defaultdict

import networkx as nx
import networkx.algorithms.centrality as centrality
import pickle
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import LinExpr
import cvxpy as cp
import pulp
from config import config
import multiprocessing as mp
from tqdm import tqdm


def solve(solver, C, E, F, D, K, uq):
    start_time = time.time()
    utilization = solver.solve_by_gurobi(C, E, F, D, K)
    end_time = time.time()
    solve_time = end_time - start_time
    uq.put((utilization, solve_time))


# def solve_by_cvxpy(g_dict, C, E, F, D, K):
#     num_e, num_f, num_k = len(E), len(F), len(K)
#     theta = cp.Variable(1)
#     action = cp.Variable((num_f, num_k))
#     k_sum = np.ones(num_k)
#     g_mat = np.zeros((num_e, num_k, num_f))

#     for e in range(num_e):
#         for idx, (i, j) in enumerate(F):
#             for k in range(num_k):
#                 g_mat[e][k][idx] = g_dict[i, j, K[k], e]
#     constraints = [
#         theta >= 0,
#         action >= 0,
#         action @ k_sum >= D,
#     ]
#     constraints += [
#         cp.trace(action @ g_mat[e]) <= theta * C[e] for e in range(num_e)
#     ]
#     objective = cp.Minimize(theta)
#     model = cp.Problem(objective=objective, constraints=constraints)
#     model.solve(verbose=True, solver='SCS')
#     return model.value

# def solve_by_pulp(g_dict, C, E, F, D, K):
#     num_e, num_f, num_k = len(E), len(F), len(K)
#     theta = pulp.LpVariable(name='theta')
#     indexs = [(f, k) for f in range(num_f) for k in range(num_k)]
#     action = pulp.LpVariable.dicts(name='action', indexs=indexs, lowBound=0.0)

#     model = pulp.LpProblem(name='mlu')
#     for f in range(num_f):
#         model += (pulp.lpSum(action[f, k] for k in range(num_k)) >= D[f], f'demand_constr_{f}')
#     for e in range(num_e):
#         model += (
#             pulp.lpSum(g_dict[i, j, K[k], e] * action[idx, k]
#                 for k in range(num_k) for idx, (i, j) in enumerate(F)) <= theta * C[e],
#             f'capacity_contstr_{e}'
#         )
#     model += theta
#     model.solve(solver=pulp.GLPK())
#     return theta.value()


class Topology(object):
    def __init__(self, name, data_dir="./data/"):
        self.name = name
        self.data_dir = data_dir

    def load(self):
        G = nx.DiGraph()
        try:
            filename = f"{self.data_dir}{self.name}"
            with open(filename, "r") as f:
                for line in f.readlines():
                    line = line.split()[:4]
                    src, dst, weight, cap = list(map(int, line))
                    G.add_edge(src, dst, weight=weight, cap=cap)
        except Exception as e:
            print(f"failed to load topology {self.name}")
            print(e)
        return G


class Traffic(object):
    def __init__(self, name, data_dir="./data/"):
        self.name = name
        self.data_dir = data_dir

    def load(self):
        TMs = []
        try:
            filename = f"{self.data_dir}{self.name}"
            with open(filename, "r") as f:
                for line in f.readlines():
                    line = line.split()
                    tm = np.array(list(map(float, line)))
                    tm = tm.reshape((math.isqrt(len(tm)), -1))
                    TMs.append(tm)
        except Exception as e:
            print(f"failed to load traffic matrices {self.name}")
            print(e)

        #  Unit: (100 bytes / 5 minutes)
        return np.array(TMs) * 100 * 8 / 300 / 1000  # kbps

    def load_pickle(self):
        filename = f"{self.data_dir}{self.name}TM.pkl"
        TMs = pickle.load(open(filename, "rb"))
        return TMs


class NodeSelector(object):
    """
    Select nodes as middle points
    """

    def __init__(self, topo: Topology):
        self.G = topo.load()

    def random(self, k: int = 1) -> list:
        nodes = [x for x in self.G.nodes]
        return np.random.permutation(nodes)[:k].tolist()

    def sp_centrality(self, k: int = 1) -> list:
        nodes = centrality.betweenness_centrality(self.G)
        sorted_nodes = [k for k, _ in sorted(nodes.items(), key=lambda x: -x[1])]
        return sorted_nodes[:k]

    def degree_centrality(self, k: int = 1) -> list:
        nodes = centrality.degree_centrality(self.G)
        sorted_nodes = [k for k, _ in sorted(nodes.items(), key=lambda x: -x[1])]
        return sorted_nodes[:k]

    def weighted_sp_centrality(self, k: int = 1) -> list:
        nodes = centrality.betweenness_centrality(self.G, weight="weight")
        sorted_nodes = [k for k, _ in sorted(nodes.items(), key=lambda x: -x[1])]
        return sorted_nodes[:k]


class TESolver(object):
    """
    Minimize Maximum Link Utilization
    """

    def __init__(self, topo: Topology):
        self.f_dict = {}
        # self.g_dict = {}
        self.topo = topo
        self.G = self.topo.load()

    def handle_G(self):
        edges, caps = [], []
        for src, dst, attr in self.G.edges.data():
            caps.append(attr["cap"])
            edges.append((src, dst))
        return caps, edges

    def handle_TM(self, TM: np.array):
        flows, demands = [], []
        num_nodes = len(TM)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if TM[i][j] != 0:
                    flows.append((i, j))
                    demands.append(TM[i][j])
        return flows, demands

    def precompute_f(self):
        num_nodes = len(self.G.nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    ans = self.compute_ecmp_link_frac(i, j, load=1.0)
                    self.f_dict[i, j] = ans

    def compute_ecmp_link_frac(self, src, dst, load=1.0):
        ans = defaultdict(int)
        try:
            paths = list(nx.all_shortest_paths(self.G, src, dst, weight="weight"))
            # build DAG
            dag = nx.DiGraph()
            node_succ = defaultdict(set)
            node_load = defaultdict(int)
            for p in paths:
                for s, t in zip(p, p[1:]):
                    node_succ[s].add(t)
                    dag.add_nodes_from([s, t])
                    dag.add_edge(s, t, frac=0.0)
            # compute fraction
            node_load[src] = load
            for node in nx.topological_sort(dag):
                nexthops = node_succ[node]
                if not nexthops:
                    continue
                nextload = node_load[node] / len(nexthops)
                for nexthop in nexthops:
                    dag[node][nexthop]["frac"] += nextload
                    node_load[nexthop] += nextload
            for s, t in dag.edges:
                ans[s, t] = dag[s][t]["frac"]

        except (KeyError, nx.NetworkXNoPath):
            print("Error, no path for %s to %s in apply_ecmp_flow()" % (src, dst))
        return ans

    def f(self, i, j, e):
        """ecmp fraction of edges"""
        if (i, j) not in self.f_dict:
            return 0
        return self.f_dict[i, j].get(e, 0)

    def g(self, i, j, k, e):
        return self.f(i, k, e) + self.f(k, j, e)

    def handle(self, TM):
        C, E = self.handle_G()  # Capacity, Edge
        F, D = self.handle_TM(TM)  # Flow, Demand
        return C, E, F, D

    def solve_ecmp(self, TM):
        C, E = self.handle_G()  # Capacity, Edge
        F, D = self.handle_TM(TM)
        num_e, num_f = len(E), len(F)
        U = []
        for e in range(num_e):
            u = 0
            for f in range(num_f):
                u += D[f] * self.f(F[f][0], F[f][1], E[e]) / C[e]
            U.append(u)
        print(f"Max load:{max(U)}")
        return max(U)

    def scale_TMs(self, TMs, load=1.0):
        ans = []
        for TM in tqdm(TMs):
            scaling = self.solve_ecmp(TM)
            newTM = np.array(TM) * (load / scaling)
            ans.append(newTM)
        return np.array(ans)

    def solve_by_gurobi(self, C, E, F, D, K):
        num_e, num_f, num_k = len(E), len(F), len(K)
        model = gp.Model(name="mlu")
        model.setParam("Threads", 2)
        model.setParam("Method", 2)  # barrier
        # varibles
        theta = model.addVar(name="theta", lb=0.0)
        action = model.addVars(num_f, num_k, lb=0.0)
        # demand constraints
        for f in range(num_f):
            model.addConstr(gp.quicksum(action[f, k] for k in range(num_k)) >= D[f])
        # utilization constraints
        for e in range(num_e):
            model.addConstr(
                LinExpr(
                    (self.g(i, j, K[k], E[e]), action[idx, k])
                    for k in range(num_k)
                    for idx, (i, j) in enumerate(F)
                )
                <= theta * C[e]
            )
        # Objective -> minimize \theta
        model.setObjective(theta, gp.GRB.MINIMIZE)
        model.optimize()
        # get solution
        if model.status == GRB.OPTIMAL:
            # grb_solution = model.getAttr('x', action)
            # print(grb_solution)
            ans = model.getObjective().getValue()
        model.dispose()
        del model
        return ans


if __name__ == "__main__":
    pass
