import networkx as nx
import numpy as np
import multiprocessing as mp
import torch
from sr import Topology, Traffic, NodeSelector, TESolver, solve


class SREnv(object):
    def __init__(self, toponame='Abilene', num_agents=1, k=5):
        self.topo = Topology(toponame)
        self.tms: np.array = Traffic(toponame).load_pickle()
        self.num_tm = len(self.tms)
        self.solver = TESolver(self.topo)
        self.solver.precompute_f()
        # self.g_dict = g_dict

        self.num_nodes = len(self.solver.G.nodes)
        self.state_dim = (self.num_nodes, self.num_nodes)
        self.action_dim = self.num_nodes
        self.num_agents = num_agents
        self.k = k
        self.baseline = {}
        self.C, self.E = self.solver.handle_G()

    def observe(self, idx=0):
        state = self.tms[idx]
        # state = np.array(state) / np.max(state) # normalization
        state = np.array(state)
        state = (state - state.mean()) / (state.std() + 1e-12) # normalization
        return state.tolist()
    
    def step(self, idx, action: list):
        assert idx < len(self.tms)
        tm = self.tms[idx]
        action = sorted(action)
        # u = self.solver.solve(tm, action)
        C, E = self.C, self.E
        F, D = self.solver.handle_TM(tm)
        # C, E, F, D = self.solver.handle(tm)
        uq = mp.Queue()
        p = mp.Process(target=solve, args=(self.solver, C, E, F, D, action, uq))
        p.start()
        p.join()
        u, t = uq.get()
        reward = 1 / u
        # reward = -10 * u

        # compute advantage
        total_reward, cnt = self.baseline.get(idx, (reward, 1))
        avg_reward = total_reward / cnt
        advantage = reward - avg_reward
        # update baseline
        return reward, advantage, t
    
    def update_baseline(self, idx, reward):
        total, cnt = self.baseline.get(idx, (0.0, 0))
        total += reward
        cnt += 1
        self.baseline[idx] = (total, cnt)

    def split_dataset(self, seed):
        idxes = np.arange(self.num_tm)
        np_state = np.random.RandomState(seed)
        np_state.shuffle(idxes)

        len_idxes = len(idxes)
        trainsize = int(0.7*len_idxes)
        trainset, testset = idxes[:trainsize], idxes[trainsize:]
        return trainset, testset
    
    def get_edge_index(self) -> torch.Tensor:
        edge_index = list(self.solver.G.edges())
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

if __name__ == '__main__':
    toponame = 'Abilene'
    for step in range(10):
        env = SREnv(toponame=toponame)
        ns = NodeSelector(Topology(toponame))
        action = ns.random(k=1)
        r = env.step(action)
        print(f'Step {step+1}: {r}')
