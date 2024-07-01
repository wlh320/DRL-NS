# compare SRTE solution time
import pickle
import time
from sr import NodeSelector, Topology
import logging
import os
import torch
from torch_geometric.data import Data, DataLoader
import zmq
import numpy as np

from env import SREnv
from drl import REINFORCE, PPO, PPOGCN
from config import config

logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

METHODS = ['deg']


class TestResult:
    def __init__(self, idxes, rewards, times):
        self.idxes = idxes
        self.rewards = rewards
        self.times = times

    def to_dict(self):
        result = {}
        for (idx, reward, time) in zip(self.idxes, self.rewards, self.times):
            result[idx] = (reward, time)
        return result


def generate_agent(srenv):
    device = torch.device("cpu")
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.method == 'REINFORCE':
        agent = REINFORCE(srenv.state_dim, srenv.action_dim, device)
    elif config.method == 'PPO':
        agent = PPO(srenv.state_dim, srenv.action_dim, device)
    elif config.method == 'PPOGCN':
        agent = PPOGCN(srenv.state_dim[0], srenv.action_dim, device)
    return agent


def generate_action_subset(srenv: SREnv, subset, method):
    """input a subset of index, return a set of action"""
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gen_start = time.time()
    if method == 'drl':
        agent = generate_agent(srenv)
        agent.load_parameters(f'{config.model_dir}/{config.toponame}')
        s_batch = []
        edge_index = srenv.get_edge_index()
        for idx in subset:
            state = srenv.observe(idx)
            if config.method == 'PPOGCN':
                state = torch.tensor(state, dtype=torch.float)
                s_batch.append(Data(x=state, edge_index=edge_index))
            else:
                s_batch.append([state])
        if config.method != 'PPOGCN':
            s_batch = np.array(s_batch)
        actions, _, _ = agent.select_action(s_batch, config.k)
        actions = actions.tolist()
    else:
        ns = NodeSelector(Topology(config.toponame))
        if method == 'ran':
            actions = [ns.random(config.k) for _ in subset]
        elif method == 'deg':
            actions = [ns.degree_centrality(config.k)] * len(subset)
        elif method == 'full':
            actions = [list(range(srenv.action_dim))] * len(subset)
        elif method == 'sp':
            actions = [ns.sp_centrality(config.k)] * len(subset)
        elif method == 'wsp':
            actions = [ns.weighted_sp_centrality(config.k)] * len(subset)
    gen_end = time.time()
    gen_time = gen_end - gen_start
    return actions, gen_time


def generate_test_task(srenv: SREnv, test_tm_idx_subsets, method):
    idxes = []
    for i in range(srenv.num_agents):
        subset = test_tm_idx_subsets[i]
        idxes.append(subset)
    actions = []
    t = 0
    for subset in idxes:
        action_subset, gen_time = generate_action_subset(srenv, subset, method)
        t += gen_time
        actions.append(action_subset)
    # Task -> [(idxes, actions), ...]
    return list(zip(idxes, actions)), t


def test(srenv: SREnv, test_tm_idx_set):
    c = zmq.Context()
    push_sock = c.socket(zmq.PUSH)
    push_sock.bind(f"ipc:///tmp/tasks{config.rid}")
    pull_sock = c.socket(zmq.PULL)
    pull_sock.bind(f"ipc:///tmp/results{config.rid}")

    num_agents = srenv.num_agents
    push_sock.send_pyobj(srenv)
    test_tm_idx_subsets = np.array_split(test_tm_idx_set, num_agents)
    try:
        file = open(f'{config.result_dir}{config.toponame}-compare.pkl', 'rb')
        test_results = pickle.load(file)
    except Exception as e:
        test_results = {}
    for method in METHODS:
        # push tasks
        task, gen_time = generate_test_task(
            srenv, test_tm_idx_subsets, method)
        gen_time /= len(test_tm_idx_set)
        print(gen_time)

        push_sock.send_pyobj(task)
        # pull results
        rewards, _, times = pull_sock.recv_pyobj()
        logger.debug(f'Method {method}: Pulling results from workers')
        # update
        maxr, minr, avgr = np.max(rewards), np.min(rewards), np.mean(rewards)
        avg_time = np.mean(times)
        print(f'method:{method} max:{maxr} min:{minr} avg:{avgr}', flush=True)
        print(f'average solving time: {avg_time:.3f} s', flush=True)

        times = (np.array(times) + gen_time).tolist()
        result = TestResult(test_tm_idx_set, rewards, times)
        test_results[method] = result
        os.system('clear')
    # dump results
    pickle.dump(test_results, open(
        f'{config.result_dir}{config.toponame}-compare.pkl', 'wb'))
    # signal of stop
    push_sock.send_pyobj(None)


def main():
    toponame = config.toponame
    num_agents = config.num_agents

    logger.info(
        f'Train topo {toponame} with {num_agents} agents, select {config.k} middlepoints')
    srenv = SREnv(toponame=toponame, num_agents=num_agents, k=config.k)

    _, testset = srenv.split_dataset(config.seed)
    # testset = testset[:100]
    logger.info(f'Training parameters:')
    logger.info(f'{config}')
    print(config)
    test(srenv, testset)


if __name__ == '__main__':
    main()
