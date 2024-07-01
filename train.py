import logging
import torch
from torch_geometric.data import Data, Batch
import zmq
import numpy as np

from tqdm import tqdm
from env import SREnv
from drl import PPOGCN, REINFORCE, PPO
from config import config

logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)


def generate_agent(srenv):
    # device = torch.device("cpu")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.method == 'REINFORCE':
        agent = REINFORCE(srenv.state_dim, srenv.action_dim, device)
    elif config.method == 'PPO':
        agent = PPO(srenv.state_dim, srenv.action_dim, device)
    elif config.method == 'PPOGCN':
        agent = PPOGCN(srenv.state_dim[0], srenv.action_dim, device)
    return agent


def generate_actions(agent, s_batch):
    if config.method == 'PPOGCN':
        actions, one_hot_actions, log_probs = agent.select_action(
            s_batch, config.k)
    else:
        s_batch = np.array(s_batch)
        actions, one_hot_actions, log_probs = agent.select_action(
            s_batch, config.k)
    actions = actions.reshape((config.num_agents, config.iter_steps, config.k))
    a_batch = np.array(one_hot_actions)
    lp_batch = np.array(log_probs)
    return actions, s_batch, a_batch, lp_batch


def generate_task(srenv, train_tm_idx_subsets, agent):
    idxes = []
    s_batch = []
    edge_index = srenv.get_edge_index()
    for i in range(srenv.num_agents):
        subset = train_tm_idx_subsets[i]
        # print(f'subset shape: {subset.shape}')
        subset = np.random.choice(subset, config.iter_steps, replace=False)
        # print(f'subset per step per agent: {subset}')
        idxes.append(subset)
        for idx in subset:
            state = srenv.observe(idx)
            if config.method == 'PPOGCN':
                state = torch.tensor(state, dtype=torch.float)
                s_batch.append(Data(x=state, edge_index=edge_index))
            else:
                s_batch.append([state])
    actions, s_batch, a_batch, lp_batch = generate_actions(agent, s_batch)

    # Task -> [(idxes, actions), ...]
    return list(zip(idxes, actions)), s_batch, a_batch, lp_batch


def train(srenv: SREnv, train_tm_idx_set):
    c = zmq.Context()
    push_sock = c.socket(zmq.PUSH)
    push_sock.bind(f"ipc:///tmp/tasks{config.rid}")
    pull_sock = c.socket(zmq.PULL)
    pull_sock.bind(f"ipc:///tmp/results{config.rid}")

    num_agents = srenv.num_agents

    push_sock.send_pyobj(srenv)
    agent = generate_agent(srenv)
    train_tm_idx_subsets = np.array_split(train_tm_idx_set, num_agents)
    max_steps = config.train_steps
    best = 0
    for step in tqdm(range(1, max_steps + 1)):
        # push tasks
        task, s_batch, a_batch, lp_batch = generate_task(
            srenv, train_tm_idx_subsets, agent)
        logger.debug(f'Step {step}: Sending task to workers')
        push_sock.send_pyobj(task)
        # pull results
        r_batch, ad_batch, times = pull_sock.recv_pyobj()
        logger.debug(f'Step {step}: Pulling results from workers')
        r_batch = np.array(r_batch)
        ad_batch = np.array(ad_batch)
        avg_r = np.mean(r_batch)
        avg_time = np.mean(times)
        # update
        agent.update(s_batch, a_batch, r_batch, ad_batch, lp_batch)
        print(
            f'Step {step}: train result {avg_r}, average time {avg_time}', flush=True)
        if avg_r > best:
            best = avg_r
        agent.save_parameters(f'{config.model_dir}/{config.toponame}')
        if step % 50 == 0:
            agent.save_parameters(
                f'{config.model_dir}/{config.toponame}-{step}')
    # signal of stop
    push_sock.send_pyobj(None)


def main():
    toponame = config.toponame
    num_agents = config.num_agents

    logger.info(
        f'Train topo {toponame} with {num_agents} agents, select {config.k} middlepoints')
    srenv = SREnv(toponame=toponame, num_agents=num_agents, k=config.k)
    trainset, _ = srenv.split_dataset(config.seed)
    logger.info(f'Training parameters:')
    logger.info(f'{config}')
    print(config)
    train(srenv, trainset)


if __name__ == '__main__':
    main()
