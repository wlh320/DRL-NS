import time
import logging
import zmq
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from env import SREnv
from config import config

logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)


def agent_proc(i, srenv: SREnv, task_queue, result_queue, barlock):
    while True:
        # queue get task
        tm_idx_subset, actions = task_queue.get()
        rewards = []
        advantages = []
        times = []
        # progress bar
        with barlock:
            pbar = tqdm(total=len(actions),
                        desc=f'#{i:02}', leave=False, position=i)
        # print(tm_idx_subset, actions)
        for tm_idx, action in zip(tm_idx_subset, actions):
            action = np.array(action)
            reward, advantage, solve_time = srenv.step(tm_idx, action)
            srenv.update_baseline(tm_idx, reward)
            # print(srenv.baseline)
            rewards.append(reward)
            advantages.append(advantage)
            times.append(solve_time)
            logger.debug(
                f'worker{i} tm_idx: {tm_idx} reward:{reward}, adv:{advantage}')
            with barlock:
                pbar.update(1)
        # queue send result
        # print(advantages)
        result_queue.put([rewards, advantages, times])


def run_worker():
    c = zmq.Context()
    push_sock = c.socket(zmq.PUSH)
    push_sock.connect(f"ipc:///tmp/results{config.rid}")
    pull_sock = c.socket(zmq.PULL)
    pull_sock.connect(f"ipc:///tmp/tasks{config.rid}")

    srenv = pull_sock.recv_pyobj()
    logger.debug('get SRenv object')

    num_agents = srenv.num_agents
    task_queues = []
    result_queues = []
    barlock = mp.Manager().RLock()
    for _ in range(num_agents):
        task_queues.append(mp.Queue(1))
        result_queues.append(mp.Queue(1))
    agents = []
    for i in range(num_agents):
        a = mp.Process(target=agent_proc, args=(
            i, srenv, task_queues[i], result_queues[i], barlock))
        agents.append(a)
    for i in range(num_agents):
        agents[i].start()

    while True:
        # get task
        task = pull_sock.recv_pyobj()
        logger.debug(f'Received task from master')
        logger.debug(f"Recv task")
        if task is None:
            break
        # run task
        for i in range(num_agents):
            task_queues[i].put(task[i])
        reward_batch = []
        adv_batch = []
        time_batch = []
        for i in range(num_agents):
            rewards, advantages, times = result_queues[i].get()
            reward_batch.extend(rewards)
            adv_batch.extend(advantages)
            time_batch.extend(times)
        # print('', flush=True)
        # push results
        push_sock.send_pyobj((reward_batch, adv_batch, time_batch))
        logger.debug(f'Send results to master')

    for i in range(num_agents):
        agents[i].kill()


if __name__ == '__main__':
    run_worker()
