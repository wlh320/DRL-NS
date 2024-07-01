# global config
class Config:
    # run
    rid = 0
    # basic
    toponame = 'rf6461'
    data_dir = './data/'
    model_dir = './model/'
    result_dir = './result/'
    figure_dir = './figure/'
    # method = 'PPO'
    # method = 'REINFORCE'
    method = 'PPOGCN'

    # NN model
    num_filters = 64
    kernel_size = 3
    hidden_dim = 64

    # training
    num_agents = 20  # {} agents
    # num_test_agents = 20 # {} test agents
    iter_steps = 3  # each agent collect {} samples for one step
    train_steps = 200  # update the NN parameters {} steps in total
    k = 5  # select {} candidates of middle point
    lr = 0.0001  # optimizer has learning rate {}
    entropy_lr = 0.1  # entropy item in gradient has {} weight

    # PPO
    k_epoch = 10
    eps_clip = 0.2
    ppo_lr = 0.001
    ppo_entropy_lr = 0.2
    ppo_critic_lr = 0.0001

    # PPOGCN
    # batch_size = 32
    hidden_features = 128

    # other
    log_level = 'INFO'
    seed = 1024  # random seed used for splitting dataset

    def __repr__(self):
        vs = list(filter(lambda x: '__' not in x[0], vars(Config).items()))
        return '\n'.join(f'{k}: {v}' for k, v in vs)


config = Config()

if __name__ == '__main__':
    print(config)
