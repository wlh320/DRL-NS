import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader, batch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from config import config
from models import GCN, Actor, ActorCritic, Critic, ACGCN


class REINFORCE:
    """single-step REINFORCE (Monte Carlo Policy Gradient) with baseline"""

    def __init__(self, state_dim, action_dim, device):
        self.lr = config.lr
        self.entropy_lr = config.entropy_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.optim = optim.Adam(self.actor.parameters(), lr=self.lr)

    def select_action(self, state, k):
        state = torch.from_numpy(state).float().to(self.device)
        logits = self.actor(state)
        probs = F.softmax(logits, dim=1).cpu()
        action = torch.multinomial(probs, k).squeeze()
        probs = torch.unsqueeze(probs, dim=-1)
        one_hot_action = np.eye(self.action_dim, dtype=np.float32)[
            np.array(action)]
        one_hot_action = torch.Tensor(one_hot_action)
        log_probs = torch.log(torch.squeeze(
            torch.matmul(one_hot_action, probs)+1e-9)).sum(dim=1)
        return action.detach().numpy(), one_hot_action.detach().numpy(), log_probs.detach().numpy()

    def update(self, s_batch, a_batch, r_batch, ad_batch, lp_batch):
        s_batch = torch.FloatTensor(s_batch).to(self.device)  # matrix
        a_batch = torch.Tensor(a_batch).to(self.device)  # one-hot
        ad_batch = torch.FloatTensor(ad_batch).to(self.device)  # vector
        adv = Variable(ad_batch, requires_grad=False)
        for _ in range(config.k_epoch):
            eps = 1e-12
            logits = self.actor(s_batch)
            probs = F.softmax(logits, dim=1)
            m = Categorical(probs)
            entropy = m.entropy()
            probs = torch.unsqueeze(probs, dim=-1)
            policy_loss = torch.log(torch.squeeze(
                torch.matmul(a_batch, probs)+eps)).sum(dim=1)
            policy_loss = (-adv * policy_loss).mean()
            entropy_loss = self.entropy_lr * entropy.mean()
            total_loss = policy_loss - entropy_loss

            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        print(
            f'policy: {policy_loss} entropy: {entropy_loss} total: {total_loss}', flush=True)

    def get_parameters(self):
        return self.actor.state_dict()

    def set_parameters(self, state_dict):
        self.actor.load_state_dict(state_dict)

    def save_parameters(self, name):
        torch.save(self.actor.state_dict(), f'{name}.pkl')

    def load_parameters(self, name):
        self.actor.load_state_dict(
            torch.load(f'{name}.pkl')
        )


class PPO:
    """single-step version of PPO"""

    def __init__(self, state_dim, action_dim, device):
        self.lr = config.ppo_lr
        self.entropy_lr = config.ppo_entropy_lr
        self.critic_lr = config.ppo_critic_lr
        self.k_epoch = config.k_epoch
        self.device = device
        self.eps_clip = config.eps_clip
        self.action_dim = action_dim
        # self.actor = Actor(state_dim, action_dim)
        # self.critic = Critic(state_dim, action_dim)
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optim = optim.Adam([
            {'params': self.ac.actor.parameters(), 'lr': self.lr},
            {'params': self.ac.critic.parameters(), 'lr': self.critic_lr},
        ])
        # self.coptim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        # self.optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        # self.ac_old = ActorCritic(state_dim, action_dim).to(self.device)
        # self.ac_old = Actor(state_dim, action_dim)
        # self.ac_old.load_state_dict(self.actor.state_dict())

    def select_action(self, state, k):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            logits, _ = self.ac(state)
        probs = F.softmax(logits, dim=1).cpu()
        # m = Categorical(probs)
        action = torch.multinomial(probs, k).squeeze().cpu()
        # log_prob = m.log_prob(action).sum()
        probs = torch.unsqueeze(probs, dim=-1)
        one_hot_action = np.eye(self.action_dim, dtype=np.float32)[
            np.array(action)]
        one_hot_action = torch.Tensor(one_hot_action)
        log_probs = torch.log(torch.squeeze(
            torch.matmul(one_hot_action, probs)+1e-9)).sum(dim=1)
        return action.detach().numpy(), one_hot_action.detach().numpy(), log_probs.detach().numpy()

    def update(self, s_batch, a_batch, r_batch, ad_batch, lp_batch):
        states = torch.FloatTensor(s_batch).detach().to(self.device)  # matrix
        actions = torch.Tensor(a_batch).detach().to(self.device)  # one-hot
        rewards = torch.Tensor(r_batch).detach().to(self.device)
        advantages = torch.FloatTensor(ad_batch).detach().to(self.device)
        old_log_probs = torch.FloatTensor(lp_batch).detach().to(self.device)
        eps = 1e-9

        # reward normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + eps)

        for _ in range(self.k_epoch):
            logits, v_values = self.ac(states)
            v_values = torch.squeeze(v_values)
            probs = F.softmax(logits, dim=1)
            m = Categorical(probs)
            entropy = m.entropy()
            probs = torch.unsqueeze(probs, dim=-1)
            log_probs = torch.log(torch.squeeze(
                torch.matmul(actions, probs)+eps)).sum(dim=1)
            # advantages = rewards - v_values.detach()

            # surrogate loss
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            s_loss = -torch.min(surr1, surr2).mean()
            # entropy loss
            e_loss = self.entropy_lr * entropy.mean()
            # critic loss
            c_loss = F.mse_loss(v_values, rewards.detach())
            c_loss = 0

            total_loss = s_loss + c_loss - e_loss
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        print(
            f'policy: {s_loss.mean()} critic: {c_loss} entropy: {e_loss.mean()} total: {total_loss}', flush=True)

        # self.ac_old.load_state_dict(self.actor.state_dict())

    def get_parameters(self):
        return self.ac.state_dict()

    def set_parameters(self, state_dict):
        self.ac.load_state_dict(state_dict)

    def save_parameters(self, name):
        torch.save(self.ac.state_dict(), f'{name}.pkl')

    def load_parameters(self, name):
        self.ac.load_state_dict(
            torch.load(f'{name}.pkl')
        )


class PPOGCN:
    """single-step version of PPO with GCN"""

    def __init__(self, state_dim, action_dim, device):
        self.lr = config.ppo_lr
        self.entropy_lr = config.ppo_entropy_lr
        self.k_epoch = config.k_epoch
        self.device = device
        self.eps_clip = config.eps_clip
        self.action_dim = action_dim
        self.gcn = GCN(state_dim, hidden_dim=config.hidden_features,
                       action_dim=action_dim, dropout=0.25).to(device)
        self.optim = optim.Adam(self.gcn.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optim, gamma=0.98)

    def select_action(self, state, k):
        self.gcn.eval()
        loader = DataLoader(state, batch_size=len(state))
        for data in loader:
            data = data.to(self.device)
            logits = self.gcn(data)
            # print(logits.shape)
            logits = logits.reshape((len(state), -1))
            probs = F.softmax(logits, dim=1).cpu()
            action = torch.multinomial(probs, k).squeeze()
            probs = torch.unsqueeze(probs, dim=-1)
            one_hot_action = np.eye(self.action_dim, dtype=np.float32)[
                np.array(action)]
            one_hot_action = torch.Tensor(one_hot_action)
            log_probs = torch.log(torch.squeeze(
                torch.matmul(one_hot_action, probs)+1e-9)).sum(dim=1)
        return action.detach().numpy(), one_hot_action.detach().numpy(), log_probs.detach().numpy()

    def update(self, s_batch, a_batch, r_batch, ad_batch, lp_batch):
        # s_batch = torch.FloatTensor(s_batch).detach().to(self.device) # matrix
        # s_batch = ts_batch.to(self.device)
        loader = DataLoader(s_batch, batch_size=len(s_batch))
        actions = torch.Tensor(a_batch).detach().to(self.device)  # one-hot
        rewards = torch.Tensor(r_batch).detach().to(self.device)
        advantages = torch.FloatTensor(ad_batch).detach().to(self.device)
        old_log_probs = torch.FloatTensor(lp_batch).detach().to(self.device)
        eps = 1e-9

        # reward normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + eps)

        self.gcn.train()
        for data in loader:
            data = data.to(self.device)
            for _ in range(self.k_epoch):
                logits = self.gcn(data)
                logits = logits.reshape((len(s_batch), -1))
                probs = F.softmax(logits, dim=1)
                m = Categorical(probs)
                entropy = m.entropy()
                probs = torch.unsqueeze(probs, dim=-1)
                log_probs = torch.log(torch.squeeze(
                    torch.matmul(actions, probs)+eps)).sum(dim=1)
                # advantages = rewards - v_values.detach()

                # surrogate loss
                ratios = torch.exp(log_probs - old_log_probs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                    1+self.eps_clip) * advantages
                s_loss = -torch.min(surr1, surr2).mean()
                # entropy loss
                e_loss = self.entropy_lr * entropy.mean()

                total_loss = s_loss - e_loss
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()

        print(
            f'policy: {s_loss.mean()} entropy: {e_loss.mean()} total: {total_loss} lr: {lr}', flush=True)

    def get_parameters(self):
        return self.gcn.state_dict()

    def set_parameters(self, state_dict):
        self.gcn.load_state_dict(state_dict)

    def save_parameters(self, name):
        torch.save(self.gcn.state_dict(), f'{name}-PPOGCN.pkl')

    def load_parameters(self, name):
        self.gcn.load_state_dict(
            torch.load(f'{name}-PPOGCN.pkl')
        )


if __name__ == '__main__':
    state = np.ones(shape=(1, 1, 20, 20)) * 10000
    r = REINFORCE(state_dim=(20, 20), action_dim=10,
                  device=torch.device("cuda"))
    d = r.get_parameters()
    r1 = REINFORCE(state_dim=(20, 20), action_dim=10,
                   device=torch.device("cuda"))
    print(r1.get_parameters())
    r1.set_parameters(d)

    print(r.get_parameters())
    print(r1.get_parameters())
