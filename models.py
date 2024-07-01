import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from config import config


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = config.hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, config.num_filters, config.kernel_size),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        n = self.state_dim[0]
        fc_dim = (n-config.kernel_size+1) * \
            (n-config.kernel_size+1) * config.num_filters
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out)
        # out = self.softmax(out)
        return out


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = config.hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, config.num_filters, config.kernel_size),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        n = self.state_dim[0]
        fc_dim = (n-config.kernel_size+1) * \
            (n-config.kernel_size+1) * config.num_filters
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out)
        return out


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = config.hidden_dim

        n = self.state_dim[0]
        fc_dim = (n-config.kernel_size+1) * \
            (n-config.kernel_size+1) * config.num_filters

        # actor head
        self.actor = nn.Sequential(
            nn.Conv2d(1, config.num_filters, config.kernel_size),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(fc_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Conv2d(1, config.num_filters, config.kernel_size),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(fc_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class GCN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_dim, dropout):
        super(GCN, self).__init__()
        self.state_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.fc_hidden_dim = config.hidden_dim

        self.gc1 = GCNConv(feature_dim, hidden_dim)
        # self.gc2 = GCNConv(hidden_dim, 256)
        self.dropout = dropout

        self.fc = nn.Sequential(
            nn.Linear(self.state_dim*hidden_dim, self.fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.fc_hidden_dim, self.action_dim),
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.leaky_relu(x)
        # x = self.gc2(x, edge_index)
        # x = F.leaky_relu(x)
        x = x.reshape((-1, self.state_dim*self.hidden_dim))
        x = self.fc(x)
        return x

