from .dqn_modeling import NetworkDQN
from torch import nn
import numpy as np
class SmallNetwork(NetworkDQN):
    def __init__(self, env):
        super(SmallNetwork, self).__init__()
        in_features = int(np.prod(env.observation_space.shape))  # Number of inputs in the layer

        self.net = nn.Sequential(nn.Linear(in_features, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)


class BigNetwork(NetworkDQN):
    def __init__(self, env):
        super(BigNetwork, self).__init__()
        in_features = int(np.prod(env.observation_space.shape))  # Number of inputs in the layer

        self.net = nn.Sequential(nn.Linear(in_features, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)

