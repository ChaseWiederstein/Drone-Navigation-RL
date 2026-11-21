import torch
import torch.nn as nn
from torch.distributions import Beta, Independent


class ActorCritic(nn.Module):
    def __init__(self, state_dim=8, action_dim=3):
        super().__init__()
        self.action_dim = action_dim

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.LayerNorm(128),
            nn.ELU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(128 + state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )

        self.actor_alpha = nn.Linear(256, action_dim)
        self.actor_beta = nn.Linear(256, action_dim)
        self.softplus = nn.Softplus()
        self.critic = nn.Linear(256, 1)

    def features(self, obs):
        depth_feature = self.depth_encoder(obs["depth"])
        feature = torch.cat([depth_feature, obs["state"]], dim=-1)
        return self.shared(feature)

    def distribution(self, obs):
        feature = self.features(obs)
        alpha = 1.0 + self.softplus(self.actor_alpha(feature)) + 1e-6
        beta = 1.0 + self.softplus(self.actor_beta(feature)) + 1e-6
        return Independent(Beta(alpha, beta), 1)

    def value(self, obs):
        feature = self.features(obs)
        return self.critic(feature).squeeze(-1)

    def act(self, obs):
        dist = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value(obs)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        dist = self.distribution(obs)
        actions = actions.clamp(1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs)
        return log_prob, entropy, value


class LegacyGaussianActorCritic(nn.Module):
    def __init__(self, state_dim=8, action_dim=3):
        super().__init__()
        self.action_dim = action_dim

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.LayerNorm(128),
            nn.ELU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(128 + state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )

        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(256, 1)

    def features(self, obs):
        depth_feature = self.depth_encoder(obs["depth"])
        feature = torch.cat([depth_feature, obs["state"]], dim=-1)
        return self.shared(feature)

    def mean_action(self, obs):
        return self.actor_mean(self.features(obs))

    def value(self, obs):
        feature = self.features(obs)
        return self.critic(feature).squeeze(-1)

class CNN(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
          nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
          nn.ELU(),

          nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
          nn.ELU(),

          nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
          nn.ELU(),

          nn.AdaptiveAvgPool2d((4, 4)),
          nn.Flatten(),

          nn.Linear(32 * 4 * 4, output_dim),
          nn.LayerNorm(output_dim),
          nn.ELU(),
        )

    def forward(self, depth):
      return self.net(depth)
