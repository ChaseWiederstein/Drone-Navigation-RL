from dataclasses import dataclass

import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    def __init__(self, input_shape=1, beta=0.995, epsilon=1e-5):
        super().__init__()
        self.input_shape = torch.Size((input_shape,)) if isinstance(input_shape, int) else torch.Size(input_shape)
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(self.input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(self.input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

    def running_mean_var(self):
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (mean_sq - mean**2).clamp(min=1e-2)
        return mean, var

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, *self.input_shape)
        batch_mean = values.mean(dim=0)
        batch_sq_mean = (values**2).mean(dim=0)
        weight = self.beta
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 - weight)

    def normalize(self, values):
        mean, var = self.running_mean_var()
        return ((values.unsqueeze(-1) - mean) / torch.sqrt(var)).squeeze(-1)

    def denormalize(self, values):
        mean, var = self.running_mean_var()
        return (values.unsqueeze(-1) * torch.sqrt(var) + mean).squeeze(-1)


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    clip_ratio: float = 0.1
    entropy_coef: float = 1e-3
    epochs: int = 4
    num_minibatches: int = 16
    max_grad_norm: float = 5.0
    huber_delta: float = 10.0


class PPO:
    def __init__(self, policy, config=None):
        self.policy = policy
        self.config = config or PPOConfig()
        self.value_norm = ValueNorm(1).to(next(self.policy.parameters()).device)
        self.critic_loss_fn = nn.HuberLoss(delta=self.config.huber_delta)

        feature_params = list(self.policy.depth_encoder.parameters()) + list(self.policy.shared.parameters())
        actor_params = list(self.policy.actor_alpha.parameters()) + list(self.policy.actor_beta.parameters())
        critic_params = list(self.policy.critic.parameters())

        self.feature_optimizer = torch.optim.Adam(feature_params, lr=self.config.learning_rate)
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.config.learning_rate)

    def denormalize_values(self, values):
        return self.value_norm.denormalize(values)

    def _minibatch_indices(self, batch_size, device):
        usable_size = (batch_size // self.config.num_minibatches) * self.config.num_minibatches
        if usable_size == 0:
            raise ValueError(
                f"Batch size {batch_size} is too small for {self.config.num_minibatches} minibatches."
            )

        perm = torch.randperm(usable_size, device=device).reshape(self.config.num_minibatches, -1)
        return perm

    def _obs_batch(self, obs, indices):
        return {
            "depth": obs["depth"][indices],
            "state": obs["state"][indices],
        }

    def update(self, obs, actions, old_log_probs, old_value_outputs, returns, advantages):
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-7)

        self.value_norm.update(returns)
        normalized_returns = self.value_norm.normalize(returns)

        batch_size = actions.shape[0]
        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        total_entropy = 0.0
        total_actor_grad_norm = 0.0
        total_critic_grad_norm = 0.0
        total_explained_var = 0.0
        num_updates = 0

        for _ in range(self.config.epochs):
            for indices in self._minibatch_indices(batch_size, actions.device):
                mb_obs = self._obs_batch(obs, indices)
                mb_actions = actions[indices]
                mb_old_log_probs = old_log_probs[indices]
                mb_old_values = old_value_outputs[indices]
                mb_returns = normalized_returns[indices]
                mb_advantages = advantages[indices]

                log_probs, entropy, values = self.policy.evaluate_actions(mb_obs, mb_actions)

                entropy_loss = -self.config.entropy_coef * entropy.mean()

                ratio = torch.exp(log_probs - mb_old_log_probs)
                unclipped = mb_advantages * ratio
                clipped = mb_advantages * ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                actor_loss = -torch.min(unclipped, clipped).mean() * self.policy.action_dim

                value_clipped = mb_old_values + (values - mb_old_values).clamp(
                    -self.config.clip_ratio,
                    self.config.clip_ratio,
                )
                critic_loss_clipped = self.critic_loss_fn(mb_returns, value_clipped)
                critic_loss_original = self.critic_loss_fn(mb_returns, values)
                critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

                loss = entropy_loss + actor_loss + critic_loss

                self.feature_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                actor_grad_norm = nn.utils.clip_grad_norm_(
                    list(self.policy.actor_alpha.parameters()) + list(self.policy.actor_beta.parameters()),
                    max_norm=self.config.max_grad_norm,
                )
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(),
                    max_norm=self.config.max_grad_norm,
                )

                self.feature_optimizer.step()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                explained_var = 1.0 - (values - mb_returns).pow(2).mean() / mb_returns.var().clamp_min(1e-7)

                total_loss += float(loss.detach().cpu())
                total_actor_loss += float(actor_loss.detach().cpu())
                total_critic_loss += float(critic_loss.detach().cpu())
                total_entropy_loss += float(entropy_loss.detach().cpu())
                total_entropy += float(entropy.mean().detach().cpu())
                total_actor_grad_norm += float(actor_grad_norm.detach().cpu())
                total_critic_grad_norm += float(critic_grad_norm.detach().cpu())
                total_explained_var += float(explained_var.detach().cpu())
                num_updates += 1

        return {
            "loss": total_loss / num_updates,
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "actor_grad_norm": total_actor_grad_norm / num_updates,
            "critic_grad_norm": total_critic_grad_norm / num_updates,
            "explained_var": total_explained_var / num_updates,
            "ppo_minibatches": num_updates,
        }
