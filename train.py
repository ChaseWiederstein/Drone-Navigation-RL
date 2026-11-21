from isaaclab.app import AppLauncher


HEADLESS = True
TOTAL_UPDATES = 400
ROLLOUT_STEPS = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CHECKPOINT_INTERVAL = 5
CHECKPOINT_PATH = r"C:\robotics\droneRL\checkpoints"
ACTION_SCALE = (2.0, 2.0, 2.0)


app_launcher = AppLauncher(headless=HEADLESS, device="cuda:0", enable_cameras=True)
simulation_app = app_launcher.app


import os
import time

import torch
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.sim import SimulationCfg, SimulationContext

enable_extension("isaacsim.sensors.camera")

from drone_env import DroneEnv
from NeurNetworks import ActorCritic
from ppo import PPO, PPOConfig
from scene import DEVICE, NUM_ENVS, DroneScene


def scale_actions(normalized_actions, device):
    action_scale = torch.tensor(ACTION_SCALE, device=device, dtype=torch.float32)
    return (2.0 * normalized_actions - 1.0) * action_scale


def stack_obs(obs_list):
    return {
        "depth": torch.stack([obs["depth"] for obs in obs_list], dim=0),
        "state": torch.stack([obs["state"] for obs in obs_list], dim=0),
    }


def flatten_obs(obs):
    return {
        "depth": obs["depth"].flatten(0, 1),
        "state": obs["state"].flatten(0, 1),
    }


def compute_gae(rewards, dones, values, last_value):
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    next_value = last_value

    for step in reversed(range(ROLLOUT_STEPS)):
        not_done = 1.0 - dones[step].float()
        delta = rewards[step] + GAMMA * next_value * not_done - values[step]
        gae = delta + GAMMA * GAE_LAMBDA * not_done * gae
        advantages[step] = gae
        next_value = values[step]

    returns = advantages + values
    return returns, advantages


def main():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    sim = SimulationContext(SimulationCfg(device=DEVICE))

    scene = DroneScene(sim)
    scene.build()

    sim.reset()
    scene.connect_drone()

    env = DroneEnv(scene, num_envs=NUM_ENVS, device=DEVICE)
    obs = env.reset()
    episode_returns = torch.zeros(NUM_ENVS, device=env.device)
    last_episode_returns = []

    policy = ActorCritic().to(env.device)
    policy.load_state_dict(torch.load(
        r"C:\robotics\droneRL\checkpoints\policy_475_good_tree_policy.pt",
        map_location=env.device,
    ))
    ppo = PPO(policy, PPOConfig())
    train_start_time = time.perf_counter()
    last_print_time = train_start_time

    success_window = []
    success_window_size = 20
    success_threshold = .8
    success_rate = 0.0
    difficulty = 0
    max_difficulty = 5

    for update in range(TOTAL_UPDATES):
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_output_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []

        for _ in range(ROLLOUT_STEPS):
            obs_buffer.append({k: v.detach().clone() for k, v in obs.items()})

            with torch.no_grad():
                normalized_actions, log_probs, values = policy.act(obs)

            actions = scale_actions(normalized_actions, env.device)
            next_obs, rewards, dones = env.step(actions)
            episode_returns += rewards.detach()

            action_buffer.append(normalized_actions.detach())
            log_prob_buffer.append(log_probs.detach())
            value_output_buffer.append(values.detach())
            value_buffer.append(ppo.denormalize_values(values).detach())
            reward_buffer.append(rewards.detach())
            done_buffer.append(dones.detach().clone())

            if dones.any():
                done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                finished_returns = episode_returns[done_ids].detach().cpu().tolist()
                last_episode_returns.extend(finished_returns)
                last_episode_returns = last_episode_returns[-8:]
                episode_returns[done_ids] = 0.0

                episode_successes = env.success[done_ids].detach().cpu().tolist()
                success_window.extend(float(success) for success in episode_successes)
                success_window = success_window[-success_window_size:]

                if success_window:
                    success_rate = sum(success_window) / len(success_window)

                if len(success_window) == success_window_size and success_rate >= success_threshold:
                    difficulty = min(difficulty + 1, max_difficulty)
                    success_window.clear()
                    env.set_difficulty(difficulty)

                next_obs = env.reset(done_ids)

            obs = next_obs

        with torch.no_grad():
            last_value_output = policy.value(obs)
            last_value = ppo.denormalize_values(last_value_output)

        rollout_obs = flatten_obs(stack_obs(obs_buffer))
        rollout_actions = torch.stack(action_buffer, dim=0).flatten(0, 1)
        rollout_log_probs = torch.stack(log_prob_buffer, dim=0).flatten(0, 1)
        rollout_value_outputs = torch.stack(value_output_buffer, dim=0)
        rollout_values = torch.stack(value_buffer, dim=0)
        rollout_rewards = torch.stack(reward_buffer, dim=0)
        rollout_dones = torch.stack(done_buffer, dim=0)

        returns, advantages = compute_gae(rollout_rewards, rollout_dones, rollout_values, last_value)
        stats = ppo.update(
            rollout_obs,
            rollout_actions,
            rollout_log_probs,
            rollout_value_outputs.flatten(0, 1),
            returns.flatten(0, 1),
            advantages.flatten(0, 1),
        )

        if update % 5 == 0:
            now = time.perf_counter()
            normalized_action_stack = torch.stack(action_buffer)
            applied_action_stack = scale_actions(normalized_action_stack, env.device)
            _, _, final_linear_vel, _ = env.get_drone_state()
            final_linear_vel = torch.as_tensor(final_linear_vel, device=env.device, dtype=torch.float32)
            mean_episode_return = sum(last_episode_returns) / max(len(last_episode_returns), 1)

            print(
                "\n"
                f"Update: {update}  "
                f"Time: {now - last_print_time:.2f}s    "
                f"Avg Reward: {float(rollout_rewards.mean().detach().cpu()):.3f}    "
                f"Mean Episode Return: {mean_episode_return:.3f}    "
                f"Done Rate: {float(rollout_dones.float().mean().detach().cpu()):.3f}   "
                f"Success Rate: {float(success_rate):.3f}   "
                f"Difficulty: {difficulty}\n"
                f"Last Episode Returns: {[round(x, 2) for x in last_episode_returns]}\n"
                f"Applied Action Dim Mean: {[round(x, 3) for x in applied_action_stack.mean(dim=(0, 1)).detach().cpu().tolist()]}\n"
                f"Loss: {stats['loss']:.3f} | Actor: {stats['actor_loss']:.3f} | Critic: {stats['critic_loss']:.3f}\n",
                flush=True,
            )
            last_print_time = now

        if update % CHECKPOINT_INTERVAL == 0:
            torch.save(policy.state_dict(), os.path.join(CHECKPOINT_PATH, f"policy_{update}.pt"))

        if not simulation_app.is_running():
            break

    torch.save(policy.state_dict(), os.path.join(CHECKPOINT_PATH, "policy_final.pt"))
    simulation_app.close()


if __name__ == "__main__":
    main()
