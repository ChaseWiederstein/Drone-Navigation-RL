import argparse
import random
from isaaclab.app import AppLauncher


DEFAULT_CHECKPOINT = r"C:\robotics\droneRL\checkpoints\best_tuned_policy.pt"
TEST_DIFFICULTIES = [0, 1, 2, 3, 4, 5]
ACTION_SCALE = (2.0, 2.0, 2.0)
TEST_NUM_ENVS = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to a saved policy .pt file.")
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps. 0 means run until closed.")
    parser.add_argument("--print-every", type=int, default=2)
    return parser.parse_args()


args = parse_args()

app_launcher = AppLauncher(headless=args.headless, device="cuda:0", enable_cameras=True)
simulation_app = app_launcher.app


import torch
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.sim import SimulationCfg, SimulationContext

enable_extension("isaacsim.sensors.camera")

from drone_env import DroneEnv
from NeurNetworks import ActorCritic, LegacyGaussianActorCritic
import scene as scene_module

scene_module.NUM_ENVS = TEST_NUM_ENVS
DEVICE = scene_module.DEVICE
DroneScene = scene_module.DroneScene


def scale_actions(normalized_actions, device):
    action_scale = torch.tensor(ACTION_SCALE, device=device, dtype=torch.float32)
    return (2.0 * normalized_actions - 1.0) * action_scale


def deterministic_action(policy, obs):
    if isinstance(policy, LegacyGaussianActorCritic):
        return policy.mean_action(obs)

    dist = policy.distribution(obs)
    return dist.mean.clamp(1e-6, 1.0 - 1e-6)


def load_policy(checkpoint, device):
    state_dict = torch.load(checkpoint, map_location=device)
    if "actor_mean.weight" in state_dict:
        policy = LegacyGaussianActorCritic().to(device)
        policy.load_state_dict(state_dict)
        return policy, "legacy_gaussian"

    policy = ActorCritic().to(device)
    policy.load_state_dict(state_dict)
    return policy, "beta"


def actions_to_controller(policy_type, policy_actions, device):
    if policy_type == "legacy_gaussian":
        return policy_actions
    return scale_actions(policy_actions, device)


def reset_test_episode(env, env_ids=None, episode=0):
    difficulty = random.choice(TEST_DIFFICULTIES)
    env.set_difficulty(difficulty)
    print({"episode": episode, "test_difficulty": difficulty}, flush=True)
    return env.reset(env_ids), difficulty


def main():
    print({"loading_policy": args.checkpoint}, flush=True)

    sim = SimulationContext(SimulationCfg(device=DEVICE))

    scene = DroneScene(sim)
    scene.build()

    sim.reset()
    scene.connect_drone()

    env = DroneEnv(scene, num_envs=TEST_NUM_ENVS, device=DEVICE)
    episode = 0
    obs, episode_difficulty = reset_test_episode(env, episode=episode)
    episode_return = torch.zeros(TEST_NUM_ENVS, device=env.device)
    episode_length = torch.zeros(TEST_NUM_ENVS, dtype=torch.long, device=env.device)

    policy, policy_type = load_policy(args.checkpoint, env.device)
    policy.eval()
    print({"policy_type": policy_type}, flush=True)

    step = 0
    while simulation_app.is_running():
        with torch.no_grad():
            normalized_actions = deterministic_action(policy, obs)
            actions = actions_to_controller(policy_type, normalized_actions, env.device)

        obs, reward, done = env.step(actions)
        episode_return += reward.detach()
        episode_length += 1

        if done.any():
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)

            # print(
            #     {
            #         "episode_done": episode,
            #         "difficulty": episode_difficulty,
            #         "return": float(episode_return[done_ids[0]].detach().cpu()),
            #         "length": int(episode_length[done_ids[0]].detach().cpu()),
            #         "success": bool(env.success[done_ids[0]].detach().cpu()),
            #     },
            #     flush=True,
            # )
            episode_return[done_ids] = 0.0
            episode_length[done_ids] = 0
            episode += 1
            obs, episode_difficulty = reset_test_episode(env, done_ids, episode)

        if step % args.print_every == 0:
            _, _, linear_vel, _ = env.get_drone_state()
            linear_vel = torch.as_tensor(linear_vel, device=env.device, dtype=torch.float32)
            # print(
            #     {
            #         "step": step,
            #         "reward": float(reward[0].detach().cpu()),
            #         "done": bool(done[0].detach().cpu()),
            #         "normalized_action": normalized_actions[0].detach().cpu().tolist(),
            #         "applied_action": actions[0].detach().cpu().tolist(),
            #         "velocity_norm": float(linear_vel[0].norm().detach().cpu()),
            #     },
            #     flush=True,
            # )

        step += 1
        if args.steps > 0 and step >= args.steps:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
