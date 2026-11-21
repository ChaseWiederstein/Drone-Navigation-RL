import argparse

from isaaclab.app import AppLauncher


DEFAULT_CHECKPOINT = r"C:\robotics\droneRL\checkpoints\policy_50.pt"
ACTION_SCALE = (2.0, 2.0, 2.0, 1.5)
TEST_NUM_ENVS = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps. 0 means run until closed.")
    parser.add_argument("--print-every", type=int, default=30)
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


def obs_with_depth(obs, depth):
    return {
        "depth": depth,
        "state": obs["state"],
    }


def main():
    print({"loading_policy": args.checkpoint}, flush=True)

    sim = SimulationContext(SimulationCfg(device=DEVICE))

    scene = DroneScene(sim)
    scene.build()

    sim.reset()
    scene.connect_drone()

    env = DroneEnv(scene, num_envs=TEST_NUM_ENVS, device=DEVICE)
    obs = env.reset()

    policy, policy_type = load_policy(args.checkpoint, env.device)
    policy.eval()
    print({"policy_type": policy_type}, flush=True)

    step = 0
    while simulation_app.is_running():
        with torch.no_grad():
            real_normalized = deterministic_action(policy, obs)

            far_depth = torch.zeros_like(obs["depth"])
            close_depth = torch.ones_like(obs["depth"])

            far_normalized = deterministic_action(policy, obs_with_depth(obs, far_depth))
            close_normalized = deterministic_action(policy, obs_with_depth(obs, close_depth))

            real_action = actions_to_controller(policy_type, real_normalized, env.device)
            far_action = actions_to_controller(policy_type, far_normalized, env.device)
            close_action = actions_to_controller(policy_type, close_normalized, env.device)

        obs, reward, done = env.step(real_action)

        if done.any():
            obs = env.reset(torch.nonzero(done, as_tuple=False).squeeze(-1))

        if step % args.print_every == 0:
            print(
                {
                    "step": step,
                    "reward": float(reward[0].detach().cpu()),
                    "real_action": real_action[0].detach().cpu().tolist(),
                    "far_depth_action": far_action[0].detach().cpu().tolist(),
                    "close_depth_action": close_action[0].detach().cpu().tolist(),
                    "far_action_delta": float((real_action - far_action).norm(dim=1)[0].detach().cpu()),
                    "close_action_delta": float((real_action - close_action).norm(dim=1)[0].detach().cpu()),
                    "real_depth_min": float(obs["depth"][0].min().detach().cpu()),
                    "real_depth_max": float(obs["depth"][0].max().detach().cpu()),
                },
                flush=True,
            )

        step += 1
        if args.steps > 0 and step >= args.steps:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
