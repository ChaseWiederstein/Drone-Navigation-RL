from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=False, device="cuda:0")
simulation_app = app_launcher.app


from isaaclab.sim import SimulationCfg, SimulationContext
import torch

from drone_env import DroneEnv
from scene import DEVICE, NUM_ENVS, DroneScene


def straight_turn_policy(obs, num_envs=NUM_ENVS, device=DEVICE):
    actions = torch.zeros((num_envs, 4), device=device)
    actions[:, 0] = 1.0
    actions[:, 3] = -3.0 * torch.pi / 4.0
    return actions


def main():
    sim = SimulationContext(SimulationCfg(device=DEVICE))

    scene = DroneScene(sim)
    scene.build()

    sim.reset()
    scene.connect_drone()

    env = DroneEnv(scene, num_envs=NUM_ENVS, device=DEVICE)
    obs = env.reset()
    step = 0

    while simulation_app.is_running():
        with torch.no_grad():
            actions = straight_turn_policy(obs, env.num_envs, env.device)

        obs, reward, done = env.step(actions)

        if step % 30 == 0:
            depth_env0 = obs["depth"][0]
            debug_obs = {
                "step": step,
                "actions": actions.shape,
                "depth": obs["depth"].shape,
                "state": obs["state"].shape,
                "state_env0": obs["state"][0].detach().cpu().tolist(),
                "reward_env0": float(reward[0].detach().cpu()),
                "done_env0": bool(done[0].detach().cpu()),
            }
            if depth_env0.numel() > 0:
                debug_obs["depth_env0_min"] = float(depth_env0.min().detach().cpu())
                debug_obs["depth_env0_max"] = float(depth_env0.max().detach().cpu())
            else:
                debug_obs["depth_env0_empty"] = True

            print(debug_obs, flush=True)
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
