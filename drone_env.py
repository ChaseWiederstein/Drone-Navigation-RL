import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from low_level_controller import Drone, LowLevelController


MAX_DEPTH = 10
DEPTH_OBS_SIZE = (20, 20)
CONTACT_FORCE_THRESHOLD = 0.0


class DroneEnv:
    def __init__(self, scene, num_envs=1, device="cuda:0"):
        self.scene = scene
        self.sim = scene.sim
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.drones = scene.drone_body
        self.controlled_drones = [
            Drone(self.drones, device=self.device, env_index=env_id)
            for env_id in range(self.num_envs)
        ]
        self.controllers = [
            LowLevelController(drone, device=self.device)
            for drone in self.controlled_drones
        ]

        self.progress = torch.zeros(self.num_envs, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_linear_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_dir = torch.zeros((self.num_envs, 3), device=self.device)
        self.horizon = 1400 # steps
        self.min_height = .1
        self.max_height = 6
        self.contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_radius = .25
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.printed_depth_step_10 = False

    def to_tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def max_pool_depth_obs(self, depth_obs):
        return F.adaptive_max_pool2d(depth_obs, DEPTH_OBS_SIZE)

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        self.scene.set_difficulty(difficulty)

    def goal_frame_basis(self, goal_direction_world):
        goal_x = goal_direction_world.clone()
        goal_x[:, 2] = 0.0
        goal_x = goal_x / torch.linalg.norm(goal_x, dim=1, keepdim=True).clamp_min(1e-6)

        world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(goal_x)
        goal_y = torch.cross(world_z, goal_x, dim=1)
        goal_y = goal_y / torch.linalg.norm(goal_y, dim=1, keepdim=True).clamp_min(1e-6)

        goal_z = torch.cross(goal_x, goal_y, dim=1)
        goal_z = goal_z / torch.linalg.norm(goal_z, dim=1, keepdim=True).clamp_min(1e-6)
        return goal_x, goal_y, goal_z

    def world_to_goal_frame(self, vec_world, goal_direction_world):
        goal_x, goal_y, goal_z = self.goal_frame_basis(goal_direction_world)

        return torch.stack(
            [
                (vec_world * goal_x).sum(dim=1),
                (vec_world * goal_y).sum(dim=1),
                (vec_world * goal_z).sum(dim=1),
            ],
            dim=1,
        )

    def goal_frame_to_world(self, vec_goal, goal_direction_world):
        goal_x, goal_y, goal_z = self.goal_frame_basis(goal_direction_world)
        return (
            vec_goal[:, 0:1] * goal_x
            + vec_goal[:, 1:2] * goal_y
            + vec_goal[:, 2:3] * goal_z
        )

    def get_env_offsets(self):
        env_offsets = self.scene.env_positions
        if env_offsets is None:
            return torch.zeros((self.num_envs, 3), device=self.device)
        return self.to_tensor(env_offsets)

    def get_target_world(self):
        if hasattr(self.scene, "targets"):
            target_local = torch.tensor(self.scene.targets, device=self.device, dtype=torch.float32)
            return target_local + self.get_env_offsets()
        target_local = torch.tensor(self.scene.target, device=self.device, dtype=torch.float32)
        return target_local.unsqueeze(0) + self.get_env_offsets()

    def clone_envs(self):
        # This is where the Isaac GridCloner / Isaac Lab cloner code will go.
        # For now our scene is still single-env, so keep this explicit.
        if self.num_envs != 1:
            raise NotImplementedError("Parallel env cloning is not wired yet.")

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.progress[env_ids] = 0
        self.done[env_ids] = False
        self.prev_linear_vel[env_ids] = 0.0
        self.contact[env_ids] = False
        self.success[env_ids] = False
        self.scene.randomize_layout(env_ids)

        start = torch.tensor(self.scene.start, device=self.device, dtype=torch.float32)
        env_offsets = self.get_env_offsets()
        positions = start.unsqueeze(0) + env_offsets[env_ids]
        orientations = torch.tensor(
            [0.3826834324, 0.0, 0.0, -0.9238795325],
            device=self.device,
            dtype=torch.float32,
        ).repeat(env_ids.numel(), 1)
        velocities = torch.zeros((env_ids.numel(), 6), device=self.device)

        self.drones.set_world_poses(positions=positions, orientations=orientations, indices=env_ids)
        self.drones.set_velocities(velocities, indices=env_ids)

        target_delta = self.get_target_world()[env_ids] - positions
        target_delta[:, 2] = 0.0
        self.target_dir[env_ids] = target_delta
        return self.get_obs()

    def step(self, actions):
        self.apply_actions(actions)
        self.sim.step()

        self.progress += 1
        obs = self.get_obs()
        reward = self.compute_reward()
        done = self.compute_done()

        return obs, reward, done

    def apply_actions(self, actions):
        actions = self.to_tensor(actions)
        velocity_world = self.goal_frame_to_world(actions[:, :3], self.target_dir)

        for env_id, controller in enumerate(self.controllers):
            state = controller.drone.get_state()

            velocity_xy = velocity_world[env_id, :2]
            if torch.linalg.norm(velocity_xy) > 0.1:
                desired_yaw = torch.atan2(velocity_xy[1], velocity_xy[0])
            else:
                desired_yaw = state["yaw"]

            p_des = torch.cat((state["pos"], desired_yaw.reshape(1)))
            v_des = torch.cat(
                [
                    velocity_world[env_id],
                    torch.zeros(1, device=self.device),
                ]
            )

            acc_des = torch.zeros(4, device=self.device)
            jerk_des = torch.zeros(4, device=self.device)

            u, _ = controller.PDControlLaw(p_des, v_des, acc_des, jerk_des)
            controller.drone.apply_control(u)

    def get_drone_state(self):
        positions, quats = self.drones.get_world_poses()
        linear_vel = self.drones.get_linear_velocities()
        angular_vel = self.drones.get_angular_velocities()

        return positions, quats, linear_vel, angular_vel

    def get_obs(self):
        return {
            "depth": self.get_depth_obs(),
            "state": self.get_state_obs(),
        }

    def get_depth_obs(self):
        camera_width, camera_height = self.scene.camera_resolution
        depth_frames = []
        for camera in self.scene.depth_cameras:
            depth = camera.get_depth()
            depth = torch.as_tensor(depth, device=self.device, dtype=torch.float32)

            if depth.numel() == 0:
                depth = torch.zeros((camera_height, camera_width), device=self.device)
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = torch.where(depth <= 0.0, torch.full_like(depth, MAX_DEPTH), depth)
            depth_frames.append(depth)

        if not depth_frames:
            depth_obs = torch.zeros((self.num_envs, 1, camera_height, camera_width), device=self.device)
            return self.max_pool_depth_obs(depth_obs)

        depth = torch.stack(depth_frames, dim=0).unsqueeze(1)
        depth = depth.clamp(0.0, MAX_DEPTH)
        depth_obs = (MAX_DEPTH - depth) / MAX_DEPTH
        pooled_depth_obs = self.max_pool_depth_obs(depth_obs)

        if not self.printed_depth_step_10 and self.progress[0] >= 10:
            self.printed_depth_step_10 = True
            plt.imsave(
                r"C:\robotics\droneRL\depth_raw_step10.png",
                depth[0, 0].detach().cpu().numpy(),
                cmap="gray",
            )
            plt.imsave(
                r"C:\robotics\droneRL\depth_pooled_step10.png",
                pooled_depth_obs[0, 0].detach().cpu().numpy(),
                cmap="gray",
            )

        return pooled_depth_obs

    def get_state_obs(self):
        positions, _, linear_vel, _ = self.get_drone_state()

        positions = self.to_tensor(positions)
        linear_vel = self.to_tensor(linear_vel)

        target_delta = self.get_target_world() - positions
        target_distance = torch.linalg.norm(target_delta, dim=1, keepdim=True).clamp_min(1e-6)
        target_direction = target_delta / target_distance
        target_distance_xy = torch.linalg.norm(target_delta[:, :2], dim=1, keepdim=True)
        target_distance_z = target_delta[:, 2:3]
        target_frame = self.target_dir.clone()
        target_frame[:, 2] = 0.0
        target_direction_goal = self.world_to_goal_frame(target_direction, target_frame)
        linear_vel_goal = self.world_to_goal_frame(linear_vel, target_frame)

        return torch.cat(
            [
                target_direction_goal,
                target_distance_xy,
                target_distance_z,
                linear_vel_goal,
            ],
            dim=1,
        )

    def compute_reward(self):
        positions, _, linear_vel, _ = self.get_drone_state()
        positions = self.to_tensor(positions)
        linear_vel = self.to_tensor(linear_vel)

        depth_closeness = self.get_depth_obs()
        distance = (1.0 - depth_closeness) * MAX_DEPTH
        reward_safety= torch.log(distance.clamp(min=1e-6, max=MAX_DEPTH)).mean(dim=(1, 2, 3))

        target_delta = self.get_target_world() - positions
        target_direction = target_delta / torch.linalg.norm(target_delta, dim=1, keepdim=True).clamp_min(1e-6)
        reward_vel = (linear_vel * target_direction).sum(dim=1)

        penalty_smooth = torch.linalg.norm(linear_vel - self.prev_linear_vel, dim=1)

        z = positions[:, 2]
        penalty_height = torch.zeros(self.num_envs, device=self.device)
        penalty_height += torch.where(z < 1.5, (1.5 - z) ** 2, torch.zeros_like(z))
        penalty_height += torch.where(z > 2.5, (z - 2.5) ** 2, torch.zeros_like(z))

        contact_forces = self.scene.contact_view.get_net_contact_forces()
        if contact_forces is None:
            self.contact[:] = False
        else:
            contact_forces = self.to_tensor(contact_forces)
            self.contact = torch.linalg.norm(contact_forces, dim=1) > CONTACT_FORCE_THRESHOLD
        penalty_contact = self.contact.float() * 1000

        target_distance = torch.linalg.norm(target_delta, dim=1)
        reward_goal = (target_distance < self.goal_radius).float() * 2000

        self.prev_linear_vel = linear_vel.clone()

        return 3*reward_vel + reward_safety - 0.1 * penalty_smooth - 8.0 * penalty_height - penalty_contact + reward_goal

    def reached_goal(self, positions):
        target_delta = self.get_target_world() - positions
        return torch.linalg.norm(target_delta, dim=1) < self.goal_radius

    def compute_done(self):
        positions, _, _, _ = self.get_drone_state()
        positions = self.to_tensor(positions)

        contact_forces = self.scene.contact_view.get_net_contact_forces()
        if contact_forces is None:
            contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            contact_forces = self.to_tensor(contact_forces)
            contact_force_norm = torch.linalg.norm(contact_forces, dim=1)
            contact = contact_force_norm > CONTACT_FORCE_THRESHOLD
            # if contact.any():
            #     contact_ids = torch.nonzero(contact, as_tuple=False).squeeze(-1)
            #     print(
            #         {
            #             "contact_env_ids": contact_ids.detach().cpu().tolist(),
            #             "contact_force_norms": contact_force_norm[contact_ids].detach().cpu().tolist(),
            #         },
            #         flush=True,
            #     )

        too_low = positions[:, 2] < self.min_height
        too_high = positions[:, 2] > self.max_height
        timeout = self.progress >= self.horizon
        reached_goal = self.reached_goal(positions)

        self.contact = contact
        self.success = reached_goal
        self.done = reached_goal | contact | too_low | too_high | timeout
        return self.done
