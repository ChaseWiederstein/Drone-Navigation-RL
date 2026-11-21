import math
import random

import isaaclab.sim as sim_utils
import torch
from isaacsim.core.cloner import GridCloner
from isaacsim.core.api.sensors import RigidContactView
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils import prims as prim_utils
from isaacsim.sensors.camera import Camera
import omni.usd
from pxr import PhysxSchema, UsdGeom, UsdPhysics

from low_level_controller import IRIS_INERTIA, IRIS_MASS


TRASH_USD = r"C:\robotics\droneRL\assets\props\plastic_bottle_1.usd"
DRONE_USD = r"C:\robotics\droneRL\assets\drone\iris_with_gemini.usda"
ENV_ROOT = "/World/envs"
TEMPLATE_ENV_PATH = f"{ENV_ROOT}/env_0"
DRONE_NAME = "drone"
DRONE_PATH = f"{TEMPLATE_ENV_PATH}/{DRONE_NAME}"
RGB_CAMERA_REL_PATH = f"{DRONE_NAME}/visual/body/camera_joint/Gemini2/camera_rgb/camera_rgb/Stream_rgb"
DEPTH_CAMERA_REL_PATH = f"{DRONE_NAME}/visual/body/camera_joint/Gemini2/camera_ir_left/camera_left/Stream_depth"

DEVICE = "cuda:0"
NUM_ENVS = 64

ENV_SPACING = 24.0




class DroneScene:
    def __init__(self, sim):
        self.sim = sim
        self.env_path = TEMPLATE_ENV_PATH
        self.drone_path = f"{self.env_path}/{DRONE_NAME}"
        self.drone_body = None
        self.contact_view = None
        self.obstacle_bodies = None
        self.obstacle_body_indices = {}
        self.rgb_cameras = []
        self.depth_cameras = []
        self.env_paths = []
        self.env_positions = None
        self.target = (5, 5, 2.0)
        self.TRASH_FLOOR_Z = 0.0
        self.TARGET_HEIGHT_ABOVE_TRASH = 2.0
        self.START = (9, 9, 2.0)
        self.WORLD_SIZE = (20, 20, .1)
        self.CAMERA_RESOLUTION = (32, 32)
        self.SCENE_SEED = 7
        self.difficulty = 0
        self.layout_counter = 0
        self.MIN_OBSTACLES = 20
        self.MAX_OBSTACLES = 40
        self.NUM_OBSTACLES = self.MIN_OBSTACLES
        self.OBSTACLE_VARIANTS = self.build_obstacle_variants()
        self.obstacle_slots = self.build_obstacle_slots()
        self.TARGET_MIN_DISTANCE = 5.0
        self.TARGET_MAX_DISTANCE = 18.0
        self.TARGET_DISTANCE_MAX_BY_LEVEL = (7.0, 9.0, 11.0, 13.0, 15.0, 18.0)
        self.PLACEMENT_MARGIN = 1.25
        self.START_CLEARANCE_RADIUS = 1.5
        self.TARGET_CLEARANCE_RADIUS = 1.5
        self.trash_position = self.get_trash_position(self.target)
        self.targets = [self.target for _ in range(NUM_ENVS)]
        self.trash_positions = [self.trash_position for _ in range(NUM_ENVS)]
        self.start = self.START
        self.world_size = self.WORLD_SIZE
        self.camera_resolution = self.CAMERA_RESOLUTION

    def build(self):
        self.sim.set_camera_view(eye=(4.0, 4.0, 3.0), target=(0.0, 0.0, 0.5))
        self.create_env_root()
        # self.generate_layout()
        self.spawn_ground()
        self.spawn_walls()
        self.spawn_obstacles()
        self.spawn_target()
        self.spawn_drone()
        self.clone_envs()
        self.spawn_lights()

    def create_env_root(self):
        if not prim_utils.is_prim_path_valid(ENV_ROOT):
            prim_utils.define_prim(ENV_ROOT, "Xform")
        if not prim_utils.is_prim_path_valid(self.env_path):
            prim_utils.define_prim(self.env_path, "Xform")

    def env_prim(self, name):
        return f"{self.env_path}/{name}"

    def get_trash_position(self, target):
        return (target[0], target[1], self.TRASH_FLOOR_Z)

    def set_difficulty(self, difficulty):
        self.difficulty = max(0, int(difficulty))
        self.NUM_OBSTACLES = min(self.MIN_OBSTACLES + max(self.difficulty - 1, 0) * 3, self.MAX_OBSTACLES)
        self.PLACEMENT_MARGIN = max(1.25 - self.difficulty * 0.1, 0.75)
        self.START_CLEARANCE_RADIUS = max(1.5 - self.difficulty * 0.05, 1.0)
        self.TARGET_CLEARANCE_RADIUS = max(1.5 - self.difficulty * 0.05, 1.0)

    def generate_layout(self):
        self.target = self.sample_target()
        self.trash_position = self.get_trash_position(self.target)

    def randomize_layout(self, env_ids=None):
        env_ids = self.normalize_env_ids(env_ids)
        for env_id in env_ids:
            seed = self.SCENE_SEED + self.difficulty * 1000 + self.layout_counter * 37 + env_id
            self.layout_counter += 1
            target = self.sample_target(seed)
            trash_position = self.get_trash_position(target)
            obstacle_layout = self.sample_obstacle_layout(target, seed + 1)

            self.targets[env_id] = target
            self.trash_positions[env_id] = trash_position
            if env_id == 0:
                self.target = target
                self.trash_position = trash_position

            self.move_layout_prims(env_id, trash_position, obstacle_layout)

    def normalize_env_ids(self, env_ids):
        if env_ids is None:
            return list(range(len(self.env_paths)))
        if hasattr(env_ids, "detach"):
            env_ids = env_ids.detach().cpu().tolist()
        return [int(env_id) for env_id in env_ids]

    def sample_target(self, seed=None):
        rng = random.Random(self.SCENE_SEED + self.difficulty * 1000 if seed is None else seed)
        min_distance, max_distance = self.target_distance_range()
        target_radius = self.TARGET_CLEARANCE_RADIUS
        half_x = self.WORLD_SIZE[0] / 2.0 - self.PLACEMENT_MARGIN - target_radius
        half_y = self.WORLD_SIZE[1] / 2.0 - self.PLACEMENT_MARGIN - target_radius

        for _ in range(1000):
            distance = rng.uniform(min_distance, max_distance)
            angle = rng.uniform(-math.pi, math.pi)
            x = self.start[0] + math.cos(angle) * distance
            y = self.start[1] + math.sin(angle) * distance
            if not (-half_x <= x <= half_x and -half_y <= y <= half_y):
                continue
            if self.xy_distance((x, y), (self.start[0], self.start[1])) >= target_radius + self.START_CLEARANCE_RADIUS:
                return (x, y, self.TRASH_FLOOR_Z + self.TARGET_HEIGHT_ABOVE_TRASH)

        raise RuntimeError("Could not sample a valid target position.")

    def target_distance_range(self):
        level = min(max(self.difficulty, 0), len(self.TARGET_DISTANCE_MAX_BY_LEVEL) - 1)
        return self.TARGET_MIN_DISTANCE, self.TARGET_DISTANCE_MAX_BY_LEVEL[level]

    def sample_non_overlapping_positions(self, count, radius, blocked_positions, seed, max_attempts=1000):
        rng = random.Random(seed)
        half_x = self.WORLD_SIZE[0] / 2.0 - self.PLACEMENT_MARGIN - radius
        half_y = self.WORLD_SIZE[1] / 2.0 - self.PLACEMENT_MARGIN - radius
        placed = []

        for _ in range(count):
            for _ in range(max_attempts):
                candidate = (rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y))
                if self.is_position_clear(candidate, radius, placed, blocked_positions):
                    placed.append(candidate)
                    break
            else:
                raise RuntimeError(f"Could only place {len(placed)} of {count} obstacles without overlap.")

        return placed

    def is_position_clear(self, candidate, radius, placed, blocked_positions):
        for position, blocked_radius in blocked_positions:
            if self.xy_distance(candidate, position) < radius + blocked_radius:
                return False

        for position in placed:
            if self.xy_distance(candidate, position) < radius * 2.0:
                return False

        return True

    def xy_distance(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    def build_obstacle_variants(self):
        return [
            {
                "name": "small_tree",
                "min_difficulty": 0,
                "footprint_radius": 0.65,
                "parts": [
                    {"name": "trunk", "shape": "cylinder", "radius": 0.08, "height": 1.2, "z": 0.6, "color": (0.35, 0.18, 0.08)},
                    {"name": "canopy", "shape": "sphere", "radius": 0.45, "z": 1.35, "color": (0.05, 0.35, 0.08)},
                ],
            },
            {
                "name": "large_tree",
                "min_difficulty": 1,
                "footprint_radius": 1.05,
                "parts": [
                    {"name": "trunk", "shape": "cylinder", "radius": 0.15, "height": 2.0, "z": 1.0, "color": (0.32, 0.16, 0.07)},
                    {"name": "canopy", "shape": "sphere", "radius": 0.85, "z": 2.15, "color": (0.03, 0.28, 0.07)},
                ],
            },
            {
                "name": "pole",
                "min_difficulty": 0,
                "footprint_radius": 0.45,
                "parts": [
                    {"name": "body", "shape": "cylinder", "radius": 0.18, "height": 6.0, "z": 1.5, "color": (0.45, 0.45, 0.42)},
                ],
            },
            {
                "name": "box",
                "min_difficulty": 0,
                "footprint_radius": 0.95,
                "parts": [
                    {"name": "body", "shape": "box", "size": (1.0, 1.0, 1.3), "z": 0.65, "color": (0.45, 0.34, 0.18)},
                ],
            },
            {
                "name": "tall_box",
                "min_difficulty": 1,
                "footprint_radius": 0.85,
                "parts": [
                    {"name": "body", "shape": "box", "size": (0.8, 0.8, 2.5), "z": 1.25, "color": (0.38, 0.38, 0.46)},
                ],
            },
            {
                "name": "wall_x",
                "min_difficulty": 2,
                "footprint_radius": 2.05,
                "parts": [
                    {"name": "body", "shape": "box", "size": (3.5, 0.25, 4), "z": 1.1, "color": (0.50, 0.50, 0.52)},
                ],
            },
            {
                "name": "wall_y",
                "min_difficulty": 2,
                "footprint_radius": 2.05,
                "parts": [
                    {"name": "body", "shape": "box", "size": (0.25, 4.0, 2.2), "z": 1.1, "color": (0.50, 0.50, 0.52)},
                ],
            },
            {
                "name": "ceiling",
                "min_difficulty": 3,
                "footprint_radius": 1.95,
                "parts": [
                    {"name": "body", "shape": "box", "size": (2.6, 2.6, 0.2), "z": 2.8, "color": (0.42, 0.42, 0.44)},
                ],
            },
            {
                "name": "sphere",
                "min_difficulty": 2,
                "footprint_radius": 0.80,
                "parts": [
                    {"name": "body", "shape": "sphere", "radius": 0.65, "z": 0.9, "color": (0.30, 0.38, 0.55)},
                ],
            },
        ]

    def available_obstacle_variants(self):
        return [
            variant
            for variant in self.OBSTACLE_VARIANTS
            if self.difficulty >= variant["min_difficulty"]
        ]

    def obstacle_variant_by_name(self, name):
        for variant in self.OBSTACLE_VARIANTS:
            if variant["name"] == name:
                return variant
        raise RuntimeError(f"Unknown obstacle variant: {name}")

    def build_obstacle_slots(self):
        slot_variant_names = [
            "small_tree", "pole", "box", "small_tree", "pole",
            "box", "small_tree", "pole", "box", "small_tree",
            "large_tree", "tall_box", "large_tree", "tall_box", "box",
            "wall_x", "wall_y", "sphere", "wall_x", "wall_y",
            "large_tree", "sphere", "ceiling", "wall_x", "ceiling",
            "small_tree", "pole", "box", "large_tree", "tall_box",
            "wall_y", "sphere", "small_tree", "wall_x", "ceiling",
            "pole", "box", "large_tree", "wall_y", "sphere",
        ]
        return [self.obstacle_variant_by_name(name) for name in slot_variant_names[: self.MAX_OBSTACLES]]

    def connect_drone(self):
        self.drone_body = RigidPrim(
            prim_paths_expr=f"{ENV_ROOT}/env_.*/{DRONE_NAME}",
            name="drones",
            reset_xform_properties=False,
        )
        self.drone_body.initialize()

        self.contact_view = RigidContactView(
            prim_paths_expr=f"{ENV_ROOT}/env_.*/{DRONE_NAME}",
            filter_paths_expr=[],
            name="drone_contact_view",
            prepare_contact_sensors=True,
        )
        self.contact_view.initialize()
        self.connect_obstacle_views()
        self.connect_cameras()

    def connect_obstacle_views(self):
        self.obstacle_bodies = RigidPrim(
            prim_paths_expr=f"{ENV_ROOT}/env_.*/obstacle_.*/.*",
            name="obstacle_bodies",
            reset_xform_properties=False,
        )
        self.obstacle_bodies.initialize()
        self.obstacle_body_indices = {path: index for index, path in enumerate(self.obstacle_bodies._prim_paths)}

    def connect_cameras(self):
        self.rgb_cameras = []
        self.depth_cameras = []

        for env_id, env_path in enumerate(self.env_paths):
            depth_camera = Camera(
                prim_path=f"{env_path}/{DEPTH_CAMERA_REL_PATH}",
                name=f"depth_camera_{env_id}",
                resolution=self.CAMERA_RESOLUTION,
            )

            depth_camera.initialize()
            depth_camera.add_distance_to_image_plane_to_frame()

            self.depth_cameras.append(depth_camera)

        print(f"CONNECTED CAMERAS: {len(self.depth_cameras)} depth, resolution={self.camera_resolution}", flush=True)

    def spawn_ground(self):
        ground = sim_utils.CuboidCfg(
            size=self.WORLD_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        )
        ground.func(self.env_prim("ground"), ground, translation=(0.0, 0.0, -0.1))

    def spawn_walls(self):
        wall_height = 6.0
        wall_thickness = 0.2
        scene_half_size = 10.0
        wall_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))

        wall_x = sim_utils.CuboidCfg(
            size=(wall_thickness, scene_half_size * 2.0, wall_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_material,
        )
        wall_y = sim_utils.CuboidCfg(
            size=(scene_half_size * 2.0, wall_thickness, wall_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=wall_material,
        )

        wall_x.func(self.env_prim("wall_left"), wall_x, translation=(-scene_half_size, 0.0, wall_height / 2.0))
        wall_x.func(self.env_prim("wall_right"), wall_x, translation=(scene_half_size, 0.0, wall_height / 2.0))
        wall_y.func(self.env_prim("wall_back"), wall_y, translation=(0.0, -scene_half_size, wall_height / 2.0))
        wall_y.func(self.env_prim("wall_front"), wall_y, translation=(0.0, scene_half_size, wall_height / 2.0))

    def spawn_obstacles(self):
        for slot, variant in enumerate(self.obstacle_slots):
            for part_index, part in enumerate(variant["parts"]):
                cfg = self.make_obstacle_part_cfg(part)
                x, y, z = self.hidden_obstacle_position(slot, part_index, part)
                cfg.func(
                    self.env_prim(self.obstacle_part_name(slot, variant, part)),
                    cfg,
                    translation=(x, y, z),
                )

    def make_obstacle_part_cfg(self, part):
        rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        collision_props = sim_utils.CollisionPropertiesCfg()
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=part["color"])

        if part["shape"] == "cylinder":
            return sim_utils.CylinderCfg(
                radius=part["radius"],
                height=part["height"],
                rigid_props=rigid_props,
                collision_props=collision_props,
                visual_material=material,
            )
        if part["shape"] == "sphere":
            return sim_utils.SphereCfg(
                radius=part["radius"],
                rigid_props=rigid_props,
                collision_props=collision_props,
                visual_material=material,
            )
        if part["shape"] == "box":
            return sim_utils.CuboidCfg(
                size=part["size"],
                rigid_props=rigid_props,
                collision_props=collision_props,
                visual_material=material,
            )
        raise RuntimeError(f"Unknown obstacle shape: {part['shape']}")

    def obstacle_part_name(self, slot, variant, part):
        return f"obstacle_{slot}/{variant['name']}_{part['name']}"

    def sample_obstacle_layout(self, target=None, seed=None):
        target = self.target if target is None else target
        rng = random.Random(self.SCENE_SEED + self.difficulty * 1000 + 1 if seed is None else seed)
        blocked_positions = [
            ((self.start[0], self.start[1]), self.START_CLEARANCE_RADIUS),
            ((target[0], target[1]), self.TARGET_CLEARANCE_RADIUS),
        ]
        available_slots = [
            (slot, variant)
            for slot, variant in enumerate(self.obstacle_slots)
            if self.difficulty >= variant["min_difficulty"]
        ]
        rng.shuffle(available_slots)
        placed = []

        for slot, variant in available_slots[: self.NUM_OBSTACLES]:
            for _ in range(1000):
                radius = variant["footprint_radius"]
                half_x = self.WORLD_SIZE[0] / 2.0 - self.PLACEMENT_MARGIN - radius
                half_y = self.WORLD_SIZE[1] / 2.0 - self.PLACEMENT_MARGIN - radius
                if half_x <= 0.0 or half_y <= 0.0:
                    continue

                candidate = (rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y))
                if self.is_obstacle_position_clear(candidate, radius, placed, blocked_positions):
                    placed.append(
                        {
                            "slot": slot,
                            "variant": variant["name"],
                            "position": candidate,
                            "radius": radius,
                        }
                    )
                    break
            else:
                break

        return placed

    def is_obstacle_position_clear(self, candidate, radius, placed, blocked_positions):
        for position, blocked_radius in blocked_positions:
            if self.xy_distance(candidate, position) < radius + blocked_radius:
                return False

        for obstacle in placed:
            if self.xy_distance(candidate, obstacle["position"]) < radius + obstacle["radius"]:
                return False

        return True

    def hidden_obstacle_position(self, slot, part_index, part):
        return (
            1000.0 + slot * 12.0,
            1000.0 + part_index * 2.0,
            part["z"],
        )

    def move_layout_prims(self, env_id, trash_position, obstacle_layout):
        env_path = self.env_paths[env_id]
        self.set_prim_translation(f"{env_path}/target_trash", trash_position)

        body_indices = []
        body_positions = []
        active_obstacles = {obstacle["slot"]: obstacle for obstacle in obstacle_layout}
        for slot in range(self.MAX_OBSTACLES):
            variant = self.obstacle_slots[slot]
            active_obstacle = active_obstacles.get(slot)

            for part_index, part in enumerate(variant["parts"]):
                part_path = f"{env_path}/{self.obstacle_part_name(slot, variant, part)}"

                if active_obstacle is not None:
                    x, y = active_obstacle["position"]
                    position = (x, y, part["z"])
                else:
                    position = self.hidden_obstacle_position(slot, part_index, part)

                if self.obstacle_bodies is not None and part_path in self.obstacle_body_indices:
                    body_indices.append(self.obstacle_body_indices[part_path])
                    body_positions.append(position)
                else:
                    self.set_prim_translation(part_path, position)

        if body_indices:
            self.obstacle_bodies.set_world_poses(
                positions=self.to_world_positions(env_id, body_positions),
                indices=body_indices,
            )

    def to_world_positions(self, env_id, local_positions):
        positions = []
        env_offset = self.env_positions[env_id] if self.env_positions is not None else (0.0, 0.0, 0.0)
        for position in local_positions:
            positions.append(
                (
                    float(position[0] + env_offset[0]),
                    float(position[1] + env_offset[1]),
                    float(position[2] + env_offset[2]),
                )
            )
        return torch.tensor(positions, device=DEVICE, dtype=torch.float32)

    def set_prim_translation(self, prim_path, translation):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim path: {prim_path}")

        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        translate_op.Set(tuple(float(value) for value in translation))


    def spawn_target(self):
        trash = sim_utils.UsdFileCfg(usd_path=TRASH_USD)
        self.trash_position = self.get_trash_position(self.target)
        trash.func(self.env_prim("target_trash"), trash, translation=self.trash_position)

    def spawn_drone(self):
        drone = sim_utils.UsdFileCfg(usd_path=DRONE_USD)
        drone.func(
            self.drone_path,
            drone,
            translation=self.START,
            orientation=(0.3826834324, 0.0, 0.0, -0.9238795325),
        )
        self.make_drone_physical()

    def clone_envs(self):
        cloner = GridCloner(spacing=ENV_SPACING)
        cloner.define_base_env(ENV_ROOT)
        self.env_paths = cloner.generate_paths(f"{ENV_ROOT}/env", NUM_ENVS)
        self.env_positions = cloner.clone(
            source_prim_path=TEMPLATE_ENV_PATH,
            prim_paths=self.env_paths,
            replicate_physics=True,
            base_env_path=ENV_ROOT,
        )
        print(f"CLONED ENVS: {self.env_paths}", flush=True)

    def spawn_lights(self):
        light = sim_utils.DistantLightCfg(intensity=3000.0)
        light.func("/World/light", light, translation=(0.0, 0.0, 10.0))

    def make_drone_physical(self):
        stage = omni.usd.get_context().get_stage()
        drone_prim = stage.GetPrimAtPath(self.drone_path)
        if drone_prim.HasAPI(UsdPhysics.CollisionAPI):
            drone_prim.RemoveAPI(UsdPhysics.CollisionAPI)

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim_path == self.drone_path or not prim_path.startswith(self.drone_path):
                continue

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            if prim.IsA(UsdPhysics.Joint):
                prim.SetActive(False)
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)

        UsdPhysics.RigidBodyAPI.Apply(drone_prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(drone_prim)

        mass = UsdPhysics.MassAPI.Apply(drone_prim)
        mass.CreateMassAttr(float(IRIS_MASS))
        mass.CreateCenterOfMassAttr((0.0, 0.0, 0.0))
        mass.CreateDiagonalInertiaAttr(tuple(float(x) for x in IRIS_INERTIA.diagonal()))

        collider = UsdGeom.Cube.Define(stage, f"{self.drone_path}/collision_box")
        collider.CreateSizeAttr(1.0)
        collider.AddScaleOp().Set((0.7, 0.7, 0.25))
        collider.AddTranslateOp().Set((0.0, 0.0, 0.0))
        collider.GetVisibilityAttr().Set("invisible")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
