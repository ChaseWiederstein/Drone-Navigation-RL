import numpy as np
import torch


IRIS_MASS = 1.5
IRIS_MAXS_ACC = 17.78 #estimated
IRIS_INERTIA = np.diag([0.0291249994, 0.0291249994, 0.0552249998])
IRIS_KF = 8.54858e-6
IRIS_KM = 1e-6
IRIS_MIN_ROTOR_SPEED = 0.0
IRIS_MAX_ROTOR_SPEED = 1100.0
IRIS_ROTOR_DIRECTIONS = np.array([-1.0, -1.0, 1.0, 1.0], dtype=float)
IRIS_ROTOR_POSITIONS = np.array(
    [
        [0.1379854530, -0.2067163885, 0.0230000000],
        [-0.1251116842, 0.2187541425, 0.0230000000],
        [0.1379999965, 0.2025757730, 0.0230000000],
        [-0.1241535991, -0.2223457992, 0.0230000000],
    ],
    dtype=float,
)


def normalize(v, eps=1e-8):
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def torch_normalize(v, eps=1e-8):
    norm = torch.linalg.norm(v)
    if norm < eps:
        return torch.zeros_like(v)
    return v / norm


def torch_vee(S):
    return torch.stack((S[2, 1], S[0, 2], S[1, 0]))


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def torch_quat_wxyz_to_matrix(quat):
    w, x, y, z = quat
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    )


def torch_quat_wxyz_to_matrix_batch(quat):
    w, x, y, z = quat.unbind(dim=1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=1),
        ),
        dim=1,
    )


def yaw_from_R(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def torch_yaw_from_R(R):
    return torch.atan2(R[1, 0], R[0, 0])


def torch_yaw_from_R_batch(R):
    return torch.atan2(R[:, 1, 0], R[:, 0, 0])


def torch_vee_batch(S):
    return torch.stack((S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]), dim=1)


def _iris_wrench_limits_from_rotors(
    rotor_positions=IRIS_ROTOR_POSITIONS,
    kf=IRIS_KF,
    km=IRIS_KM,
    rotor_directions=IRIS_ROTOR_DIRECTIONS,
    omega_max=IRIS_MAX_ROTOR_SPEED,
):
    rotor_positions = np.asarray(rotor_positions, dtype=float)
    x = rotor_positions[:, 0]
    y = rotor_positions[:, 1]
    rotor_directions = np.asarray(rotor_directions, dtype=float)

    omega2 = float(omega_max) ** 2

    f_max = 4.0 * float(kf) * omega2

    # Axis-wise symmetric maxima implied by independent rotor saturation bounds.
    # These are conservative "component" bounds; simultaneous (F, tau) feasibility
    # is a smaller polytope.
    tx_pos = float(np.sum(y[y > 0.0])) * float(kf) * omega2
    tx_neg = float(np.sum(-y[y < 0.0])) * float(kf) * omega2
    tx_max = min(tx_pos, tx_neg)

    ty_pos = float(np.sum(-x[x < 0.0])) * float(kf) * omega2  # (-x) positive when x < 0
    ty_neg = float(np.sum(x[x > 0.0])) * float(kf) * omega2   # (-x) negative when x > 0
    ty_max = min(ty_pos, ty_neg)

    tz_pos = float(np.sum(rotor_directions[rotor_directions > 0.0])) * float(km) * omega2
    tz_neg = float(np.sum(-rotor_directions[rotor_directions < 0.0])) * float(km) * omega2
    tz_max = min(tz_pos, tz_neg)

    return float(f_max), np.array([tx_max, ty_max, tz_max], dtype=float)


IRIS_MAX_THRUST, IRIS_MAX_TORQUE = _iris_wrench_limits_from_rotors()


class Drone:
    def __init__(
        self,
        body,
        articulation=None,
        m=IRIS_MASS,
        J=IRIS_INERTIA,
        g=9.81,
        rotor_joint_names=("joint0", "joint1", "joint2", "joint3"),
        device=None,
        env_index=0,
    ):
        self.body = body
        self.articulation = articulation
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.env_index = int(env_index)
        self.env_indices = torch.tensor([self.env_index], device=self.device, dtype=torch.int64)
        self.m = float(m)
        self.J = self._to_tensor(J)
        self.J_inv = torch.linalg.inv(self.J)
        self.g = float(g)
        self.rotor_joint_names = tuple(rotor_joint_names)
        self.rotor_joint_indices = self._resolve_rotor_joint_indices()
        self.last_applied_wrench = torch.zeros(4, device=self.device)

    def _to_tensor(self, value):
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _body_view(self):
        if hasattr(self.body, "_rigid_prim_view"):
            return self.body._rigid_prim_view
        return self.body

    def get_state(self):
        body_view = self._body_view()
        positions, orientations = body_view.get_world_poses()
        linear_velocities = body_view.get_linear_velocities()
        angular_velocities = body_view.get_angular_velocities()

        pos = self._to_tensor(positions[self.env_index])
        vel = self._to_tensor(linear_velocities[self.env_index])
        quat = self._to_tensor(orientations[self.env_index])
        R = torch_quat_wxyz_to_matrix(quat)

        omega_world = self._to_tensor(angular_velocities[self.env_index])
        omega_body = R.T @ omega_world

        return {
            "pos": pos,
            "vel": vel,
            "R": R,
            "omega": omega_body,
            "yaw": torch_yaw_from_R(R),
        }

    def apply_control(self, u):
        u = self._to_tensor(u)
        actual_thrust = u[0]
        actual_torque_body = u[1:4]

        state = self.get_state()
        R = state["R"]
        zB_world = R[:, 2]

        force_world = actual_thrust * zB_world
        torque_world = R @ actual_torque_body

        forces = force_world.reshape(1, 3).to(dtype=torch.float32)
        torques = torque_world.reshape(1, 3).to(dtype=torch.float32)
        self._body_view().apply_forces_and_torques_at_pos(
            forces=forces,
            torques=torques,
            positions=None,
            indices=self.env_indices,
            is_global=True,
        )

        self.last_applied_wrench = u

    def _resolve_rotor_joint_indices(self):
        if self.articulation is None:
            return None

        indices = []
        for name in self.rotor_joint_names:
            try:
                indices.append(self.articulation.get_dof_index(name))
            except Exception:
                return None
        return np.asarray(indices, dtype=np.int32)


class LowLevelController:
    def __init__(
        self,
        drone,
        omegap=2.0,
        omegaR=8.0,
        max_thrust=None,
        max_torque=None,
        device=None,
    ):
        self.drone = drone
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.J = self._to_tensor(drone.J)
        self.e3 = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        self.Kp = drone.m * omegap**2 * torch.eye(3, device=self.device)
        self.Kv = 2.0 * drone.m * omegap * torch.eye(3, device=self.device)
        self.Kr = self.J * omegaR**2
        self.Kw = 2.0 * self.J * omegaR

        self.max_thrust = float(max_thrust) if max_thrust is not None else float(IRIS_MAX_THRUST)
        if max_torque is None:
            self.max_torque = self._to_tensor(IRIS_MAX_TORQUE)
        else:
            self.max_torque = self._to_tensor(max_torque)

    def _to_tensor(self, value):
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def PDControlLaw(self, sigma, sigmad, sigmadd, sigmaddd):
        state = self.drone.get_state()
        pos = self._to_tensor(state["pos"])
        vel = self._to_tensor(state["vel"])
        R = self._to_tensor(state["R"])
        omega = self._to_tensor(state["omega"])

        sigma = self._to_tensor(sigma)
        sigmad = self._to_tensor(sigmad)
        sigmadd = self._to_tensor(sigmadd)
        sigmaddd = self._to_tensor(sigmaddd)

        ep = pos - sigma[0:3]
        ev = vel - sigmad[0:3]

        e3 = self.e3
        Fdes = -self.Kp @ ep - self.Kv @ ev + self.drone.m * self.drone.g * e3 + self.drone.m * sigmadd[0:3]

        zB = R[:, 2]
        u1 = torch.dot(Fdes, zB)
        u1 = torch.clamp(u1, 0.0, self.max_thrust)

        zBdes = torch_normalize(Fdes)
        if torch.linalg.norm(zBdes) < 1e-8:
            zBdes = e3

        xCdes = torch.stack((torch.cos(sigma[3]), torch.sin(sigma[3]), sigma.new_tensor(0.0)))
        yBdes = torch_normalize(torch.cross(zBdes, xCdes, dim=0))
        if torch.linalg.norm(yBdes) < 1e-8:
            yBdes = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        xBdes = torch.cross(yBdes, zBdes, dim=0)

        Rdes = torch.stack((xBdes, yBdes, zBdes), dim=1)
        E = 0.5 * (Rdes.T @ R - R.T @ Rdes)
        eR = torch_vee(E)

        a_dot = sigmaddd[0:3]
        thrust_for_reference = torch.clamp(torch.abs(u1), min=1e-6)
        h = (self.drone.m / thrust_for_reference) * (a_dot - (zBdes @ a_dot) * zBdes)

        p_des = -h @ yBdes
        q_des = h @ xBdes
        r_des = sigmad[3] * (e3 @ zBdes)
        omega_des = torch.stack((p_des, q_des, r_des))

        ew = omega - omega_des
        torque = -self.Kr @ eR - self.Kw @ ew + torch.cross(omega, self.J @ omega, dim=0)
        torque = torch.maximum(torch.minimum(torque, self.max_torque), -self.max_torque)

        u = torch.cat((u1.reshape(1), torque))
        debug = {
            "ep": ep,
            "ev": ev,
            "eR": eR,
            "ew": ew,
            "Fdes": Fdes,
            "Rdes": Rdes,
        }

        return u, debug

    def PDControlLawBatched(self, positions, orientations, linear_vel, angular_vel, sigma, sigmad, sigmadd, sigmaddd):
        positions = self._to_tensor(positions)
        orientations = self._to_tensor(orientations)
        linear_vel = self._to_tensor(linear_vel)
        angular_vel = self._to_tensor(angular_vel)
        sigma = self._to_tensor(sigma)
        sigmad = self._to_tensor(sigmad)
        sigmadd = self._to_tensor(sigmadd)
        sigmaddd = self._to_tensor(sigmaddd)

        R = torch_quat_wxyz_to_matrix_batch(orientations)
        omega = torch.bmm(R.transpose(1, 2), angular_vel.unsqueeze(-1)).squeeze(-1)

        ep = positions - sigma[:, 0:3]
        ev = linear_vel - sigmad[:, 0:3]

        e3 = self.e3.unsqueeze(0).expand_as(positions)
        Fdes = (
            -(ep @ self.Kp.T)
            - (ev @ self.Kv.T)
            + self.drone.m * self.drone.g * e3
            + self.drone.m * sigmadd[:, 0:3]
        )

        zB = R[:, :, 2]
        u1 = (Fdes * zB).sum(dim=1)
        u1 = torch.clamp(u1, 0.0, self.max_thrust)

        zBdes_norm = torch.linalg.norm(Fdes, dim=1, keepdim=True)
        zBdes = Fdes / zBdes_norm.clamp_min(1e-8)
        zBdes = torch.where(zBdes_norm < 1e-8, e3, zBdes)

        xCdes = torch.stack(
            (
                torch.cos(sigma[:, 3]),
                torch.sin(sigma[:, 3]),
                torch.zeros_like(sigma[:, 3]),
            ),
            dim=1,
        )
        yBdes_raw = torch.cross(zBdes, xCdes, dim=1)
        yBdes_norm = torch.linalg.norm(yBdes_raw, dim=1, keepdim=True)
        yBdes = yBdes_raw / yBdes_norm.clamp_min(1e-8)
        y_fallback = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand_as(yBdes)
        yBdes = torch.where(yBdes_norm < 1e-8, y_fallback, yBdes)
        xBdes = torch.cross(yBdes, zBdes, dim=1)

        Rdes = torch.stack((xBdes, yBdes, zBdes), dim=2)
        E = 0.5 * (torch.bmm(Rdes.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), Rdes))
        eR = torch_vee_batch(E)

        a_dot = sigmaddd[:, 0:3]
        thrust_for_reference = torch.clamp(torch.abs(u1), min=1e-6).unsqueeze(1)
        zBdes_dot_adot = (zBdes * a_dot).sum(dim=1, keepdim=True)
        h = (self.drone.m / thrust_for_reference) * (a_dot - zBdes_dot_adot * zBdes)

        p_des = -(h * yBdes).sum(dim=1)
        q_des = (h * xBdes).sum(dim=1)
        r_des = sigmad[:, 3] * (e3 * zBdes).sum(dim=1)
        omega_des = torch.stack((p_des, q_des, r_des), dim=1)

        ew = omega - omega_des
        gyroscopic = torch.cross(omega, omega @ self.J.T, dim=1)
        torque = -(eR @ self.Kr.T) - (ew @ self.Kw.T) + gyroscopic
        max_torque = self.max_torque.unsqueeze(0)
        torque = torch.maximum(torch.minimum(torque, max_torque), -max_torque)

        u = torch.cat((u1.unsqueeze(1), torque), dim=1)
        debug = {
            "ep": ep,
            "ev": ev,
            "eR": eR,
            "ew": ew,
            "Fdes": Fdes,
            "Rdes": Rdes,
        }

        return u, debug
