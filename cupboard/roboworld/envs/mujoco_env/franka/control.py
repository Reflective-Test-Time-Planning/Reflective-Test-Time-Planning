# Adapted from https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py

import mujoco
import mujoco.viewer
import numpy as np


class ControllerBase:

    def __init__(self, model, data, dof_ids, actuator_ids):
        assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

        # Load the model and data.
        self.model = model
        self.data = data

        self.dof_ids = dof_ids
        self.actuator_ids = actuator_ids

        # End-effector site we wish to control.
        self.site_id = model.site("mocap_anchor").id

        # initial qpos
        self.q0 = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    def get_control(self, target_pos, target_quat):
        raise NotImplementedError


class DiffIKNullspaceController(ControllerBase):

    def __init__(self, model, data, dof_ids, actuator_ids):
        super(DiffIKNullspaceController, self).__init__(model, data, dof_ids, actuator_ids)

        self.integration_dt: float = 0.1
        self.damping: float = 1e-4
        self.Kpos: float = 0.95
        self.Kori: float = 0.95
        self.dt = model.opt.timestep
        self.Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 0.0])
        self.max_angvel = 2.0

        # Pre-allocate numpy arrays.
        self.jac = np.zeros((6, model.nv))
        self.diag = self.damping * np.eye(6)
        self.eye = np.eye(len(self.dof_ids))
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

    def get_control(self, target_pos, target_quat):
        # Spatial velocity (aka twist).
        dx = target_pos - self.data.site(self.site_id).xpos
        self.twist[:3] = self.Kpos * dx / self.integration_dt
        mujoco.mju_mat2Quat(self.site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.Kori / self.integration_dt

        # Jacobian.
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
        jac = self.jac[:, self.dof_ids]

        # Damped least squares.
        dq = jac.T @ np.linalg.solve(jac @ jac.T + self.diag, self.twist)

        # Nullspace control biasing joint velocities towards the home configuration.
        dq += (self.eye - np.linalg.pinv(jac) @ jac) @ (self.Kn * (self.q0 - self.data.qpos[self.dof_ids]))

        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > self.max_angvel:
            dq *= self.max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q_full = self.data.qpos.copy()  # Note the copy here is important.
        dq_full = np.zeros(self.model.nv)
        dq_full[self.dof_ids] = dq
        mujoco.mj_integratePos(self.model, q_full, dq_full, self.integration_dt)

        ctrl = q_full[self.dof_ids]

        return ctrl

    def set_control(self, ctrl=None, target_pos=None, target_quat=None):
        if ctrl is not None:
            pass
        else:
            assert target_pos is not None and target_quat is not None
            ctrl = self.get_control(target_pos, target_quat)
        self.data.ctrl[self.actuator_ids] = ctrl.copy()
