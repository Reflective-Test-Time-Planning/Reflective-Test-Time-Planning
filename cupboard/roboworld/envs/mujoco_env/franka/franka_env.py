import abc
import copy
import pickle

import mujoco
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from typing import List

from roboworld.envs.mujoco_env.mujoco_env import MujocoEnv
from roboworld.envs.mujoco_env.utils.rotation import axisangle2mat, euler2quat, mat2quat, euler2mat, quat2axisangle, mat2euler
from roboworld.envs.mujoco_env.utils.transform import (get_rel_quat, get_rel_rotmat, get_rel_axisangle, rotate_quat, quat2mat, quat2euler)
from roboworld.envs.mujoco_env.franka.control import DiffIKNullspaceController
np.set_printoptions(precision=4)


class FrankaBase(MujocoEnv, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Franka Mujoco envs with eef pose control.
    """
    mocap_low = np.array([-0.5, -0.5, 0.06])
    mocap_high = np.array([0.5, 0.5, 1.2])

    robot_init_qpos = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]

    def __init__(self, model_name, frame_skip=5, render_mode='offscreen', render_width=640, render_height=480):
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip, render_mode=render_mode,
                           render_width=render_width, render_height=render_height)
        self.data.qpos[:9] = self.robot_init_qpos
        self.reset_mocap_welds()
        self.eef_site_name = "mocap_anchor"

    @property
    def eef_pos(self):
        return self._get_site_pos(self.eef_site_name)

    @property
    def eef_quat(self):
        return self._get_site_quat(self.eef_site_name)

    @property
    def gripper_interval(self):
        """The distance between gripper's 2 fingers

        Returns:
            (float): distance
        """
        right_finger_pos = self._get_site_pos('rightEndEffector')
        left_finger_pos = self._get_site_pos('leftEndEffector')
        return np.linalg.norm(right_finger_pos - left_finger_pos)

    def get_env_state(self):
        state = {
            "eq": (self.data.eq_active if mujoco.mj_version() >= 300 else self.model.eq_active, self.model.eq_data),    # mujoco 3
            "joint": (self.data.qpos, self.data.qvel),
            "mocap": (self.data.mocap_pos, self.data.mocap_quat)
        }
        if hasattr(self, "prev_action"):
            state["prev_action"] = self.prev_action
        if hasattr(self, "curr_path_length"):
            state["curr_path_length"] = self.curr_path_length
        return copy.deepcopy(state)

    def set_env_state(self, state):
        if mujoco.mj_version() >= 300:  # mujoco 3
            self.data.eq_active, self.model.eq_data = copy.deepcopy(state['eq'])
        else:
            self.model.eq_active, self.model.eq_data = copy.deepcopy(state['eq'])
        self.data.qpos, self.data.qvel = copy.deepcopy(state['joint'])
        self.data.mocap_pos, self.data.mocap_quat = copy.deepcopy(state['mocap'])
        mujoco.mj_forward(self.model, self.data)
        if hasattr(self, "prev_action"):
            self.prev_action = state["prev_action"].copy()
        if hasattr(self, "curr_path_length"):
            self.curr_path_length = state["curr_path_length"]

    def __getstate__(self):
        state = self.__dict__.copy()
        return {'state': state, 'env_state': self.get_env_state()}

    def __setstate__(self, state):
        self.__dict__ = state['state']
        self.set_env_state(state['env_state'])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        eq_name = "mocap_hand"
        eq_id = self.model.equality(eq_name).id

        self.model.eq_data[eq_id, :] = np.array(
            [0, 0, 0.0584 + 0.045,
             0, 0, 0,
             0., 0.7071415, 0.70707206, 0.,
             1]
        )

        mujoco.mj_forward(self.model, self.data)


class FrankaEnv(FrankaBase, metaclass=abc.ABCMeta):

    max_path_length = 5000

    TARGET_RADIUS = 0.05
    ZERO_ACTION = np.array([0., 0., 0., 1., 0., 0., 0., 0.04])
    GRIPPER_CLOSE_ACT = 0.0
    GRIPPER_OPEN_ACT = 0.04
    POS_ERROR_THRESH = 0.001
    DEFAULT_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04])

    def __init__(
            self,
            model_name,
            frame_skip=5,
            render_mode='offscreen',
            render_width=640,
            render_height=480,
            hand_low=(-1, -1, -1),
            hand_high=(1, 1, 1.5),
            mocap_low=None,
            mocap_high=None,
            translation_action_scale=1.0,
            rotation_action_scale=1.0,
            magic_attaching=False,
            store_trajectory=False,
            action_noise=False,
    ):
        super().__init__(model_name, frame_skip=frame_skip, render_mode=render_mode,
                         render_width=render_width, render_height=render_height)
        self.random_init = True
        self.translation_action_scale = translation_action_scale
        self.rotation_action_scale = rotation_action_scale
        self.magic_attaching = magic_attaching
        self.action_noise = action_noise
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.curr_path_length = 0
        self.seeded_rand_vec = False
        self._freeze_rand_vec = True
        self._last_rand_vec = None
        self.prev_action = self.ZERO_ACTION.copy()

        self.init_left_pad = self.get_body_pos('left_finger')
        self.init_right_pad = self.get_body_pos('right_finger')

        self.action_space = Box(
            np.full(8, np.float32(-1)),
            np.full(8, np.float32(1)),
        )

        self.hand_init_pos = None  # OVERRIDE ME
        self.hand_init_quat = None  # OVERRIDE ME

        self._prev_eef_pose = None
        self._last_stable_obs = None
        self._prev_obs = self._get_obs()

        self.object_init_poses = {}
        self.object_init_poses_raw = {}
        self.object_grasp_site_names = {}

        self.store_trajectory = store_trajectory
        self.flush_trajectory()

        self.controller = DiffIKNullspaceController(
            model=self.model,
            data=self.data,
            dof_ids=np.array([self.model.joint(f"joint{i+1}").id for i in range(7)]),
            actuator_ids=np.array([self.model.actuator(f"actuator{i+1}").id for i in range(7)])
        )

    def get_delta_action(self, action):
        """
        Convert to relative action.
        :param action: 8-d (relative pos, absolute quat, gripper)
        :return: relative pos, relative rpy, gripper
        """
        assert len(action) == 8
        rel_quat = get_rel_quat(self.eef_quat, action[3:7], global_frame=True)
        delta_action = np.zeros(7)
        delta_action[:3] = action[:3]
        delta_action[3:6] = quat2euler(rel_quat)
        delta_action[-1] = action[-1]

        return delta_action

    def _clip_delta_xyz(self, xyz, max_norm=0.08):
        # xyz: delta position in global frame
        clipped_xyz = xyz.copy()
        xyz_norm = np.linalg.norm(xyz)
        if xyz_norm > max_norm:
            clipped_xyz *= (max_norm / xyz_norm)
        return clipped_xyz

    def _clip_delta_rpy(self, rpy, max_angle=0.08):
        # rpy: delta rotation in global frame
        rel_quat = euler2quat(rpy)  # global
        axis, theta = quat2axisangle(rel_quat)
        theta = min(theta, max_angle) if theta > 0 else max(theta, -max_angle)
        rel_rot_mat = axisangle2mat(axis, theta)
        clipped_rpy = mat2euler(rel_rot_mat)
        return clipped_rpy

    def clip_delta_action(self, action, translation_max_norm=0.2, rotation_max_angle=0.2):
        assert len(action) == 7
        clipped_action = action.copy()
        clipped_action[:3] = self._clip_delta_xyz(action[:3], max_norm=translation_max_norm)
        clipped_action[3:6] = self._clip_delta_rpy(action[3:6], max_angle=rotation_max_angle)

        return clipped_action

    def get_target_mocap_pose(self, action, translation_max_norm=0.02, rotation_max_angle=0.2, delta_rpy=False):
        if delta_rpy:
            # action[3:6] is 3-d delta rpy (global frame)
            assert len(action) == 7, f"Action length should be 7, got {len(action)}."
        else:
            # action[3:7] is 4-d absolute quat (global frame)
            assert len(action) == 8, f"Action length should be 8, got {len(action)}."
        # action[:3] is always delta position

        pos_action = action[:3].copy()
        pos_action_norm = np.linalg.norm(pos_action)
        if pos_action_norm > translation_max_norm:
            pos_action *= (translation_max_norm / pos_action_norm)
        pos_delta = pos_action * self.translation_action_scale
        target_mocap_pos = self.eef_pos + pos_delta[None]

        target_mocap_pos[0, :] = np.clip(
            target_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )

        if delta_rpy:
            rel_quat = euler2quat(action[3:6])  # global
            axis, theta = quat2axisangle(rel_quat)
            theta = min(theta, rotation_max_angle) if theta > 0 else max(theta, -rotation_max_angle)
            theta *= self.rotation_action_scale
            rel_rot_mat = axisangle2mat(axis, theta)
            target_mocap_quat = rotate_quat(self.eef_quat, rel_rot_mat, global_frame=True)
        else:
            raise NotImplementedError

        return target_mocap_pos, target_mocap_quat

    def touching_object(self, object_geom_id=None):
        """Determines whether the gripper is touching the object with given id or touching any object when id=None

        Args:
            object_geom_id (int): the ID of the object in question

        Returns:
            (bool): whether the gripper is touching the object

        """
        leftpad_geom_ids = [self.unwrapped.model.geom(f'leftpad_c{i+1}').id for i in range(1)]
        rightpad_geom_ids = [self.unwrapped.model.geom(f'rightpad_c{i+1}').id for i in range(1)]

        leftpad_object_contacts = [
            x for x in self.unwrapped.data.contact
            if ((x.geom1 in leftpad_geom_ids or x.geom2 in leftpad_geom_ids)
                and (object_geom_id is None or object_geom_id in (x.geom1, x.geom2)))
        ]
        rightpad_object_contacts = [
            x for x in self.unwrapped.data.contact
            if ((x.geom1 in rightpad_geom_ids or x.geom2 in rightpad_geom_ids)
                and (object_geom_id is None or object_geom_id in (x.geom1, x.geom2)))
        ]

        leftpad_object_contact_forces = np.array([
            self.unwrapped.data.efc_force[x.efc_address]
            for x in leftpad_object_contacts
        ])
        rightpad_object_contact_forces = np.array([
            self.unwrapped.data.efc_force[x.efc_address]
            for x in rightpad_object_contacts
        ])

        return np.any(leftpad_object_contact_forces > 0) and np.any(rightpad_object_contact_forces > 0)

    def _get_eef_pose(self):
        eef_xyz = self.eef_pos
        eef_quat = self.eef_quat
        eef_rpy = quat2euler(eef_quat)
        obs = np.hstack((eef_xyz, eef_rpy))
        return obs

    def _get_obs(self):
        curr_pose = self._get_eef_pose()
        delta_pose = np.zeros(6)
        if self._prev_eef_pose is not None:
            delta_pose[:3] = curr_pose[:3] - self._prev_eef_pose[:3]
            prev_quat = euler2quat(self._prev_eef_pose[3:])
            curr_quat = euler2quat(curr_pose[3:])
            delta_quat = get_rel_quat(prev_quat, curr_quat, global_frame=True)
            delta_pose[3:] = quat2euler(delta_quat)
        return np.hstack((curr_pose, delta_pose))

    def _get_obs_dict(self):
        obs = self._get_obs()
        return dict(
            state_observation=obs,
        )

    @property
    def observation_space(self):
        return Box(np.full(12, np.float32(-np.inf)), np.full(12, np.float32(np.inf)))

    def step(self, action, translation_max_norm=0.02, rotation_max_angle=0.2, debug=False):

        if len(action) == 8:
            delta_action = self.get_delta_action(action)
        else:
            assert len(action) == 7
            delta_action = action.copy()

        delta_action = self.clip_delta_action(delta_action,
                                              translation_max_norm=translation_max_norm,
                                              rotation_max_angle=rotation_max_angle)

        if self.action_noise:
            delta_action[:3] += np.random.uniform(-0.005, 0.005, size=(3,))
            delta_action[3:6] += np.random.uniform(-0.02, 0.02, size=(3,))
            delta_action = self.clip_delta_action(delta_action,
                                                  translation_max_norm=translation_max_norm,
                                                  rotation_max_angle=rotation_max_angle)

        target_pos, target_quat = self.get_target_mocap_pose(delta_action, delta_rpy=True)

        if debug:
            target_pos0, target_quat0 = self.get_target_mocap_pose(action)
            assert np.allclose(target_pos0, target_pos), f"{target_pos0} {target_pos}"
            assert np.allclose(target_quat0, target_quat), f"{target_quat0} {target_quat}"

        self.data.mocap_pos[:] = target_pos
        self.data.mocap_quat[:] = target_quat

        ctrl = self.DEFAULT_CTRL.copy()
        ctrl[:7] = self.controller.get_control(target_pos, target_quat)
        ctrl[-1] = action[-1]
        self.do_simulation(ctrl)
        self.curr_path_length += 1

        obs, reward, done, info = None, 0., False, {}

        if self._did_see_sim_exception:
            obs = self._last_stable_obs  # observation just before going unstable
        else:
            self._last_stable_obs = self._get_obs()
            obs = self._last_stable_obs 

        self.prev_action = action.copy()
        self._prev_eef_pose = self._get_eef_pose()

        return obs, reward, done, info

    def flush_trajectory(self):
        self.trajectory = {
            'observations': [],
            'actions': [],
            'images': [],
        }

    def retrieve_trajectory(self):
        return {k: np.array(v) for k, v in self.trajectory.items()}

    def reset(self, seed=None):
        self.curr_path_length = 0
        obs = super().reset(seed=seed)
        self.curr_path_length = 0
        self.prev_action = self.ZERO_ACTION.copy()
        self._prev_eef_pose = self._get_eef_pose()
        self.flush_trajectory()
        return obs

    def _reset_hand(self, steps=50):
        self.data.qpos[:9] = self.robot_init_qpos
        self.reset_mocap_welds()
        for _ in range(steps):
            self.data.mocap_pos = self.hand_init_pos
            self.data.mocap_quat = self.hand_init_quat
            self.do_simulation([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255], self.frame_skip)
        self.init_tcp = self.eef_pos

    def control_gripper(self, target_interval):
        action = self.prev_action.copy()
        action[:3] = 0
        action[-1] = target_interval / 2
        while np.abs(self.gripper_interval - target_interval) > 0.001:
            self.step(action)
            self.render()

    def close_gripper(self, grasp_obj_name=None):
        action = self.prev_action.copy()
        action[:3] = 0
        action[-1] = self.GRIPPER_CLOSE_ACT
        while not self.touching_object() and self.gripper_interval > 0.001:
            self.step(action)
            self.render()
        if self.touching_object():
            action[-1] = 0
            for _ in range(10):
                self.step(action)
                self.render()

        if grasp_obj_name is not None and self.magic_attaching:
            self._set_weld_eq_by_current_state(eq_name=f"{grasp_obj_name}_grasp_hand", active=True)

    def _get_grasp_site_names(self, obj_name, suffix="", max_grasps=10):
        key = f"{obj_name}{suffix}"
        if key in self.object_grasp_site_names:
            return self.object_grasp_site_names[key]
        self.object_grasp_site_names[key] = []
        for i in range(max_grasps):
            try:
                self.data.site(f"{key}_grasp{i}")
                self.object_grasp_site_names[key].append(f"{key}_grasp{i}")
            except:
                break
        return self.object_grasp_site_names[key]

    def get_canonical_grasp_pose(self, obj_name):
        grasp_name = f"{obj_name}_grasp0"
        return self._get_site_pos(grasp_name), self._get_site_quat(grasp_name)

    def get_grasp_quats(self, obj_name, grasp_pos=None):
        grasp_poses = self.get_grasp_poses(obj_name)
        grasp_quats = [quat for (pos, quat) in grasp_poses if grasp_pos is None or np.allclose(grasp_pos, pos)]
        return grasp_quats

    def get_grasp_poses(self, obj_name: str, suffix: str = ""):
        return [(self._get_site_pos(grasp_name), self._get_site_quat(grasp_name))
                for grasp_name in self._get_grasp_site_names(obj_name, suffix=suffix)]

    def get_best_grasp_pose(self, obj_name: str = None, grasp_poses: List[tuple[np.ndarray, np.ndarray]] = None):
        assert obj_name is not None or grasp_poses is not None
        if grasp_poses is None:
            grasp_poses = self.get_grasp_poses(obj_name)
        min_err = float("inf")
        best_pos, best_quat = None, None
        for (pos, quat) in grasp_poses:
            curr_err = self.quat_err(self.eef_quat, quat)
            if curr_err < min_err:
                min_err = curr_err
                best_pos, best_quat = pos, quat
        return best_pos, best_quat

    def _set_weld_eq_by_current_state(self, eq_name, active=True):
        eq = self.model.equality(eq_name)
        obj1_name = self.model.body(eq.obj1id).name
        obj2_name = self.model.body(eq.obj2id).name
        obj1_pos, obj1_quat = self.get_body_pos(obj1_name), self.get_body_quat(obj1_name)
        obj2_pos, obj2_quat = self.get_body_pos(obj2_name), self.get_body_quat(obj2_name)
        T1, T2 = np.eye(4), np.eye(4)
        T1[:3, :3] = quat2mat(obj1_quat)
        T1[:3, 3] = obj1_pos
        T2[:3, :3] = quat2mat(obj2_quat)
        T2[:3, 3] = obj2_pos
        T2in1 = np.linalg.inv(T1) @ T2
        self.set_weld_data(eq_name, relpos=T2in1[:3, 3], relquat=mat2quat(T2in1[:3, :3]))
        self.set_eq_active(eq_name, active=active)

    def open_gripper(self, grasp_obj_name=None, target_interval=0.07):
        if grasp_obj_name is not None and self.magic_attaching:
            self.set_eq_active(f"{grasp_obj_name}_grasp_hand", False)

        action = self.prev_action.copy()
        action[:3] = 0  # no arm movement

        # Estimate current interval from previous action
        current_interval = action[-1] * 2

        while current_interval < target_interval:
            current_interval += 0.003
            action[-1] = current_interval / 2  # symmetric gripper width control
            self.step(action)
            self.render()

            # Optional early exit if object has been released
            if grasp_obj_name is not None and not self.touching_object():
                for _ in range(3):
                    current_interval += 0.003
                    action[-1] = current_interval / 2
                    self.step(action)
                    self.render()
                break

    def goto(self, pos=None, quat=None, pos_err_th=0.01, quat_err_th=0.01, max_steps=2000, 
             translation_max_norm=0.05, rotation_max_angle=0.2, grasped_object=None,
             debug=False):
        action = self.prev_action
        if pos is None:
            action[:3] = 0
        steps_st = self.curr_path_length
        while True:
            if grasped_object and not self.object_is_in_hand(grasped_object):
                return False
            pos_err = None if pos is None else self.pos_err(self.eef_pos, pos)
            quat_err = None if quat is None else self.quat_err(self.eef_quat, quat)
            if (pos is None or pos_err <= pos_err_th) and \
                    (quat is None or quat_err <= quat_err_th):
                break
            if debug:
                err_txt = f"[Goto] Error: "
                if pos is not None:
                    err_txt += f" pos {self.pos_err(self.eef_pos, pos)}"
                if quat is not None:
                    err_txt += f" quat {self.quat_err(self.eef_quat, quat)}"
                print(err_txt)
            if pos is not None and self.pos_err(self.eef_pos, pos) > pos_err_th:
                delta_pos = pos - self.eef_pos
                action[:3] = delta_pos
            if quat is not None and self.quat_err(self.eef_quat, quat) > quat_err_th:
                action[3:7] = quat
            self.step(action, translation_max_norm=translation_max_norm, rotation_max_angle=rotation_max_angle)
            self.render()
            
            if self.curr_path_length >= steps_st + max_steps:
                print("Max steps exceeded for `goto`.")
                return False
        return True

    @property
    def dummy_action(self):
        action = self.prev_action.copy()
        action[:3] = 0
        return action

    def pickup(self, obj_name):
        grasp_poses = []
        for suffix in ["", "_side0", "_side1"]:
            grasp_poses += self.get_grasp_poses(obj_name, suffix=suffix)

        valid_grasp_poses = [pose for pose in grasp_poses if quat2mat(pose[1])[2, 2] > 0.6]
        # valid_grasp_poses = grasp_poses
        if len(valid_grasp_poses) > 0:
            grasp_pose = self.get_best_grasp_pose(grasp_poses=valid_grasp_poses)
            self._pickup_aux(obj_name, obj_grasp_pose=grasp_pose)
        else:
            #grasp_pose = self.get_best_grasp_pose(grasp_poses=grasp_poses)
            #self._pickup_aux(obj_name, obj_grasp_pose=grasp_pose)

            print(f"Object `{obj_name}` is not graspable")
            return False

        return True

    def reorient(self, obj_name):
        if self.object_is_upright(obj_name):
            return
        self.put_on_fixture(obj_name)
        self.grasp_from_fixture(obj_name)

    def put_on_fixture(self, obj_name):
        assert self.object_is_in_hand(obj_name)
        is_nail = ("nail" in self.peg_desc[obj_name])
        fixture_pos = self.get_body_pos("fixture")
        if is_nail:
            waypoints = np.array([[0.05, 0.05, z] for z in [0.2, 0.07, 0.045, 0.2]])
        else:
            waypoints = np.array([[0, 0.04, z] for z in [0.2, 0.12, 0.06, 0.20]])
        self.goto(pos=fixture_pos + waypoints[0], quat=euler2quat([0, 0, 0]), grasped_object=obj_name)
        obj_quat = self.get_body_quat(obj_name)
        if is_nail:
            yaw = 0
        else:
            if (quat2mat(obj_quat)[:, 2]) @ np.array([1, 0, 0]) > 0:    # obj_z points to positive x
                yaw = np.pi / 2
            else:
                yaw = -np.pi / 2
        self.goto(pos=fixture_pos + waypoints[1], quat=euler2quat([0, 0, yaw]), grasped_object=obj_name)
        self.goto(pos=fixture_pos + waypoints[2], quat=euler2quat([0, 0, yaw]), grasped_object=obj_name)
        self.open_gripper(obj_name, target_interval=min(self.gripper_interval + 0.02, 0.10))
        self.goto(pos=fixture_pos + waypoints[3], quat=euler2quat([0, 0, yaw]))

    def grasp_from_fixture(self, obj_name):
        fixture_pos = self.get_body_pos("fixture")
        grasp_poses = self.get_grasp_poses(obj_name)
        is_nail = ("nail" in self.peg_desc[obj_name])
        if is_nail:
            grasp_poses += self.get_grasp_poses(obj_name, suffix="_ridge")
            grasp_poses.sort(key=lambda x: np.abs(np.dot(
                quat2mat(x[1])[:, 0],   # x of grasp orientation
                np.array([0, 0, 1])     # z of global frame
            )))   # we want x of grasp pose pointing along (negative or positive) z
            grasp_poses = grasp_poses[:2]

        grasp_poses.sort(key=lambda x: self.quat_err(x[1], self.eef_quat))
        obj_grasp_pos, obj_grasp_quat = grasp_poses[0]

        prepare_pos = obj_grasp_pos + quat2mat(obj_grasp_quat) @ np.array([0, 0, 0.05])
        self.goto(pos=fixture_pos + np.array([0, 0.2, 0.2]))
        self.goto(pos=prepare_pos, quat=obj_grasp_quat)
        self.control_gripper(target_interval=0.05)
        self.goto(pos=obj_grasp_pos, quat=obj_grasp_quat)

        self.close_gripper(grasp_obj_name=obj_name)
        liftup_pos = self.eef_pos.copy()
        liftup_pos[-1] = 0.5 + 0.4
        self.goto(pos=liftup_pos, pos_err_th=0.05, quat_err_th=0.05)
        self.goto(quat=rotate_quat(self.eef_quat, euler2mat([np.pi/2, 0, 0]), global_frame=True),
                  pos_err_th=0.05, quat_err_th=0.01, rotation_max_angle=0.04)
        self.goto(quat=[1, 0, 0, 0], pos_err_th=0.05, quat_err_th=0.01)

    def _pickup_aux(self, obj_name, obj_grasp_pose=None, gripper_interval=None):
        if gripper_interval is None:
            self.open_gripper()
        else:
            self.control_gripper(target_interval=gripper_interval)

        if obj_grasp_pose is None:
            obj_grasp_pose = self.get_best_grasp_pose(obj_name)
        obj_grasp_pos, obj_grasp_quat = obj_grasp_pose

        prepare_pos = obj_grasp_pos + quat2mat(obj_grasp_quat) @ np.array([0, 0, 0.2])
        self.goto(pos=prepare_pos, quat=[1, 0, 0, 0])
        self.goto(quat=obj_grasp_quat)
        self.goto(pos=obj_grasp_pos, quat=obj_grasp_quat)
        self.close_gripper(grasp_obj_name=obj_name)
        liftup_pos = self.eef_pos.copy()
        liftup_pos[-1] = 0.5 + 0.4
        self.goto(pos=liftup_pos, pos_err_th=0.05, quat_err_th=0.05)
        self.goto(quat=[1, 0, 0, 0], pos_err_th=0.05, quat_err_th=0.01)

    def pickup_nail(self, obj_name):
        canonical_grasp_pos, canonical_grasp_quat = self.get_canonical_grasp_pose(obj_name)
        z = quat2mat(canonical_grasp_quat)[:, 2]
        assert np.abs(np.linalg.norm(z) - 1) < 1e-5
        if z @ np.array([0, 0, 1]) >= 0.5:
            self._pickup_aux(obj_name)
            return

        grasp_y = z.copy()
        grasp_z = np.array([0, 0, 1])
        grasp_x = np.cross(grasp_y, grasp_z)
        grasp_rotmat = np.column_stack((grasp_x, grasp_y, grasp_z))
        grasp_quat = mat2quat(grasp_rotmat)

        self._pickup_aux(obj_name, obj_grasp_pose=(canonical_grasp_pos, grasp_quat))

    def put_on_table(self, obj_name):
        if not self.object_is_in_hand(obj_name):
            return
        target_pos = self.object_init_poses_raw[obj_name]['pos']
        target_quat = self.object_init_poses_raw[obj_name]['quat']
        self.goto(pos=target_pos + np.array([0, 0, 0.2]), quat=target_quat, debug=False)
        if self.object_is_upright(obj_name):
            self.align(src_body_name=obj_name, dst_pose=(target_pos, target_quat), pos_err_th=0.005, quat_err_th=0.05,
                       check_stuck=True)
        else:
            for site_name in [f"{obj_name}_side0_base", f"{obj_name}_side1_base"]:
                site_quat = self._get_site_quat(site_name)
                if quat2mat(site_quat)[:, 2] @ np.array([0, 0, 1]) > 0.9:
                    self.align(src_body_name=obj_name, src_site_name=site_name, dst_pose=(target_pos, target_quat),
                               pos_err_th=0.005, quat_err_th=0.05, check_stuck=True)
        self.open_gripper(obj_name, target_interval=min(self.gripper_interval + 0.01, 0.08))
        self.home()

    def insert(self, src_name, dst_name):
        target_align_pos = self._get_site_pos(f"{dst_name}_align")
        prepare_pos = target_align_pos + np.array([0, 0, 0.30])
        self.goto(pos=prepare_pos)
        if not self.object_is_upright(src_name):
            print(f"Object `{src_name}`({self.peg_desc[src_name]}) is in bad orientation for insertion.")
            q_hand, q_obj = self.eef_quat, self.get_body_quat(src_name)
            rot_hand2obj = get_rel_rotmat(q_hand, q_obj)
            rot_align2grasp = get_rel_rotmat(self._get_site_quat(f"{src_name}_align"), self.eef_quat)
            q_target = rotate_quat(self._get_site_quat(f"{dst_name}_align"), rot_mat=rot_align2grasp)
            q_target = rotate_quat(q_target, rot_mat=rot_hand2obj)
            p_target = self._get_site_pos(f"{dst_name}_align")
            self.align(src_body_name=src_name, src_site_name=f"{src_name}_align", dst_pose=(p_target, q_target),
                       offset=[0, 0, 0.15])
            succ = self.align(src_body_name=src_name, src_site_name=f"{src_name}_align", dst_pose=(p_target, q_target),
                              quat_err_th=0.005, translation_max_norm=0.02, check_stuck=True)

        else:
            self.align(src_body_name=src_name, src_site_name=f"{src_name}_align", dst_site_name=f"{dst_name}_align",
                       translation_max_norm=0.01, rotation_max_angle=0.1,
                       offset=[0, 0, 0.15], debug=False)
            succ = self.align(src_body_name=src_name, src_site_name=f"{src_name}_align", dst_site_name=f"{dst_name}_align",
                              quat_err_th=0.005, translation_max_norm=0.01, rotation_max_angle=0.1, check_stuck=True)
        if succ:
            self.open_gripper(grasp_obj_name=src_name)
            self.home()
        else:
            self.home()

    def align(self, src_body_name, src_site_name=None, dst_site_name=None, dst_pose=None, offset=None,
              pos_err_th=0.002, quat_err_th=0.002, translation_max_norm=0.02, rotation_max_angle=0.2,
              check_stuck=False, max_steps=500, debug=False):
        # align src pose to dst pose + offset
        # either dst_site_name or dst_pose should be specified

        def get_src_dst_poses():
            if src_site_name is not None:
                _src_pos = self._get_site_pos(src_site_name)
                _src_quat = self._get_site_quat(src_site_name)
            else:
                # if src_site_name is not specified, use body pose
                _src_pos = self.get_body_pos(src_body_name)
                _src_quat = self.get_body_quat(src_body_name)

            if dst_site_name is not None:
                assert dst_pose is None
                _dst_pos = self._get_site_pos(dst_site_name)
                _dst_quat = self._get_site_quat(dst_site_name)
            else:
                assert dst_pose is not None
                _dst_pos, _dst_quat = dst_pose[0].copy(), dst_pose[1].copy()
            if offset is not None:
                _dst_pos += offset

            return _src_pos, _src_quat, _dst_pos, _dst_quat
        src_pos, src_quat, dst_pos, dst_quat = get_src_dst_poses()
        if debug:
            print("src_pos", src_pos)
            print("src_quat", src_quat)
            print("dst_pos", dst_pos)
            print("dst_quat", dst_quat)

        def get_eef_target_quat():
            rel_rotmat = get_rel_rotmat(src_quat, dst_quat, global_frame=True)
            eef_target_quat = rotate_quat(self.eef_quat, rel_rotmat, global_frame=True)
            return eef_target_quat

        steps_st = self.curr_path_length

        while True:
            if not self.object_is_in_hand(src_body_name):
                print(f"{src_body_name} dropped!")
                return False
            pos_err = self.pos_err(src_pos, dst_pos)
            quat_err = self.quat_err(src_quat, dst_quat)
            if pos_err <= pos_err_th and quat_err <= quat_err_th:
                break
            if debug:
                print(f"Alignment error: [pos] {self.pos_err(src_pos, dst_pos):.5f} "
                      f"[quat] {self.quat_err(src_quat, dst_quat):.5f}")
            delta_pos = dst_pos - src_pos
            # check collision
            if self.has_large_cfrc(src_body_name, cfrc_th=10.0):
                pos_act_limit = 0.001
                if debug:
                    print("large contact force!")
            else:
                pos_act_limit = translation_max_norm
            delta_pos_norm = np.linalg.norm(delta_pos)
            if delta_pos_norm > pos_act_limit:
                delta_pos *= (pos_act_limit / delta_pos_norm)
            action = self.prev_action.copy()
            action[:3] = delta_pos
            if debug:
                print(action[:3])

            action[3:7] = get_eef_target_quat()
            self.step(action, translation_max_norm=translation_max_norm, rotation_max_angle=rotation_max_angle)
            self.render()
            if check_stuck and self.is_stuck(src_body_name):
                return False
            src_pos, src_quat, dst_pos, dst_quat = get_src_dst_poses()
            if self.curr_path_length >= steps_st + max_steps:
                print("Max steps exceeded for aligning.")
                return False

        return True

    def home(self):
        self.goto(pos=self.hand_init_pos, quat=self.hand_init_quat)

    def reset_object(self, body_name):
        self.set_body_pos(body_name, self.object_init_poses[body_name]["pos"])
        self.set_body_quat(body_name, self.object_init_poses[body_name]["quat"])

    @staticmethod
    def pos_err(pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    @staticmethod
    def quat_err(quat1, quat2):
        # Mujoco quat (w,x,y,z)
        err = min(np.linalg.norm(quat1 - quat2), np.linalg.norm(quat1 + quat2))
        err /= np.sqrt(2)
        return err

    def get_lin_vel(self, body_name):
        body_id = self.model.body(body_name).id
        assert self.model.body_dofnum[body_id] == 6
        qvel_start_idx = self.model.body_dofadr[body_id]
        body_lin_vel = self.data.qvel[qvel_start_idx: qvel_start_idx + 6]
        return body_lin_vel

    def get_contact_force(self, body_name):
        geom_adr = self.model.body(body_name).geomadr[0]
        geom_num = self.model.body(body_name).geomnum[0]
        assert geom_adr != -1
        object_geom_ids = [geom_adr + i for i in range(geom_num)]

        forces = []
        for c in self.unwrapped.data.contact:
            if c.geom1 not in object_geom_ids and c.geom2 not in object_geom_ids:
                continue
            f = self.unwrapped.data.efc_force[c.efc_address]
            forces.append(f)
        return np.array(forces)

    def has_large_cfrc(self, body_name, cfrc_th=10.0):
        forces = self.get_contact_force(body_name)
        return len(forces) > 0 and np.max(forces) > cfrc_th

    def is_stuck(self, object_in_hand, contact_force_th=20., lin_vel_th=0.02):
        lin_vel = self.get_lin_vel(object_in_hand)
        return self.has_large_cfrc(object_in_hand, cfrc_th=contact_force_th) and np.abs(lin_vel[2]) < lin_vel_th

    def object_is_upright(self, body_name, thresh=0.9):
        _, body_quat = self.get_canonical_grasp_pose(body_name)
        rot_mat = quat2mat(body_quat)
        z = rot_mat @ np.array([0, 0, 1])
        return z @ np.array([0, 0, 1]) > thresh

    def object_is_in_hand(self, obj_name):
        if self.magic_attaching:
            eq = self.model.equality(f"{obj_name}_grasp_hand")
            if mujoco.mj_version() >= 300:  # mujoco 3
                return self.data.eq_active[eq.id]
            else:
                return self.model.eq_active[eq.id]
        else:
            pad_geom_ids = []
            for i in range(1):
                pad_geom_ids.extend([f'leftpad_c{i+1}', f'rightpad_c{i+1}'])
            return self.check_collision(
                geom_ids1=self.get_geom_ids(obj_name),
                geom_ids2=[self.model.geom(name).id for name in pad_geom_ids],
                require_nonzero_force=True
            )

    def get_object_in_hand(self):
        if self.magic_attaching:
            for i in range(self.model.neq):
                eq = self.model.equality(i)
                eq_act = self.data.eq_active[i] if mujoco.mj_version() >= 300 else self.model.eq_active[i]  # mujoco 3
                if eq_act and "grasp_hand" in eq.name:
                    obj_id = eq.obj2id
                    return self.model.body(obj_id).name
            return None

        for obj in self.peg_names:
            if self.object_is_in_hand(obj):
                return obj
        return None
