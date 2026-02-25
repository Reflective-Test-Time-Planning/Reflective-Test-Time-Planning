import warnings
import numpy as np

from roboworld.envs.asset_path_utils import full_path_for
from roboworld.envs.mujoco_env.franka.franka_env import FrankaEnv
from roboworld.envs.mujoco_env.utils.rotation import euler2quat, quat2mat, mat2quat
from roboworld.envs.generator import get_color_name


class FrankaAssemblyEnv(FrankaEnv):

    def __init__(self, board_name, fixture_name, peg_names, peg_descriptions, model_name="main.xml",
                 render_mode='offscreen', frame_skip=5, max_episode_length=5000, grid_arrangement=False,
                 control_mode="script",
                 **kwargs):
        hand_low = (-0.5, -0.7, 0)
        hand_high = (1, 0.5, 1.5)

        self._model_name = model_name

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            frame_skip=frame_skip,
            **kwargs
        )
        self.max_path_length = max_episode_length
        self.grid_arrangement=grid_arrangement
        assert control_mode in {"script"}, f"Unknown control mode: {control_mode}."
        self.control_mode = control_mode

        self.init_config = {
            'obj_init_pos': np.array((0, 0.6, 0.02), dtype=np.float32),
            'hand_init_pos': np.array((0., 0., 0.9), dtype=np.float32),
            'hand_init_quat': np.array((1., 0., 0., 0.), dtype=np.float32)
        }
        self.goal = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.hand_init_quat = self.init_config['hand_init_quat']
        self.robot_base_pos = self.get_body_pos("link0")

        self.board_name = board_name
        self.board_id = self.model.body(board_name).id
        self.fixture_name = fixture_name
        self.fixture_id = self.model.body(fixture_name).id if fixture_name is not None else -1
        self.peg_names = peg_names
        self.peg_desc = {k: v for k, v in zip(peg_names, peg_descriptions)}
        self.peg_colors = []
        for name in peg_names:
            geom_adr = self.model.body(name).geomadr
            assert geom_adr != -1
            color = get_color_name(self.model.geom(geom_adr).rgba[:3])
            self.peg_colors.append(color)
        self.peg_ids = [self.model.body(name).id for name in peg_names]

        self.board_pos_low    = ( 0.00,  0.15, 0.50)
        self.board_pos_high   = ( 0.09,  0.20, 0.50)
        self.fixture_pos_low  = (-0.20,  0.10, 0.50)
        self.fixture_pos_high = (-0.20,  0.20, 0.50)
        self.peg_pos_low      = (-0.30, -0.50, 0.50)
        self.peg_pos_high     = ( 0.20, -0.05, 0.50)

        self.object_init_poses = {peg_name: {
            "pos": self.get_body_pos(peg_name), "quat": self.get_body_quat(peg_name)
        } for peg_name in peg_names}

        self.goal_images = None

    @property
    def model_name(self):
        return full_path_for(self._model_name)

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        return obs_dict

    def reset_model(self, max_trials=50):
        self._reset_hand()

        if self.random_init:
            for i, body_name in enumerate(self.peg_names):
                self.set_body_pos(body_name, np.array([2., i, 0.]))
            is_valid = False
            for k in range(max_trials):
                is_valid = self.randomize_object_poses()
                if is_valid:
                    break
            if not is_valid:
                warnings.warn("Unstable initial state")

            self.randomize_partial_assembly(camera_names=["table_back"])

        self._reset_hand()

        return self._get_obs()

    def randomize_partial_assembly(self, min_pegs_to_assemble=1, max_pegs_to_assemble=None,
                                   camera_names=None, full_assembly=False):

        if max_pegs_to_assemble is None:
            max_pegs_to_assemble = len(self.peg_names)

        for peg_name in self.peg_names:
            # record initial poses
            self.object_init_poses[peg_name]["pos"] = self.get_body_pos(peg_name)
            self.object_init_poses[peg_name]["quat"] = self.get_body_quat(peg_name)
            # assemble all pegs by setting their poses directly
            self.set_body_pos(peg_name, self._get_site_pos(f"{peg_name}_hole_align"))
            self.set_body_quat(peg_name, self._get_site_quat(f"{peg_name}_hole_align"))

        if camera_names is None:
            camera_names = ["table_back"]
        if self.render_mode == "offscreen":
            self.goal_images = {cam: self.read_pixels(camera_name=cam) for cam in camera_names}

        if full_assembly:
            pegs_to_assemble = self.peg_names
        else:
            pegs_to_assemble = list(np.random.choice(
                self.peg_names,
                size=np.random.randint(min_pegs_to_assemble, max(min_pegs_to_assemble + 1, max_pegs_to_assemble)),
                replace=False
            ))

        # reset initial poses of the pegs to be assembled
        for peg_name in reversed(self.peg_names):
            if peg_name in pegs_to_assemble:
                self.set_body_pos(peg_name, self.object_init_poses[peg_name]["pos"])
                self.set_body_quat(peg_name, self.object_init_poses[peg_name]["quat"])

        return self.goal_images

    def compute_reward(self, actions, obs):
        raise NotImplementedError

    def randomize_object_poses(self, max_trials_per_object=50, debug=False):

        def get_rand_quat(euler_init=None, free_axes="z"):
            if euler_init is None:
                rand_euler = np.zeros(3)
            else:
                rand_euler = np.array(euler_init)
            for i, axis in enumerate(["x", "y", "z"]):
                if axis in free_axes:
                    rand_euler[i] = self.np_random.uniform(-np.pi, np.pi)
            return euler2quat(rand_euler)

        # randomize pose of the board
        board_rand_pos = self.np_random.uniform(self.board_pos_low, self.board_pos_high)
        board_rand_quat = get_rand_quat()
        self.set_body_pos(self.board_name, board_rand_pos)
        self.set_body_quat(self.board_name, board_rand_quat)
        self.object_init_poses[self.board_name] = {"pos": board_rand_pos, "quat": board_rand_quat}

        # randomize poses of the other components
        if self.grid_arrangement:
            n_pegs = len(self.peg_names)
            n_rows = 1 if n_pegs <= 3 else 2
            n_cols = (n_pegs + n_rows - 1) // n_rows
            xs = np.linspace(self.peg_pos_low[0] + 0.01, self.peg_pos_high[0] - 0.10, n_cols)
            ys = np.linspace(self.peg_pos_low[1] + 0.15, self.peg_pos_high[1] - 0.10, n_rows)
            xv, yv = np.meshgrid(xs, ys)
            grid_positions = np.array([[x, y, 0.5] for x, y in zip(xv.flatten(), yv.flatten())])
            self.peg2grid_idxs = self.np_random.permutation(n_pegs)
            self.grid2peg_idxs = np.ones((n_rows, n_cols), dtype=int) * (-1)
            self.grid2peg_idxs = self.grid2peg_idxs.flatten()
            for peg_i, idx in enumerate(self.peg2grid_idxs):
                self.grid2peg_idxs[idx] = peg_i
            self.grid2peg_idxs = self.grid2peg_idxs.reshape(n_rows, n_cols)
        else:
            grid_positions = None

        for i, body_name in enumerate(self.peg_names):
            is_valid = False
            body_id = self.model.body(body_name).id
            upright = self.np_random.uniform() < 0.5
            for k in range(max_trials_per_object):
                # reset previously decided poses in case some objects are moved because of collision
                for obj_name, obj_pose in self.object_init_poses.items():
                    self.set_body_pos(obj_name, obj_pose["pos"])
                    self.set_body_quat(obj_name, obj_pose["quat"])
                if grid_positions is None:
                    rand_pos = self.np_random.uniform(self.peg_pos_low, self.peg_pos_high)
                else:
                    rand_pos = grid_positions[self.peg2grid_idxs[i]]
                    rand_pos[:2] += self.np_random.uniform(-0.02, 0.02)
                rand_quat = get_rand_quat()
                self.object_init_poses_raw[body_name] = {"pos": rand_pos, "quat": rand_quat}
                if not upright:
                    # set the pose of either side0_base or side1_base to (rand_pos, rand_quat)
                    side_choice = self.np_random.choice([0, 1])
                    site_name = f"{body_name}_side{side_choice}_base"
                    transf = np.eye(4)
                    transf[:3, :3] = quat2mat(self.model.site(site_name).quat)
                    transf[:3, 3] = self.model.site(site_name).pos
                    T_site, T_body, T_rand = np.eye(4), np.eye(4), np.eye(4)
                    T_rand[:3, :3] = quat2mat(rand_quat)
                    T_rand[:3, 3] = rand_pos
                    T_rand_new = T_rand @ np.linalg.inv(transf)
                    rand_quat = mat2quat(T_rand_new[:3, :3])
                    rand_pos = T_rand_new[:3, 3]

                self.set_body_pos(body_name, rand_pos)
                self.set_body_quat(body_name, rand_quat)

                curr_body_pos = self.get_body_pos(body_name)
                curr_body_quat = self.get_body_quat(body_name)
                self.object_init_poses[body_name] = {"pos": curr_body_pos, "quat": curr_body_quat}

                # Constraint 1: no collision
                collision = False
                for con in self.data.contact:
                    body_id1 = self.model.geom(con.geom1).bodyid
                    assert len(body_id1) == 1
                    body_id1 = body_id1[0]
                    body_id2 = self.model.geom(con.geom2).bodyid
                    assert len(body_id2) == 1
                    body_id2 = body_id2[0]
                    if body_id1 in self.peg_ids[:i] + [self.board_id] and body_id2 in self.peg_ids[:i] + [self.board_id]:
                        collision = True
                        if debug:
                            print("Constraint 1 violated")
                        break

                # Constraint 2: not too close to other objects
                too_close = False
                if not self.grid_arrangement:
                    for body_name2 in self.peg_names:
                        if body_name2 == body_name:
                            continue
                        if np.linalg.norm(curr_body_pos - self.get_body_pos(body_name2)) < 0.1:
                            too_close = True
                            if debug:
                                print("Constraint 2 violated")
                            break

                # Constraint 3: object grasp pose is reachable
                reachable = True
                for name in self.peg_names[:i+1]:
                    pos = self.get_body_pos(name)
                    dis_to_base = np.linalg.norm(pos - self.robot_base_pos)
                    if not (0.3 <= dis_to_base <= 0.74):
                        reachable = False
                        if debug:
                            print("Constraint 3 violated")
                        break

                if not collision and not too_close and reachable:
                    is_valid = True
                    break
                if debug:
                    self.render()

            if not is_valid:
                return False

        return True

    def act_pick_up(self, obj_name):
        err = 0
        obj_in_hand = self.get_object_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand != obj_name:
                print(f"Cannot pick up `{obj_name}` since another object is grasped in hand. "
                      f"(`{obj_in_hand}` is in hand)")
                err = -1
            else:
                print(f"Should not pick up `{obj_name}` since it is already in hand.")
                err = -2
            return err
        if "nail" in self.peg_desc[obj_name]:
            self.pickup_nail(obj_name)
        else:
            self.pickup(obj_name)
        return err

    def done_pick_up(self, obj_name):
        self.goto(quat=[1, 0, 0, 0], pos_err_th=0.05, quat_err_th=0.01)
        return self.object_is_in_hand(obj_name) and \
            np.abs(self.eef_pos[-1] - 0.8) < 0.05 and \
            self.quat_err(self.eef_quat, np.array([1, 0, 0, 0])) < 0.01

    def act_put_down(self, obj_name):
        err = 0
        if self.object_is_in_hand(obj_name):
            self.put_on_table(obj_name)
        else:
            print(f"Cannot put down `{obj_name}` since it is not grasped in hand.")
            err = -1
        return err

    def done_put_down(self, obj_name):
        return not self.object_is_in_hand(obj_name) and \
            self.pos_err(self.eef_pos, self.hand_init_pos) < 0.002 and \
            self.quat_err(self.eef_quat, self.hand_init_quat) < 0.002

    def act_insert(self, obj_name):
        err = 0
        if self.object_is_in_hand(obj_name):
            self.insert(obj_name, f"{obj_name}_hole")
        else:
            print(f"Cannot insert object `{obj_name}` since it is not in hand.")
            err = -1
        return err

    def done_insert(self, obj_name):
        return self.pos_err(self.eef_pos, self.hand_init_pos) < 0.002 and \
            self.quat_err(self.eef_quat, self.hand_init_quat) < 0.002

    def act_reorient(self, obj_name):
        err = 0
        if self.object_is_in_hand(obj_name):
            self.reorient(obj_name)
        else:
            print(f"Cannot reorient object `{obj_name}` since it is not in hand.")
            err = -1
        return err

    def done_reorient(self, obj_name):
        return self.object_is_in_hand(obj_name) and self.object_is_upright(obj_name) and \
            np.abs(self.eef_pos[-1] - 1.0) < 0.05 and self.quat_err(self.eef_quat, np.array([1, 0, 0, 0])) < 0.01

    def act_txt(self, text):
        text_split = text.split()
        act = " ".join(text_split[:max(1, len(text_split) - 1)])
        obj = text_split[-1]

        peg_ind = self.peg_colors.index(obj)
        body_name = self.peg_names[peg_ind]

        err = 0
        if self.control_mode == "script":
            if act == "pick up":
                err = self.act_pick_up(body_name)
            elif act == "put down":
                err = self.act_put_down(body_name)
            elif act == "reorient":
                err = self.act_reorient(body_name)
            elif act == "insert":
                err = self.act_insert(body_name)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

        return err

    def is_success(self, pos_err_th=0.005, quat_err_th=0.02, debug=False):
        for i, body_name in enumerate(self.peg_names):
            if not self.object_is_success(body_name, pos_err_th, quat_err_th, debug=debug):
                if debug:
                    print(f"{body_name} ({self.peg_colors[i]}) is not correctly placed.")
                return False
        return True

    def object_is_success(self, body_name, pos_err_th=0.005, quat_err_th=0.02, debug=False):
        src_pos = self._get_site_pos(f"{body_name}_align")
        dst_pos = self._get_site_pos(f"{body_name}_hole_align")
        src_quat = self._get_site_quat(f"{body_name}_align")
        dst_quat = self._get_site_quat(f"{body_name}_hole_align")
        pos_err = self.pos_err(src_pos, dst_pos)
        quat_err = self.quat_err(src_quat, dst_quat)
        if debug:
            print(body_name, "pos err:", pos_err, "quat err", quat_err)
        if self._is_nail(body_name):
            return pos_err <= pos_err_th and self.object_is_upright(body_name)
        else:
            return pos_err <= pos_err_th and quat_err <= quat_err_th

    def _is_nail(self, obj_name):
        return "nail" in self.peg_desc[obj_name]


from enum import Enum
class State(Enum):
    DONE = 1        # brick: is properly inserted into board
                    # global: all assembled
    READY = 2       # brick: is not inserted yet but ready to be manipulated. (Note: this only ensures that
                    # the bricks that should be inserted BEFORE this one are inserted already. It's still
                    # possible that this brick cannot be inserted right now because some brick that should
                    # be inserted AFTER this one is inserted first and causes blocking, in which case
                    # that brick should have a BAD state and should be removed first.)
                    # global: some brick should be READY or BAD_D
    BAD_B = 3       # brick: is in BAD state since it's Blocking other bricks, need to be removed
                    # global: need to reset some brick(s) to proceed
    BAD_D = 4       # brick: is in BAD state since it's Down, need to be reoriented
    BLOCKED_P = 5   # brick: is BLOCKED since some Predecessor brick(s) should be inserted before
    BLOCKED_S = 6   # brick: is BLOCKED since some Successor brick(s) is inserted before

    @property
    def description(self):
        return {
            self.DONE: "DONE",
            self.READY: "READY",
            self.BAD_B: "BAD (blocking other bricks)",
            self.BAD_D: "BAD (is down)",
            self.BLOCKED_P: "BLOCKED (by predecessor)",
            self.BLOCKED_S: "BLOCKED (by successor)"
        }[self]


class AssemblyOracle(object):

    def __init__(self, brick_ids: list, brick_descriptions: list, dependencies: dict, env: FrankaAssemblyEnv = None):
        super(AssemblyOracle, self).__init__()
        self.brick_ids = brick_ids.copy()

        self.state = {ind: State.READY for ind in brick_ids}
        self._obj_in_hand = None
        self._obj_is_upright = {ind: False for ind in brick_ids}

        self.to_neighbors = {ind: [] for ind in brick_ids}
        self.from_neighbors = {ind: [] for ind in brick_ids}
        self.in_deg = {ind: 0 for ind in brick_ids}
        for (u, v) in dependencies:
            self.to_neighbors[u].append(v)
            self.from_neighbors[v].append(u)
            self.in_deg[v] += 1
        for ind in self.brick_ids:
            if self.in_deg[ind] > 0:
                self.state[ind] = State.BLOCKED_P
        self.brick_descriptions = brick_descriptions.copy()
        self.env = env

    def update_state_from_env(self):
        assert self.env is not None
        for ind in self.brick_ids:
            brick_state = self._check_brick_state(ind, symbolic=False)
            self._update_state(ind, brick_state)
            self._obj_is_upright[ind] = self._check_obj_upright(ind, symbolic=False)
        self._obj_in_hand = self.env.get_object_in_hand()

    def _check_brick_state(self, ind, symbolic=False):
        if self._check_obj_done(ind, symbolic=symbolic):
            if self.in_deg[ind] == 0:
                return State.DONE  # is properly placed and not blocking other bricks
            else:
                return State.BAD_B  # is blocking other bricks
        else:
            if self._check_obj_inboard(ind, symbolic=symbolic):  # is in board area
                return State.BAD_B
            if not self._check_obj_upright(ind, symbolic=symbolic):
                return State.BAD_D
            if self.in_deg[ind] == 0:
                if self._has_successor_in_board(ind, symbolic=symbolic):
                    return State.BLOCKED_S
                return State.READY
            else:
                return State.BLOCKED_P

    def _check_obj_inboard(self, ind, symbolic=False):
        if symbolic:
            return self.state[ind] in {State.DONE, State.BAD_B} and self._obj_in_hand != f"brick_{ind}"
        body_name = f"brick_{ind}"
        return self.env.get_body_pos(body_name)[1] > 0 and not self.env.object_is_in_hand(body_name)

    def _check_obj_done(self, ind, symbolic=False):
        if symbolic:
            return self.state[ind] == State.DONE
        return self.env.object_is_success(f"brick_{ind}")

    def _check_obj_upright(self, ind, symbolic=False):
        if symbolic:
            return self._obj_is_upright[ind]
        return self.env.object_is_upright(f"brick_{ind}")

    def _has_successor_in_board(self, ind, symbolic=False):
        # check if any successor is blocking `body_name`
        for v in self.to_neighbors[ind]:
            if self._check_obj_inboard(v, symbolic=symbolic):
                return True
            if self._has_successor_in_board(v, symbolic=symbolic):
                return True
        return False
    
    def _has_direct_successor_in_board(self, ind, symbolic=False):
        for v in self.to_neighbors[ind]:
            if self._check_obj_inboard(v, symbolic=symbolic):
                return True
        return False

    def _all_predecessors_done(self, ind, symbolic=False):
        for v in self.from_neighbors[ind]:
            if not self._check_obj_done(v, symbolic=symbolic):
                return False
            if not self._all_predecessors_done(v, symbolic=symbolic):
                return False
        return True

    def _update_state(self, ind, new_state, symbolic=False):
        # update the state of brick_{ind} and propagate to others
        if self.state[ind] == new_state:
            return  # do not trigger propagation
        old_state = self.state[ind]
        self.state[ind] = new_state
        if new_state == State.DONE:
            for v in self.to_neighbors[ind]:
                self.in_deg[v] -= 1
                if self.in_deg[v] == 0:
                    if self.state[v] == State.BLOCKED_P and self._all_predecessors_done(v, symbolic=symbolic):
                        self.state[v] = State.READY
        elif old_state == State.DONE:
            for v in self.to_neighbors[ind]:
                self.in_deg[v] += 1
                if self.in_deg[v] == 1:
                    if self.state[v] == State.READY or self.state[v] == State.BLOCKED_S:
                        self.state[v] = State.BLOCKED_P
                    elif self.state[v] == State.DONE:
                        self._update_state(v, State.BAD_B, symbolic=symbolic)

    def get_action(self, symbolic=False):
        if symbolic:
            actions = self.get_plan(max_steps=1)
            if len(actions) == 0:
                return "done"
            return actions[0]
        if self._obj_in_hand is not None:
            obj_id = int(self._obj_in_hand.split("_")[-1])
            obj_desc = self.brick_descriptions[obj_id - min(self.brick_ids)]
            print(f"`{self._obj_in_hand}`({obj_desc}) is in hand.")
            if self._all_predecessors_done(obj_id):
                if self.state[obj_id] == State.BAD_D:
                    return f"reorient the {obj_desc}"
                elif self.state[obj_id] == State.BLOCKED_S:
                    return f"put down the {obj_desc}"
                else:
                    return f"insert the {obj_desc}"
            else:
                return f"put down the {obj_desc}"
        else:
            if self.global_state == State.READY:
                # pick one feasible brick
                for i, ind in enumerate(self.brick_ids):
                    if self.state[ind] in {State.READY, State.BAD_D}:
                        return f"pick up the {self.brick_descriptions[i]}"
            elif self.global_state == State.BAD_B:
                for (ind, desc) in zip(reversed(self.brick_ids), reversed(self.brick_descriptions)):
                    if self.state[ind] == State.BAD_B:
                        return f"pick up the {desc}"
            elif self.global_state == State.DONE:
                return "done"
            # should not reach here
            raise RuntimeError(f"Bad global status: {self.state2name(self.global_state)}")
    
    def get_all_feasible_actions(self):
        if self._obj_in_hand is not None:
            obj_id = int(self._obj_in_hand.split("_")[-1])
            obj_desc = self.brick_descriptions[obj_id - min(self.brick_ids)]
            print(f"`{self._obj_in_hand}`({obj_desc}) is in hand.")
            if self._all_predecessors_done(obj_id):
                if self.state[obj_id] == State.BAD_D:
                    return [f"reorient the {obj_desc}"]
                elif self.state[obj_id] == State.BLOCKED_S:
                    return [f"put down the {obj_desc}"]
                else:
                    return [f"insert the {obj_desc}"]
            else:
                return [f"put down the {obj_desc}"]
        else:
            if self.global_state == State.READY:
                return [
                    f"pick up the {self.brick_descriptions[i]}" 
                    for i, ind in enumerate(self.brick_ids) 
                    if self.state[ind] in {State.READY, State.BAD_D}
                ]
            elif self.global_state == State.BAD_B:
                actions = []
                for (ind, desc) in zip(reversed(self.brick_ids), reversed(self.brick_descriptions)):
                    if self.state[ind] == State.BAD_B and not self._has_direct_successor_in_board(ind):
                        actions.append(f"pick up the {desc}")
                return actions
            elif self.global_state == State.DONE:
                return ["done"]
            # should not reach here
            raise RuntimeError(f"Bad global status: {self.state2name(self.global_state)}")

    def get_plan(self, max_steps: int = None):
        if max_steps is not None:
            assert max_steps > 0

        actions = []
        n_iters = 0
        while self.global_state != State.DONE:
            n_iters += 1
            if n_iters > 100:
                print("Infinite loop in getting plan")
                return None
            if self._obj_in_hand is not None:
                obj_id = int(self._obj_in_hand.split("_")[-1])
                obj_desc = self.brick_descriptions[obj_id - min(self.brick_ids)]
                if self._all_predecessors_done(obj_id, symbolic=True):
                    if self.state[obj_id] == State.BAD_D:
                        actions.append(f"reorient the {obj_desc}")
                        self._obj_is_upright[obj_id] = True
                        self._update_state(obj_id, State.READY, symbolic=True)
                    elif self.state[obj_id] == State.BLOCKED_S:
                        actions.append(f"put down the {obj_desc}")
                        self._obj_in_hand = None
                    else:
                        actions.append(f"insert the {obj_desc}")
                        self._obj_in_hand = None
                        self._update_state(obj_id, State.DONE, symbolic=True)
                else:
                    actions.append(f"put down the {obj_desc}")
                    self._obj_in_hand = None
                    self._update_state(obj_id, State.BLOCKED_P, symbolic=True)
            else:
                if self.global_state == State.READY:
                    # pick one feasible brick
                    for i, ind in enumerate(self.brick_ids):
                        if self.state[ind] in {State.READY, State.BAD_D}:
                            actions.append(f"pick up the {self.brick_descriptions[i]}")
                            self._obj_in_hand = f"brick_{ind}"
                            break
                elif self.global_state == State.BAD_B:
                    for (ind, desc) in zip(reversed(self.brick_ids), reversed(self.brick_descriptions)):
                        if self.state[ind] == State.BAD_B:
                            actions.append(f"pick up the {desc}")
                            self._obj_in_hand = f"brick_{ind}"
                            break

            if max_steps is not None and len(actions) >= max_steps:
                return actions[:max_steps]

            for ind in self.brick_ids:
                brick_state = self._check_brick_state(ind, symbolic=True)
                self._update_state(ind, brick_state)

        return actions

    def state2name(self, state):
        return State(state).description

    def get_states(self):
        states = {}
        for i, ind in enumerate(self.brick_ids):
            states[self.brick_descriptions[i]] = self.state2name(self.state[ind])
        return states

    @property
    def global_state(self):
        done_cnt = 0
        for k, v in self.state.items():
            if v == State.BAD_B:
                return State.BAD_B
            done_cnt += (v == State.DONE)
        if done_cnt == len(self.brick_ids):
            return State.DONE
        return State.READY

    @property
    def global_state_value(self):
        done_cnt = 0
        for k, v in self.state.items():
            done_cnt += (v == State.DONE)
        return done_cnt / len(self.brick_ids)

    def get_oracle_state(self):
        state = {
            'state': self.state.copy(),
            'in_deg': self.in_deg.copy()
        }
        return state

    def set_oracle_state(self, state):
        self.state = state['state'].copy()
        self.in_deg = state['in_deg'].copy()
