import abc
import random
import warnings
import glfw
from gym.utils import seeding
import numpy as np
from os import path
import gym
import mujoco

from roboworld.envs.mujoco_env.utils import rotation as R


class MujocoEnv(gym.Env, abc.ABC):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    max_path_length = 100000

    def __init__(self, model_path, frame_skip, render_mode='offscreen', render_width=640, render_height=480):
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.model_path = model_path
        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        if self.render_mode == 'offscreen':
            self.model.vis.global_.offwidth = render_width
            self.model.vis.global_.offheight = render_height

        self.renderer = mujoco.Renderer(self.model, width=render_width, height=render_height)   # for offscreen rendering
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        if mujoco.mj_version() >= 300:  # mujoco 3
            self._init_eq_active = self.model.eq_active0.copy()
        else:
            self._init_eq_active = self.model.eq_active.copy()

        self._did_see_sim_exception = False

        self.np_random, _ = seeding.np_random(None)
        self.frames = []
        self._record = False
        self._record_frame_skip = 5
        self._record_skipped_frames = 0

    def seed(self, seed):
        assert seed is not None
        random.seed(seed)
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        self.viewer.cam.azimuth = 0       # Any value is fine since we're looking from the top
        self.viewer.cam.elevation = -90   # -90 means looking straight down
        self.viewer.cam.distance = 1.0    # Adjust as needed for zoom level
        self.viewer.cam.lookat = [0.0, 0.0, 0.65]

        # self.viewer.cam.azimuth = 45        # View at an angle from the diagonal
        # self.viewer.cam.elevation = -60     # Slightly tilted, not directly top-down
        # self.viewer.cam.distance = 1.5      # Keep zoom level as is, or tweak if needed
        # self.viewer.cam.lookat = [0.0, 0.0, 0.65]  # Focus point stays the same

    def reset(self, seed=None):
        self._did_see_sim_exception = False
        mujoco.mj_resetData(self.model, self.data)
        if mujoco.mj_version() >= 300:  # mujoco 3
            self.model.eq_active0 = self._init_eq_active.copy()
            self.data.eq_active = self._init_eq_active.copy()
        else:
            self.model.eq_active = self._init_eq_active.copy()
        if seed is not None:
            self.seed(seed)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        self.frames = []
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

        self.data.qpos = qpos.copy()
        self.data.qvel = qvel.copy()
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        self.max_path_length = 100000000000
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise RuntimeError(f'Maximum path length ({self.max_path_length}) exceeded')
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip
        self.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            try:
                mujoco.mj_step(self.model, self.data)
            except Exception as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def read_pixels(self, camera_name):
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def render(self, offscreen=False, camera_name="table_back", resolution=(640, 480)):

        if self.render_mode == 'offscreen':
            offscreen = True
        if offscreen:
            if self._record:
                img = self.read_pixels(camera_name=camera_name)
                self._record_img(img.copy())
                return img
            else:
                return None
        else:
            img = self._get_viewer('window').render()
            if self._record:
                self._record_img(img.copy())
            return img

    def record_on(self, record_frame_skip=5):
        self.frames = []
        self._record = True
        self._record_frame_skip = record_frame_skip
        self._record_skipped_frames = 0

    def record_off(self):
        self._record = False
    
    def _record_img(self, img):
        if self._record_skipped_frames == 0:
            self.frames.append(img.copy())
        self._record_skipped_frames += 1
        if self._record_skipped_frames == self._record_frame_skip:
            self._record_skipped_frames = 0

    @property
    def is_recording(self):
        return self._record

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def _get_viewer(self, mode):
        assert mode in {'window', 'offscreen'}
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'window':
                import mujoco_viewer
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, mode='window')
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def get_body_pos(self, body_name):
        return self.data.body(body_name).xpos.copy()

    def get_body_quat(self, body_name):
        return self.data.body(body_name).xquat.copy()

    def set_body_pos(self, body_name, pos):
        body_id = self.model.body(body_name).id
        assert self.model.body_dofnum[body_id] == 6
        qpos_start_idx = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
        self.data.qpos[qpos_start_idx: qpos_start_idx + 3] = pos.copy()
        mujoco.mj_forward(self.model, self.data)

    def set_body_quat(self, body_name, quat):
        body_id = self.model.body(body_name).id
        assert self.model.body_dofnum[body_id] == 6
        qpos_start_idx = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
        self.data.qpos[qpos_start_idx + 3: qpos_start_idx + 7] = quat.copy()
        mujoco.mj_forward(self.model, self.data)

    def _get_site_pos(self, site_name):
        return self.data.site(site_name).xpos.copy()

    def _get_site_quat(self, site_name):
        site_mat = self.data.site(site_name).xmat.copy().reshape(3, 3)
        return R.mat2quat(site_mat)

    def _set_site_pos(self, name, pos):
        """Sets the position of the site corresponding to `name`

        Args:
            name (str): The site's name
            pos (np.ndarray): Flat, 3 element array indicating site's location
        """
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1

        self.data.site_xpos[self.model.site(name).id] = pos[:3]

    def set_eq_active(self, eq_name, active=True):
        eq_id = self.model.equality(eq_name).id
        if mujoco.mj_version() >= 300:  # mujoco 3
            self.data.eq_active[eq_id] = active
        else:
            self.model.eq_active[eq_id] = active

        mujoco.mj_forward(self.model, self.data)

    def set_weld_data(self, eq_name, data=None, relpos=None, relquat=None):
        eq_id = self.model.equality(eq_name).id
        if data is not None:
            self.model.eq_data[eq_id, :] = data
            return
        if relpos is not None:
            self.model.eq_data[eq_id, 3:6] = relpos
        if relquat is not None:
            self.model.eq_data[eq_id, 6:10] = relquat

    def get_geom_ids(self, body_name):
        geom_adr = self.model.body(body_name).geomadr[0]
        geom_num = self.model.body(body_name).geomnum[0]
        assert geom_adr != -1
        geom_ids = [geom_adr + i for i in range(geom_num)]
        return geom_ids

    def check_collision(self, geom_ids1, geom_ids2, require_nonzero_force=False):
        """
        Check if there's collision between geoms in sets `geom_ids1` and `geom_ids2`
        """
        contacts = [
            c for c in self.data.contact
            if ((c.geom1 in geom_ids1 and c.geom2 in geom_ids2)
                or (c.geom1 in geom_ids2 and c.geom2 in geom_ids1))
        ]
        if not require_nonzero_force:
            return len(contacts) > 0

        contacts_with_nonzero_force = [
            c for c in contacts if self.data.efc_force[c.efc_address] > 0
        ]
        return len(contacts_with_nonzero_force) > 0
