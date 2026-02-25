from __future__ import annotations
from typing import Dict, List, Optional
from numpy.typing import ArrayLike
from xml.etree.ElementTree import Element
import numpy as np
import random
import glob
import pathlib

from roboworld.envs.xml_utils import set_attributes, create_element, XmlMaker
from roboworld.envs.mujoco_env.utils.rotation import euler2quat

ASSETS_DIR = pathlib.Path(__file__).parent.resolve() / "assets"

COLORS = {
    "blue": (0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
    "orange": (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
    "green": (0.41568627450980394, 0.8, 0.39215686274509803),
    "red": (0.8392156862745098, 0.37254901960784315, 0.37254901960784315),
    "purple": (0.5843137254901961, 0.4235294117647059, 0.7058823529411765),
    "brown": (0.5490196078431373, 0.3803921568627451, 0.23529411764705882),
    "pink": (0.8627450980392157, 0.49411764705882355, 0.7529411764705882),
    "gray": (0.4745098039215686, 0.4745098039215686, 0.4745098039215686),
    "yellow": (0.9098, 0.6784, 0.1373)
}
SHAPES = ["arch", "circle", "cross", "flower", "heart", "hexagon", "moon", "oval", "parallelogram",
          "pentagon", "square", "star", "trapezoid", "trefoil", "triangle"]
VOXEL_SIZE = 0.01


def get_color_name(rgb: ArrayLike) -> str:
    """
    Get the color name from the given RGB value
    """
    for k, v in COLORS.items():
        if np.all(np.abs(np.array(rgb) - np.array(v)) < 1e-3):
            return k
    return ""


def slice3d(start: ArrayLike, end: ArrayLike) -> tuple[slice, ...]:
    """
    Create a 3D slice from start and end indices
    """
    s = []
    for (l, r) in zip(start, end):
        assert l % 1 == 0 and r % 1 == 0
        s.append(slice(int(l), int(r)))
    return tuple(s)


def random_color() -> tuple:
    """
    Sample a random color (RGB)
    """
    idx = np.random.choice(len(COLORS))
    color_name = list(COLORS)[idx]
    return COLORS[color_name]


def random_rgba(a: Optional[float] = None) -> tuple:
    """
    sample a random RGBA value
    """
    return (random_color()) + (np.random.rand() if a is None else a,)


def random_size(x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None):
    """
    Sample a random size (x, y, z) for a cuboid brick
    """
    swap_xy = x is None and y is None and np.random.random() < 0.5
    if x is None:
        x = np.random.randint(1, 8)
    if y is None:
        y = np.random.randint(1, 8 if x <= 3 else 3)
    if swap_xy:
        x, y = y, x
    if z is None:
        z = np.random.randint(2, 6)
    return x, y, z


class Brick(object):
    """
    Class of a brick, which is represented by a 3D array of voxels
    The value of each voxel is an integer, with 0 representing empty space, and 1 representing occupancy.
    """

    def __init__(self, name: str, size: ArrayLike, offset: Optional[ArrayLike] = None, rgba: Optional[tuple] = None,
                 parent_board: Optional[Board] = None):
        self.name = name
        self._rgba = rgba if rgba is not None else random_rgba(a=1.0)
        self._voxels = np.ones(size, dtype=int)
        self._voxel_size = VOXEL_SIZE
        self.description = ""   # language description of this brick
        self.parent_board = parent_board
        self.offset = np.array(offset) if offset is not None else np.zeros(3)  # voxel offset relative to parent

    def _get_voxel_center(self, idx: ArrayLike):
        """
        Get the center position of a voxel indexed by `idx`
        """
        return (np.array(idx) + 0.5) * self._voxel_size - self.base_center

    def _get_cuboid_center(self, idx: ArrayLike, size: ArrayLike):
        """
        Get the center position of a cuboid indexed by `idx` and with size `size`
        """
        return (np.array(idx) + size / 2) * self._voxel_size - self.base_center

    def get_body(self, pos: Optional[ArrayLike] = None, quat: Optional[ArrayLike] = None,
                 freejoint: bool = True, optimize: bool = True):
        """
        Get the body element of this brick
        """
        body = create_element("body", attributes={"name": self.name})
        aux_assets = []
        if pos is not None:
            set_attributes(body, {"pos": pos})
        if quat is not None:
            set_attributes(body, {"quat": quat})
        if freejoint:
            create_element(tag="freejoint", parent=body)
        # create visual geoms
        visual_geoms, visual_aux_assets = self._get_geoms("visual", optimize=optimize)
        body.extend(visual_geoms)
        aux_assets.extend(visual_aux_assets)
        aux_asset_names = [x.get("name") for x in aux_assets]
        # create collision geoms
        collision_geoms, collision_aux_assets = self._get_geoms("collision", optimize=optimize)
        for collision_geom in collision_geoms:
            set_attributes(collision_geom, attributes={"name": collision_geom.get("name") + "_c", "class": "collision"})
            geom_type = collision_geom.get("type")
            if geom_type in {"box", "cylinder"}:
                original_size = np.array(collision_geom.get("size").split()).astype(float)
                set_attributes(collision_geom, attributes={
                    "size": original_size - np.array([0.001, 0.001, 0]) if collision_geom.get("type") == "box"
                                else original_size - np.array([0.002, 0])
                })
            body.append(collision_geom)
        for collision_aux_asset in collision_aux_assets:
            if collision_aux_asset.get("name") not in aux_asset_names:
                aux_assets.append(collision_aux_asset)

        # create a site for alignment
        create_element(tag="site", parent=body, attributes={
            "name": f"{self.name}_align", "pos": [0, 0, 0], "class": "invisible_site"
        })
        # create sites for grasping
        grasp_poses = self._get_grasp_poses()
        for name, (grasp_pos, grasp_quat) in grasp_poses.items():
            create_element(tag="site", parent=body, attributes={
                "name": name, "pos": grasp_pos, "quat": grasp_quat, "class": "invisible_site"
            })

        # create sites for auxiliary poses
        grasp_pos = list(grasp_poses.values())[0][0]
        poses = self._get_auxiliary_poses(grasp_pos)
        for site_name, (pos, quat) in poses.items():
            attributes = {"name": site_name, "class": "invisible_site"}
            if pos is not None:
                attributes["pos"] = pos
            if quat is not None:
                attributes["quat"] = quat
            create_element(tag="site", parent=body, attributes=attributes)
        return body, aux_assets

    def _get_grasp_poses(self) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Get grasp poses of this brick
        """
        main_axis = self._get_main_axis()
        assert np.any(self._voxels[:, :, -1] > 0)
        candidates = []
        if main_axis == 0:
            for i in range(self.size[main_axis] - 1):
                if np.all(self._voxels[i: i + 2, :, -2:] != 0):
                    candidates.append(i)
        else:  # main_axis == 1
            for i in range(self.size[main_axis] - 1):
                if np.all(self._voxels[:, i: i + 2, -2:] != 0):
                    candidates.append(i)
        mid = (self.size[main_axis] - 1) // 2
        if len(candidates) == 0:
            print(f"Cannot find graspable pose for object `{self.name}`")
            pos_main_axis = self.size[main_axis] * self._voxel_size / 2
        else:
            candidates.sort(key=lambda x: abs(x - mid))
            best_idx = candidates[0]
            pos_main_axis = (best_idx + 1) * self._voxel_size
        pos = [pos_main_axis, self.base_center[1 - main_axis], (self.size[-1] - 0.5) * self._voxel_size]
        if main_axis == 1:
            pos[0], pos[1] = pos[1], pos[0]
        pos = np.array(pos) - self.base_center
        euler_zs = [0., np.pi] if main_axis == 1 else [-np.pi / 2, np.pi / 2]
        euler_candidates = [np.array([0, 0, z]) for z in euler_zs]

        grasp_poses = {f"{self.name}_grasp{i}": (pos, euler2quat(euler)) for i, euler in enumerate(euler_candidates)}

        return grasp_poses

    def _get_auxiliary_poses(self, grasp_pos: np.ndarray) -> dict[str, tuple[np.ndarray | None, np.ndarray | None]]:
        """
        Get the auxiliary poses of this brick
        """
        main_axis = self._get_main_axis()

        height = self.size[-1] * self._voxel_size
        width = self.size[1 - main_axis] * self._voxel_size
        waist_idx_along_main_axis = round((grasp_pos[main_axis] + self.base_center[main_axis]) / self._voxel_size)
        thickness = np.sum(self.voxels[waist_idx_along_main_axis, 0, :] != 0) * self._voxel_size if main_axis == 0 \
            else np.sum(self.voxels[0, waist_idx_along_main_axis, :] != 0) * self._voxel_size
        side0_base_pos = np.array([0, -width / 2, height / 2])
        side1_base_pos = np.array([0, width / 2, height / 2])
        side0_grasp_pos = np.array([grasp_pos[main_axis], width / 2 - 0.5 * self._voxel_size, height - thickness / 2])
        side1_grasp_pos = np.array([grasp_pos[main_axis], -width / 2 + 0.5 * self._voxel_size, height - thickness / 2])
        if main_axis == 1:
            for p in [side0_base_pos, side1_base_pos, side0_grasp_pos, side1_grasp_pos]:
                p[0], p[1] = p[1], p[0]

        if main_axis == 0:
            side0_base_quat = euler2quat([-np.pi / 2, 0, 0])
            side1_base_quat = euler2quat([np.pi / 2, 0, 0])
            side0_grasp_quat0 = euler2quat([-np.pi / 2, 0, np.pi / 2])
            side0_grasp_quat1 = euler2quat([-np.pi / 2, 0, -np.pi / 2])
            side1_grasp_quat0 = euler2quat([np.pi / 2, 0, np.pi / 2])
            side1_grasp_quat1 = euler2quat([np.pi / 2, 0, -np.pi / 2])
        else:
            side0_base_quat = euler2quat([0, np.pi / 2, 0])
            side1_base_quat = euler2quat([0, -np.pi / 2, 0])
            side0_grasp_quat0 = euler2quat([0, np.pi / 2, 0])
            side0_grasp_quat1 = euler2quat([0, np.pi / 2, np.pi])
            side1_grasp_quat0 = euler2quat([0, -np.pi / 2, 0])
            side1_grasp_quat1 = euler2quat([0, -np.pi / 2, np.pi])

        return {
            f"{self.name}_side0_base": (side0_base_pos, side0_base_quat),
            f"{self.name}_side1_base": (side1_base_pos, side1_base_quat),
            f"{self.name}_side0_grasp0": (side0_grasp_pos, side0_grasp_quat0),
            f"{self.name}_side0_grasp1": (side0_grasp_pos, side0_grasp_quat1),
            f"{self.name}_side1_grasp0": (side1_grasp_pos, side1_grasp_quat0),
            f"{self.name}_side1_grasp1": (side1_grasp_pos, side1_grasp_quat1),
        }

    def _get_main_axis(self):
        """
        Get the main axis of this brick. 0 for x-axis, 1 for y-axis
        """
        main_axis = 1 if self.size[0] <= 5 else 0
        return main_axis

    def _get_geoms(self, geom_class: str, optimize: bool = True) -> tuple[List[Element], List[Element]]:
        """
        Get the visual or collision geoms of this brick
        :param geom_class: "visual" or "collision"
        :param optimize: whether to optimize the geoms
        :return: lists of geoms and lists of auxiliary assets
        """
        if optimize:
            return self._get_optimized_geoms(geom_class)
        geoms, aux_assets = [], []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                for k in range(self.size[2]):
                    if self._voxels[i, j, k] != 0:
                        geom = create_element(tag="geom", attributes={
                            "name": f"{self.name}_g{len(geoms)}_{get_color_name(self._rgba[:3])}",
                            "pos": self._get_voxel_center((i, j, k)),
                            "type": "box",
                            "size": np.array([self._voxel_size / 2] * 3),
                            "rgba": np.array(self._rgba),
                            "class": geom_class
                        })
                        geoms.append(geom)
        return geoms, aux_assets

    def _cuboid2geom(self, idx: ArrayLike, size: ArrayLike, val: int, geom_class: str, gidx: int = -1):
        """
        Convert a cuboid to geoms
        :param idx: index (pos) of the cuboid
        :param size: size of the cuboid
        :param val: value of the cuboid in the voxel array
        :param geom_class: "visual" or "collision"
        :param gidx: geom index
        :return: lists of geoms and lists of auxiliary assets
        """
        assert val != 0
        geoms, aux_assets = [], []
        if val > 0:
            geom = create_element(tag="geom", attributes={
                "name": f"{self.name}_g{gidx}_{get_color_name(self._rgba[:3])}",
                "pos": self._get_cuboid_center(idx, size),
                "type": "box",
                "size": np.array(size) * self._voxel_size / 2,
                "rgba": np.array(self._rgba),
                "class": geom_class
            })
            geoms.append(geom)
        else:
            # hole
            if self.parent_board is not None and val in self.parent_board.hole_shapes:
                shape = self.parent_board.hole_shapes[val]
                if geom_class == "collision":
                    for p in glob.glob(f"{ASSETS_DIR}/objects/pegs_and_holes/{shape}_hole/*.obj"):
                        geom = create_element(tag="geom", attributes={
                            "name": f"{self.name}_g{gidx}_{len(geoms)}_{get_color_name(self._rgba[:3])}",
                            "pos": self._get_cuboid_center(idx, size),
                            "type": "mesh",
                            "mesh": f"{self.name}_{shape}_hole_g{gidx}_{len(geoms)}",
                            "rgba": np.array(self._rgba),
                            "class": geom_class
                        })
                        aux_asset = create_element("mesh", attributes=dict(
                            name=f"{self.name}_{shape}_hole_g{gidx}_{len(geoms)}",
                            file=f"./objects/pegs_and_holes/{shape}_hole/{p.split('/')[-1]}",
                            scale=size,
                        ))
                        geoms.append(geom)
                        aux_assets.append(aux_asset)
                else:
                    geom = create_element(tag="geom", attributes={
                        "name": f"{self.name}_g{gidx}_{get_color_name(self._rgba[:3])}",
                        "pos": self._get_cuboid_center(idx, size),
                        "type": "mesh",
                        "mesh": f"{self.name}_{shape}_hole_g{gidx}",
                        "rgba": np.array(self._rgba),
                        "class": geom_class
                    })
                    aux_asset = create_element("mesh", attributes=dict(
                        name=f"{self.name}_{shape}_hole_g{gidx}",
                        file=f"./objects/pegs_and_holes/{shape}_hole.stl",
                        scale=size,
                    ))
                    geoms.append(geom)
                    aux_assets.append(aux_asset)

            else:
                # nail
                if geom_class == "visual":
                    geom = create_element(tag="geom", attributes={
                        "name": f"{self.name}_g{gidx}_{get_color_name(self._rgba[:3])}",
                        "pos": self._get_cuboid_center(idx, size),
                        "type": "mesh",
                        "mesh": f"{self.name}_hole_{val}",
                        "rgba": np.array(self._rgba),
                        "class": geom_class
                    })
                    aux_asset = create_element("mesh", attributes=dict(
                        name=f"{self.name}_hole_{val}",
                        file="./objects/circle_hole/circle_hole.stl",
                        scale=size,
                    ))
                    geoms.append(geom)
                    aux_assets.append(aux_asset)
                else:
                    pass    # no collision shape for nailhole
        return geoms, aux_assets

    @staticmethod
    def _find_cuboid_for_voxel(voxels: np.array, idx: ArrayLike):
        """
        Find the maximal cuboid located at `idx` in the `voxels` array
        """
        val = voxels[idx]
        size, maximized = np.ones(3, dtype=int), np.zeros(3, dtype=bool)
        expanding_dim = 0
        size[0] += 1
        while not np.all(maximized):
            if idx[expanding_dim] + size[expanding_dim] <= voxels.shape[expanding_dim] \
                    and np.all(voxels[slice3d(idx, np.array(idx) + size)] == val):
                # decide the dim to expand
                for i in range(3):
                    if maximized[i]:
                        continue
                    size[i] += 1
                    expanding_dim = i
                    break
            else:
                maximized[expanding_dim] = True
                size[expanding_dim] -= 1
        assert np.all(size > 0)
        return size

    def _get_optimized_geoms(self, geom_class: str) -> tuple[List[Element], List[Element]]:
        """
        Get the visual or collision geoms of this brick with optimization to reduce the number of geoms
        """
        voxels = self._voxels.copy()
        cuboids = []

        for i in range(voxels.shape[0]):
            for j in range(voxels.shape[1]):
                for k in range(voxels.shape[2]):
                    idx = (i, j, k)
                    if voxels[idx] != 0:
                        size = self._find_cuboid_for_voxel(voxels, idx)
                        cuboids.append((idx, size, voxels[idx]))
                        voxels[slice3d(idx, np.array(idx) + size)] = 0

        geoms, aux_assets = [], []
        for gidx, (cuboid_idx, cuboid_size, cuboid_val) in enumerate(cuboids):
            _geoms, _aux_assets = self._cuboid2geom(cuboid_idx, cuboid_size, cuboid_val, geom_class=geom_class, gidx=gidx)
            geoms.extend(_geoms)
            aux_assets.extend(_aux_assets)
        return geoms, aux_assets

    @property
    def rgba(self):
        return self._rgba

    @rgba.setter
    def rgba(self, rgba):
        self._rgba = np.array(rgba).copy()

    @property
    def voxels(self):
        return self._voxels

    @voxels.setter
    def voxels(self, voxels):
        self._voxels = voxels

    @property
    def base_center(self):
        x = self._voxel_size * self.size[0] / 2
        y = self._voxel_size * self.size[1] / 2
        return np.array([x, y, 0])

    @property
    def size(self):
        return self._voxels.shape


class Nail(Brick):
    """
    Class of a nail, overriding the `Brick` class
    """

    def __init__(self, name, size, shank_length, offset=None, rgba=None, parent_board=None):
        super().__init__(name, size, offset=offset, rgba=rgba, parent_board=parent_board)
        self.shank_length = shank_length
        self.voxels[:, :, :self.shank_length] = 0
        self.voxels[1:-1, 1:-1, :self.shank_length] = -1

    def _get_geoms(self, geom_class: str, optimize: bool = True) -> tuple[List[Element], List[Element]]:
        color_name = get_color_name(self._rgba[:3])
        geom_shank = create_element(tag="geom", attributes={
            "name": f"{self.name}_g0_{color_name}",
            "pos": self.shank_center,
            "type": "cylinder",
            "size": np.array([(self.size[0] - 2) * self._voxel_size / 2, self.shank_length * self._voxel_size / 2]),
            "rgba": np.array(self._rgba),
            "class": geom_class
        })
        geom_head = create_element(tag="geom", attributes={
            "name": f"{self.name}_g1_{color_name}",
            "pos": self.head_center,
            "type": "mesh",
            "mesh": f"{self.name}_nail_head",
            "rgba": np.array(self._rgba),
            "class": geom_class
        })
        aux_assets = [create_element("mesh", attributes=dict(
            name=f"{self.name}_nail_head",
            file="./objects/hexagon_prism/hexagon_prism.stl",
            scale=np.array([self.size[0], self.size[1], (self.size[-1] - self.shank_length)]),
        ))]
        return [geom_shank, geom_head], aux_assets

    def _get_grasp_poses(self) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        pos = np.zeros(3)
        pos[-1] = (self.size[-1] - 0.5) * self._voxel_size
        face_grasp_euler_candidates = [np.array([0, 0, np.pi / 6 + k * np.pi / 3]) for k in range(6)]
        ridge_grasp_euler_candidates = [np.array([0, 0, k * np.pi / 3]) for k in range(6)]
        grasp_poses = {}
        for i, euler in enumerate(face_grasp_euler_candidates):
            grasp_poses[f"{self.name}_grasp{i}"] = (pos, euler2quat(euler))
        for i, euler in enumerate(ridge_grasp_euler_candidates):
            grasp_poses[f"{self.name}_ridge_grasp{i}"] = (pos, euler2quat(euler))
        return grasp_poses

    @property
    def shank_center(self):
        shank_size = np.array([self.size[0], self.size[1], self.shank_length])
        return shank_size * self._voxel_size / 2 - self.base_center

    @property
    def head_center(self):
        center = self.base_center
        center[-1] = (self.size[-1] + self.shank_length) / 2 * self._voxel_size
        return center - self.base_center


class Peg(Brick):
    """
    Class of a peg, overriding the `Brick` class
    """

    def __init__(self, name, size, shape, offset=None, rgba=None, parent_board=None):
        super().__init__(name, size, offset=offset, rgba=rgba, parent_board=parent_board)
        self.shape = shape
        self.voxels[:, :, :] = -1

    def _get_geoms(self, geom_class, optimize=True) -> tuple[List[Element], List[Element]]:
        color_name = get_color_name(self._rgba[:3])
        geoms, aux_assets = [], []
        for p in glob.glob(f"{ASSETS_DIR}/objects/pegs_and_holes/{self.shape}_peg/*.obj"):
            geom = create_element(tag="geom", attributes={
                "name": f"{self.name}_g{len(geoms)}_{color_name}",
                "pos": self.peg_center,
                "type": "mesh",
                "mesh": f"{self.name}_{self.shape}_peg_g{len(geoms)}",
                "rgba": np.array(self._rgba),
                "class": geom_class
            })
            aux_asset = create_element("mesh", attributes=dict(
                name=f"{self.name}_{self.shape}_peg_g{len(geoms)}",
                file=f"./objects/pegs_and_holes/{self.shape}_peg/{p.split('/')[-1]}",
                scale=self.size,
            ))
            geoms.append(geom)
            aux_assets.append(aux_asset)
        return geoms, aux_assets

    def _get_grasp_poses(self) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        pos = np.zeros(3)
        pos[-1] = (self.size[-1] - 0.5) * self._voxel_size
        quat = np.array([1., 0., 0., 0.])
        # TODO: consider symmetry
        return {f"{self.name}_grasp0": (pos, quat)}

    @property
    def peg_center(self):
        x = self._voxel_size * self.size[0] / 2
        y = self._voxel_size * self.size[1] / 2
        z = self._voxel_size * self.size[2] / 2
        return np.array([x, y, z]) - self.base_center


class Board(object):
    """
    Class of an assembly board.
    An assembly board is represented by a 3D array of voxels, with each voxel representing a unit of space.
    The value of each voxel is an integer, with 0 representing empty space, and positive integers representing
    different bricks. The value of -1 represents a hole.
    """

    def __init__(self, size: ArrayLike):
        self._voxels = np.zeros(size, dtype=int)
        self._voxel_size = VOXEL_SIZE
        self.bricks: List[Brick] = []
        self.hole_shapes = {}
        self.dependencies = set()

    def add_brick(self, brick: Brick, compliant: bool = False) -> None:
        """
        Add a brick to the assembly board
        :param brick: a Brick object to be added
        :param compliant: if the brick is not compliant, it only intersects with the base board.
            Otherwise, it also occupies the voxels intersecting with existing bricks.
        """
        self.bricks.append(brick)
        new_id = self.num_bricks
        if isinstance(brick, Peg):
            self.hole_shapes[-new_id] = brick.shape
        self.update_voxels(brick, new_id, compliant=compliant)

    def get_bodies(self, base_pos, base_quat, optimize=True) -> tuple[List[Element], List[Element]]:
        """
        Get bodies and auxiliary assets for the bricks
        """
        bodies, aux_assets = [], []
        for brick in self.bricks:
            body, aux = brick.get_body(pos=base_pos, quat=base_quat, freejoint=True, optimize=optimize)
            bodies.append(body)
            aux_assets.extend(aux)
        base_body = bodies[0]
        for i, brick in enumerate(self.bricks[1:]):
            create_element(tag="site", parent=base_body, attributes={
                "name": f"brick_{i + 2}_hole_align", "pos": self._get_relative_pos(brick), "class": "invisible_site"
            })
        return bodies, aux_assets

    def get_equalities(self) -> List[Element]:
        """
        Get equality constraints for grasping
            e.g. <weld name="brick_2_grasp_hand" body1="hand" body2="brick_2" active="false"/>
        """
        equalities = []
        for i, brick in enumerate(self.bricks[1:]):
            eq1 = create_element(tag="weld", attributes={
                "name": f"brick_{i + 2}_grasp_hand", "body1": "hand", "body2": f"brick_{i + 2}", "active": "false",
                "solimp": [0.99, 0.999, 0.001], "solref": [0.01, 1]
            })
            eq2 = create_element(tag="weld", attributes={
                "name": f"brick_{i + 2}_on_fixture", "body1": "fixture", "body2": f"brick_{i + 2}", "active": "false",
            })
            equalities.extend([eq1, eq2])

        return equalities

    def _get_relative_pos(self, brick: Brick) -> np.ndarray:
        """
        Get the position of child center relative to parent center
        """
        return brick.base_center + brick.offset * VOXEL_SIZE - self.base_center

    def update_voxels(self, new_brick: Brick, new_id: int, compliant: bool = False) -> None:
        """
        Update the voxel representation of the whole board
        :param new_brick: the new brick to be added
        :param new_id: the id of the new brick
        :param compliant: if the brick is compliant, it only intersects with the base board.
        """
        idxs = slice3d(new_brick.offset, np.array(new_brick.offset) + np.array(new_brick.voxels.shape))
        critical_brick_ids = self._voxels[idxs][(self._voxels[idxs] != 0) & (new_brick.voxels != 0)]
        critical_brick_ids = list(set(critical_brick_ids))
        for i in critical_brick_ids:
            if i != 1:  # exclude base board
                self.dependencies.add((i, new_id))  # brick_{i} should be manipulated before brick_{new_id}

        if compliant:
            # a compliant brick should only intersect with the base brick
            intersect_voxels = (
                (self._voxels[idxs] > 1) & (new_brick.voxels != 0)
            )   # an array of 0/1, with 1 representing intersection
            # update voxels of the new brick
            new_brick.voxels[intersect_voxels != 0] = 0
            # update voxel representation of the whole board
            self._voxels[idxs][new_brick.voxels != 0] = 0
            self._voxels[idxs] += new_brick.voxels * new_id
            # update voxels of the base brick
            base_brick = self.bricks[0]

            _idxs = slice3d(base_brick.offset, np.array(base_brick.offset) + np.array(
                base_brick.voxels.shape))
            _voxels = self._voxels[_idxs]
            base_brick.voxels[_voxels > 1] = 0
            _idxs_hole = (base_brick.voxels != 0) & (_voxels < 0)
            base_brick.voxels[_idxs_hole] = _voxels[_idxs_hole]

        else:
            self._voxels[idxs][new_brick.voxels != 0] = 0
            self._voxels[idxs] += new_brick.voxels * new_id
            for id in critical_brick_ids:
                # update voxels of each critical brick
                brick = self.bricks[id - 1]  # brick ids are 1-indexed; the base brick has id 1
                _idxs = slice3d(brick.offset, np.array(brick.offset) + np.array(brick.voxels.shape))
                _voxels = self._voxels[_idxs]
                brick.voxels[(_voxels > 0) & (_voxels != id)] = 0
                _idxs_hole = (brick.voxels != 0) & (_voxels < 0)
                brick.voxels[_idxs_hole] = _voxels[_idxs_hole]

    def sample_nail_point(self, thickness=1, max_trials=10):
        """
        Sample a position on the board for a nail
        """
        voxels = self._voxels.reshape(-1, self.size[-1])
        cnt = map(lambda v: np.unique(v).size, voxels)
        cnt = np.array(list(cnt)).reshape(self.size[:2])
        # cnt = minimum_filter(cnt, size=(2, 2))
        grid = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        grid = np.array(grid).transpose((1, 2, 0))
        candidates = grid[cnt > 2]
        for _ in range(max_trials):
            x, y = candidates[np.random.randint(len(candidates))]
            if np.all(self._voxels[x - 1:x + thickness + 1, y - 1:y + thickness + 1] == self._voxels[x, y]):
                empties = np.where(self._voxels[x, y] == 0)
                assert empties[0].size > 0
                z = empties[0][0]
                return np.array([x, y, z])
        return None

    def sample_peg_point(self, thickness=2, max_trials=20):
        """
        Sample a position on the board for a peg
        """
        voxels = self._voxels.reshape(-1, self.size[-1])
        cnt = map(lambda v: np.unique(v).size, voxels)
        cnt = np.array(list(cnt)).reshape(self.size[:2])
        grid = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]), indexing='ij')
        grid = np.array(grid).transpose((1, 2, 0))
        mask = np.zeros_like(cnt, dtype=bool)
        mask[4:-4-thickness, 4:-4-thickness] = True
        candidates = grid[(cnt == 2) & (mask)]
        for _ in range(max_trials):
            x, y = candidates[np.random.randint(len(candidates))]
            if np.all(self._voxels[x - 1:x + thickness + 1, y - 1:y + thickness + 1] == self._voxels[x, y]):
                empties = np.where(self._voxels[x, y] == 0)
                assert empties[0].size > 0
                z = empties[0][0]
                return np.array([x, y, z])
        return None

    @property
    def num_bricks(self):
        return len(self.bricks)

    @property
    def size(self):
        return self._voxels.shape

    @property
    def base_center(self):
        x = self._voxel_size * self.size[0] / 2
        y = self._voxel_size * self.size[1] / 2
        return np.array([x, y, 0])


def generate_rectangular_brick(name: str, size: ArrayLike,
                               offset: Optional[ArrayLike] = None,
                               rgba: Optional[tuple] = None,
                               parent_board: Optional[Board] = None
                               ) -> Brick:
    """
    Generate a rectangular brick
    """
    if offset is not None:
        for i, x in enumerate(offset):
            assert x % 1 == 0
            assert isinstance(x, int), f"Offset should be integers, got {offset}: {type(x)} at index {i}"
    brick = Brick(name=name, size=size, offset=offset, rgba=rgba, parent_board=parent_board)
    return brick


def generate_board(max_bodies: int = 20) -> Board:
    """
    Generate a board
    """
    color_names = list(COLORS)
    np.random.shuffle(color_names)
    rgba_list = [tuple(COLORS[color_name]) + (1.,) for color_name in color_names]
    size = (np.random.randint(25, 40),
            np.random.randint(25, 40),
            50)
    board = Board(size=size)
    # generate base
    base_height = np.random.randint(4, 8)
    base = generate_rectangular_brick(
        name="brick_1",
        size=(size[0], size[1], base_height), rgba=rgba_list[0]
    )
    base.description = f"{color_names[0]} board"
    board.add_brick(base)

    lower_bound = 1
    upper_bound = base_height + 1

    occupied = [np.zeros(size[0], dtype=bool), np.zeros(size[1], dtype=bool)]

    def sample_beam(dir, max_iterations=10):
        nonlocal lower_bound, upper_bound
        assert dir in {0, 1}
        success = False
        iterations = 0
        while not success:
            iterations += 1
            if iterations > max_iterations:
                return None, None
            a = np.random.randint(10, size[dir] - 6)
            a += (size[dir] - a) % 2
            b = np.random.randint(3, 4)
            b += (size[1 - dir] - b) % 2
            offset_z = np.random.randint(lower_bound, min(lower_bound + 1, base_height))
            # we want: offset_z + h >= upperbound + 1
            h_min = upper_bound - offset_z + 1
            h = np.random.randint(h_min, h_min + 1)
            if dir == 0:
                sz = (a, b, h)
                offset_x = (size[dir] - a) / 2
                offset_y = np.random.randint(5, size[1] - b - 5)
            else:
                sz = (b, a, h)
                offset_x = np.random.randint(5, size[0] - b - 5)
                offset_y = (size[dir] - a) / 2
            assert offset_x % 1 == 0 and offset_y % 1 == 0
            offset = [int(offset_x), int(offset_y), offset_z]

            st, ed = offset[1 - dir], offset[1 - dir] + sz[1 - dir]
            if np.all(~occupied[1 - dir][max(0, st - 1): min(ed + 1, len(occupied[1 - dir]))]):
                success = True
                occupied[1 - dir][max(0, st - 1): min(ed + 1, len(occupied[1 - dir]))] = True

        lower_bound = max(lower_bound, offset_z + 1)
        upper_bound = max(upper_bound, offset_z + h)

        return offset, sz

    gen_compliant_brick = False

    id = 2
    while id <= 7 and id <= max_bodies:
        if lower_bound >= base_height:
            break
        brick_offset, brick_size = sample_beam(dir=np.random.choice([0, 1]))
        if brick_offset is None:
            break
        brick = generate_rectangular_brick(name=f"brick_{id}", size=brick_size, offset=brick_offset)
        brick.rgba = rgba_list[id - 1]
        brick.description = f"{color_names[id - 1]} block"
        if id > 3 and np.random.rand() < 0.5:
            gen_compliant_brick = True
        board.add_brick(brick, compliant=gen_compliant_brick)
        id += 1

    for _ in range(2):
        if id > max_bodies:
            break
        thickness = 2
        nail_point = board.sample_nail_point(thickness)
        if nail_point is not None:
            x, y, z = nail_point
            nail = Nail(name=f"brick_{id}",
                        size=(thickness + 2, thickness + 2, z + np.random.randint(2, 5)),
                        offset=(x-1, y-1, 1), shank_length=z-1)
            nail.rgba = rgba_list[id - 1]
            nail.description = f"{color_names[id - 1]} nail"
            board.add_brick(nail)
            id += 1

    return board


def generate_xml(seed: int) -> tuple[XmlMaker, dict]:
    """
    Generate an XML file for a board
    """
    np.random.seed(seed)
    random.seed(seed)

    xml = XmlMaker()
    board = generate_board()

    bodies, aux_assets = board.get_bodies(base_pos=None, base_quat=None)
    for asset in aux_assets:
        xml.add_asset(asset)
    eqs = board.get_equalities()

    n_objects = board.num_bricks
    xs = np.linspace(-0.2, 0.8, 4).reshape(1, -1).repeat(3, axis=0).flatten()
    ys = np.array([-0.5, 0.0, 0.5]).repeat(4, axis=0)

    for x, y, peg in zip(xs[:n_objects], ys[:n_objects], bodies):
        set_attributes(peg, {"pos": [x, y, 0.5]})
        xml.add_object(peg)
    for eq in eqs:
        xml.add_equality(eq)

    info = {
        "n_bodies": len(bodies),
        "brick_descriptions": {brick.name: brick.description for brick in board.bricks},
        "dependencies": board.dependencies.copy()
    }
    return xml, info
