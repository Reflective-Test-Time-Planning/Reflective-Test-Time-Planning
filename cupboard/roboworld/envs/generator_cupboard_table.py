from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from numpy.typing import ArrayLike
from xml.etree.ElementTree import Element
import numpy as np
import random
import os
import pathlib

from roboworld.envs.xml_utils import set_attributes, create_element, XmlMaker
from roboworld.envs.mujoco_env.utils.rotation import euler2quat
from scipy.spatial.transform import Rotation as R

# Import your cupboard generation classes
from generate_cupboard_objects import (
    CupboardPartitioner, FittingTaskGenerator, Shape3D, COLORS, WOOD_COLOR
)

ASSETS_DIR = pathlib.Path(__file__).parent.resolve() / "assets"
VOXEL_SIZE = 0.01


def multiply_quaternions(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
    
    Returns:
        Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ]

def quat_inverse(q):
    """Invert a quaternion in MuJoCo format [w, x, y, z]."""
    w, x, y, z = q
    norm_sq = w*w + x*x + y*y + z*z
    return [w / norm_sq, -x / norm_sq, -y / norm_sq, -z / norm_sq]

def compose_rotations(orientation_quat, quat):
    """
    Compose two rotations: apply orientation_quat first, then quat.
    
    Args:
        orientation_quat: Initial orientation quaternion [w, x, y, z]
        quat: Second rotation quaternion [w, x, y, z]
    
    Returns:
        Final composed quaternion [w, x, y, z]
    """
    # To apply orientation_quat first, then quat:
    # final_quat = quat * orientation_quat
    return multiply_quaternions(quat, orientation_quat)

def orientation_to_quat(orientation):
    """
    Converts an axis permutation to a valid right-handed rotation quaternion in MuJoCo format (w, x, y, z).

    Args:
        orientation (tuple): A permutation of (0, 1, 2) representing new axis mapping.
                             For example, (1, 0, 2) means X→Y, Y→X, Z→Z

    Returns:
        list: Quaternion [w, x, y, z] representing the rotation.
    """
    # Identity basis
    base_axes = np.eye(3)  # X, Y, Z = [1,0,0], [0,1,0], [0,0,1]

    # Permute and transpose to get the rotation matrix
    rot_matrix = base_axes[list(orientation)].T

    # Ensure right-handedness (det > 0)
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 0] *= -1  # Flip one axis to make it right-handed

    # Convert to quaternion: scipy gives [x, y, z, w]
    quat_xyzw = R.from_matrix(rot_matrix).as_quat()

    # Convert to MuJoCo format: [w, x, y, z]
    quat_wxyz = np.roll(quat_xyzw, 1)

    return quat_wxyz.tolist()

def get_color_name(rgb: ArrayLike) -> str:
    """Get the color name from the given RGB value"""
    for k, v in COLORS.items():
        if np.all(np.abs(np.array(rgb) - np.array(v)) < 1e-3):
            return k
    return ""

def random_color() -> tuple:
    """Sample a random color (RGB)"""
    idx = np.random.choice(len(COLORS))
    color_name = list(COLORS)[idx]
    return COLORS[color_name]

def random_rgba(a: Optional[float] = None) -> tuple:
    """sample a random RGBA value"""
    return (random_color()) + (np.random.rand() if a is None else a,)

class CupboardStructure(object):
    """Class representing the cupboard structure for MuJoCo"""
    
    def __init__(self, cupboard: CupboardPartitioner, scale: float = 0.025):
        self.cupboard = cupboard
        self.scale = scale
        self.name = "brick_1"  # Match scene.xml expectation
        
    def get_body(self, pos: Optional[ArrayLike] = None, quat: Optional[ArrayLike] = None):
        """Get the cupboard body element"""
        body = create_element("body", attributes={"name": "brick_1"})
        
        if pos is not None:
            set_attributes(body, {"pos": pos})
        if quat is not None:
            set_attributes(body, {"quat": quat})
            
        # Generate cupboard structure
        self._add_base_structure(body)
        self._add_partitions(body)
        self._add_back_panels(body)
        
        return body, []
    
    def _add_base_structure(self, parent_body):
        """Add floor and walls to cupboard"""
        # Floor - should match the cupboard width x depth
        floor_geom = create_element("geom", attributes={
            "name": "cupboard_floor",
            "type": "box",
            "size": [
                self.cupboard.width * self.scale / 2,
                self.cupboard.height * self.scale / 2, 
                0.01
            ],
            "pos": [
                self.cupboard.width * self.scale / 2,
                self.cupboard.height * self.scale / 2,
                0.01
            ],
            "rgba": list(WOOD_COLOR) + [1.0],
            "class": "visual"
        })
        parent_body.append(floor_geom)
        
        floor_collision = create_element("geom", attributes={
            "name": "cupboard_floor_c",
            "type": "box",
            "size": [
                self.cupboard.width * self.scale / 2,
                self.cupboard.height * self.scale / 2, 
                0.01
            ],
            "pos": [
                self.cupboard.width * self.scale / 2,
                self.cupboard.height * self.scale / 2,
                0.01
            ],
            "class": "collision"
        })
        parent_body.append(floor_collision)
        
        # Walls
        wall_thickness = 0.02
        
        # Left wall (X=0, extends along Y=depth, height=height)
        left_wall = create_element("geom", attributes={
            "name": "cupboard_left_wall",
            "type": "box",
            "size": [wall_thickness/2, self.cupboard.height * self.scale / 2, self.cupboard.depth * self.scale / 2],
            "pos": [0, self.cupboard.height * self.scale / 2, self.cupboard.depth * self.scale / 2],
            "rgba": list(WOOD_COLOR) + [1.0],
            "class": "visual"
        })
        parent_body.append(left_wall)
        
        left_wall_c = create_element("geom", attributes={
            "name": "cupboard_left_wall_c",
            "type": "box",
            "size": [wall_thickness/2, self.cupboard.height * self.scale / 2, self.cupboard.depth * self.scale / 2],
            "pos": [0, self.cupboard.height * self.scale / 2, self.cupboard.depth * self.scale / 2],
            "class": "collision"
        })
        parent_body.append(left_wall_c)
        
        # Right wall (X=width, extends along Y=depth, height=height)
        right_wall = create_element("geom", attributes={
            "name": "cupboard_right_wall", 
            "type": "box",
            "size": [wall_thickness/2, self.cupboard.height * self.scale / 2, self.cupboard.depth * self.scale / 2],
            "pos": [self.cupboard.width * self.scale, self.cupboard.height * self.scale / 2, self.cupboard.depth * self.scale / 2],
            "rgba": list(WOOD_COLOR) + [1.0],
            "class": "visual"
        })
        parent_body.append(right_wall)
        
        right_wall_c = create_element("geom", attributes={
            "name": "cupboard_right_wall_c", 
            "type": "box",
            "size": [wall_thickness/2, self.cupboard.height * self.scale / 2, self.cupboard.depth * self.scale / 2],
            "pos": [self.cupboard.width * self.scale, self.cupboard.height * self.scale / 2, self.cupboard.depth * self.scale / 2],
            "class": "collision"
        })
        parent_body.append(right_wall_c)
    
    def _add_partitions(self, parent_body):
        """Add partition boards"""
        board_thickness = 0.01
        
        for i, partition in enumerate(self.cupboard.partitions):
            if partition.axis == 'x':
                x = partition.position * self.scale
                y_start = partition.start[0] * self.scale
                y_end = partition.end[0] * self.scale  
                z_start = partition.start[1] * self.scale
                z_end = partition.end[1] * self.scale
                
                partition_geom = create_element("geom", attributes={
                    "name": f"partition_x_{i}",
                    "type": "box",
                    "size": [board_thickness/2, (y_end - y_start)/2, (z_end - z_start)/2],
                    "pos": [x, (y_start + y_end)/2, (z_start + z_end)/2],
                    "rgba": list(WOOD_COLOR) + [1.0],
                    "class": "visual"
                })
                parent_body.append(partition_geom)
                
                partition_geom_c = create_element("geom", attributes={
                    "name": f"partition_x_{i}_c",
                    "type": "box",
                    "size": [board_thickness/2, (y_end - y_start)/2, (z_end - z_start)/2],
                    "pos": [x, (y_start + y_end)/2, (z_start + z_end)/2],
                    "class": "collision"
                })
                parent_body.append(partition_geom_c)
                
            elif partition.axis == 'z':
                x_start = partition.start[0] * self.scale
                x_end = partition.end[0] * self.scale
                y_start = partition.start[1] * self.scale  
                y_end = partition.end[1] * self.scale
                z = partition.position * self.scale
                
                partition_geom = create_element("geom", attributes={
                    "name": f"partition_z_{i}",
                    "type": "box",
                    "size": [(x_end - x_start)/2, (y_end - y_start)/2, board_thickness/2],
                    "pos": [(x_start + x_end)/2, (y_start + y_end)/2, z],
                    "rgba": list(WOOD_COLOR) + [1.0],
                    "class": "visual"
                })
                parent_body.append(partition_geom)
                
                partition_geom_c = create_element("geom", attributes={
                    "name": f"partition_z_{i}_c",
                    "type": "box",
                    "size": [(x_end - x_start)/2, (y_end - y_start)/2, board_thickness/2],
                    "pos": [(x_start + x_end)/2, (y_start + y_end)/2, z],
                    "class": "collision"
                })
                parent_body.append(partition_geom_c)
            elif partition.axis == 'y':
                # Y-axis partitions (horizontal shelves)
                x_start = partition.start[0] * self.scale
                x_end = partition.end[0] * self.scale
                y = partition.position * self.scale
                z_start = partition.start[1] * self.scale
                z_end = partition.end[1] * self.scale
                
                partition_geom = create_element("geom", attributes={
                    "name": f"partition_y_{i}",
                    "type": "box",
                    "size": [(x_end - x_start)/2, board_thickness/2, (z_end - z_start)/2],
                    "pos": [(x_start + x_end)/2, y, (z_start + z_end)/2],
                    "rgba": list(WOOD_COLOR) + [1.0],
                    "class": "visual"
                })
                parent_body.append(partition_geom)
                
                partition_geom_c = create_element("geom", attributes={
                    "name": f"partition_y_{i}_c",
                    "type": "box",
                    "size": [(x_end - x_start)/2, board_thickness/2, (z_end - z_start)/2],
                    "pos": [(x_start + x_end)/2, y, (z_start + z_end)/2],
                    "class": "collision"
                })
                parent_body.append(partition_geom_c)

    def _add_back_panels(self, parent_body):
        """Add colored back panels for compartments"""
        back_thickness = 0.01
        
        for i, compartment in enumerate(self.cupboard.compartments):
            if compartment.color is None:
                continue

            x_start = compartment.min_x * self.scale
            x_end = compartment.max_x * self.scale
            y_pos = self.cupboard.height * self.scale - back_thickness/2
            z_start = compartment.min_z * self.scale
            z_end = compartment.max_z * self.scale
            
            if x_end > x_start and z_end > z_start:
                panel_geom = create_element("geom", attributes={
                    "name": f"back_panel_{i}",
                    "type": "box",
                    "size": [(x_end - x_start)/2, back_thickness/2, (z_end - z_start)/2],
                    "pos": [(x_start + x_end)/2, y_pos, (z_start + z_end)/2],
                    "rgba": list(compartment.color) + [1.0],
                    "class": "visual"
                })
                parent_body.append(panel_geom)

                panel_geom = create_element("geom", attributes={
                    "name": f"back_panel_{i}_c",
                    "type": "box",
                    "size": [(x_end - x_start)/2, back_thickness/2, (z_end - z_start)/2],
                    "pos": [(x_start + x_end)/2, y_pos, (z_start + z_end)/2],
                    "class": "collision"
                })
                parent_body.append(panel_geom)

class CupboardShape(object):
    """Class representing a 3D shape for the fitting task using loaded assets"""
    
    def __init__(self, name: str, shape_info: Shape3D, asset_path: str = None, rgba: Optional[tuple] = None):
        self.name = name
        self.shape_info = shape_info
        self.asset_path = asset_path
        self._rgba = rgba if rgba is not None else random_rgba(a=1.0)
        self.description = f"{get_color_name(self._rgba[:3])} object"
        
    def get_body(self, pos: Optional[ArrayLike] = None, quat: Optional[ArrayLike] = None, 
                 freejoint: bool = True):
        """Get the body element of this shape"""
        body = create_element("body", attributes={"name": self.name})
        aux_assets = []
        
        if pos is not None:
            set_attributes(body, {"pos": pos})
        if quat is not None:
            set_attributes(body, {"quat": quat})
        if freejoint:
            create_element(tag="freejoint", parent=body)
            
        # Create visual and collision geoms
        if self.asset_path and os.path.exists(self.asset_path):
            visual_geom, collision_geom, mesh_assets, orientation_quat, _ = self._create_mesh_geoms()
            set_attributes(body, {"quat": orientation_quat})
            aux_assets.extend(mesh_assets)
        else:
            # Fallback to basic geometry
            visual_geom = self._create_basic_geom("visual")
            collision_geom = self._create_basic_geom("collision")
        
        body.append(visual_geom)
        body.append(collision_geom)
        
        # Add grasp sites
        self._add_grasp_sites(body)
        
        return body, aux_assets
    
    def get_body_in_the_compartments(self, pos: Optional[ArrayLike] = None, quat: Optional[ArrayLike] = None, 
                 freejoint: bool = True):
        """Get the body element of this shape"""
        body = create_element("body", attributes={"name": self.name})
        aux_assets = []
        
        w, h, d = self.shape_info.dimensions
        pos[2] = 0.5 + w/2

        if pos is not None:
            set_attributes(body, {"pos": pos})
        if quat is not None:
            set_attributes(body, {"quat": quat})
        if freejoint:
            create_element(tag="freejoint", parent=body)
            
        visual_geom, collision_geom, mesh_assets, orientation_quat, target_dims = self._create_mesh_geoms()
        
        quat = compose_rotations(orientation_quat, quat)
        set_attributes(body, {"quat": quat})
    
        aux_assets.extend(mesh_assets)

        body.append(visual_geom)
        body.append(collision_geom)
        
        grasp_height = w/2

        # Add grasp sites
        self._add_grasp_sites(body, grasp_height)
        
        return body, aux_assets
    
    def collision(self, pos, poses, size, sizes):
        """
        Check if a box at `pos` with bounding box `size` overlaps with any existing boxes.

        Parameters:
            pos: [x, y, z] position of the current box.
            poses: list of [x, y, z] positions of existing boxes.
            size: [sx, sy, sz] size of the current box.
            sizes: list of [sx, sy, sz] sizes for each existing box.

        Returns:
            True if there is any overlap, False otherwise.
        """
        if pos[0] == 0 and pos[1] == 0 and pos[2] == 0: return True

        for other_pos, other_size in zip(poses, sizes):
            if all(abs(p - o) - 0.05 < (s / 2 + os / 2)
                for p, o, s, os in zip(pos, other_pos, size, other_size)):
                return True
        return False

    def get_body_on_the_table(self, pos: Optional[ArrayLike] = None, quat: Optional[ArrayLike] = None, 
                 freejoint: bool = True, poses = [], sizes = []):
        """Get the body element of this shape"""
        body = create_element("body", attributes={"name": self.name})
        aux_assets = []
        
        if pos is not None:
            set_attributes(body, {"pos": pos})
        # if quat is not None:
        #     set_attributes(body, {"quat": quat})
        if freejoint:
            create_element(tag="freejoint", parent=body)
            
        visual_geom, collision_geom, mesh_assets, orientation_quat, target_dims = self._create_mesh_geoms()
        
        quat = compose_rotations(orientation_quat, quat)
        set_attributes(body, {"quat": quat})
    
        aux_assets.extend(mesh_assets)

        x_range=[-0.2, 0.1]; y_range_1=[-0.35,-0.15]; y_range_2=[0.2,0.5]; y_range_3 = [-0.15, 0.2]
        x_range_1 = [-0.15, -0.05]
        x_range_2 = [-0.2, 0.15]
        w, h, d = self.shape_info.dimensions

        x, y, z, = 0, 0, 0
        pos = [x, y, z]

        while self.collision(pos, poses, self.shape_info.dimensions, sizes):
            # x = random.uniform(x_range[0] + d/2, x_range[1] - d/2)
            # a = random.random()
            # if a < 0.2 and x_range_1[1] - d/2 > x_range_1[0] + d/2 and y_range_3[1] - h/2 > y_range_3[0] + h/2:
            #     x = random.uniform(x_range_1[0] + d/2, x_range_1[1] - d/2)
            #     y = random.uniform(y_range_3[0] + h/2, y_range_3[1] - h/2)
            # else:
            a = random.random()
            if a > 0.5 or  x_range_2[1] - d/2 < x_range_2[0] + d/2 and y_range_2[1] - h/2 < y_range_2[0] + h/2:
                x = random.uniform(x_range_2[0] + d/2, x_range_2[1] - d/2)
                y = random.uniform(y_range_1[0] + h/2, y_range_1[1] - h/2)
            else:
                x = random.uniform(x_range_2[0] + d/2, x_range_2[1] - d/2)
                y = random.uniform(y_range_2[0] + h/2, y_range_2[1] - h/2)

            # if random.random() > 0.5:
            #     y = random.uniform(y_range_1[0] + h/2, y_range_1[1] - h/2)
            # else:
            #     y = random.uniform(y_range_2[0] + h/2, y_range_2[1] - h/2)

            z = 0.5 + w/2 
            pos = [x, y, z]

        set_attributes(body, {"pos": pos})

        grasp_height = w/2

        body.append(visual_geom)
        body.append(collision_geom)
        
        # Add grasp sites
        self._add_grasp_sites(body, grasp_height, quat)
        
        return body, aux_assets, pos, self.shape_info.dimensions
    
    def _load_obj_file(self, obj_path: str) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, ...]]]:
        """Load vertices and faces from an OBJ file (ignore materials)"""
        vertices = []
        faces = []
        
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):
                        # Vertex line: v x y z
                        parts = line.split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            vertices.append((x, y, z))
                    elif line.startswith('f '):
                        # Face line: f v1 v2 v3 (or more vertices)
                        parts = line.split()[1:]  # Skip the 'f'
                        face_vertices = []
                        
                        for part in parts:
                            # Take only the vertex index (before any '/')
                            vertex_index_str = part.split('/')[0]
                            vertex_index = int(vertex_index_str)
                            
                            # Handle negative indices
                            if vertex_index < 0:
                                vertex_index = len(vertices) + vertex_index
                            else:
                                vertex_index = vertex_index - 1  # OBJ uses 1-based indexing
                            
                            face_vertices.append(vertex_index)
                        
                        # Only add valid faces
                        if len(face_vertices) >= 3 and all(0 <= idx < len(vertices) for idx in face_vertices):
                            faces.append(tuple(face_vertices))
            
            print(f"    Loaded: {len(vertices)} vertices, {len(faces)} faces")
            return vertices, faces
            
        except Exception as e:
            print(f"    Error loading {obj_path}: {e}")
            return [], []
        
    def _create_mesh_geoms(self):
        """Create mesh-based geometries from pre-normalized assets"""
        # Create mesh asset
        mesh_name = f"{self.name}_mesh"

        vertices, faces = self._load_obj_file(self.asset_path)
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        
        # Current dimensions
        curr_width = max_x - min_x
        curr_height = max_y - min_y
        curr_depth = max_z - min_z
        current_dims = [curr_width, curr_height, curr_depth]
        
        # For normalized assets, scale to exact target dimensions
        
        target_dims = self.shape_info.dimensions

        # Find best orientation by trying all 6 possible axis alignments
        orientations = [
            (0, 1, 2),  # X→X, Y→Y, Z→Z (no rotation)
            (0, 2, 1),  # X→X, Y→Z, Z→Y  
            (1, 0, 2),  # X→Y, Y→X, Z→Z
            (1, 2, 0),  # X→Y, Y→Z, Z→X
            (2, 0, 1),  # X→Z, Y→X, Z→Y
            (2, 1, 0),  # X→Z, Y→Y, Z→X
        ]
        
        best_orientation = None
        best_score = float('inf')
        
        # Find orientation that best matches target aspect ratios
        for orientation in orientations:
            # How current dimensions map to target dimensions in this orientation
            mapped_dims = [current_dims[orientation[i]] for i in range(3)]
            
            # Calculate how well this orientation matches (minimize dimension mismatches)
            score = 0
            for i in range(3):
                ratio = mapped_dims[i] / target_dims[i] if target_dims[i] > 0 else 1
                score += abs(1.0 - ratio)  # Penalty for dimension mismatch
            
            if score < best_score:
                best_score = score
                best_orientation = orientation

        # orientation_quat = [1,0,0,0] # default, degraded

        orientation_quat = orientation_to_quat(best_orientation) 

        target_dims = [target_dims[best_orientation[0]], target_dims[best_orientation[1]], target_dims[best_orientation[2]]]

        scale = [target_dims[0] / current_dims[0], target_dims[1] / current_dims[1], target_dims[2] / current_dims[2]]

        mesh_asset = create_element("mesh", attributes={
            "name": mesh_name,
            "file": self.asset_path,
            "scale": scale  # Scale to exact target dimensions
        })
        
        # Visual geom with orientation
        visual_geom = create_element("geom", attributes={
            "name": f"{self.name}_visual",
            "type": "mesh",
            "mesh": mesh_name,
            "rgba": self._rgba,
            "class": "visual"
        })

        # Collision geom - use box approximation matching shape dimensions
        collision_geom = create_element("geom", attributes={
            "name": f"{self.name}_collision",
            "type": "box",
            "size": [d/2 - 0.0005 for d in target_dims],
            "class": "collision"
        })
        
        return visual_geom, collision_geom, [mesh_asset], orientation_quat, target_dims
    
    def _create_basic_geom(self, geom_class: str):
        """Create basic geometry as fallback"""
        dims = self.shape_info.dimensions
        size_adjustment = -0.0005 if geom_class == "collision" else 0
        
        # Default to box shape
        geom = create_element("geom", attributes={
            "name": f"{self.name}_{geom_class}",
            "type": "box",
            "size": [(d/2 + size_adjustment) for d in dims],
            "rgba": self._rgba,
            "class": geom_class
        })
        return geom
    
    def _add_grasp_sites(self, parent_body, grasp_height, quat):
        """Add grasping sites to the shape based on bounding box"""
        quat = quat_inverse(quat)
        body = create_element("body", parent=parent_body, attributes={"name": f"{self.name}_grasp", "quat":quat})

        dims = self.shape_info.dimensions
        # Use the largest dimension as height for grasping
        grasp_height = grasp_height + 0.005
            
        create_element(tag="site", parent=body, attributes={
            "name": f"{self.name}_grasp0",
            "pos": [0, 0, grasp_height],
            "class": "invisible_site"
        })
        
        # Alignment site at center
        create_element(tag="site", parent=body, attributes={
            "name": f"{self.name}_align", 
            "pos": [0, 0, 0],
            "class": "invisible_site"
        })

def position_shape_in_compartment(shape: Shape3D, compartment, scale: float):
    """Position a shape within a compartment - CORRECT COORDINATE MAPPING"""
    # Original system: X=width, Y=height, Z=depth
    # Your MuJoCo cupboard: X=width, Y=depth, Z=height
    # So compartment.min_y/max_y is actually DEPTH (front/back)
    # And compartment.min_z/max_z is actually HEIGHT (up/down)
    
    comp_min_x = compartment.min_x * scale  # left edge (width)
    comp_max_x = compartment.max_x * scale  # right edge (width)
    comp_min_y = compartment.min_y * scale  # bottom edge (original height = MuJoCo depth)  
    comp_max_y = compartment.max_y * scale  # top edge (original height = MuJoCo depth)
    comp_min_z = compartment.min_z * scale  # front edge (original depth = MuJoCo height)
    comp_max_z = compartment.max_z * scale  # back edge (original depth = MuJoCo height)
    
    print(f"    Positioning {shape.shape_type} in compartment:")
    print(f"      X: {comp_min_x:.3f} to {comp_max_x:.3f} (width: left-right)")
    print(f"      Y: {comp_min_y:.3f} to {comp_max_y:.3f} (depth: front-back)")  
    print(f"      Z: {comp_min_z:.3f} to {comp_max_z:.3f} (height: up-down)")
    
    # Shape dimensions - assuming width, height, depth from original
    
    w, h, d = shape.dimensions
    print(f"      Shape dims: {w:.3f} × {h:.3f} × {d:.3f} (w×h×d from original)")
    
    # Calculate available space with margin
    margin = 0.005
    available_x = comp_max_x - comp_min_x - w - 2*margin  # width space
    available_y = comp_max_y - comp_min_y - h - 2*margin  # depth space (original depth -> MuJoCo Y)
    available_z = comp_max_z - comp_min_z - d - 2*margin  # height space (original height -> MuJoCo Z)

    # if available_x < 0 or available_y < 0 or available_z < 0:
    #     print(f"    WARNING: Shape too big! Available: {available_x:.3f}, {available_y:.3f}, {available_z:.3f}")
    #     # Force fit at center
    #     pos_x = comp_min_x + (comp_max_x - comp_min_x) / 2  # center in width
    #     pos_y = comp_min_y + (comp_max_y - comp_min_y) / 2  # center in depth
    #     pos_z = comp_min_z + (comp_max_z - comp_min_z) / 2  # center in height
    # else:
    #     # Center in available space
    pos_x = comp_min_x + margin + w/2  # left + margin + half width
    pos_y = comp_min_y + margin + h/2  # front + margin + half depth (original depth -> MuJoCo Y)
    pos_z = comp_min_z + margin + d/2  # bottom + margin + half height (original height -> MuJoCo Z)
    
    # Final clamp
    pos_x = max(comp_min_x + w/2, min(pos_x, comp_max_x - w/2))
    pos_y = max(comp_min_y + h/2, min(pos_y, comp_max_y - h/2))  # depth
    pos_z = max(comp_min_z + d/2, min(pos_z, comp_max_z - d/2))  # height
    
    # Store position in MuJoCo coordinates (X=width, Y=depth, Z=height)
    shape.position = (pos_x, pos_y, pos_z)
    print(f"    → Final MuJoCo position: ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})")
    print(f"      X={pos_x:.3f} (width), Y={pos_y:.3f} (depth), Z={pos_z:.3f} (height)")

def load_random_assets_normalized(shapes: List[Shape3D], asset_folder: str = "assets_simple_normalize"):
    """Load random asset paths from pre-normalized folder"""
    if not os.path.exists(asset_folder):
        print(f"Asset folder '{asset_folder}' not found! Using fallback.")
        return [None] * len(shapes)
    
    subfolders = [f for f in os.listdir(asset_folder) 
                 if os.path.isdir(os.path.join(asset_folder, f))]
    
    if not subfolders:
        print(f"No subfolders found in '{asset_folder}'! Using fallback.")
        return [None] * len(shapes)
    
    # Create randomized list to avoid duplicates
    available_assets = subfolders.copy()
    np.random.shuffle(available_assets)
    
    # Extend if needed
    while len(available_assets) < len(shapes):
        available_assets.extend(subfolders)
        np.random.shuffle(available_assets)
    
    # Load asset paths
    asset_paths = []
    for i, shape in enumerate(shapes):
        subfolder = available_assets[i]
        subfolder_path = os.path.join(asset_folder, subfolder)
        
        obj_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.obj')]
        if obj_files:
            asset_path = os.path.join(subfolder_path, obj_files[0])
            asset_paths.append(asset_path)
            print(f"Shape {i+1}: Using normalized asset {subfolder}/{obj_files[0]}")
        else:
            asset_paths.append(None)
            print(f"Shape {i+1}: No OBJ in {subfolder}, using fallback")
    
    return asset_paths

def generate_cupboard_task(max_shapes: int = 8) -> tuple:
    """Generate a cupboard fitting task"""
    # Generate cupboard structure exactly like the original
    cupboard = CupboardPartitioner(width=10, height=3, depth=10)
    target_compartments = np.random.randint(5, 8)
    cupboard.generate_partial_partitions(target_compartments)
    
    print(f"Generated cupboard with {len(cupboard.compartments)} compartments")
    
    # Generate shapes for fitting task using the original logic
    fitting_generator = FittingTaskGenerator(cupboard, scale=0.025)
    shapes, placement_options, sorted_compartments, original_indices = fitting_generator.generate_fitting_task()

    # Limit number of shapes
    if len(shapes) > max_shapes:
        shapes = shapes[:max_shapes]
    
    print(f"Generated {len(shapes)} shapes for fitting")
    
    # Load pre-normalized asset paths
    asset_paths = load_random_assets_normalized(shapes, "assets_simple_normalize")
    
    # NOW position shapes properly using original logic
    for comp_idx, shape in placement_options.items():
        compartment = sorted_compartments[comp_idx]
        position_shape_in_compartment(shape, compartment, 0.025)
        shape.compartment_id = original_indices[comp_idx]
    
    return cupboard, shapes, asset_paths


def transform_to_world(local_pos, cupboard_pos, cupboard_quat):
    """Apply rotation and translation to get world position."""
    rot = R.from_quat([cupboard_quat[1], cupboard_quat[2], cupboard_quat[3], cupboard_quat[0]])  # [x, y, z, w]
    return rot.apply(local_pos) + cupboard_pos

def transform_local_to_world(local_pos, origin_pos, origin_quat):
    rot = R.from_quat([origin_quat[1], origin_quat[2], origin_quat[3], origin_quat[0]])  # MuJoCo wxyz → scipy xyzw
    world_pos = rot.apply(local_pos) + origin_pos
    return world_pos

def add_compartment_sites(cupboard_body, cupboard, origin_pos, origin_quat, scale=0.025):
    compartment_poses = []
    for i, compartment in enumerate(cupboard.compartments):
        # Center in local (scaled) coordinates
        center_local = np.array([
            (compartment.min_x + compartment.max_x) * 0.5 * scale,
            (compartment.min_y + compartment.max_y) * 0.5 * scale,
            (compartment.min_z + compartment.max_z) * 0.5 * scale
        ])

        # Transform to world coordinates
        world_pos = transform_local_to_world(center_local, np.array(origin_pos), np.array(origin_quat))

        compartment_poses.append(world_pos)

    return compartment_poses

def generate_xml(seed: int) -> tuple[XmlMaker, dict]:
    """Generate an XML file for cupboard fitting task"""
    np.random.seed(seed)
    random.seed(seed)
    
    xml = XmlMaker()
    cupboard, shapes, asset_paths = generate_cupboard_task(max_shapes=8)
    
    quat = euler2quat([0, 0, 0])

    # Create cupboard structure - position on table
    cupboard_structure = CupboardStructure(cupboard, scale=0.025)
    cupboard_body, cupboard_assets = cupboard_structure.get_body(
        pos=[-0.1, 0.15, 0.65],  # Position on table
        quat=[0.5, -0.5, 0.5, -0.5]
    )
    xml.add_object(cupboard_body)

    compartment_poses = add_compartment_sites(
        cupboard_body=cupboard_body,
        cupboard=cupboard,
        origin_pos=[-0.1, 0.15, 0.65],
        origin_quat=[0.5, -0.5, 0.5, -0.5],
        scale=0.025
    )
    
    # Add cupboard assets
    for asset in cupboard_assets:
        xml.add_asset(asset)
    
    # Create shapes and position them using the fitted positions
    shape_bodies = []
    
    poses, sizes = [], []

    for i, (shape_info, asset_path) in enumerate(zip(shapes, asset_paths)):
        # Create shape object
        shape_name = f"shape_{i+1}"
        shape_obj = CupboardShape(
            shape_name, 
            shape_info, 
            asset_path=asset_path,
            rgba=tuple(shape_info.color) + (1.0,)
        )
        
        # Use the position calculated by fitting algorithm and add cupboard world offset
        # shape.position is in cupboard-local coordinates (X=width, Y=height, Z=depth)
        # Your cupboard is positioned at [0.1, 0.3, 0.5] in world coordinates

        pos_x = shape_info.position[0] + 0.1  # shape X + cupboard world X
        pos_y = shape_info.position[1] + 0.3 # shape Y + cupboard world Y  
        pos_z = shape_info.position[2] + 0.5 + 0.02 # shape Z + cupboard world Z + floor height
        
        print(f"Shape {i+1}:")
        print(f"  Compartment position: {shape_info.position}")
        print(f"  World position: ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})")
        print(f"  Target compartment: {shape_info.compartment_id}")
        print(f"  Asset: {os.path.basename(os.path.dirname(asset_path)) if asset_path else 'fallback'}")
        
        # Minimal random orientation
        quat = euler2quat([0, 0, 0])
        

        shape_body, shape_assets, pose, size = shape_obj.get_body_on_the_table(
            pos=[pos_x, pos_y, pos_z],
            quat=[0.5, -0.5, 0.5, -0.5],
            freejoint=True,
            poses=poses,
            sizes=sizes
        )
        # shape_body, shape_assets = shape_obj.get_body_in_the_compartments(
        #     pos=compartment_poses[shape_info.compartment_id],
        #     quat=[0.5, -0.5, 0.5, -0.5],
        #     freejoint=True
        # )

        poses.append(pose)
        sizes.append(size)
        
        xml.add_object(shape_body)
        
        # Add shape assets
        for asset in shape_assets:
            xml.add_asset(asset)
            
        shape_bodies.append(shape_body)
    
    # Add equality constraints for grasping
    for i, shape_info in enumerate(shapes):
        shape_name = f"shape_{i+1}"
        eq = create_element("weld", attributes={
            "name": f"{shape_name}_grasp_hand",
            "body1": "hand", 
            "body2": shape_name,
            "active": "false",
            "solimp": [0.99, 0.999, 0.001],
            "solref": [0.01, 1]
        })
        xml.add_equality(eq)
    
    # Add target sites in compartments for alignment
    for i, compartment in enumerate(cupboard.compartments):
        if i < len(shapes):  # Only add sites for shapes we have
            target_site = create_element("site", parent=cupboard_body, attributes={
                "name": f"shape_{i+1}_target",
                "pos": [
                    (compartment.min_x + compartment.max_x) / 2 * 0.025,
                    (compartment.min_y + compartment.max_y) / 2 * 0.025,
                    (compartment.min_z + compartment.max_z) / 2 * 0.025 + 0.02
                ],
                "class": "invisible_site"
            })
    
    # Create comprehensive task info
    info = {
        "n_shapes": len(shapes),
        "n_compartments": len(cupboard.compartments),
        "shape_descriptions": {f"shape_{i+1}": f"{get_color_name(shape.color[:3])} object" 
                             for i, shape in enumerate(shapes)},
        "cupboard_info": {
            "dimensions": (cupboard.width, cupboard.height, cupboard.depth),
            "scale": 0.025,
            "compartments": len(cupboard.compartments)
        },
        "compartment_descriptions": {f"back_panel_{i+1}": f"{get_color_name(compartment.color[:3])} compartment" 
                        for i, compartment in enumerate(cupboard.compartments)},
        "shapes_info": [
            {
                "id": i+1,
                "type": shape.shape_type,
                "dimensions": shape.dimensions,
                "color": shape.color,
                "position": shape.position,
                "target_compartment": shape.compartment_id,
                "asset_path": asset_paths[i]
            }
            for i, shape in enumerate(shapes)
        ],
        "compartments_info": [
            {
                "id": i+1,
                "bbox": {
                    "min_x": comp.min_x, "min_y": comp.min_y, "min_z": comp.min_z,
                    "max_x": comp.max_x, "max_y": comp.max_y, "max_z": comp.max_z
                },
                "physical_bbox": {
                    "min_x": comp.min_x * 0.025, "min_y": comp.min_y * 0.025, "min_z": comp.min_z * 0.025,
                    "max_x": comp.max_x * 0.025, "max_y": comp.max_y * 0.025, "max_z": comp.max_z * 0.025
                },
                "color": comp.color,
                "pos": compartment_poses[i]
            }
            for i, comp in enumerate(cupboard.compartments)
        ]
    }
    
    return xml, info

# def main():
#     """Main function to test cupboard fitting task generation"""
#     print("🏠🤖 Cupboard Fitting Task Generator for MuJoCo")
#     print("=" * 50)
    
#     # Create output directory
#     output_dir = "fitting_tasks_table"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Test with a few different seeds
#     for i in range(3):
#         seed = 3000 + i
#         print(f"\n🔄 Generating task {i+1}/3 (seed={seed})")
        
#         try:
#             xml, info = generate_xml(seed)
            
#             # Write XML file to fitting_tasks directory
#             output_filename = os.path.join(output_dir, f"cupboard_task_{seed}.xml")
#             xml.write_to_file(output_filename)
            
#             # # Also generate the original OBJ/JSON/PNG files for reference
#             # from generate_cupboard_objects import generate_complete_fitting_task
#             # task_info = generate_complete_fitting_task(seed=seed, output_dir=output_dir)
            
#             # Print summary
#             print(f"✅ Generated: {output_filename}")
#             print(f"   - Shapes: {info['n_shapes']}")
#             print(f"   - Compartments: {info['n_compartments']}")
#             print(f"   - Cupboard size: {info['cupboard_info']['dimensions']}")
#             print(f"   - Also generated: OBJ, PNG, JSON files")
            
#             # Print shape details
#             print("   - Shape positions:")
#             for shape_info in info['shapes_info']:
#                 pos = shape_info['position']
#                 comp_id = shape_info['target_compartment']
#                 asset_name = os.path.basename(os.path.dirname(shape_info['asset_path'])) if shape_info['asset_path'] else 'fallback'
#                 print(f"     * Shape {shape_info['id']}: {shape_info['type']} ({asset_name})")
#                 print(f"       Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
#                 print(f"       Compartment: {comp_id}")
            
#         except Exception as e:
#             print(f"❌ Error generating task {i+1}: {e}")
#             import traceback
#             traceback.print_exc()
    
#     print(f"\n🎉 Generation complete!")
#     print(f"All files saved to '{output_dir}/' directory:")
#     print("  - XML files for MuJoCo environments")
#     print("  - OBJ/MTL files for 3D visualization") 
#     print("  - PNG files for task visualization")
#     print("  - JSON files for task data")
#     print("\nONLY SHAPE LOGIC CHANGES:")
#     print("  ✅ Using pre-normalized assets from assets_simple_normalize folder")
#     print("  ✅ Reorienting objects based on largest dimension")
#     print("  ✅ Proper fitting of shapes into compartments using original algorithm")
#     print("  ✅ Objects positioned inside compartments (not falling!)")
#     print("  ✅ Cupboard structure UNCHANGED from your working version")

# if __name__ == "__main__":
#     main()