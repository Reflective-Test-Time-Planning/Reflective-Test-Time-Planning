import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import math

# Color definitions (RGB values 0-1)
COLORS = {
    "blue": (0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
    "orange": (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
    "green": (0.41568627450980394, 0.8, 0.39215686274509803),
    "red": (0.8392156862745098, 0.37254901960784315, 0.37254901960784315),
    "purple": (0.5843137254901961, 0.4235294117647059, 0.7058823529411765),
    "pink": (0.8627450980392157, 0.49411764705882355, 0.7529411764705882),
    "gray": (0.4745098039215686, 0.4745098039215686, 0.4745098039215686),
    "yellow": (0.9098, 0.6784, 0.1373),
    "cyan": (0.2, 0.8, 0.9),
    "brown": (0.5, 0.3, 0.1),
}

WOOD_COLOR = (0.6, 0.4, 0.2)  # Brown wood color

class Partition:
    """Represents a partition board in 3D space"""
    def __init__(self, axis: str, position: int, start: Tuple[int, int], end: Tuple[int, int]):
        self.axis = axis  # 'x', 'y', or 'z'
        self.position = position  # position along the axis
        self.start = start  # (coord1, coord2) start point
        self.end = end  # (coord1, coord2) end point
    
    def __repr__(self):
        return f"Partition({self.axis}={self.position}, span={self.start}->{self.end})"

class Compartment:
    """Represents a closed compartment in the cupboard"""
    def __init__(self, min_coords: Tuple[int, int, int], max_coords: Tuple[int, int, int]):
        self.min_x, self.min_y, self.min_z = min_coords
        self.max_x, self.max_y, self.max_z = max_coords
        self.volume = (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)
        self.color = None  # Will be assigned during generation
        self.occupied_objects = []  # List of objects placed in this compartment
    
    def __repr__(self):
        return f"Compartment({self.min_x},{self.min_y},{self.min_z})->({self.max_x},{self.max_y},{self.max_z}) Vol:{self.volume}"

class Shape3D:
    """Represents a 3D shape with position and properties"""
    def __init__(self, shape_type: str, dimensions: Tuple[float, float, float], position: Tuple[float, float, float] = (0, 0, 0)):
        self.shape_type = shape_type  # 'cube', 'cylinder', 'sphere', 'elongated_cube', 'truncated_cone'
        self.dimensions = dimensions  # (width, height, depth) or (radius, height, _) for cylinder
        self.position = position  # (x, y, z) position
        self.color = None
        self.compartment_id = None  # Which compartment this shape is placed in
        
    def get_bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get the axis-aligned bounding box of the shape"""
        x, y, z = self.position
        
        if self.shape_type == 'cube' or self.shape_type == 'elongated_cube':
            w, h, d = self.dimensions
            return (x, y, z), (x + w, y + h, z + d)
        elif self.shape_type == 'cylinder':
            r, h, _ = self.dimensions
            return (x - r, y, z - r), (x + r, y + h, z + r)
        elif self.shape_type == 'sphere':
            r, _, _ = self.dimensions
            return (x - r, y - r, z - r), (x + r, y + r, z + r)
        elif self.shape_type == 'truncated_cone':
            r, h, _ = self.dimensions
            return (x - r, y, z - r), (x + r, y + h, z + r)
        else:
            w, h, d = self.dimensions
            return (x, y, z), (x + w, y + h, z + d)
    
    def check_unique(self, sorted_compartments, comp_dimensions, shape_index):
        unique = True
        if self is None: return False
        
        for j in range(len(sorted_compartments)):
            if j == shape_index or comp_dimensions[shape_index][0] == comp_dimensions[j][0] or comp_dimensions[shape_index][1] == comp_dimensions[j][1] or comp_dimensions[shape_index][2] == comp_dimensions[j][2]: continue
            if self.fits_in_compartment(sorted_compartments[j], self.scale):
                unique = False
                break
        return unique

    def fits_in_compartment(self, compartment: Compartment, scale) -> bool:
        """Check if this shape can fit in the given compartment"""
        # Convert compartment dimensions to physical units
        comp_width = (compartment.max_x - compartment.min_x) * scale
        comp_height = (compartment.max_y - compartment.min_y) * scale
        comp_depth = (compartment.max_z - compartment.min_z) * scale
        
        if self.shape_type == 'cube':
            w, h, d = self.dimensions
            return w <= comp_width and h <= comp_height and d <= comp_depth
            
        elif self.shape_type == 'elongated_cube' or self.shape_type == 'rectangular_box':
            w, h, d = self.dimensions
            # Try all orientations for rectangular boxes
            orientations = [
                (w, h, d),  # Original
                (w, d, h),  # Rotated around X
                (h, w, d),  # Rotated around Z
                (h, d, w),  # Rotated around Y
                (d, w, h),  # Another rotation
                (d, h, w)   # Final rotation
            ]
            
            for ow, oh, od in orientations:
                if ow <= comp_width and oh <= comp_height and od <= comp_depth:
                    return True
            return False
            
        elif self.shape_type == 'cylinder':
            r, h, _ = self.dimensions
            # Cylinder can be oriented vertically or horizontally
            # Vertical: needs 2r×2r base, h height
            vertical_fits = (2*r <= comp_width and h <= comp_height and 2*r <= comp_depth)
            # Horizontal (lying on side): needs h×2r base, 2r height
            horizontal_fits = (h <= comp_width and 2*r <= comp_height and 2*r <= comp_depth) or \
                            (2*r <= comp_width and 2*r <= comp_height and h <= comp_depth)
            return vertical_fits or horizontal_fits
            
        elif self.shape_type == 'sphere':
            r, _, _ = self.dimensions
            return 2*r <= comp_width and 2*r <= comp_height and 2*r <= comp_depth
            
        elif self.shape_type == 'truncated_cone':
            r, h, _ = self.dimensions
            # Similar to cylinder
            vertical_fits = (2*r <= comp_width and h <= comp_height and 2*r <= comp_depth)
            horizontal_fits = (h <= comp_width and 2*r <= comp_height and 2*r <= comp_depth) or \
                            (2*r <= comp_width and 2*r <= comp_height and h <= comp_depth)
            return vertical_fits or horizontal_fits
        
        return False
    
    def __repr__(self):
        return f"Shape3D({self.shape_type}, {self.dimensions}, pos={self.position})"

class CupboardPartitioner:
    def __init__(self, width: int = 10, height: int = 3, depth: int = 10):
        self.width = width   # x dimension
        self.height = height # y dimension  
        self.depth = depth   # z dimension
        self.partitions = []
        self.compartments = []
    
    def add_partition(self, axis: str, position: int, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Add a partition board to the cupboard with specified span"""
        # Validate position
        if axis == 'x' and (position <= 0 or position >= self.width):
            return False
        elif axis == 'y' and (position <= 0 or position >= self.height):
            return False
        elif axis == 'z' and (position <= 0 or position >= self.depth):
            return False
        
        # Validate span coordinates based on axis
        if axis == 'x':  # YZ plane
            if start[0] < 0 or end[0] > self.height or start[1] < 0 or end[1] > self.depth:
                return False
        elif axis == 'y':  # XZ plane
            if start[0] < 0 or end[0] > self.width or start[1] < 0 or end[1] > self.depth:
                return False
        else:  # axis == 'z', XY plane
            if start[0] < 0 or end[0] > self.width or start[1] < 0 or end[1] > self.height:
                return False
        
        partition = Partition(axis, position, start, end)
        self.partitions.append(partition)
        return True

    def _has_unique_dimensions(self, compartments: List[Compartment]) -> bool:
        """Check if all compartments have unique dimensions"""
        dimensions_set = set()
        
        for comp in compartments:
            width = comp.max_x - comp.min_x
            height = comp.max_y - comp.min_y
            depth = comp.max_z - comp.min_z
            
            # Create a tuple of sorted dimensions to handle rotational equivalence
            # e.g., (2,3,4) is the same as (3,2,4) or (4,2,3)
            dims = tuple(sorted([width, height, depth]))
            
            if dims in dimensions_set:
                return False
            dimensions_set.add(dims)
        
        return True

    def _validate_compartment_dimensions(self, compartments: List[Compartment], min_dimension: int = 2) -> bool:
        """Check if all compartments have minimum dimensions along x and z axes"""
        for comp in compartments:
            width = comp.max_x - comp.min_x
            depth = comp.max_z - comp.min_z
            
            if width < min_dimension or depth < min_dimension:
                return False
        
        return True

    def generate_partial_partitions(self, target_compartments: int = None) -> List[Partition]:
        """Generate partitions that create properly enclosed compartments with unique dimensions"""
        max_attempts = 40  # Limit attempts to avoid infinite loops
        
        for attempt in range(max_attempts):
            self.partitions = []
            
            # # Try different generation strategies
            # if attempt < 10:
            self._create_grid_with_partial_divisions()
            # else:
            #     self._create_varied_asymmetric_layout()
            
            # Find compartments
            self.compartments = self.find_compartments_flood_fill()
            
            # Check if we have enough compartments, they're all unique, AND meet minimum dimensions
            if (len(self.compartments) >= 4 and 
                self._has_unique_dimensions(self.compartments) and 
                self._validate_compartment_dimensions(self.compartments, min_dimension=2)):
                print(f"✅ Generated {len(self.compartments)} compartments with unique dimensions and minimum size (attempt {attempt + 1})")
                break
            else:
                issues = []
                if len(self.compartments) < 4:
                    issues.append(f"only {len(self.compartments)} compartments")
                if not self._has_unique_dimensions(self.compartments):
                    issues.append("non-unique dimensions")
                if not self._validate_compartment_dimensions(self.compartments, min_dimension=2):
                    issues.append("compartments too small")
                print(f"❌ Attempt {attempt + 1}: {', '.join(issues)}")
        
        else:
            print("⚠️ Warning: Could not generate valid compartments after maximum attempts")
        
        # Assign colors to compartments
        self._assign_colors()
        
        return self.partitions
    
    def _create_grid_with_partial_divisions(self):
        """Create a grid-like structure with some partial divisions"""
        main_x = random.randint(2, 8)
        main_z = random.randint(2, 8)
        
        # Add main full partitions
        self.add_partition('x', main_x, (0, 0), (self.height, self.depth))
        self.add_partition('z', main_z, (0, 0), (self.width, self.height))
        
        # Left region (0 to main_x)
        if main_x > 4:
            z_pos = random.randint(2, main_z - 1) if main_z > 3 else main_z // 2
            self.add_partition('z', z_pos, (0, 0), (main_x, self.height))
        
        # Right region (main_x to width)
        remaining_width = self.width - main_x
        if remaining_width > 4:
            z_pos = random.randint(main_z + 1, self.depth - 1) if main_z < self.depth - 2 else main_z + 1
            self.add_partition('z', z_pos, (main_x, 0), (self.width, self.height))
        
        # Top region (0 to main_z)
        if main_z > 4:
            x_pos = random.randint(2, main_x - 1) if main_x > 3 else main_x // 2
            self.add_partition('x', x_pos, (0, 0), (self.height, main_z))
        
        # Bottom region (main_z to depth)
        remaining_depth = self.depth - main_z
        if remaining_depth > 4:
            x_pos = random.randint(main_x + 1, self.width - 1) if main_x < self.width - 2 else main_x + 1
            self.add_partition('x', x_pos, (0, main_z), (self.height, self.depth))
    
    def _create_nested_compartments(self):
        """Create nested/recursive compartments"""
        # Divide into 3 main regions along X
        x1 = random.randint(2, 4)
        x2 = random.randint(6, 8)
        
        self.add_partition('x', x1, (0, 0), (self.height, self.depth))
        self.add_partition('x', x2, (0, 0), (self.height, self.depth))
        
        # Left region (0 to x1): divide with Z-partitions
        z_mid = random.randint(2, self.depth - 2)
        self.add_partition('z', z_mid, (0, 0), (x1, self.height))
        
        # Middle region (x1 to x2): divide with Z-partitions
        if x2 - x1 > 3:
            z_pos = random.randint(z_mid + 1, self.depth - 1) if z_mid < self.depth - 2 else (z_mid + self.depth) // 2
            self.add_partition('z', z_pos, (x1, 0), (x2, self.height))
        
        # Add one more subdivision for complexity
        if random.random() < 0.7:
            corner_x = random.randint(1, 2)
            corner_z = random.randint(1, 3)
            self.add_partition('x', corner_x, (0, 0), (self.height, corner_z))
            self.add_partition('z', corner_z, (0, 0), (corner_x, self.height))
    
    def find_compartments_flood_fill(self) -> List[Compartment]:
        """Find compartments - partitions are boundaries, not blocked cells"""
        grid = np.zeros((self.width, self.height, self.depth), dtype=int)
        
        compartments = []
        region_id = 1
        
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    if grid[x, y, z] == 0:  # Unvisited space
                        region_coords = self._flood_fill_with_partition_boundaries(grid, x, y, z, region_id)
                        if region_coords:
                            xs, ys, zs = zip(*region_coords)
                            min_coords = (min(xs), min(ys), min(zs))
                            max_coords = (max(xs) + 1, max(ys) + 1, max(zs) + 1)
                            compartment = Compartment(min_coords, max_coords)
                            compartments.append(compartment)
                            region_id += 1
        
        return compartments
    
    def _flood_fill_with_partition_boundaries(self, grid: np.ndarray, start_x: int, start_y: int, start_z: int, fill_value: int) -> List[Tuple[int, int, int]]:
        """Flood fill that stops at partition boundaries"""
        region_coords = []
        stack = [(start_x, start_y, start_z)]
        
        while stack:
            x, y, z = stack.pop()
            
            if x < 0 or x >= self.width or y < 0 or y >= self.height or z < 0 or z >= self.depth:
                continue
                
            if grid[x, y, z] != 0:
                continue
            
            grid[x, y, z] = fill_value
            region_coords.append((x, y, z))
            
            neighbors = [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]
            
            for nx, ny, nz in neighbors:
                if self._can_move_to(x, y, z, nx, ny, nz):
                    stack.append((nx, ny, nz))
        
        return region_coords
    
    def _can_move_to(self, from_x: int, from_y: int, from_z: int, to_x: int, to_y: int, to_z: int) -> bool:
        """Check if we can move from one cell to another (no partition blocking)"""
        if to_x < 0 or to_x >= self.width or to_y < 0 or to_y >= self.height or to_z < 0 or to_z >= self.depth:
            return False
        
        for partition in self.partitions:
            if partition.axis == 'x':
                if (from_x < partition.position <= to_x or to_x < partition.position <= from_x):
                    if (partition.start[0] <= from_y < partition.end[0] and 
                        partition.start[1] <= from_z < partition.end[1]):
                        return False
            elif partition.axis == 'z':
                if (from_z < partition.position <= to_z or to_z < partition.position <= from_z):
                    if (partition.start[0] <= from_x < partition.end[0] and 
                        partition.start[1] <= from_y < partition.end[1]):
                        return False
        
        return True
    
    def _assign_colors(self):
        """Assign random colors to compartments"""
        color_names = list(COLORS.keys())
        random.shuffle(color_names)
        
        for i, compartment in enumerate(self.compartments):
            color_name = color_names[i % len(color_names)]
            compartment.color = COLORS[color_name]

class ShapeGenerator:
    """Generate various 3D shapes for the fitting task"""
    
    def __init__(self, scale: float = 0.1):
        self.scale = scale  # Scale factor matching cupboard
        
    def generate_shape(self, shape_type: str = None, max_dimension: float = None) -> Shape3D:
        """Generate a random shape of specified type"""
        if shape_type is None:
            shape_type = random.choice(['cube', 'cylinder', 'sphere', 'elongated_cube', 'truncated_cone'])
        
        if max_dimension is None:
            max_dimension = 0.25  # Maximum dimension in physical units
        
        if shape_type == 'cube':
            size = random.uniform(0.08, min(max_dimension, 0.25))
            return Shape3D('cube', (size, size, size))
            
        elif shape_type == 'elongated_cube':
            width = random.uniform(0.06, min(max_dimension * 0.7, 0.18))
            height = random.uniform(0.08, min(max_dimension, 0.28))
            depth = random.uniform(0.06, min(max_dimension * 0.7, 0.18))
            return Shape3D('elongated_cube', (width, height, depth))
            
        elif shape_type == 'cylinder':
            radius = random.uniform(0.04, min(max_dimension * 0.5, 0.12))
            height = random.uniform(0.06, min(max_dimension, 0.28))
            return Shape3D('cylinder', (radius, height, 0))
            
        elif shape_type == 'sphere':
            radius = random.uniform(0.04, min(max_dimension * 0.5, 0.12))
            return Shape3D('sphere', (radius, 0, 0))
            
        elif shape_type == 'truncated_cone':
            radius = random.uniform(0.05, min(max_dimension * 0.5, 0.13))
            height = random.uniform(0.08, min(max_dimension, 0.25))
            return Shape3D('truncated_cone', (radius, height, 0))
        
        # Default to cube
        size = random.uniform(0.08, min(max_dimension, 0.25))
        return Shape3D('cube', (size, size, size))

class FittingTaskGenerator:
    """Generate fitting tasks with shapes and compartments"""
    
    def __init__(self, cupboard: CupboardPartitioner, scale: float = 0.1):
        self.cupboard = cupboard
        self.scale = scale
        self.shapes = []
        self.shape_generator = ShapeGenerator(scale)
        
    def generate_fitting_task(self) -> List[Shape3D]:
        """Generate a fitting task with 5-8 shapes that have limited placement options"""
        num_shapes = 0
        self.shapes = []
        
        # Sort compartments by volume (largest first)
        # Attach original indexes to each compartment
        compartments_with_index = list(enumerate(self.cupboard.compartments))

        # Sort by volume (compartment is at index 1 in the tuple)
        sorted_with_index = sorted(compartments_with_index, key=lambda x: x[1].volume, reverse=True)

        # Extract sorted compartments and their original indices
        sorted_compartments = [comp for idx, comp in sorted_with_index]
        original_indices = [idx for idx, comp in sorted_with_index]
        
        print(f"\n=== GENERATING FITTING TASK ===")
        print(f"Target shapes: {num_shapes}")
        print(f"Available compartments: {len(sorted_compartments)}")
        
        # Calculate compartment physical dimensions
        comp_dimensions = []
        for comp in sorted_compartments:
            phys_w = (comp.max_x - comp.min_x) * self.scale
            phys_h = (comp.max_y - comp.min_y) * self.scale 
            phys_d = (comp.max_z - comp.min_z) * self.scale
            comp_dimensions.append((phys_w, phys_h, phys_d))
            print(f"Compartment {len(comp_dimensions)}: {phys_w:.3f} × {phys_h:.3f} × {phys_d:.3f}")
        
        # Strategy: Create shapes that can only fit in specific compartments
        placement_options = {}
        
        for i in range(len(sorted_compartments)):
            # if i != 0 and random.random() < 0.15: continue

            # Create a shape with deliberately constrained fitting options
            shape = self._generate_strategically_constrained_shape(sorted_compartments, comp_dimensions, i)
            
            self.shapes.append(shape)
            placement_options[i] = shape
        
        # Now assign shapes to compartments using a greedy approach
        self._assign_shapes_to_compartments(placement_options, sorted_compartments)
        
        # Assign colors to shapes
        self._assign_shape_colors()
        
        print(f"✅ Generated {len(self.shapes)} shapes with constrained placement")
        return self.shapes, placement_options, sorted_compartments, original_indices
    
    def _generate_strategically_constrained_shape(self, sorted_compartments: List[Compartment], comp_dimensions: List[Tuple[float, float, float]], shape_index: int) -> Tuple[Optional[Shape3D], List[int]]:
        """Generate a shape that can only fit in specific compartments by design"""


        # Choose shape type with better distribution
        # shape_types = ['cylinder', 'elongated_cube', 'truncated_cone', 'rectangular_box']
        # weights = [0.25, 0.25, 0.25, 0.25]  # Favor more interesting shapes
        # shape_type = random.choices(shape_types, weights=weights)[0]
        shape_type = 'elongated_cube'
        
        shape = None
        max_retries = 10
        minimum = 0.95
        maximum = 0.98
        i = 0

        while shape is None or shape.check_unique(sorted_compartments, comp_dimensions, shape_index) == False and i < max_retries: 

            minimum = min(0.95, minimum+0.05)
            maximum = min(0.99, maximum+0.01)
            minimum = min(maximum-0.01, minimum)

            shape = self._create_shape_for_compartments(shape_type, comp_dimensions, shape_index, minimum, maximum)
            i += 1

        return shape
    

    def _create_shape_for_compartments(self, shape_type: str, comp_dimensions: List[Tuple[float, float, float]], shape_index, minimum=0.7, maximum=0.95) -> Optional[Shape3D]:
        """Create a shape sized to fit in specific compartments"""        

        usable_width = comp_dimensions[shape_index][0] - 0.02
        usable_height = comp_dimensions[shape_index][1] 
        usable_depth = comp_dimensions[shape_index][2] - 0.02
        
        print(f"    Creating {shape_type} with available space: {usable_width:.3f} × {usable_height:.3f} × {usable_depth:.3f}")
        
        if shape_type == 'rectangular_box' or shape_type == 'elongated_cube':
            width = usable_width * random.uniform(minimum, maximum)
            height = usable_height * random.uniform(minimum, maximum)
            depth = usable_depth * random.uniform(minimum, maximum)

            return Shape3D('elongated_cube', (width, height, depth))
        
        elif shape_type == 'cylinder':           
            max_radius = min(usable_width, usable_depth) / 2 * random.uniform(minimum, maximum)
            height = usable_height * random.uniform(minimum, maximum)
            
            return Shape3D('cylinder', (max_radius, height, 0))
        
        elif shape_type == 'truncated_cone':
            # Truncated cone: base radius constrained, height independent
            max_radius = min(usable_width, usable_depth) / 2 * random.uniform(minimum, maximum)
            height = usable_height * random.uniform(0.5, 0.95)

            return Shape3D('truncated_cone', (max_radius, height, 0))       

        return None  

    
    def _assign_shapes_to_compartments(self, placement_options: Dict, sorted_compartments: List[Compartment]):
        """Assign shapes to compartments ensuring a valid solution exists"""
        
        compartment_occupied = [False] * len(sorted_compartments)
        
        for comp_idx, shape in placement_options.items():
            compartment = sorted_compartments[comp_idx]
            self._position_shape_in_compartment(shape, compartment)
                
            compartment.occupied_objects.append(shape)
            shape.compartment_id = comp_idx

    
    def _position_shape_in_compartment(self, shape: Shape3D, compartment: Compartment):
        """Position a shape within a compartment ensuring it doesn't cross partition boundaries"""
        # Convert compartment bounds to physical coordinates
        comp_min_x = compartment.min_x * self.scale
        comp_min_y = compartment.min_y * self.scale
        comp_min_z = compartment.min_z * self.scale
        comp_max_x = compartment.max_x * self.scale
        comp_max_y = compartment.max_y * self.scale
        comp_max_z = compartment.max_z * self.scale
        
        print(f"    Positioning {shape.shape_type} in compartment bounds: ({comp_min_x:.3f},{comp_min_y:.3f},{comp_min_z:.3f}) to ({comp_max_x:.3f},{comp_max_y:.3f},{comp_max_z:.3f})")
        
        # Calculate shape bounding box requirements
        if shape.shape_type in ['cube', 'elongated_cube', 'rectangular_box']:
            w, h, d = shape.dimensions
            
            shape.dimensions = w, h, d  # Update shape with best orientation
            
            # Ensure shape fits with margin
            margin = 0.005  # Small safety margin
            available_x = comp_max_x - comp_min_x - w - 2*margin
            available_y = comp_max_y - comp_min_y - h - 2*margin
            available_z = comp_max_z - comp_min_z - d - 2*margin
            
            if available_x < 0 or available_y < 0 or available_z < 0:
                print(f"    WARNING: Shape {shape.dimensions} too big for compartment!")
                # Force fit by placing at compartment center
                pos_x = comp_min_x + (comp_max_x - comp_min_x - w) / 2
                pos_y = comp_min_y + (comp_max_y - comp_min_y - h) / 2
                pos_z = comp_min_z + (comp_max_z - comp_min_z - d) / 2
            else:
                # Random position within safe bounds
                pos_x = comp_min_x + margin
                pos_y = comp_min_y + margin
                pos_z = comp_min_z + margin
            
        elif shape.shape_type in ['cylinder', 'truncated_cone']:
            r, h, _ = shape.dimensions
            
            # Decide orientation: vertical (default) or horizontal
            vertical_fits = (2*r <= comp_max_x - comp_min_x and h <= comp_max_y - comp_min_y and 2*r <= comp_max_z - comp_min_z)
            horizontal_fits = (h <= comp_max_x - comp_min_x and 2*r <= comp_max_y - comp_min_y and 2*r <= comp_max_z - comp_min_z)
            
            if vertical_fits and (not horizontal_fits or random.choice([True, False])):
                # Vertical orientation
                margin = 0.005
                available_x = comp_max_x - comp_min_x - 2*r - 2*margin
                available_y = comp_max_y - comp_min_y - h - 2*margin
                available_z = comp_max_z - comp_min_z - 2*r - 2*margin
                
                pos_x = comp_min_x + r + margin
                pos_y = comp_min_y + r + margin
                pos_z = comp_min_z + r + margin
            else:
                # Horizontal orientation - lying on side
                margin = 0.005
                available_x = comp_max_x - comp_min_x - h - 2*margin
                available_y = comp_max_y - comp_min_y - 2*r - 2*margin
                available_z = comp_max_z - comp_min_z - 2*r - 2*margin
                
                pos_x = comp_min_x + r + margin
                pos_y = comp_min_y + r + margin
                pos_z = comp_min_z + r + margin
                
        elif shape.shape_type == 'sphere':
            r, _, _ = shape.dimensions
            margin = 0.005
            available_x = comp_max_x - comp_min_x - 2*r - 2*margin
            available_y = comp_max_y - comp_min_y - 2*r - 2*margin
            available_z = comp_max_z - comp_min_z - 2*r - 2*margin
            
            pos_x = comp_min_x + r + margin + random.uniform(0, max(0, available_x))
            pos_y = comp_min_y + r + margin + random.uniform(0, max(0, available_y))
            pos_z = comp_min_z + r + margin + random.uniform(0, max(0, available_z))
        
        # Clamp to compartment bounds as final safety check
        pos_x = max(comp_min_x, min(pos_x, comp_max_x))
        pos_y = max(comp_min_y, min(pos_y, comp_max_y))
        pos_z = max(comp_min_z, min(pos_z, comp_max_z))
        
        shape.position = (pos_x, pos_y, pos_z)
        print(f"    → Final position: ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})")
    
    def _assign_shape_colors(self):
        """Assign random colors to shapes"""
        shape_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'yellow', 'cyan', 'gray', 'brown']
        random.shuffle(shape_colors)
        
        for i, shape in enumerate(self.shapes):
            color_name = shape_colors[i % len(shape_colors)]
            shape.color = COLORS[color_name]


import os
import random
from pathlib import Path

class EnhancedOBJGenerator:
    """Generate OBJ files for cupboard with random asset shapes"""
    
    def __init__(self, cupboard: CupboardPartitioner, shapes: List[Shape3D], scale: float = 0.1, asset_folder: str = "assets_simple"):
        self.cupboard = cupboard
        self.shapes = shapes
        self.asset_folder = asset_folder
        self.scale = scale
        self.vertices = []
        self.faces = []
        self.materials = {}
        self.face_materials = []
        self.used_colors = set()
        
        # Load random assets for each shape
        self.shape_assets = self._load_random_assets_no_duplicates()

    def _get_unique_color(self) -> Tuple[float, float, float]:
        """Get a unique color that hasn't been used yet"""
        available_colors = list(COLORS.values())
        
        # Remove already used colors
        unused_colors = [color for color in available_colors if color not in self.used_colors]
        
        if not unused_colors:
            # If all colors are used, start reusing them
            print("Warning: All colors used, reusing colors")
            self.used_colors.clear()
            unused_colors = available_colors
        
        # Pick a random unused color
        chosen_color = random.choice(unused_colors)
        self.used_colors.add(chosen_color)
        
        return chosen_color
    
    def generate_obj(self, filename: str):
        """Generate complete OBJ file with cupboard and shapes"""
        self._generate_base_structure()
        self._generate_partitions()
        self._generate_back_panels()
        self._generate_shapes()
        
        # Write files
        self._write_obj_file(filename)
        mtl_filename = filename.replace('.obj', '.mtl')
        self._write_mtl_file(mtl_filename)
        
        json_filename = filename.replace('.obj', '_task_data.json')
        self.export_task_data(json_filename)
        
        print(f"Generated: {filename}, {mtl_filename}, and {json_filename}")
        return filename, mtl_filename

    
    def _generate_base_structure(self):
        """Generate the base cupboard structure"""
        # Floor
        self._add_box((0, 0, 0), 
                     (self.cupboard.width * self.scale, self.cupboard.height * self.scale, 0.02), 
                     "wood_floor")
        
        # Walls
        self._add_box((0, 0, 0), 
                     (0.02, self.cupboard.height * self.scale, self.cupboard.depth * self.scale), 
                     "wood_left")
        
        self._add_box((self.cupboard.width * self.scale - 0.02, 0, 0), 
                     (0.02, self.cupboard.height * self.scale, self.cupboard.depth * self.scale), 
                     "wood_right")
        
        self._add_box((0, 0, self.cupboard.depth * self.scale - 0.02), 
                     (self.cupboard.width * self.scale, self.cupboard.height * self.scale, 0.02), 
                     "wood_top")
    
    def _generate_partitions(self):
        """Generate partition boards"""
        board_thickness = 0.01
        
        for i, partition in enumerate(self.cupboard.partitions):
            material_name = f"partition_{i}"
            self.materials[material_name] = WOOD_COLOR
            
            if partition.axis == 'x':
                x = partition.position * self.scale
                y_start = partition.start[0] * self.scale
                y_end = partition.end[0] * self.scale
                z_start = partition.start[1] * self.scale
                z_end = partition.end[1] * self.scale
                
                self._add_box(
                    (x - board_thickness/2, y_start, z_start),
                    (board_thickness, y_end - y_start, z_end - z_start),
                    material_name
                )
            
            elif partition.axis == 'z':
                x_start = partition.start[0] * self.scale
                x_end = partition.end[0] * self.scale
                y_start = partition.start[1] * self.scale
                y_end = partition.end[1] * self.scale
                z = partition.position * self.scale
                
                self._add_box(
                    (x_start, y_start, z - board_thickness/2),
                    (x_end - x_start, y_end - y_start, board_thickness),
                    material_name
                )
    
    def _generate_back_panels(self):
        """Generate colored back panels for compartments"""
        back_thickness = 0.01
        back_y_position = self.cupboard.height * self.scale - back_thickness
        
        for i, compartment in enumerate(self.cupboard.compartments):
            if compartment.color is None:
                continue
                
            material_name = f"back_panel_{i}"
            self.materials[material_name] = compartment.color
            
            x_start = compartment.min_x * self.scale
            x_end = compartment.max_x * self.scale
            z_start = compartment.min_z * self.scale  
            z_end = compartment.max_z * self.scale
            
            if x_end > x_start and z_end > z_start:
                self._add_box(
                    (x_start, back_y_position, z_start),
                    (x_end - x_start, back_thickness, z_end - z_start),
                    material_name
                )

    def _find_mtl_files(self, subfolder_path: str) -> List[str]:
        """Find all MTL files in a subfolder"""
        mtl_files = []
        try:
            for file in os.listdir(subfolder_path):
                if file.lower().endswith('.mtl'):
                    mtl_files.append(os.path.join(subfolder_path, file))
        except:
            pass
        return mtl_files

    def _fix_texture_paths(self, texture_path: str, asset_subfolder: str, output_dir: str) -> str:
        """Fix texture paths to be relative to the output OBJ location"""
        if not texture_path:
            return texture_path
            
        # Handle different path formats
        texture_path = texture_path.replace('\\', '/')  # Normalize path separators
        
        # If it's already an absolute path or starts with ../, keep it
        if os.path.isabs(texture_path) or texture_path.startswith('../'):
            return texture_path
            
        # Build the relative path from output_dir back to the asset folder
        # output_dir is typically "fitting_tasks"
        # We need to go: fitting_tasks -> . -> asset -> subfolder -> texture
        relative_asset_path = f"../{self.asset_folder}/{asset_subfolder}/{texture_path}"
        
        print(f"    Fixed texture path: {texture_path} -> {relative_asset_path}")
        return relative_asset_path

    def _load_mtl_files(self, subfolder_path: str, subfolder_name: str, output_dir: str = "fitting_tasks") -> Dict[str, Dict]:
        """Load all MTL files in a subfolder and fix texture paths"""
        materials = {}
        
        mtl_files = self._find_mtl_files(subfolder_path)
        if not mtl_files:
            print(f"    No MTL files found in {subfolder_path}")
            return materials
            
        print(f"    Found MTL files: {[os.path.basename(f) for f in mtl_files]}")
        
        for mtl_file in mtl_files:
            try:
                with open(mtl_file, 'r') as f:
                    current_material = None
                    
                    for line in f:
                        line = line.strip()
                        if line.startswith('newmtl '):
                            # New material definition
                            current_material = line.split(' ', 1)[1].strip()
                            materials[current_material] = {}
                        elif current_material and line and not line.startswith('#'):
                            # Material property
                            if ' ' in line:
                                parts = line.split(' ', 1)
                                prop = parts[0]
                                value = parts[1]
                                
                                # Parse different material properties
                                if prop in ['Kd', 'Ka', 'Ks']:  # Colors
                                    try:
                                        rgb = [float(x) for x in value.split()]
                                        if len(rgb) >= 3:
                                            materials[current_material][prop] = rgb[:3]
                                    except:
                                        pass
                                elif prop in ['Ns', 'Ni', 'd', 'Tr']:  # Numeric values
                                    try:
                                        materials[current_material][prop] = float(value)
                                    except:
                                        pass
                                elif prop in ['map_Kd', 'map_Ka', 'map_Ks', 'map_Bump', 'map_Ns', 'norm']:  # Texture maps
                                    # Fix the texture path to be relative to output directory
                                    fixed_path = self._fix_texture_paths(value.strip(), subfolder_name, output_dir)
                                    materials[current_material][prop] = fixed_path
                                elif prop in ['illum']:  # Illumination model
                                    try:
                                        materials[current_material][prop] = int(value)
                                    except:
                                        materials[current_material][prop] = value.strip()
                                else:
                                    materials[current_material][prop] = value.strip()
                
                print(f"    Loaded {len([m for m in materials if materials[m]])} materials from {os.path.basename(mtl_file)}")
                
            except Exception as e:
                print(f"    Error loading MTL file {mtl_file}: {e}")
        
        return materials
    
    def _load_obj_with_materials(self, obj_path: str) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, ...]], List[str]]:
        """Load OBJ file and track which material each face uses"""
        vertices = []
        faces = []
        face_materials = []
        current_material = "default"
        
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
                    elif line.startswith('usemtl '):
                        # Material usage: usemtl material_name
                        current_material = line.split(' ', 1)[1].strip()
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
                            face_materials.append(current_material)
            
            print(f"    Loaded: {len(vertices)} vertices, {len(faces)} faces with materials")
            return vertices, faces, face_materials
            
        except Exception as e:
            print(f"    Error loading {obj_path}: {e}")
            return [], [], []

    def _load_random_assets_no_duplicates(self) -> List[Tuple[List[Tuple[float, float, float]], List[Tuple[int, ...]], str]]:
        """Load random OBJ files from asset subfolders for each shape, ensuring no duplicates"""
        shape_assets = []
        
        # Get all subfolders in asset directory
        if not os.path.exists(self.asset_folder):
            print(f"Asset folder '{self.asset_folder}' not found! Using fallback shapes.")
            for i in range(len(self.shapes)):
                shape_assets.append(self._create_fallback_shape())
            return shape_assets
        
        subfolders = [f for f in os.listdir(self.asset_folder) 
                     if os.path.isdir(os.path.join(self.asset_folder, f))]
        
        if not subfolders:
            print(f"No subfolders found in '{self.asset_folder}'! Using fallback shapes.")
            for i in range(len(self.shapes)):
                shape_assets.append(self._create_fallback_shape())
            return shape_assets
        
        print(f"Found {len(subfolders)} asset folders: {subfolders}")
        
        # Create a randomized list of available assets
        available_assets = subfolders.copy()
        random.shuffle(available_assets)
        
        # If we need more shapes than available assets, repeat some
        if len(self.shapes) > len(available_assets):
            print(f"Warning: Need {len(self.shapes)} shapes but only have {len(available_assets)} unique assets. Some will be reused.")
            while len(available_assets) < len(self.shapes):
                available_assets.extend(subfolders)
            random.shuffle(available_assets)
        
        # Load assets for each shape
        for i, shape in enumerate(self.shapes):
            subfolder = available_assets[i]
            subfolder_path = os.path.join(self.asset_folder, subfolder)
            
            # Find OBJ file (look for any .obj file)
            obj_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.obj')]
            
            if not obj_files:
                print(f"No OBJ files found in {subfolder}, using fallback")
                shape_assets.append(self._create_fallback_shape())
                continue
                
            obj_file = obj_files[0]  # Use the first OBJ file found
            obj_path = os.path.join(subfolder_path, obj_file)
            
            print(f"Loading asset for shape {i+1}: {subfolder}/{obj_file}")
            
            # Load OBJ (ignore materials)
            vertices, faces = self._load_obj_file(obj_path)
            
            if vertices and faces:
                normalized_vertices = self._normalize_and_orient_asset(vertices, shape.dimensions)
                shape_assets.append((normalized_vertices, faces, subfolder))
            else:
                print(f"Failed to load {obj_path}, using fallback")
                shape_assets.append(self._create_fallback_shape())
        
        return shape_assets

    def _normalize_and_orient_asset(self, vertices: List[Tuple[float, float, float]], target_dimensions: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Normalize asset and orient it to match target dimensions"""
        if not vertices:
            return vertices
        
        # Calculate current bounding box
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
        
        # Target dimensions
        target_width, target_height, target_depth = target_dimensions
        target_dims = [target_width, target_height, target_depth]
        
        print(f"    Original dimensions: {curr_width:.3f} × {curr_height:.3f} × {curr_depth:.3f}")
        print(f"    Target dimensions: {target_width:.3f} × {target_height:.3f} × {target_depth:.3f}")
        
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
        
        print(f"    Best orientation: {best_orientation} (score: {best_score:.3f})")
        
        # Apply the best orientation and normalization
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        # Normalize to unit size and apply orientation
        max_current_dim = max(current_dims)
        scale_factor = 1.0 / max_current_dim if max_current_dim > 0 else 1.0
        
        normalized_vertices = []
        for vertex in vertices:
            # Center the vertex
            centered = [
                vertex[0] - center_x,
                vertex[1] - center_y, 
                vertex[2] - center_z
            ]
            
            # Apply orientation (reorder axes)
            oriented = [
                centered[best_orientation[0]],
                centered[best_orientation[1]],
                centered[best_orientation[2]]
            ]
            
            # Scale to unit size
            normalized = [coord * scale_factor for coord in oriented]
            normalized_vertices.append(tuple(normalized))
        
        return normalized_vertices
    
    def _load_obj_with_materials(self, obj_path: str) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, ...]], List[str]]:
        """Load OBJ file and track which material each face uses"""
        vertices = []
        faces = []
        face_materials = []
        current_material = "default"
        
        try:
            with open(obj_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):
                        # Vertex line: v x y z
                        parts = line.split()
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append((x, y, z))
                    elif line.startswith('usemtl '):
                        # Material usage: usemtl material_name
                        current_material = line.split(' ', 1)[1].strip()
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
                            face_materials.append(current_material)
            
            print(f"    Loaded: {len(vertices)} vertices, {len(faces)} faces with materials")
            return vertices, faces, face_materials
            
        except Exception as e:
            print(f"    Error loading {obj_path}: {e}")
            return [], [], []

    def _create_fallback_shape(self) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, ...]], str, List[str], Dict]:
        """Create a fallback shape when asset loading fails"""
        vertices = [
            (-0.4, -0.2, -0.3), (0.4, -0.2, -0.3), (0.4, 0.2, -0.3), (-0.4, 0.2, -0.3),
            (-0.4, -0.2, 0.3), (0.4, -0.2, 0.3), (0.4, 0.2, 0.3), (-0.4, 0.2, 0.3),
        ]
        
        faces = [
            (0, 1, 2), (0, 2, 3),  # Bottom
            (4, 7, 6), (4, 6, 5),  # Top  
            (0, 4, 5), (0, 5, 1),  # Front
            (2, 6, 7), (2, 7, 3),  # Back
            (0, 3, 7), (0, 7, 4),  # Left
            (1, 5, 6), (1, 6, 2),  # Right
        ]
        
        face_materials = ["fallback_material"] * len(faces)
        mtl_materials = {
            "fallback_material": {
                "Kd": [0.7, 0.7, 0.7],
                "Ks": [0.3, 0.3, 0.3],
                "Ns": 50
            }
        }
        
        return vertices, faces, "fallback", face_materials, mtl_materials

    def _add_scaled_asset(self, position: Tuple[float, float, float], target_dimensions: Tuple[float, float, float], 
                         asset_data: Tuple, material: str):
        """Add a scaled asset to the mesh with simple material assignment"""
        vertices, faces, asset_name = asset_data
        x, y, z = position
        target_w, target_h, target_d = target_dimensions
        
        # Calculate scaling and positioning
        if vertices:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            zs = [v[2] for v in vertices]
            
            curr_w = max(xs) - min(xs)
            curr_h = max(ys) - min(ys) 
            curr_d = max(zs) - min(zs)
            
            scale_x = target_w / curr_w if curr_w > 0 else 1.0
            scale_y = target_h / curr_h if curr_h > 0 else 1.0
            scale_z = target_d / curr_d if curr_d > 0 else 1.0
        else:
            scale_x = scale_y = scale_z = 1.0
        
        # Position from corner
        offset_x = target_w / 2
        offset_y = target_h / 2
        offset_z = target_d / 2
        
        adjusted_x = x + offset_x
        adjusted_y = y + offset_y
        adjusted_z = z + offset_z
        
        print(f"    Adding {asset_name}: scale {scale_x:.2f}×{scale_y:.2f}×{scale_z:.2f}")
        
        # Transform and add vertices
        start_vertex = len(self.vertices)
        
        for vertex in vertices:
            vx, vy, vz = vertex
            new_vertex = (
                adjusted_x + vx * scale_x,
                adjusted_y + vy * scale_y,
                adjusted_z + vz * scale_z
            )
            self.vertices.append(new_vertex)
        
        # Add faces with single material
        for face in faces:
            new_face = tuple(start_vertex + vertex_idx for vertex_idx in face)
            self.faces.append(new_face)
            self.face_materials.append(material)
            
    def _generate_shapes(self):
        """Generate shapes using random assets with unique colors"""
        print(f"Generating {len(self.shapes)} shapes...")
        
        for i, shape in enumerate(self.shapes):
            # Get a unique color for this shape
            shape_color = self._get_unique_color()
            
            material_name = f"shape_{i}"
            self.materials[material_name] = shape_color
            
            # Use the pre-loaded asset for this shape
            asset_data = self.shape_assets[i]
            
            print(f"Shape {i+1}: Using color {shape_color} for asset {asset_data[2]}")
            
            self._add_scaled_asset(shape.position, shape.dimensions, asset_data, material_name)
        
        print(f"Generated {len(self.faces)} total faces with {len(self.face_materials)} face materials")

        
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
        
    def _add_box(self, position: Tuple[float, float, float], size: Tuple[float, float, float], material: str):
        """Add a box to the mesh"""
        x, y, z = position
        w, h, d = size
        
        box_vertices = [
            (x, y, z),         # 0: bottom-front-left
            (x+w, y, z),       # 1: bottom-front-right
            (x+w, y+h, z),     # 2: bottom-back-right
            (x, y+h, z),       # 3: bottom-back-left
            (x, y, z+d),       # 4: top-front-left
            (x+w, y, z+d),     # 5: top-front-right
            (x+w, y+h, z+d),   # 6: top-back-right
            (x, y+h, z+d),     # 7: top-back-left
        ]
        
        start_vertex = len(self.vertices)
        self.vertices.extend(box_vertices)
        
        box_faces = [
            (0, 1, 2), (0, 2, 3),  # Bottom
            (4, 7, 6), (4, 6, 5),  # Top
            (0, 4, 5), (0, 5, 1),  # Front
            (2, 6, 7), (2, 7, 3),  # Back
            (0, 3, 7), (0, 7, 4),  # Left
            (1, 5, 6), (1, 6, 2),  # Right
        ]
        
        for face in box_faces:
            face_indices = tuple(start_vertex + i for i in face)
            self.faces.append(face_indices)
            self.face_materials.append(material)
    
    def _add_cylinder(self, position: Tuple[float, float, float], radius: float, height: float, material: str, segments: int = 16):
        """Add a cylinder to the mesh"""
        x, y, z = position
        
        # Generate vertices
        bottom_center = len(self.vertices)
        self.vertices.append((x, y, z))  # Bottom center
        
        top_center = len(self.vertices)
        self.vertices.append((x, y + height, z))  # Top center
        
        # Bottom and top circle vertices
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            dx = radius * math.cos(angle)
            dz = radius * math.sin(angle)
            
            # Bottom circle
            self.vertices.append((x + dx, y, z + dz))
            # Top circle
            self.vertices.append((x + dx, y + height, z + dz))
        
        # Generate faces
        # Bottom faces
        for i in range(segments):
            next_i = (i + 1) % segments
            self.faces.append((bottom_center, 2 + i * 2, 2 + next_i * 2))
            self.face_materials.append(material)
        
        # Top faces
        for i in range(segments):
            next_i = (i + 1) % segments
            self.faces.append((top_center, 2 + next_i * 2 + 1, 2 + i * 2 + 1))
            self.face_materials.append(material)
        
        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            v1 = 2 + i * 2      # Bottom current
            v2 = 2 + next_i * 2  # Bottom next
            v3 = 2 + next_i * 2 + 1  # Top next
            v4 = 2 + i * 2 + 1  # Top current
            
            self.faces.append((v1, v2, v3))
            self.face_materials.append(material)
            self.faces.append((v1, v3, v4))
            self.face_materials.append(material)
    
    def _add_sphere(self, position: Tuple[float, float, float], radius: float, material: str, segments: int = 12):
        """Add a sphere to the mesh"""
        x, y, z = position
        
        # Generate vertices using spherical coordinates
        start_vertex = len(self.vertices)
        
        for i in range(segments + 1):
            phi = math.pi * i / segments  # 0 to π
            for j in range(segments * 2):
                theta = 2 * math.pi * j / (segments * 2)  # 0 to 2π
                
                sphere_x = x + radius * math.sin(phi) * math.cos(theta)
                sphere_y = y + radius * math.cos(phi)
                sphere_z = z + radius * math.sin(phi) * math.sin(theta)
                
                self.vertices.append((sphere_x, sphere_y, sphere_z))
        
        # Generate faces
        for i in range(segments):
            for j in range(segments * 2):
                next_j = (j + 1) % (segments * 2)
                
                # Current ring
                v1 = start_vertex + i * segments * 2 + j
                v2 = start_vertex + i * segments * 2 + next_j
                
                # Next ring
                v3 = start_vertex + (i + 1) * segments * 2 + next_j
                v4 = start_vertex + (i + 1) * segments * 2 + j
                
                # Add two triangles per quad
                self.faces.append((v1, v2, v3))
                self.face_materials.append(material)
                self.faces.append((v1, v3, v4))
                self.face_materials.append(material)
    
    def _add_truncated_cone(self, position: Tuple[float, float, float], radius: float, height: float, material: str, segments: int = 16):
        """Add a truncated cone to the mesh"""
        x, y, z = position
        top_radius = radius * 0.6  # Top is 60% of bottom radius
        
        # Generate vertices
        bottom_center = len(self.vertices)
        self.vertices.append((x, y, z))  # Bottom center
        
        top_center = len(self.vertices)
        self.vertices.append((x, y + height, z))  # Top center
        
        # Bottom and top circle vertices
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            
            # Bottom circle
            dx_bottom = radius * math.cos(angle)
            dz_bottom = radius * math.sin(angle)
            self.vertices.append((x + dx_bottom, y, z + dz_bottom))
            
            # Top circle
            dx_top = top_radius * math.cos(angle)
            dz_top = top_radius * math.sin(angle)
            self.vertices.append((x + dx_top, y + height, z + dz_top))
        
        # Generate faces (similar to cylinder but with different radii)
        # Bottom faces
        for i in range(segments):
            next_i = (i + 1) % segments
            self.faces.append((bottom_center, 2 + i * 2, 2 + next_i * 2))
            self.face_materials.append(material)
        
        # Top faces
        for i in range(segments):
            next_i = (i + 1) % segments
            self.faces.append((top_center, 2 + next_i * 2 + 1, 2 + i * 2 + 1))
            self.face_materials.append(material)
        
        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            v1 = 2 + i * 2      # Bottom current
            v2 = 2 + next_i * 2  # Bottom next
            v3 = 2 + next_i * 2 + 1  # Top next
            v4 = 2 + i * 2 + 1  # Top current
            
            self.faces.append((v1, v2, v3))
            self.face_materials.append(material)
            self.faces.append((v1, v3, v4))
            self.face_materials.append(material)
    
    def _write_obj_file(self, filename: str):
        """Write the OBJ file"""
        with open(filename, 'w') as f:
            f.write("# Cupboard fitting task with shapes - Generated OBJ file\n")
            f.write(f"# {len(self.cupboard.compartments)} compartments, {len(self.shapes)} shapes\n")
            f.write(f"mtllib {Path(filename).stem}.mtl\n\n")
            
            # Vertices
            for vertex in self.vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            f.write("\n")
            
            # Faces grouped by material
            current_material = None
            for i, (face, material) in enumerate(zip(self.faces, self.face_materials)):
                if material != current_material:
                    f.write(f"usemtl {material}\n")
                    current_material = material
                
                face_str = " ".join(str(idx + 1) for idx in face)
                f.write(f"f {face_str}\n")
    
    def _write_mtl_file(self, filename: str):
        """Write the MTL material file"""
        with open(filename, 'w') as f:
            f.write("# Material file for cupboard fitting task\n\n")
            
            # Wood materials
            wood_materials = ["wood_floor", "wood_left", "wood_right", "wood_top"]
            for mat in wood_materials:
                f.write(f"newmtl {mat}\n")
                f.write(f"Kd {WOOD_COLOR[0]:.6f} {WOOD_COLOR[1]:.6f} {WOOD_COLOR[2]:.6f}\n")
                f.write("Ks 0.1 0.1 0.1\n")
                f.write("Ns 10\n\n")
            
            # Other materials
            for material_name, color in self.materials.items():
                f.write(f"newmtl {material_name}\n")
                f.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
                f.write("Ks 0.3 0.3 0.3\n")
                f.write("Ns 50\n\n")
    
    def export_task_data(self, filename: str):
        """Export complete task data to JSON"""
        task_data = {
            "cupboard_dimensions": {
                "width": self.cupboard.width,
                "height": self.cupboard.height, 
                "depth": self.cupboard.depth
            },
            "scale_factor": self.scale,
            "compartments": [],
            "shapes": [],
            "fitting_analysis": self._analyze_fitting_constraints()
        }
        
        # Export compartments
        for i, compartment in enumerate(self.cupboard.compartments):
            comp_data = {
                "id": i + 1,
                "bbox": {
                    "min_x": compartment.min_x, "min_y": compartment.min_y, "min_z": compartment.min_z,
                    "max_x": compartment.max_x, "max_y": compartment.max_y, "max_z": compartment.max_z
                },
                "physical_bbox": {
                    "min_x": compartment.min_x * self.scale, "min_y": compartment.min_y * self.scale, "min_z": compartment.min_z * self.scale,
                    "max_x": compartment.max_x * self.scale, "max_y": compartment.max_y * self.scale, "max_z": compartment.max_z * self.scale
                },
                "volume": compartment.volume,
                "occupied_by": [shape.compartment_id for shape in compartment.occupied_objects] if compartment.occupied_objects else [],
                "color": {
                    "rgb": list(compartment.color) if compartment.color else None
                }
            }
            task_data["compartments"].append(comp_data)
        
        # Export shapes
        for i, shape in enumerate(self.shapes):
            shape_data = {
                "id": i + 1,
                "type": shape.shape_type,
                "dimensions": list(shape.dimensions),
                "position": list(shape.position),
                "compartment_id": shape.compartment_id,
                "color": {
                    "rgb": list(shape.color) if shape.color else None
                },
                "bounding_box": {
                    "min": list(shape.get_bounding_box()[0]),
                    "max": list(shape.get_bounding_box()[1])
                }
            }
            task_data["shapes"].append(shape_data)
        
        with open(filename, 'w') as f:
            json.dump(task_data, f, indent=2)
    
    def _analyze_fitting_constraints(self) -> Dict:
        """Analyze the fitting constraints of the task"""
        analysis = {
            "total_shapes": len(self.shapes),
            "total_compartments": len(self.cupboard.compartments),
            "shape_placement_options": {},
            "solution_uniqueness": "limited"  # Will be analyzed
        }
        
        # Analyze placement options for each shape
        for i, shape in enumerate(self.shapes):
            compatible_compartments = []
            for j, comp in enumerate(self.cupboard.compartments):
                if shape.fits_in_compartment(comp, self.scale):
                    compatible_compartments.append(j + 1)  # 1-indexed
            
            analysis["shape_placement_options"][f"shape_{i+1}"] = {
                "type": shape.shape_type,
                "compatible_compartments": compatible_compartments,
                "constraint_level": "high" if len(compatible_compartments) <= 2 else "medium" if len(compatible_compartments) <= 4 else "low"
            }
        
        return analysis

def visualize_fitting_task(cupboard: CupboardPartitioner, shapes: List[Shape3D], output_dir: str, seed: int):
    """Create a visualization of the fitting task"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Compartments only
    ax1.set_title("Compartments (Top View)", fontsize=14, fontweight='bold')
    
    # Draw cupboard outline
    cupboard_rect = patches.Rectangle((0, 0), cupboard.width, cupboard.depth, 
                                    fill=False, edgecolor='black', linewidth=3)
    ax1.add_patch(cupboard_rect)
    
    # Draw compartments with their colors
    for i, compartment in enumerate(cupboard.compartments):
        color = compartment.color if compartment.color else (0.5, 0.5, 0.5)
        
        rect = patches.Rectangle(
            (compartment.min_x, compartment.min_z),
            compartment.max_x - compartment.min_x,
            compartment.max_z - compartment.min_z,
            facecolor=color, alpha=0.7, edgecolor='black', linewidth=2
        )
        ax1.add_patch(rect)
        
        # Add compartment label
        center_x = (compartment.min_x + compartment.max_x) / 2
        center_z = (compartment.min_z + compartment.max_z) / 2
        ax1.text(center_x, center_z, f'C{i+1}', ha='center', va='center', 
               fontsize=10, weight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Right plot: Compartments + Shapes
    ax2.set_title("Fitting Solution (Top View)", fontsize=14, fontweight='bold')
    
    # Draw cupboard outline
    cupboard_rect2 = patches.Rectangle((0, 0), cupboard.width, cupboard.depth, 
                                     fill=False, edgecolor='black', linewidth=3)
    ax2.add_patch(cupboard_rect2)
    
    # Draw compartments (lighter)
    for i, compartment in enumerate(cupboard.compartments):
        color = compartment.color if compartment.color else (0.5, 0.5, 0.5)
        
        rect = patches.Rectangle(
            (compartment.min_x, compartment.min_z),
            compartment.max_x - compartment.min_x,
            compartment.max_z - compartment.min_z,
            facecolor=color, alpha=0.3, edgecolor='gray', linewidth=1
        )
        ax2.add_patch(rect)
    
    # Draw shapes with their bounding boxes
    scale = 0.1  # Match the scale used in the cupboard
    for i, shape in enumerate(shapes):
        if shape.position:
            # Convert physical position back to grid coordinates
            grid_x = shape.position[0] / scale
            grid_z = shape.position[2] / scale
            
            bbox_min, bbox_max = shape.get_bounding_box()

            bbox_width = (bbox_max[0] - bbox_min[0]) / scale
            bbox_depth = (bbox_max[2] - bbox_min[2]) / scale
            
            # Choose marker style based on shape type
            if shape.shape_type == 'cube':
                marker_shape = 's'  # square
                marker_size = 100
            elif shape.shape_type == 'cylinder':
                marker_shape = 'o'  # circle
                marker_size = 80
            elif shape.shape_type == 'sphere':
                marker_shape = 'o'  # circle
                marker_size = 60
            elif shape.shape_type == 'elongated_cube':
                marker_shape = 'D'  # diamond
                marker_size = 90
            else:  # truncated_cone
                marker_shape = '^'  # triangle
                marker_size = 70
            
            color = shape.color if shape.color else (0.3, 0.3, 0.3)

            # Draw bounding box
            bbox_rect = patches.Rectangle(
                (bbox_min[0] / scale, bbox_min[2] / scale),
                bbox_width, bbox_depth,
                fill=False, edgecolor=color, linewidth=2, linestyle='--'
            )

            ax2.add_patch(bbox_rect)
            
            # Draw shape marker
            ax2.scatter(grid_x, grid_z, c=[color], s=marker_size, marker=marker_shape, 
                       edgecolors='black', linewidth=1, alpha=0.8, zorder=10)
            
            # Add shape label
            ax2.text(grid_x, grid_z + 0.3, f'S{i+1}', ha='center', va='center', 
                   fontsize=8, weight='bold', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Configure both axes
    for ax in [ax1, ax2]:
        ax.set_xlim(0, cupboard.width)
        ax.set_ylim(0, cupboard.depth)
        ax.set_xlabel('Width (X)')
        ax.set_ylabel('Depth (Z)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Add legend for shape types
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Cube'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Cylinder/Sphere'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=10, label='Elongated Cube'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Truncated Cone'),
        plt.Line2D([0], [0], linestyle='--', color='gray', label='Shape Bounding Box')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f'fitting_task_{seed}.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Fitting task visualization saved as '{plot_filename}'")
    return plot_filename

def generate_complete_fitting_task(seed: int = None, output_dir: str = "fitting_tasks") -> Dict:
    """Generate a complete cupboard fitting task with shapes"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🏠 Generating Complete Fitting Task (seed={seed})")
    print("="*60)
    
    # Generate cupboard with partitions
    cupboard = CupboardPartitioner(width=10, height=3, depth=10)
    target_compartments = random.randint(5, 8)
    
    print(f"Generating cupboard with target: {target_compartments} compartments")
    cupboard.generate_partial_partitions(target_compartments)
    
    print(f"✅ Generated {len(cupboard.compartments)} compartments with {len(cupboard.partitions)} partitions")
    
    # Generate fitting task with shapes
    fitting_generator = FittingTaskGenerator(cupboard, scale=0.1)
    shapes, _, _, _ = fitting_generator.generate_fitting_task()
    
    # Generate 3D mesh with cupboard and shapes
    obj_generator = EnhancedOBJGenerator(cupboard, shapes, scale=0.1)
    obj_filename = os.path.join(output_dir, f"fitting_task_{seed or 'random'}.obj")
    obj_file, mtl_file = obj_generator.generate_obj(obj_filename)
    
    # Generate visualization
    viz_filename = visualize_fitting_task(cupboard, shapes, output_dir, seed)
    
    # Create comprehensive task info
    task_info = {
        "seed": seed,
        "cupboard_dimensions": (cupboard.width, cupboard.height, cupboard.depth),
        "num_compartments": len(cupboard.compartments),
        "num_partitions": len(cupboard.partitions),
        "num_shapes": len(shapes),
        "shapes_summary": {},
        "files": {
            "obj": obj_file,
            "mtl": mtl_file,
            "visualization": viz_filename,
            "task_data": obj_file.replace('.obj', '_task_data.json')
        }
    }
    
    # Analyze shape distribution
    shape_types = {}
    for shape in shapes:
        if shape.shape_type not in shape_types:
            shape_types[shape.shape_type] = 0
        shape_types[shape.shape_type] += 1
    
    task_info["shapes_summary"] = shape_types
    
    # Print summary
    print("\n" + "="*60)
    print("COMPLETE FITTING TASK GENERATED")
    print("="*60)
    print(f"Cupboard: {cupboard.width}×{cupboard.height}×{cupboard.depth}")
    print(f"Compartments: {len(cupboard.compartments)}")
    print(f"Shapes to fit: {len(shapes)}")
    print(f"Shape types: {shape_types}")
    print(f"\nGenerated files:")
    for file_type, filepath in task_info["files"].items():
        print(f"  - {file_type}: {filepath}")
    
    print(f"\nShape placement details:")
    for i, shape in enumerate(shapes):
        comp_id = shape.compartment_id
        comp_text = f"Compartment {comp_id + 1}" if comp_id is not None else "Unplaced"
        print(f"  Shape {i+1} ({shape.shape_type}): {comp_text}")
    
    print("="*60)
    
    return task_info

def main():
    """Main function to generate fitting tasks"""
    print("🏠🎲 Cupboard Fitting Task Generator")
    print("Generating cupboards with shapes for fitting challenges...\n")
    
    # Generate several examples
    for i in range(3):
        seed = 2000 + i
        print(f"\n🔄 Generating fitting task {i+1}/3 (seed={seed})")
        
        task_info = generate_complete_fitting_task(seed=seed, output_dir="fitting_tasks")
        
        print(f"✅ Fitting task {i+1} completed!")
      
    print(f"\n🎉 All fitting tasks generated!")
    print("Check the 'fitting_tasks' directory for:")
    print("  - OBJ/MTL files (3D models)")
    print("  - PNG visualizations (top-view layouts)")
    print("  - JSON task data (complete specifications)")
    print("\nYou can view the 3D models in Blender, MeshLab, or any OBJ viewer.")

# if __name__ == "__main__":
#     main()