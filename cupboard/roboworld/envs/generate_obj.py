import numpy as np
import random
from typing import List, Tuple, Dict
import os
from pathlib import Path

# Color definitions (RGB values 0-1)
COLORS = {
    "blue": (0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
    "orange": (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
    "green": (0.41568627450980394, 0.8, 0.39215686274509803),
    "red": (0.8392156862745098, 0.37254901960784315, 0.37254901960784315),
    "purple": (0.5843137254901961, 0.4235294117647059, 0.7058823529411765),
    "brown": (0.5490196078431373, 0.3803921568627451, 0.23529411764705882),
    "pink": (0.8627450980392157, 0.49411764705882355, 0.7529411764705882),
    "gray": (0.4745098039215686, 0.4745098039215686, 0.4745098039215686),
    "yellow": (0.9098, 0.6784, 0.1373),
    "cyan": (0.2, 0.8, 0.9),
    "lime": (0.5, 1.0, 0.0),
    "magenta": (1.0, 0.0, 1.0)
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
    
    def __repr__(self):
        return f"Compartment({self.min_x},{self.min_y},{self.min_z})->({self.max_x},{self.max_y},{self.max_z}) Vol:{self.volume}"

class CupboardPartitioner:
    def __init__(self, width: int = 10, height: int = 1, depth: int = 10):
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
    
    def generate_partial_partitions(self, target_compartments: int = None) -> List[Partition]:
        """Generate partitions that create properly enclosed compartments"""
        if target_compartments is None:
            target_compartments = random.randint(5, 8)
        
        self.partitions = []
        
        # Strategy: Build compartments systematically to ensure they're enclosed
        if random.choice([True, False]):
            self._create_grid_with_partial_divisions()
        else:
            self._create_nested_compartments()
        
        # Find compartments
        self.compartments = self.find_compartments_flood_fill()
        
        # If we don't have enough compartments, try the other method
        if len(self.compartments) < 4:
            self.partitions = []
            if len(self.partitions) == 0:
                self._create_grid_with_partial_divisions()
            else:
                self._create_nested_compartments()
            self.compartments = self.find_compartments_flood_fill()
        
        # Assign colors to compartments
        self._assign_colors()
        
        return self.partitions
    
    def _create_grid_with_partial_divisions(self):
        """Create a grid-like structure with some partial divisions"""
        main_x = random.randint(3, 7)
        main_z = random.randint(3, 7)
        
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
        """Find compartments using 3D flood fill algorithm"""
        # Create 3D grid
        grid = np.zeros((self.width, self.height, self.depth), dtype=int)
        
        # Mark partition locations as blocked (-1)
        for partition in self.partitions:
            self._mark_partition_in_grid(grid, partition)
        
        # Find connected regions using flood fill
        compartments = []
        region_id = 1
        
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    if grid[x, y, z] == 0:  # Unvisited empty space
                        region_coords = self._flood_fill_3d(grid, x, y, z, region_id)
                        if region_coords:
                            min_coords, max_coords = self._get_region_bounds(region_coords)
                            compartment = Compartment(min_coords, max_coords)
                            compartments.append(compartment)
                            region_id += 1
        
        return compartments
    
    def _mark_partition_in_grid(self, grid: np.ndarray, partition: Partition):
        """Mark partition location in the 3D grid"""
        if partition.axis == 'x':
            for y in range(max(0, partition.start[0]), min(self.height, partition.end[0])):
                for z in range(max(0, partition.start[1]), min(self.depth, partition.end[1])):
                    if 0 <= partition.position < self.width:
                        grid[partition.position, y, z] = -1
        elif partition.axis == 'z':
            for x in range(max(0, partition.start[0]), min(self.width, partition.end[0])):
                for y in range(max(0, partition.start[1]), min(self.height, partition.end[1])):
                    if 0 <= partition.position < self.depth:
                        grid[x, y, partition.position] = -1
        elif partition.axis == 'y':
            for x in range(max(0, partition.start[0]), min(self.width, partition.end[0])):
                for z in range(max(0, partition.start[1]), min(self.depth, partition.end[1])):
                    if 0 <= partition.position < self.height:
                        grid[x, partition.position, z] = -1
    
    def _flood_fill_3d(self, grid: np.ndarray, start_x: int, start_y: int, start_z: int, fill_value: int) -> List[Tuple[int, int, int]]:
        """3D flood fill algorithm"""
        stack = [(start_x, start_y, start_z)]
        region_coords = []
        
        while stack:
            x, y, z = stack.pop()
            
            if (x < 0 or x >= self.width or y < 0 or y >= self.height or 
                z < 0 or z >= self.depth or grid[x, y, z] != 0):
                continue
            
            grid[x, y, z] = fill_value
            region_coords.append((x, y, z))
            
            neighbors = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1)
            ]
            
            for nx, ny, nz in neighbors:
                stack.append((nx, ny, nz))
        
        return region_coords
    
    def _get_region_bounds(self, coords: List[Tuple[int, int, int]]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get bounding box of a region"""
        if not coords:
            return (0, 0, 0), (0, 0, 0)
        
        xs, ys, zs = zip(*coords)
        min_coords = (min(xs), min(ys), min(zs))
        max_coords = (max(xs) + 1, max(ys) + 1, max(zs) + 1)
        
        return min_coords, max_coords
    
    def _assign_colors(self):
        """Assign random colors to compartments"""
        color_names = list(COLORS.keys())
        random.shuffle(color_names)
        
        for i, compartment in enumerate(self.compartments):
            color_name = color_names[i % len(color_names)]
            compartment.color = COLORS[color_name]

class OBJGenerator:
    """Generate OBJ files for the cupboard with colored compartments"""
    
    def __init__(self, cupboard: CupboardPartitioner, scale: float = 0.1):
        self.cupboard = cupboard
        self.scale = scale  # Scale factor for real-world dimensions
        self.vertices = []
        self.faces = []
        self.materials = {}
        self.face_materials = []
        
    def generate_obj(self, filename: str):
        """Generate complete OBJ file with MTL materials"""
        self._generate_base_structure()
        self._generate_partitions()
        self._generate_compartment_markers()
        
        # Write OBJ file
        self._write_obj_file(filename)
        
        # Write MTL file
        mtl_filename = filename.replace('.obj', '.mtl')
        self._write_mtl_file(mtl_filename)
        
        print(f"Generated: {filename} and {mtl_filename}")
        return filename, mtl_filename
    
    def _generate_base_structure(self):
        """Generate the base cupboard structure (floor and back wall)"""
        # Floor (bottom)
        self._add_box(
            (0, 0, 0), 
            (self.cupboard.width * self.scale, self.cupboard.height * self.scale, 0.02), 
            "wood_floor"
        )
        
        # Back wall
        self._add_box(
            (0, 0, 0), 
            (0.02, self.cupboard.height * self.scale, self.cupboard.depth * self.scale), 
            "wood_back"
        )
        
        # Right wall  
        self._add_box(
            (self.cupboard.width * self.scale - 0.02, 0, 0), 
            (0.02, self.cupboard.height * self.scale, self.cupboard.depth * self.scale), 
            "wood_right"
        )
        
        # Top wall
        self._add_box(
            (0, 0, self.cupboard.depth * self.scale - 0.02), 
            (self.cupboard.width * self.scale, self.cupboard.height * self.scale, 0.02), 
            "wood_top"
        )
    
    def _generate_partitions(self):
        """Generate partition boards"""
        board_thickness = 0.01
        
        for i, partition in enumerate(self.cupboard.partitions):
            material_name = f"partition_{i}"
            color = random.choice(list(COLORS.values()))
            self.materials[material_name] = color
            
            if partition.axis == 'x':
                # YZ plane partition
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
                # XY plane partition
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
    
    def _generate_compartment_markers(self):
        """Generate colored markers for each compartment"""
        marker_thickness = 0.005
        
        for i, compartment in enumerate(self.cupboard.compartments):
            if compartment.color is None:
                continue
                
            material_name = f"compartment_{i}"
            self.materials[material_name] = compartment.color
            
            # Create a thin colored plane at the bottom of each compartment
            x_start = compartment.min_x * self.scale + 0.01
            x_end = compartment.max_x * self.scale - 0.01
            y_start = compartment.min_y * self.scale
            y_end = compartment.max_y * self.scale
            z = 0.02  # Just above the floor
            
            if x_end > x_start and y_end > y_start:
                self._add_box(
                    (x_start, y_start, z),
                    (x_end - x_start, y_end - y_start, marker_thickness),
                    material_name
                )
    
    def _add_box(self, position: Tuple[float, float, float], size: Tuple[float, float, float], material: str):
        """Add a box to the mesh"""
        x, y, z = position
        w, h, d = size
        
        # 8 vertices of the box
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
        
        # 6 faces of the box (each face has 2 triangles)
        box_faces = [
            # Bottom face (z = z)
            (0, 1, 2), (0, 2, 3),
            # Top face (z = z+d)
            (4, 7, 6), (4, 6, 5),
            # Front face (y = y)
            (0, 4, 5), (0, 5, 1),
            # Back face (y = y+h)
            (2, 6, 7), (2, 7, 3),
            # Left face (x = x)
            (0, 3, 7), (0, 7, 4),
            # Right face (x = x+w)
            (1, 5, 6), (1, 6, 2),
        ]
        
        for face in box_faces:
            face_indices = tuple(start_vertex + i for i in face)
            self.faces.append(face_indices)
            self.face_materials.append(material)
    
    def _write_obj_file(self, filename: str):
        """Write the OBJ file"""
        with open(filename, 'w') as f:
            # Header
            f.write("# Cupboard fitting task generated OBJ file\n")
            f.write(f"# Generated with {len(self.cupboard.compartments)} compartments\n")
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
                
                # OBJ faces are 1-indexed
                face_str = " ".join(str(idx + 1) for idx in face)
                f.write(f"f {face_str}\n")
    
    def _write_mtl_file(self, filename: str):
        """Write the MTL material file"""
        with open(filename, 'w') as f:
            f.write("# Material file for cupboard fitting task\n\n")
            
            # Wood materials
            f.write("newmtl wood_floor\n")
            f.write(f"Kd {WOOD_COLOR[0]:.6f} {WOOD_COLOR[1]:.6f} {WOOD_COLOR[2]:.6f}\n")
            f.write("Ks 0.1 0.1 0.1\n")
            f.write("Ns 10\n\n")
            
            f.write("newmtl wood_back\n")
            f.write(f"Kd {WOOD_COLOR[0]:.6f} {WOOD_COLOR[1]:.6f} {WOOD_COLOR[2]:.6f}\n")
            f.write("Ks 0.1 0.1 0.1\n")
            f.write("Ns 10\n\n")
            
            f.write("newmtl wood_right\n")
            f.write(f"Kd {WOOD_COLOR[0]:.6f} {WOOD_COLOR[1]:.6f} {WOOD_COLOR[2]:.6f}\n")
            f.write("Ks 0.1 0.1 0.1\n")
            f.write("Ns 10\n\n")
            
            f.write("newmtl wood_top\n")
            f.write(f"Kd {WOOD_COLOR[0]:.6f} {WOOD_COLOR[1]:.6f} {WOOD_COLOR[2]:.6f}\n")
            f.write("Ks 0.1 0.1 0.1\n")
            f.write("Ns 10\n\n")
            
            # Colored materials
            for material_name, color in self.materials.items():
                f.write(f"newmtl {material_name}\n")
                f.write(f"Kd {color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")
                f.write("Ks 0.3 0.3 0.3\n")
                f.write("Ns 50\n\n")

def generate_cupboard_task(seed: int = None, output_dir: str = "output") -> Dict:
    """Generate a complete cupboard fitting task"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate cupboard with partitions
    cupboard = CupboardPartitioner(width=10, height=1, depth=10)
    target_compartments = random.randint(5, 8)
    
    print(f"Generating cupboard with target: {target_compartments} compartments")
    cupboard.generate_partial_partitions(target_compartments)
    
    print(f"✅ Generated {len(cupboard.compartments)} compartments with {len(cupboard.partitions)} partitions")
    
    # Generate 3D mesh
    obj_generator = OBJGenerator(cupboard, scale=0.1)
    obj_filename = os.path.join(output_dir, f"cupboard_task_{seed or 'random'}.obj")
    obj_file, mtl_file = obj_generator.generate_obj(obj_filename)
    
    # Create task description
    task_info = {
        "seed": seed,
        "cupboard_dimensions": (cupboard.width, cupboard.height, cupboard.depth),
        "num_compartments": len(cupboard.compartments),
        "num_partitions": len(cupboard.partitions),
        "compartments": [],
        "partitions": [],
        "files": {
            "obj": obj_file,
            "mtl": mtl_file
        }
    }
    
    # Add compartment details
    for i, comp in enumerate(cupboard.compartments):
        comp_info = {
            "id": i + 1,
            "position": (comp.min_x, comp.min_y, comp.min_z),
            "size": (comp.max_x - comp.min_x, comp.max_y - comp.min_y, comp.max_z - comp.min_z),
            "volume": comp.volume,
            "color": comp.color
        }
        task_info["compartments"].append(comp_info)
    
    # Add partition details
    for i, partition in enumerate(cupboard.partitions):
        part_info = {
            "id": i + 1,
            "axis": partition.axis,
            "position": partition.position,
            "span": {"start": partition.start, "end": partition.end}
        }
        task_info["partitions"].append(part_info)
    
    # Print summary
    print("\n" + "="*60)
    print("CUPBOARD FITTING TASK GENERATED")
    print("="*60)
    print(f"Dimensions: {cupboard.width}×{cupboard.height}×{cupboard.depth}")
    print(f"Compartments: {len(cupboard.compartments)}")
    print(f"Partitions: {len(cupboard.partitions)}")
    print(f"Generated files:")
    print(f"  - {obj_file}")
    print(f"  - {mtl_file}")
    
    print(f"\nCompartment Details:")
    for comp_info in task_info["compartments"]:
        print(f"  C{comp_info['id']}: {comp_info['size']} at {comp_info['position']}, vol={comp_info['volume']}")
    
    print("="*60)
    
    return task_info

def main():
    """Main function to generate multiple tasks"""
    print("🏠 Cupboard Fitting Task Generator")
    print("Generating 3D meshes with colored compartments...\n")
    
    # Generate several examples
    for i in range(3):
        seed = 1000 + i
        print(f"\n🔄 Generating task {i+1}/3 (seed={seed})")
        
        task_info = generate_cupboard_task(seed=seed, output_dir="cupboard_tasks")
        
        print(f"✅ Task {i+1} completed!")
    
    print(f"\n🎉 All tasks generated! Check the 'cupboard_tasks' directory for OBJ files.")
    print("You can view these files in Blender, MeshLab, or any 3D viewer that supports OBJ+MTL.")

if __name__ == "__main__":
    main()