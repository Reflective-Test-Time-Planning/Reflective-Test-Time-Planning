import os
import numpy as np

def parse_obj_file(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('f '):
                face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices), faces

def write_obj_file(file_path, vertices, faces):
    with open(file_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")

def clean_and_normalize_obj(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if not file.endswith('.obj'):
                continue

            obj_path = os.path.join(subdir, file)
            rel_path = os.path.relpath(subdir, input_folder)
            out_dir = os.path.join(output_folder, rel_path)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, file)

            vertices, faces = parse_obj_file(obj_path)
            if len(vertices) == 0 or len(faces) == 0:
                continue

            # Filter: only keep vertices used in at least one face
            used_indices = set(idx for face in faces for idx in face)
            index_map = {}
            new_vertices = []
            for new_idx, old_idx in enumerate(sorted(used_indices)):
                index_map[old_idx] = new_idx
                new_vertices.append(vertices[old_idx])
            new_faces = [[index_map[idx] for idx in face] for face in faces]
            vertices = np.array(new_vertices)

            # Filter: remove outlier vertices
            dists = np.linalg.norm(vertices - np.median(vertices, axis=0), axis=1)
            threshold = np.median(dists) + 2.5 * np.std(dists)
            mask = dists < threshold
            kept_indices = {i for i, keep in enumerate(mask) if keep}
            index_map = {}
            filtered_vertices = []
            for new_idx, old_idx in enumerate(sorted(kept_indices)):
                index_map[old_idx] = new_idx
                filtered_vertices.append(vertices[old_idx])
            filtered_faces = [
                [index_map[idx] for idx in face if idx in kept_indices]
                for face in new_faces
                if all(idx in kept_indices for idx in face)
            ]

            vertices = np.array(filtered_vertices)
            faces = filtered_faces
            if len(vertices) == 0 or len(faces) == 0:
                continue

            # Normalize size and align base to Y = 0
            center_xy = np.mean(vertices[:, [0, 2]], axis=0)
            vertices[:, 0] -= center_xy[0]
            vertices[:, 2] -= center_xy[1]
            min_y = np.min(vertices[:, 1])
            vertices[:, 1] -= min_y
            scale = np.max(np.linalg.norm(vertices, axis=1))
            vertices /= scale

            write_obj_file(out_path, vertices, faces)

if __name__ == "__main__":
    clean_and_normalize_obj("./assets_simple", "./assets_simple_normalize2")
