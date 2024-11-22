import os

import open3d as o3d

from my_rendering import call_renderer

# Variables
obj_id = 1
dataset_name = "hope"
obj_id_str = str(obj_id).zfill(6)
obj_name = "obj_" + obj_id_str
custom_dataset_name = "custom_"+dataset_name

root_dir = os.path.dirname(os.path.abspath(__file__))
# Source paths
src_models_dir = os.path.join(root_dir, "gigaPose_datasets",
                          "datasets", dataset_name, "models")
src_obj_file_path = os.path.join(src_models_dir, obj_name+".ply")

# Target paths
tgt_models_dir = os.path.join(root_dir, "gigaPose_datasets",
                              "datasets", custom_dataset_name, "models")
tgt_obj_file_path = os.path.join(tgt_models_dir, obj_name+".ply")

os.makedirs(tgt_models_dir, exist_ok=True)


# Save Colorless meshfile
mesh = o3d.io.read_triangle_mesh(src_obj_file_path)
mesh.triangle_uvs = o3d.utility.Vector2dVector([])
mesh.textures = []
if mesh.has_vertices() and mesh.has_triangles():
    o3d.io.write_triangle_mesh(tgt_obj_file_path, mesh)
else:
    print("OIS OASCH")


# Target render path
templates_dir = os.path.join(root_dir, "gigaPose_datasets",
                          "datasets", "templates", custom_dataset_name)
renderings_dir = os.path.join(templates_dir, obj_id_str)
poses_dir = os.path.join(templates_dir, "object_poses")
tar_renderings_dir = os.path.join(templates_dir, obj_id_str)

# Pose path
pose_path = os.path.join(poses_dir, obj_id_str+".npy")

# Render
rendered = call_renderer(tgt_obj_file_path, pose_path, tar_renderings_dir)

print("WTF?")




