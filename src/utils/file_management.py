import os
import shutil

import numpy as np


def get_cam_K_from_yaml(yaml_path):

    with open(yaml_path, 'r') as file:
        for line in file:
            if "f_u:" in line:
                f_u = float(line.split(':')[1])
            if "f_v:" in line:
                f_v = float(line.split(':')[1])
            if "pp_x:" in line:
                pp_x = float(line.split(':')[1])
            if "pp_y:" in line:
                pp_y = float(line.split(':')[1])
    # Create the 3x3 camera calibration matrix (intrinsic matrix)
    cam_K = np.array([
        [f_u,  0,    pp_x],
        [0,    f_v,  pp_y],
        [0,    0,    1]
    ])

    return cam_K


def write_detector_yaml(detector_yaml_path, pose_estimate: np.ndarray):

    pose_meter = np.eye(pose_estimate.shape[0])
    pose_meter[:3, :3] = pose_estimate[:3, :3]
    pose_meter[:3, 3] = pose_estimate[:3, 3] * 1e-3

    with open(detector_yaml_path, 'r+') as file:
        content = file.read()

        # Delete everything after "["
        index = content.find("[")
        file.seek(index + 1)
        file.truncate()

        # Write new values
        for idx, entry in enumerate(pose_meter.flatten()):                
            if idx != 0 and idx % 4 == 0:
                file.write(f"\n")
                file.write(f"      ")
            if idx != 15:
                file.write(f"{entry:.6f}" + ", ")
            else:
                file.write(f"{entry:.6f}")

        file.write("]")


def copy_and_rename(path_og, dir_new, filename_new=None):
    assert os.path.exists(path_og), "Original path does not exist"

    os.makedirs(dir_new, exist_ok=True)
    if filename_new == None:
        path_new = os.path.join(dir_new, "scene.png")
    else:
        _, file_extension = os.path.splitext(path_og)
        path_new = os.path.join(dir_new, filename_new, file_extension)
    
    shutil.copy(path_og, path_new)
    print(f"Image copied to path {path_new}.")


    