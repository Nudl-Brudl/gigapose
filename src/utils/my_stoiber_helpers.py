"""
Functions to facilitate yaml file manipulation for M3T
"""


import os
import shutil

import numpy as np


def get_cam_K_from_yaml(yaml_path):
    ''' Read calibration matrix from .yaml file for M3T'''

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


def update_detector_yaml(detector_yaml_path, pose_estimate: np.ndarray):
    '''Writes pose into a MyDetector .yaml file for M3T'''

    # Convert pose from millimeter to meter
    pose_meter = np.ones_like(pose_estimate)
    pose_meter[:3, :3] = pose_estimate[:3, :3]
    pose_meter[:3, 3] = pose_estimate[:3, 3] *1e-3

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
                file.write(f"         ")
            if idx != 15:
                file.write(f"{entry}" + ", ")
            else:
                file.write(f"{entry}")

        file.write("]")


def copy_and_rename(path_og, dir_new, filename_new=None):
    '''Copies image into new folder and names it "scene.png" '''
    
    assert os.path.exists(path_og), f"Original path: {path_og} does not exist"

    os.makedirs(dir_new, exist_ok=True)
    if filename_new == None:
        path_new = os.path.join(dir_new, "scene.png")
    else:
        _, file_extension = os.path.splitext(path_og)
        path_new = os.path.join(dir_new, filename_new, file_extension)
    
    shutil.copy(path_og, path_new)
    print(f"Image copied to path {path_new}.")



def delete_all_contents(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate through all the contents of the folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)  # Get full path
            # Check if it's a file or directory and delete accordingly
            if os.path.isfile(item_path):
                os.remove(item_path)  # Delete file
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Delete directory
        print(f"All contents of '{folder_path}' have been deleted.")
    else:
        print(f"The folder '{folder_path}' does not exist.")


