'''

'''
import sys
import subprocess

import glob
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
from PIL import Image
import torch
import warnings
import sys
sys.path.append("../")

from src.utils.logging import get_logger


warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def get_rot_matrix(axis: int, phi):
    assert axis == 0 or axis == 1 or axis == 2, "Invalid axis"

    if axis == 0:
        rot_matrix = np.array([
            [1, 0,              0],
            [0, np.cos(phi),    -np.sin(phi)],
            [0, np.sin(phi),    np.cos(phi)]
        ])
    elif axis == 1:
        rot_matrix = np.array([
            [np.cos(phi),   0, np.sin(phi)],
            [0,             1, 0],
            [-np.sin(phi),  0, np.cos(phi)]
        ])
    elif axis == 2:
        rot_matrix = np.array([
            [np.cos(phi),   -np.sin(phi), 0],
            [np.sin(phi),   np.cos(phi),  0],
            [0,             0,            1]
        ])
    
    hom_trafo = np.eye(4)
    hom_trafo[:3, :3] = rot_matrix
    return hom_trafo


def get_trans_matrix(axis: int, d):
    assert axis == 0 or axis == 1 or axis == 2, "Invalid axis"

    trans = np.zeros((3,))
    trans[axis] = d

    hom_trafo = np.eye(4)
    hom_trafo[:3, 3] = trans

    return hom_trafo


def call_renderer(cad_path, obj_pose_path, output_dir, 
                  scale_translation=None, blenderproc=False):

    root_dir = os.path.abspath(os.path.dirname(__file__))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if blenderproc:
        blender_proc_path = os.path.join(root_dir, "src","lib3d", "blenderproc.py")
        #command = f"blenderproc run ./src/lib3d/blenderproc.py {cad_path} {obj_pose_path} {output_dir} {0}"
        command = f"blenderproc run {blender_proc_path} {cad_path} {obj_pose_path} {output_dir} {0}"
        os.system(command)
    else:
        current_wd = os.getcwd()
        # command = f"python3 -m src.custom_megapose.call_panda3d {cad_path} {obj_pose_path} {output_dir} {0}"
        #panda3d_path = os.path.join(root_dir, "src", "custom_megapose", "call_panda3d.py")
        #panda3d_path = "src.custom_megapose.call_panda3d"
        panda3d_dir = os.path.join(root_dir, "src","custom_megapose")
        os.chdir(panda3d_dir)
        panda3d_path = "call_panda3d"
        panda3d_args = ["python3",
                        "-m",
                        panda3d_path,
                        cad_path,
                        obj_pose_path,
                        output_dir,
                        "0",
                        "False",
                        str(scale_translation)
                        ]
        try:
            # result = subprocess.run(panda3d_args, 
            #                         check=True, capture_output=True, text=True)

            command = f"python3 -m {panda3d_path} {cad_path} {obj_pose_path} {output_dir} 0 False {str(scale_translation)}"
            os.system(command)
            os.chdir(current_wd)
        except subprocess.CalledProcessError as e:
            print(f"Error occured while running callpanda3d")

    # Check if all images have been rendered
    num_images = len(glob.glob(f"{output_dir}/*.png"))
    if num_images == len(np.load(obj_pose_path)) * 2:
        return True
    else:
        logger.info(f"Found only {num_images} images for  {cad_path} {obj_pose_path} {output_dir}")
        return False
    


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    obj_id = 4
    obj_id_str = str(obj_id).zfill(6)



    root_dir = os.path.abspath(os.path.dirname(__file__))    
    datasets_dir = os.path.join(root_dir, "gigaPose_datasets", "datasets")
    cad_path = os.path.join(datasets_dir, "custom", "models", f"{obj_id_str}.obj")

    output_dir = os.path.join(datasets_dir, "templates", "custom", obj_id_str)

    pose_path = os.path.join(datasets_dir, 
                             "templates", 
                             "custom",
                             'object_poses', 
                             f"{obj_id_str}.npy")
    sys.path.append(root_dir)
    print(f"Rootdir = {root_dir}")
    ret = call_renderer(cad_path, pose_path, output_dir, blenderproc=False)
        


