'''

'''
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
        command = f"blenderproc run ./src/lib3d/blenderproc.py {cad_path} {obj_pose_path} {output_dir} {0}"
        os.system(command)
    else:
        # command = f"python3 -m src.custom_megapose.call_panda3d {cad_path} {obj_pose_path} {output_dir} {0}"
        #panda3d_path = os.path.join(root_dir, "src", "custom_megapose", "call_panda3d.py")
        panda3d_path = "src.custom_megapose.call_panda3d"
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
            result = subprocess.run(panda3d_args, 
                                    check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occured while running callpanda3d")

    # Check if all images have been rendered
    num_images = len(glob.glob(f"{output_dir}/*.png"))
    if num_images == len(np.load(obj_pose_path)) * 2:
        return True
    else:
        logger.info(f"Found only {num_images} images for  {cad_path} {obj_pose_path}")
        return False
    


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    root_dir = os.path.abspath(os.path.dirname(__file__))
    pose_path = os.path.join('src', 'lib3d', 'predefined_poses', 'obj_poses_level1.npy')


    if False:
        
        cad_path = os.path.join(os.path.dirname(__file__), os.pardir, 'Koenig_Johannes', 
                                'LabSetup', 'LegoBlock.stl')
        poses = np.load(pose_path)
        
        output_dir = os.path.join(root_dir, "gigaPose_datasets", "datasets", "templates", 
                                  "custom_hope", "000002")
        #output_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'Koenig_Johannes', 
        #                        'LabSetup', 'renderings')
        

        # Update poses to zoom in
        poses[:, :3, 3] /= 3    
        pose_path = os.path.join(os.path.dirname(__file__), os.pardir, 'Koenig_Johannes', 
                                'LabSetup', 'poses.npy')
        np.save(pose_path, poses)
    elif False:
        cad_path = os.path.join(root_dir, "gigaPose_datasets", "datasets", "hope", 
                                "models", "obj_000002.ply")

        output_dir = os.path.join(root_dir, "gigaPose_datasets", "datasets", 
                                  "templates", "custom_hope", "000002")

        poses = np.load(pose_path)
        poses[:, :3, 3] *= 0.4    
        pose_path = os.path.join(os.path.dirname(__file__), os.pardir, 'Koenig_Johannes', 
                                'LabSetup', 'hope_poses.npy')
        np.save(pose_path, poses)

    elif True:
        cad_path = os.path.join(root_dir, "my_renderings",  "LegoBlock.obj")

        output_dir = os.path.join(root_dir, "my_renderings",  "000001_new")

        pose1 = np.eye(4)
        pose2 = np.eye(4)
        pose3 = np.eye(4)
        pose4 = np.eye(4)
        pose_rot_y_90 = get_rot_matrix(1, np.pi/2)
        pose_rot_y_390 = get_rot_matrix(1, 3*np.pi/2)

        pose1[2, 3] = 300

        pose2[2, 3] = 300
        pose2[0, 3] = 30

        pose3[2, 3] = 300
        pose3[1, 3] = 30

        pose4[2, 3] = 600

        pose_rot_y_90[2, 3] = 300
        pose_rot_y_390[2, 3] = 300

        poses = np.array([pose1, pose2, pose3, pose_rot_y_90, pose_rot_y_390])

        pose_path = os.path.join(root_dir, "my_renderings", 'my_poses.npy')
        np.save(pose_path, poses)

    print("Exists:")
    print(f"CAD: {os.path.exists(cad_path)}")
    print(f"Pose: {os.path.exists(pose_path)}")
    ret = call_renderer(cad_path, pose_path, output_dir, blenderproc=True)
        


