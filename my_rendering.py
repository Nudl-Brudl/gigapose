'''

'''
import os

import warnings
import sys

from src.utils.logging import get_logger


warnings.filterwarnings("ignore")
logger = get_logger(__name__)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    obj_id = 1
    scale_translation = None

    scale_translation = str(scale_translation)
    obj_id_str = str(obj_id).zfill(6)


    root_dir = os.path.abspath(os.path.dirname(__file__))    
    datasets_dir = os.path.join(root_dir, "gigaPose_datasets", "datasets")
    cad_path = os.path.join(datasets_dir, "custom", "models", f"{obj_id_str}.obj")

    output_dir = os.path.join(datasets_dir, "templates", "custom", obj_id_str)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    obj_pose_path = os.path.join(datasets_dir, 
                             "templates", 
                             "custom",
                             "object_poses", 
                             "default_poses.npy" #f"{obj_id_str}.npy",
                             )
    
    obj_pose_path_id = os.path.join(datasets_dir, 
                             "templates", 
                             "custom",
                             "object_poses", 
                             f"{obj_id_str}.npy",
                             )
    
    data = np.load(obj_pose_path)
    np.save(obj_pose_path_id, data)
    
    callpanda_path = os.path.join(root_dir, "my_call_panda3d.py")

    command = f"python3 {callpanda_path} {cad_path} {obj_pose_path} {output_dir} 0 False {str(scale_translation)}"

    os.system(command)
        


