'''
Run inference on a single image with depth, mask...



Axes:

0------------ x axis = idx[1]
|
|
|
|
|
y axis = idx[0]
'''

import os
import cv2
import hydra
import json
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import wandb
import warnings

import src.utils.bbox as bbox
from src.utils.crop import CropResizePad
from src.utils.logging import get_logger
from src.utils.logging import start_disable_output, stop_disable_output
from src.utils.batch import BatchedData
import src.megapose.utils.tensor_collection as tc


warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def load_model(cfg):
    '''
    Loads the model in eval mode with its checkpoints
    '''

    model = instantiate(cfg.model)
    logger.info("Model initialized!")

    checkpoint = torch.load(cfg.model.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return model


def retrieve_scene_data(img_type: str, 
                        scene_id: int, 
                        view_id: int,
                        obj_id: int):
    '''
    Retrieve data from a scene
    '''
    
    assert img_type in ["val", "test"], f"Invalid img_type: {img_type}. Must be 'val' or 'test'"

    names = ["rgb", "depth", "mask", "mask_visib"]

    # Create path string where data is stored
    scene_id_str = str(scene_id).zfill(6)
    view_id_str = str(view_id).zfill(6)
    DATA_DIR = os.path.join(os.path.dirname(__file__), 
                            "gigaPose_datasets",
                            "datasets",
                            "hope",
                            img_type,
                            scene_id_str
                            )
    
    # Get Paths
    RGB_PATH = os.path.join(DATA_DIR, "rgb", view_id_str + ".png")
    DEPTH_PATH = os.path.join(DATA_DIR, "depth", view_id_str + ".png")
    JSON_PATH = os.path.join(DATA_DIR, "scene_gt.json")

    # Get rgb depth
    rgb = cv2.imread(RGB_PATH).astype(np.float32) / 255.
    depth = cv2.imread(DEPTH_PATH).astype(np.float32) / 255.
    mask = None
    mask_visib = None

    if img_type == "val":
        # Get index of mask
        with open(JSON_PATH, 'r') as file:
            gt_dict = json.load(file)

        list_obj_dicts = gt_dict[str(scene_id)]

        idx_mask = None
        for idx, obj_dict in enumerate(list_obj_dicts):
            if obj_dict['obj_id'] == obj_id:
                idx_mask = idx
                break
        
        #if idx_mask == None:
        #    raise ValueError(f"Object number {obj_id} is not present in the picture {scene_id_str+"_"+view_id_str}")

        idx_mask_str = str(idx_mask).zfill(6)
        MASK_PATH = os.path.join(DATA_DIR, "mask", view_id_str + "_" + idx_mask_str + ".png")
        MASK_VISIB_PATH = os.path.join(DATA_DIR, "mask_visib", 
                                    view_id_str + "_" + idx_mask_str + ".png")    
        mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        mask_visib = cv2.imread(MASK_VISIB_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

    data_dict = {"rgb": rgb,    "depth": depth, 
                 "mask": mask,  "mask_visib": mask_visib}
    return data_dict


def retrieve_bounding_box(mask: np.ndarray):
    msk_idx = np.where(mask)
    idx0_min = np.min(msk_idx[0])
    idx0_max = np.max(msk_idx[0])
    idx1_min = np.min(msk_idx[1])
    idx1_max = np.max(msk_idx[1])
    return (idx0_min, idx0_max, idx1_min, idx1_max)



@hydra.main(version_base=None, config_path="configs", config_name="test")
def run_inference(cfg: DictConfig):
    
    # Can still add and modify keys of cfg
    OmegaConf.set_struct(cfg, False)

    # Set logger and initialize it
    logger.info("Initializing logger, callbacks and trainer")

    # Set type of image (test, val), scene id, view id, object id
    IMAGE_TYPE = "val"
    SCENE_ID = 1
    VIEW_ID = 0
    OBJECT_ID = 15
    DEBUG = True
    device = "cuda"

    # Retrieve image
    scene_dict = retrieve_scene_data(IMAGE_TYPE, SCENE_ID, VIEW_ID, OBJECT_ID)
    
    # Get object boundaries
    

    if DEBUG:
        cv2.imshow('Image', scene_dict["mask"][798:964, 1200:1318])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Make Bounding Box
    y1, y2, x1, x2 = retrieve_bounding_box(scene_dict["mask"])
    box = torch.tensor([x1, y1, x2, y2], device=device)
    bound_box = bbox.BoundingBox(box)

    # Eliminate all pixels in background
    m_rgb = scene_dict["rgb"] * scene_dict["mask"][:, :, None]
    m_rgb = np.transpose(m_rgb, (2, 0, 1))
    m_rgba = np.concatenate([m_rgb, scene_dict["mask"][None, :, :]], axis=0)

    #Crop image
    crop_function = CropResizePad()
    m_rgba = torch.tensor(m_rgba, device=device)

    # There is a problem with the type of the box. If box is tensor
    # the len of the shape has to be 2, why? no idea
    out = crop_function(bound_box.xyxy_box, m_rgba)
    
    '''
    # Crop Image
    rgba = np.concatenate([scene_dict["rgb"], 
                           scene_dict["mask"][:,:, None]],
                           axis=2)
    rgba = torch.from_numpy(rgba).to("cuda")
    
    crop_function = CropResizePad()
    out = crop_function(np.array([798, 964, 1200, 1318]), rgba)
    '''

    # Create model
    # model = load_model(cfg)

    # Set template data

    # Compute ae features of templates

    # Compute ae features of query
    

if __name__ == "__main__":
    run_inference()