'''
Provides dataset for inference with GigaPose for bop set
'''

import os

import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import src.megapose.utils.tensor_collection as tc
import src.utils.bbox as bbox



class SingleInferenceDataset(Dataset):
    
    def __init__(self, 
                 set_name: str,
                 val_or_test: str, 
                 scene_id: int, 
                 view_id: int,
                 obj_id: int,
                 obj_instance_id: int,
                 transforms,
                 device='cuda',
                 obj_instance_idxs=[] 
                 ):
        '''
        Initializes InferenceDataset

        Args:
            set_name (str): name of the dataset in which the query image is
            val_or_test (str): is either 'val' or 'test'
            scene_id (int): indicates the scene id of the photo
            view_id (int): indicates the view id of the photo
            obj_id (int): is the label of the object of which the pose should be detected
            obj_instance (int): is the index of the object istance, if more instances of the object of interest are in the image
            transforms (F.Transform): has the crop and normalize transformations
            device (str): device
            obj_instance_idxs (List): is a list of the masks of the object instances
        '''
        
        assert val_or_test in ["test", "val"], "val_or_test is not 'test' or 'val'!"

        self.val_or_test = val_or_test
        self.scene_id = scene_id
        self.view_id = view_id
        self.obj_id = obj_id
        self.obj_instance_id = obj_instance_id
        self.transforms = transforms
        self.obj_instance_idxs = obj_instance_idxs
        self.device = device
        self.gt_dict = {}

        scene_id_str = str(scene_id).zfill(6)
        view_id_str = str(view_id).zfill(6)

        file_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(file_dir, "..", ".."))
        self.datasets_dir = os.path.join(root_dir, "gigaPose_datasets", "datasets")
        self.scene_path = os.path.join(self.datasets_dir, set_name, 
                                       val_or_test, scene_id_str)
        
        
    def __getitem__(self, index) -> dict:

        # Get ground truth if available
        if self.val_or_test == "val":
            gt_dict_path = os.path.join(self.scene_path,
                                        "scene_gt.json")
            
            with open(gt_dict_path, 'r') as file:
                views_gt_dicts = json.load(file)
            view_gt_dicts = views_gt_dicts["0"]

            # Get correspondig mask ids
            self.obj_instance_idxs = [idx for idx, d in enumerate(view_gt_dicts) 
                        if d["obj_id"] == self.obj_id]
            assert len(self.obj_instance_idxs) > 0, f"Object with label {self.obj_id} not in image {str(self.scene_id).zfill(6)}_{str(self.view_id).zfill(6)}."
            self.gt_dict = view_gt_dicts[self.obj_instance_idxs[self.obj_instance_id]]

        # Create query paths
        mask_idx = self.obj_instance_idxs[self.obj_instance_id]
        mask_id_str = str(mask_idx).zfill(6)
        scene_id_str = str(self.scene_id).zfill(6)
        view_id_str = str(self.view_id).zfill(6)

        rgb_path = os.path.join(self.scene_path, "rgb", (view_id_str + ".png"))
        mask_path = os.path.join(self.scene_path, "mask",
                                 (view_id_str + "_" + mask_id_str + ".png"))
        mask_visib_path = os.path.join(self.scene_path, "mask_visib",
                                       (view_id_str + "_" + mask_id_str + ".png"))
        
        # Load data
        rgb = cv2.imread(rgb_path).astype(np.float32) / 255.
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        mask_visib = cv2.imread(mask_visib_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

        # Make Bounding Box
        y1, y2, x1, x2 = self.retrieve_bounding_box(mask_visib)
        box = torch.tensor([[x1, y1, x2, y2]], device=self.device)
        bounding_box = bbox.BoundingBox(box)

        # Eliminate all pixels in background and change to CxHxW
        m_rgb = rgb * mask_visib[:, :, None]
        m_rgb = np.transpose(m_rgb, (2, 0, 1))
        m_rgba = np.concatenate([m_rgb, mask_visib[None, :, :]], axis=0)
        m_rgba = torch.tensor(m_rgba, device=self.device)

        # Crop image & extract crop info       
        query_cropped = self.transforms.crop_transform(bounding_box.xyxy_box, 
                                                       m_rgba[None, :, :, :])

        rgb_cropped = query_cropped["images"][:, :3]
        mask_cropped = query_cropped["images"][:, -1]

        # Get camera data
        camera_path = os.path.join(self.scene_path, "scene_camera.json")

        with open(camera_path, 'r') as file:
            camera_dict = json.load(file)
            camera_K = torch.tensor(camera_dict["0"]["cam_K"], device=self.device)
            camera_K = torch.reshape(camera_K, (3, 3))

        # Get output from cropped data & camera
        output = {
            "tar_img": self.transforms.normalize(rgb_cropped).squeeze(),
            "tar_mask": mask_cropped.squeeze(),
            "tar_K": camera_K,
            "tar_M": query_cropped["M"].squeeze(),
            "tar_label": self.obj_id
        }

        return output
    

    def __len__(self):
        return 1
    
    
    def retrieve_bounding_box(self, mask: np.ndarray):
        """
        Retrieves limits of bounding box given a binary mask

        Axes:
            0------------ x axis = idx[1]
            |
            |
            |
            |
            |
            y axis = idx[0]
        """

        msk_idx = np.where(mask)

        idx0_min = np.min(msk_idx[0])
        idx0_max = np.max(msk_idx[0])
        idx1_min = np.min(msk_idx[1])
        idx1_max = np.max(msk_idx[1])

        return (idx0_min, idx0_max, idx1_min, idx1_max)
    


class PipelineInferenceDataset(Dataset):
    
    def __init__(self,
                 obj_id, 
                 rgb_path: str,
                 mask_path: str,
                 camera_params: np.ndarray,
                 transforms,
                 device='cuda',
                 ):
        '''
        Initializes InferenceDataset

        Args:
            set_name (str): name of the dataset in which the query image is
            obj_id (int): is the label of the object of which the pose should be detected
            transforms (F.Transform): has the crop and normalize transformations
            device (str): device
        '''
        self.obj_id = obj_id
        self.rgb_path = rgb_path
        self.mask_path = mask_path
        self.camera_params = camera_params
        self.transforms = transforms
        self.device = device
        
        
    def __getitem__(self, index) -> dict:
        
        # Load data
        rgb = cv2.imread(self.rgb_path).astype(np.float32) / 255.
        mask_visib = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        mask_visib = np.where(mask_visib > 0.0, 1.0, 0.0).astype(np.float32)
        # Make Bounding Box
        y1, y2, x1, x2 = self.retrieve_bounding_box(mask_visib)
        box = torch.tensor([[x1, y1, x2, y2]], device=self.device)
        bounding_box = bbox.BoundingBox(box)

        # Eliminate all pixels in background and change to CxHxW
        m_rgb = rgb * mask_visib[:, :, None]
        m_rgb = np.transpose(m_rgb, (2, 0, 1))
        m_rgba = np.concatenate([m_rgb, mask_visib[None, :, :]], axis=0)
        m_rgba = torch.tensor(m_rgba, device=self.device)

        # Crop image & extract crop info       
        query_cropped = self.transforms.crop_transform(bounding_box.xyxy_box, 
                                                       m_rgba[None, :, :, :])

        rgb_cropped = query_cropped["images"][:, :3]
        mask_cropped = query_cropped["images"][:, -1]

        # Get camera data

        camera_K = torch.from_numpy(self.camera_params).float().to(self.device)
        # Get output from cropped data & camera
        output = {
            "tar_img": self.transforms.normalize(rgb_cropped).squeeze(),
            "tar_mask": mask_cropped.squeeze(),
            "tar_K": camera_K,
            "tar_M": query_cropped["M"].squeeze(),
            "tar_label": self.obj_id
        }

        return output
    

    def __len__(self):
        return 1
    
    
    def retrieve_bounding_box(self, mask: np.ndarray):
        """
        Retrieves limits of bounding box given a binary mask

        Axes:
            0------------ x axis = idx[1]
            |
            |
            |
            |
            |
            y axis = idx[0]
        """

        msk_idx = np.where(mask)

        idx0_min = np.min(msk_idx[0])
        idx0_max = np.max(msk_idx[0])
        idx1_min = np.min(msk_idx[1])
        idx1_max = np.max(msk_idx[1])

        return (idx0_min, idx0_max, idx1_min, idx1_max)
    

class SameObjectPipepline(Dataset):
    
    def __init__(self,
                 obj_id,
                 num_obj,
                 rgb_path: str,
                 mask_path: str,
                 camera_params: np.ndarray,
                 transforms,
                 device='cuda',
                 ):
        '''
        Initializes InferenceDataset

        Args:
            set_name (str): name of the dataset in which the query image is
            obj_id (int): is the label of the object of which the pose should be detected
            transforms (F.Transform): has the crop and normalize transformations
            device (str): device
        '''
        self.obj_id = obj_id
        self.rgb_path = rgb_path
        self.mask_path = mask_path
        self.camera_params = camera_params
        self.transforms = transforms
        self.device = device
        self.num_obj = num_obj
        
        
    def __getitem__(self, idx) -> dict:
        
        # Load data
        rgb = cv2.imread(self.rgb_path).astype(np.float32) / 255.
        mask_visib = cv2.imread(self.mask_path, 
                                cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        mask_visib = np.where(mask_visib == np.unique(mask_visib)[idx+1], 
                              1.0, 0.0).astype(np.float32)
        # Make Bounding Box
        y1, y2, x1, x2 = self.retrieve_bounding_box(mask_visib)
        box = torch.tensor([[x1, y1, x2, y2]], device=self.device)
        bounding_box = bbox.BoundingBox(box)

        # Eliminate all pixels in background and change to CxHxW
        m_rgb = rgb * mask_visib[:, :, None]
        m_rgb = np.transpose(m_rgb, (2, 0, 1))
        m_rgba = np.concatenate([m_rgb, mask_visib[None, :, :]], axis=0)
        m_rgba = torch.tensor(m_rgba, device=self.device)

        # Crop image & extract crop info       
        query_cropped = self.transforms.crop_transform(bounding_box.xyxy_box, 
                                                       m_rgba[None, :, :, :])

        rgb_cropped = query_cropped["images"][:, :3]
        mask_cropped = query_cropped["images"][:, -1]

        # Get camera data

        camera_K = torch.from_numpy(self.camera_params).float().to(self.device)
        # Get output from cropped data & camera
        output = {
            "tar_img": self.transforms.normalize(rgb_cropped).squeeze(),
            "tar_mask": mask_cropped.squeeze(),
            "tar_K": camera_K,
            "tar_M": query_cropped["M"].squeeze(),
            "tar_label": self.obj_id
        }

        return output
    

    def __len__(self):
        return self.num_obj
    
    
    def retrieve_bounding_box(self, mask: np.ndarray):
        """
        Retrieves limits of bounding box given a binary mask

        Axes:
            0------------ x axis = idx[1]
            |
            |
            |
            |
            |
            y axis = idx[0]
        """

        msk_idx = np.where(mask)

        idx0_min = np.min(msk_idx[0])
        idx0_max = np.max(msk_idx[0])
        idx1_min = np.min(msk_idx[1])
        idx1_max = np.max(msk_idx[1])

        return (idx0_min, idx0_max, idx1_min, idx1_max)
    


class RosSingleInferenceDataset(Dataset):
    
    def __init__(self,
                 obj_type_id: int, 
                 rgb: np.ndarray,
                 mask: np.ndarray,
                 camera_k: np.ndarray,
                 transforms,
                 device='cuda',
                 ):
        '''
        Initializes InferenceDataset

        Args:
            transforms (F.Transform): has the crop and normalize transformations
            device (str): device
        '''
        self.obj_id = obj_type_id
        self.rgb = rgb
        self.mask = mask
        self.camera_k = camera_k
        self.transforms = transforms
        self.device = device
        
        
        
    def __getitem__(self, index) -> dict:
        
        rgb = self.rgb.astype(np.float32) / 255.
        mask_visib = self.mask.astype(np.float32) / 255.
        mask_visib = np.where(mask_visib > 0.0, 1.0, 0.0).astype(np.float32)
        # Make Bounding Box
        y1, y2, x1, x2 = self.retrieve_bounding_box(mask_visib)
        box = torch.tensor([[x1, y1, x2, y2]], device=self.device)
        bounding_box = bbox.BoundingBox(box)

        # Eliminate all pixels in background and change to CxHxW
        m_rgb = rgb * mask_visib[:, :, None]
        m_rgb = np.transpose(m_rgb, (2, 0, 1))
        m_rgba = np.concatenate([m_rgb, mask_visib[None, :, :]], axis=0)
        m_rgba = torch.tensor(m_rgba, device=self.device)

        # Crop image & extract crop info       
        query_cropped = self.transforms.crop_transform(bounding_box.xyxy_box, 
                                                       m_rgba[None, :, :, :])

        rgb_cropped = query_cropped["images"][:, :3]
        mask_cropped = query_cropped["images"][:, -1]

        # Get camera data
        camera_K = torch.from_numpy(self.camera_k)  

        # Get output from cropped data & camera
        output = {
            "tar_img": self.transforms.normalize(rgb_cropped).squeeze(),
            "tar_mask": mask_cropped.squeeze(),
            "tar_K": camera_K,
            "tar_M": query_cropped["M"].squeeze(),
            "tar_label": self.obj_id
        }

        return output
    

    def __len__(self):
        return 1
    
    
    def retrieve_bounding_box(self, mask: np.ndarray):
        """
        Retrieves limits of bounding box given a binary mask

        Axes:
            0------------ x axis = idx[1]
            |
            |
            |
            |
            |
            y axis = idx[0]
        """

        msk_idx = np.where(mask)

        idx0_min = np.min(msk_idx[0])
        idx0_max = np.max(msk_idx[0])
        idx1_min = np.min(msk_idx[1])
        idx1_max = np.max(msk_idx[1])

        return (idx0_min, idx0_max, idx1_min, idx1_max)