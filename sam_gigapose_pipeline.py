''' 
Script to do GigaPose inference on multiple objects

At the moment only multiple objects from the same type can be estimated.
Useful information: GigaPose uses millimeters, Stoiber tracking uses meters.

'''


import os
import sys
import shutil
import subprocess

import argparse
import cv2
from hydra import initialize, compose
from hydra.utils import instantiate
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageTk
import subprocess
import tkinter as tk
import torch
from torch.utils.data import DataLoader
import trimesh
import warnings

from segment_anything import SamPredictor, sam_model_registry

from my_image_capture import capture_scene
from my_rendering import call_renderer
from src.utils.my_stoiber_helpers import (copy_and_rename, 
                                          get_cam_K_from_yaml, 
                                          update_detector_yaml,
                                          delete_all_contents)
from src.utils.inference_data import SameObjectPipepline
from src.utils.logging import get_logger
from src.custom_megapose.template_dataset import TemplateData, TemplateDataset
from src.custom_megapose.transform import ScaleTransform
from src.dataloader.template import MyTemplateSet
from src.utils.my_visualization import draw_xyz_axis, draw_posed_3d_box
from segment_interface import SegmentInterface



default_body = """%YAML:1.2
geometry_path: INFER_FROM_NAME
geometry_unit_in_meter: 1.0
geometry_counterclockwise: 1
geometry_enable_culling: 1
geometry2body_pose: !!opencv-matrix
  rows: 4
  cols: 4
  dt: d
  data: [ 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1 ]"""


default_detector = """%YAML:1.2
link2world_pose: !!opencv-matrix
  rows: 4
  cols: 4
  dt: d
  data: [1.0, 0, 0.0, 0.0, 
         0.0, 1.0, 0.0, 0, 
         0.0, 0.0, 1.0, 0.3, 
         1.0, 1.0, 1.0, 1.0]"""


default_region_modality = """%YAML:1.2
use_adaptive_coverage: 1
reference_contour_length: 0.23
measured_occlusion_threshold: 0.01

# visualize_lines_correspondence: 1"""



def get_key_from_value(my_dict: dict, value: int) -> str:
    key = None
    for k, v in my_dict.items():
        if v == value:
            key = k
            break
    return key



if __name__ == "__main__":
    '''
    Potential arguments
        dataset_name: str 
        obj_list: list of object names 
        num_obj_list: list of number of instances of corresponding objects in obj_list
        just_detection : bool indicates if only detection or also tracking
        render_scale: 



    TODO Create folder structure that makes sense for the pipeline
            Try not to mess with gigapose structure too much
    TODO Push everything relevant onto git
    TODO In Stoiber write code so that camera params are saved to a yaml file
    TODO In Stoiber check the update rate
    TODO In GigaPose check the inference time
    '''


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    warnings.filterwarnings("ignore")
    logger = get_logger(__name__)


    ################################# Variables ##################################
    obj_id_list = [3]
    num_obj_list = [1]
    render_scale_list = [1]
    dataset_name = "custom"

    USE_CAMERA = True
    DO_SEGMENTATION = True
    SHOW_RESTULTS = True
    DO_STOIBER = False

    num_obj = sum(num_obj_list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
            
    
    ########################## Paths and Preliminaries ###########################
    NUM_TEMPLATES = 162
    cad_type = ".obj"
    obj_id_list_str = [str(id).zfill(6) for id in obj_id_list]

    root_dir = os.path.abspath(os.path.dirname(__file__))

    datasets_dir = os.path.join(root_dir, "gigaPose_datasets", "datasets")
    desired_dataset_dir = os.path.join(datasets_dir, dataset_name)

    # Stoiber Files
    stoiber_dir = os.path.join(root_dir, "M3T")
    stoiber_recording_dir = os.path.join(stoiber_dir, "my_single_image")
    stoiber_data_dir = os.path.join(stoiber_dir, "data", "my_tracker")

    cam_K_yaml_path = os.path.join(stoiber_recording_dir, "color_camera.yaml")
    detector_yaml_path = os.path.join(stoiber_data_dir, 
                                      "mydetector.yaml")

    # Templates dirs
    templates_dir = os.path.join(datasets_dir, "templates", dataset_name)
    template_dir_list = [os.path.join(templates_dir, id_str) 
                         for id_str in obj_id_list_str]

    templates_poses_dir = os.path.join(templates_dir, "object_poses")
    templates_default_poses_path = os.path.join(templates_poses_dir, 
                                                "default_poses.npy")
    templates_poses_path_list = [os.path.join(templates_poses_dir, id_str + ".npy") 
                                 for id_str in obj_id_list_str]

    templates_models_dir = os.path.join(datasets_dir, dataset_name, "models")
    templates_cad_path_list = [os.path.join(templates_models_dir, id_str + cad_type) 
                               for id_str in obj_id_list_str]
    
    # SAM paths and config
    sam_checkpt_path = os.path.join(root_dir, 
                                    os.pardir, 
                                    "checkpoints", 
                                    "sam_vit_h_4b8939.pth")
    sam_model_type = "vit_h"
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpt_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    

    # Scene data
    camera_data_dir = os.path.join(datasets_dir, dataset_name, "camera_data")
    img_data_path  = os.path.join(camera_data_dir, "scene", "image", "scene.png")
    img_dir = os.path.dirname(img_data_path)
    

    # Tracker data dir
    tracker_data_dir = os.path.join(root_dir, "my_tracker_data")
    os.makedirs(tracker_data_dir, exist_ok=True)
        
    # Read the name_to_id.json file
    with open(os.path.join(desired_dataset_dir, "name_to_id.json"), 'r') as file:
        name_to_id_dict = json.load(file)

    # ################################## Rendering #################################
    # logger.info("Check if Objects need to be rendered...")
    # for idx, obj_id in enumerate(obj_id_list):

    #     template_cad_path = templates_cad_path_list[idx]
    #     template_dir = template_dir_list[idx]
    #     obj_name = get_key_from_value(name_to_id_dict, obj_id)
    #     render_scale = render_scale_list[idx]

    #     if True:#(not os.path.exists(template_dir) or
    #         #len(os.listdir(template_dir)) != NUM_TEMPLATES*2):
    #         obj_poses = np.load(templates_default_poses_path)
    #         obj_poses[:, :3, 3] *= render_scale
    #         np.save(templates_poses_path_list[idx], obj_poses)

    #         logger.info(f"Rendering {obj_name}...")
    #         call_renderer(template_cad_path, 
    #                       templates_poses_path_list[idx], 
    #                       template_dir, 
    #                       render_scale)
    #         logger.info(f"{obj_name} has been rendered.")
    # logger.info("All object renderings exist.")                    

    ################################# Intrinsics #################################
    if USE_CAMERA:
        camera_params = capture_scene(camera_data_dir)
    else:
        camera_params = np.array([[921.76849365,   0.        , 654.91455078],
                                [  0.        , 920.54547119, 350.65985107],
                                [  0.        ,   0.        ,   1.        ]])


    ################################ Segmentation ################################
    if DO_SEGMENTATION:
        seg_interface = SegmentInterface(sam_checkpt_path, device, img_data_path)
        seg_interface.run()



    ################################## GigaPose ##################################
    with initialize(version_base=None, config_path="my_configs"):
        cfg = compose(config_name='test.yaml')

    transforms = instantiate(cfg.data.test.dataloader.transforms)

    logger.info("Create Template Dataset...")
    template_data_list = [TemplateData(label=str(obj_id_list[idx]),
                                       template_dir=template_dir_list[idx],
                                       num_templates=NUM_TEMPLATES,
                                       TWO_init=ScaleTransform(1.0),
                                       pose_path=templates_poses_path_list[idx])
                                       for idx in range(len(obj_id_list))]
    
    template_dataset = TemplateDataset(template_data_list)

    template_config = {'dir': templates_dir, 
                        'level_templates': 1, 
                        'pose_distribution': 'all', 
                        'scale_factor': 1.0, 
                        'num_templates': NUM_TEMPLATES,
                        'image_name': 'OBJECT_ID/VIEW_ID.png', 
                        'pose_name': 'object_poses/OBJECT_ID.npy'}
    
    template_config = OmegaConf.create(template_config)

    my_template_set = MyTemplateSet(datasets_dir,
                                    dataset_name,
                                    template_dataset,
                                    template_config,
                                    transforms)
    logger.info("Template dataset created!")

    cfg_trainer = cfg.machine.trainer

    if "WandbLogger" in cfg_trainer.logger._target_:
        os.environ["WANDB_API_KEY"] = cfg.user.wandb_api_key
        if cfg.machine.dryrun:
            os.environ["WANDB_MODE"] = "offline"
        logger.info(f"Wandb logger initialized at {cfg.save_dir}")
    elif "TensorBoardLogger" in cfg_trainer.logger._target_:
        tensorboard_dir = f"{cfg.save_dir}/{cfg_trainer.logger.name}"
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Tensorboard logger initialized at {tensorboard_dir}")
    else:
        raise NotImplementedError("Only Wandb and Tensorboard loggers are supported")
        
    # Create dir where results are saved:
    # ./gigaPose_datasets/results/large_1
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Instantiate pytorch_lightning.Trainer with predefined config:
    # max_epochs, accelerator, devices...
    trainer = instantiate(cfg_trainer)
    logger.info("Trainer initialized!")

    model = instantiate(cfg.model)
    model.test_dataset_name = dataset_name
    model.template_datasets = {dataset_name: my_template_set}
    # model.eval()
    logger.info("Model initialized!")


    rgb_path = os.path.join(xmem_workspace, "images", "scene.png")
    mask_path = os.path.join(xmem_workspace, "masks", "scene.png")

    rgb_orig = cv2.imread(rgb_path).astype(np.float32) / 255.
    mask_visib = cv2.imread(mask_path, 
                            cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    
    inference_dataset = SameObjectPipepline(obj_id_list[0], 
                                            num_obj,
                                            rgb_path,
                                            mask_path,
                                            camera_params,
                                            transforms,
                                            "cuda")

    dataloader = DataLoader(inference_dataset, batch_size=1)
    logger.info("Start Inference...")

    cfg.model.checkpoint_path = datasets_dir = os.path.join(root_dir, 
                                                            "gigaPose_datasets", 
                                                            "pretrained",
                                                            "gigaPose_v1.ckpt")
    prediction_list = trainer.predict(model,
                                dataloaders=dataloader,
                                ckpt_path=cfg.model.checkpoint_path)
    logger.info("Inference Done")
    logger.info(f"Pose : {prediction_list[0].pred_poses[0].numpy()[0]}")


    if SHOW_RESTULTS:
        mesh = trimesh.load(templates_cad_path_list[0])
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        to_origin[:3, 3] *= 1000. #* render_scale_list[0]

        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3) * 1000# / render_scale_list[0]
        logger.info(f"bbox: {bbox}\n" + 
                    f"to_origin: {to_origin}")
        
        # Show 5 best results
        for idx_obj in range(num_obj):
            poses_est = prediction_list[idx_obj].pred_poses[0].numpy()
            rgb_orig = cv2.imread(rgb_path).astype(np.float32)

            for idx_pose in range(poses_est.shape[0]):
                obj_pose = poses_est[idx_pose]
                obj_pose_o = obj_pose@np.linalg.inv(to_origin)

                boxed_img = draw_posed_3d_box(camera_params, 
                                              rgb_orig.copy(), 
                                              obj_pose_o, 
                                              bbox)
                boxed_img = draw_xyz_axis(boxed_img, 
                                          obj_pose, 
                                          scale=20, 
                                          K=camera_params)
                cv2.imshow(f"Obj {idx_obj} Pose number {idx_pose+1}", boxed_img/255.)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    ################################## Tracking ##################################
    delete_all_contents(stoiber_data_dir)

    default_body = default_body.replace("INFER_FROM_NAME", template_cad_path)
    arg_body_base_path = stoiber_data_dir
    for instance_id in range(num_obj_list[0]):
        pose_stoiber = prediction_list[instance_id].pred_poses[0, 0].numpy()
        body_name = f"body_{instance_id}"

        body_yaml_path = os.path.join(arg_body_base_path, 
                                      body_name + ".yaml")
        detector_yaml_path = os.path.join(arg_body_base_path,
                                          body_name + "_detector.yaml")
        region_modality_path = os.path.join(arg_body_base_path,
                                            body_name + "_region_modality.yaml")
        
        # Create body files
        with open(body_yaml_path, 'w') as file:
            file.write(default_body)
        # Create detector files
        with open(detector_yaml_path, 'w') as file:
            file.write(default_detector)
        # Create region modality files
        with open(region_modality_path, 'w') as file:
            file.write(default_region_modality)
        update_detector_yaml(detector_yaml_path, pose_stoiber)
    
    ############################## Stoiber Arguments ##############################
    if DO_STOIBER:
        arg_obj_id = str(obj_id_list[0])
        arg_num_obj = str(num_obj_list[0])
        arg_body_base_path = stoiber_data_dir
        

        path_stoiber = os.path.join(root_dir, "M3T")
        path_executables = os.path.join(path_stoiber, 
                                        "build_debug", 
                                        "examples")
        path_rs_rgb = os.path.join(path_executables, "my_tracker_rs_rgb")

        stoiber_args = [
            path_rs_rgb,
            arg_obj_id,
            arg_num_obj,
            arg_body_base_path
        ]

        try:
            result = subprocess.run(stoiber_args, check=True, 
                                    capture_output=True, text=True)
            print(f"Stoiber Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running Stoiber script:\n{e.stderr}")

        print("End")

    print("The End")

    '''
    cd /home/my_gigapose/M3T/build_debug/examples
    ./my_tracker_rs_rgb 2 1 /home/my_gigapose/M3T/data/my_tracker
    '''
