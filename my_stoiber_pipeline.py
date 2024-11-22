import os
import sys
import shutil

import cv2
from hydra import initialize, compose
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import subprocess
from torch.utils.data import DataLoader
import warnings
import yaml

from my_image_capture import capture_scene
from my_rendering import call_renderer
from src.utils.inference_data import PipelineInferenceDataset
from src.utils.logging import get_logger
from src.utils.my_stoiber_helpers import (copy_and_rename, 
                                          get_cam_K_from_yaml, 
                                          update_detector_yaml)
from src.custom_megapose.template_dataset import TemplateData, TemplateDataset
from src.custom_megapose.transform import ScaleTransform
from src.dataloader.template import MyTemplateSet

from src.utils.my_visualization import draw_xyz_axis



if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    warnings.filterwarnings("ignore")
    logger = get_logger(__name__)


    ################################# Variables ###################################
    dataset_name = "custom"
    cad_type = ".obj"
    obj_id = 2
    NUM_TEMPLATES = 162
    obj_name = "LegoBlock"
    show_results = False

    obj_id_str = str(obj_id).zfill(6)


    ################################ Create Paths #################################
    root_dir = os.path.abspath(os.path.dirname(__file__))

    datasets_dir = os.path.join(root_dir, "gigaPose_datasets", "datasets")

    # Stoiber Files
    stoiber_dir = os.path.join(root_dir, "M3T")
    stoiber_recording_dir = os.path.join(stoiber_dir, "my_single_image")
    stoiber_data_dir = os.path.join(stoiber_dir, "data", "my_tracker")

    cam_K_yaml_path = os.path.join(stoiber_recording_dir, "color_camera.yaml")
    detector_yaml_path = os.path.join(stoiber_data_dir, 
                                      "mydetector.yaml")

    # Templates dirs
    templates_dir = os.path.join(datasets_dir, "templates", dataset_name)
    template_dir = os.path.join(templates_dir, obj_id_str)
    template_poses_dir = os.path.join(templates_dir, "object_poses")
    template_pose_path = os.path.join(template_poses_dir, 
                                      (obj_id_str+".npy"))

    template_cad_path = os.path.join(datasets_dir, dataset_name, 
                                    "models", obj_id_str+cad_type)

    # Scene data Stoiber
    img_data_path_og  = os.path.join(stoiber_recording_dir, "scene.png")
    img_dir_og = os.path.dirname(img_data_path_og)

    # Target Folder
    camera_data_dir = os.path.join(datasets_dir, dataset_name, "camera_data")
    img_data_path  = os.path.join(camera_data_dir, "scene", "image", "scene.png")
    img_dir = os.path.dirname(img_data_path)

    # Copy first image to target folder
    copy_and_rename(img_data_path_og, img_dir)

    # XMem paths
    xmem_workspace = os.path.join(root_dir, "workspace")
    xmem_path = os.path.join(root_dir, "..", "Koenig_Johannes", "XMem", 
                             "interactive_demo.py")


    ################################## Configs ####################################
    with initialize(version_base=None, config_path="my_configs"):
        cfg = compose(config_name='test.yaml')


    ################################# Rendering ###################################
    logger.info(f"Check if object {obj_id} has to be rendered...")
    if (not os.path.exists(template_dir) or 
        len(os.listdir(template_dir)) != NUM_TEMPLATES*2):
        logger.info(f"Rendering {obj_id}...")
        call_renderer(template_cad_path, template_pose_path, template_dir)
        logger.info(f"{obj_id} has been rendered.")
    else:
        logger.info(f"Object {obj_id} already rendered...")


    ################################ Create Model #################################
    transforms = instantiate(cfg.data.test.dataloader.transforms)

    logger.info("Create Template dataset...")
    template_data = TemplateData(label=str(obj_id),
                                template_dir=template_dir,
                                num_templates=NUM_TEMPLATES,
                                TWO_init=ScaleTransform(1.0),
                                pose_path=template_pose_path)

    template_dataset = TemplateDataset([template_data])

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


    ############################## Segment Object #################################

    if os.path.exists(xmem_workspace):
        shutil.rmtree(xmem_workspace)

    xmem_args = [
        "python3", 
        xmem_path, 
        "--images", img_dir,
        "--num_objects", "1",
        "--size", "-1",
        "--workspace", xmem_workspace
        ]

    try:
        result = subprocess.run(xmem_args, check=True, 
                                capture_output=True, text=True)
        print(f"XMem script output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running XMem script:\n{e.stderr}")

    # Change value of mask to 1
    rgb_path = os.path.join(xmem_workspace, "images", "scene.png")
    mask_path = os.path.join(xmem_workspace, "masks", "scene.png")

    # Get the camera_parameters
    camera_params = get_cam_K_from_yaml(cam_K_yaml_path)

    inference_dataset = PipelineInferenceDataset(obj_id, rgb_path, mask_path, 
                                                camera_params, transforms)
    
    dataloader = DataLoader(inference_dataset, batch_size=1)
    logger.info("Start Inference...")
    cfg.model.checkpoint_path = "/home/my_gigapose/gigaPose_datasets/pretrained/gigaPose_v1.ckpt"
    prediction_list = trainer.predict(model,
                                dataloaders=dataloader,
                                ckpt_path=cfg.model.checkpoint_path)


    if show_results:
        # Show 5 best results
        poses_est = prediction_list[0].pred_poses[0].numpy()
        rgb_orig = cv2.imread(rgb_path).astype(np.float32)


        K_render = np.array([572.4114, 0.0, 320, 
                             0.0, 573.57043, 240, 
                             0.0, 0.0, 1.0]).reshape(
                (3, 3))
        template_poses = np.load(template_pose_path)
        for idx in range(5):
            idx_used = idx * 10
            idx_str = str(idx_used).zfill(6)
            render_path = os.path.join(template_dir, idx_str+".png")
            obj_pose = template_poses[idx_used]

            render = cv2.imread(render_path).astype(np.float32)
            render_draw = draw_xyz_axis(render, obj_pose, scale=20, K=K_render)

            cv2.imshow(f"Render Pose {idx_used+1}", render_draw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for idx in range(poses_est.shape[0]):
            boxed_img = draw_xyz_axis(rgb_orig, poses_est[idx], 
                                      scale=20, K=camera_params)
            cv2.imshow(f"Pose number {idx+1}", boxed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    logger.info("Inference Done")

    pose_stoiber = prediction_list[0].pred_poses[0, 0].numpy()
    update_detector_yaml(detector_yaml_path, pose_stoiber)
    print("End Initial Pose Detector.")
