'''
Script for inference on a single image with mask



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

from src.utils.inference_data import SingleInferenceDataset
import src.utils.bbox as bbox
from src.utils.crop import CropResizePad
from src.utils.logging import get_logger
from src.utils.logging import start_disable_output, stop_disable_output
from src.utils.my_visualization import draw_xyz_axis
from src.utils.batch import BatchedData
import src.megapose.utils.tensor_collection as tc
from src.custom_megapose.template_dataset import TemplateData, TemplateDataset
from src.custom_megapose.transform import Transform, ScaleTransform
from src.dataloader.template import MyTemplateSet


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="test")
def run_inference(cfg: DictConfig):

    ############################################################################
    ############################### Preliminaries ##############################
    ############################################################################
    device = 'cuda'
    debug = True
    # Change if needed
    val_or_test = "val"
    scene_id = 1
    view_id = 0
    obj_id = 27
    obj_instance_id = 0
    mask_ids = []
    dataset_name = "hope" # custom
    transforms = instantiate(cfg.data.test.dataloader.transforms)


    # Config constants
    NUM_TEMPLATES = 162

    scene_id_str = str(scene_id).zfill(6)
    view_id_str = str(view_id).zfill(6)
    obj_id_str = str(obj_id).zfill(6)
    
    datasets_dir = os.path.join("gigaPose_datasets", "datasets")

    # Template dirs
    templates_dir = os.path.join(datasets_dir, "templates", dataset_name)
    desired_template_dir = os.path.join(templates_dir, obj_id_str)
    template_poses_dir = os.path.join(templates_dir, "object_poses")
    template_pose_path = os.path.join(template_poses_dir, (obj_id_str+".npy"))


    ############################################################################
    ############################ Test Template Data ############################
    ############################################################################

    # load the object poses
    #template_poses = np.load(template_pose_path)

    '''
    TemplateData(   label='1', 
                    template_dir='./gigaPose_datasets/datasets/templates//hope/000001', 
                    num_templates=162,
                    TWO_init=<src.custom_megapose.transform.ScaleTransform object at 0x7a484c1fcb20>, 
                    pose_path='./gigaPose_datasets/datasets/templates//hope/object_poses/000001.npy', 
                    unique_id=None, 
                    TWO=None, 
                    box_amodal=None)
    '''
    logger.info("Create Template dataset.")

    template_data = TemplateData(label=str(obj_id), 
                                template_dir=desired_template_dir, 
                                num_templates=NUM_TEMPLATES, 
                                TWO_init=ScaleTransform(1.0),
                                pose_path=template_pose_path)

    template_dataset = TemplateDataset([template_data])

    # template_config = {'dir': '${machine.root_dir}/datasets/templates/', 'level_templates': 1, 'pose_distribution': 'all', 'scale_factor': 1.0, 'num_templates': 162, 'image_name': 'OBJECT_ID/VIEW_ID.png', 'pose_name': 'object_poses/OBJECT_ID.npy'}
    # template_config = {'dir': './gigaPose_datasets/datasets/templates//hope', 'level_templates': 1, 'pose_distribution': 'all', 'scale_factor': 1.0, 'num_templates': 162, 'image_name': 'OBJECT_ID/VIEW_ID.png', 'pose_name': 'object_poses/OBJECT_ID.npy'}
    # root_dir = './gigaPose_datasets/datasets/'

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


    ############################################################################
    ############################### Create Model ###############################
    ############################################################################
    
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

    ############################################################################
    ############################### Do inference ###############################
    ############################################################################

    # Create Dataset & dataloader
    inference_dataset = SingleInferenceDataset(dataset_name, 
                                               val_or_test, 
                                               scene_id, 
                                               view_id, 
                                               obj_id, 
                                               obj_instance_id,
                                               transforms)
    
    dataloader = DataLoader(inference_dataset, batch_size=1)
    ckpt_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(ckpt_path, "gigaPose_datasets/pretrained/gigaPose_v1.ckpt")
    prediction_list = trainer.predict(model, 
                                 dataloaders=dataloader, 
                                 ckpt_path=ckpt_path)

    poses_est = prediction_list[0].pred_poses[0].numpy()

    # Prepare visualization
    rgb_path = os.path.join(inference_dataset.scene_path, 
                            "rgb", (view_id_str + ".png"))
    
    rgb_orig = cv2.imread(rgb_path).astype(np.float32)

    camera_path = os.path.join(inference_dataset.scene_path, "scene_camera.json")
    with open(camera_path, 'r') as file:
            camera_dict = json.load(file)
            camera_K = np.array(camera_dict["0"]["cam_K"]).reshape((3, 3))


    template_poses = np.load(template_pose_path)
    K_render = np.array([572.4114, 0.0, 320, 0.0, 573.57043, 240, 0.0, 0.0, 1.0]).reshape(
        (3, 3))
    
    for idx in range(5):
         idx_str = str(idx).zfill(6)
         render_path = os.path.join(desired_template_dir, idx_str+".png")
         obj_pose = template_poses[idx]

         render = cv2.imread(render_path).astype(np.float32)
         render_draw = draw_xyz_axis(render, obj_pose, scale=40, K=K_render)

         cv2.imshow(f"Render Pose {idx+1}", render_draw)
         cv2.waitKey(0)
         cv2.destroyAllWindows()

    
    for idx in range(poses_est.shape[0]):
        boxed_img = draw_xyz_axis(rgb_orig, poses_est[idx], scale=40, K=camera_K)
        # Resize image
        scaling_factor = 0.6
        new_width = int(boxed_img.shape[0] * scaling_factor)
        new_height = int(boxed_img.shape[1] * scaling_factor)
        new_dim = (new_height, new_width)
        boxed_img = cv2.resize(boxed_img, new_dim, interpolation=cv2.INTER_AREA)

        cv2.imshow(f"Pose number {idx+1}", boxed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    ############################################################################
    ############################ Compare Pred to GT ############################
    ############################################################################
    gt_rot = np.array(inference_dataset.gt_dict["cam_R_m2c"])
    gt_rot = gt_rot.reshape(3, 3)
    gt_trans = np.array(inference_dataset.gt_dict["cam_t_m2c"])
    
    err_rot = np.abs(poses_est[0, :3, :3] - gt_rot)
    err_trans = np.abs(poses_est[0, :3, 3] - gt_trans)

    print(err_rot)
    print(err_trans)


if __name__ == "__main__":
    # I could write a script, that runs inference on multiple objects
    # to then compute the mean inference time and its standard deviation.

    run_inference()