"""
Starts the tracking process

Calls the Stoiber executable
"""

import os
import sys
import subprocess

from src.utils.my_stoiber_helpers import (copy_and_rename, 
                                          get_cam_K_from_yaml, 
                                          update_detector_yaml,
                                          delete_all_contents)


default_model = """%YAML:1.2
model_path: \"INFER_FROM_NAME\"
sphere_radius: 0.5"""


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


if __name__ == "__main__":

    # Model information
    obj_id_list = [1]
    num_obj_list = [3]
    render_scale = [1]
    dataset_name = "custom"

    obj_id = obj_id_list[0]
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
                                    "models", obj_id_str + ".obj")

    # Scene data Stoiber
    img_data_path_og  = os.path.join(stoiber_recording_dir, "scene.png")
    img_dir_og = os.path.dirname(img_data_path_og)

    # Target Folder
    camera_data_dir = os.path.join(datasets_dir, dataset_name, "camera_data")
    img_data_path  = os.path.join(camera_data_dir, "scene", "image", "scene.png")
    img_dir = os.path.dirname(img_data_path)

    
    ############################## Stoiber Arguments ##############################
    arg_obj_id = str(obj_id_list[0])
    arg_num_obj = str(num_obj_list[0])
    arg_body_base_path = stoiber_data_dir
     

    path_stoiber = os.path.join(root_dir, "M3T")
    path_executables = os.path.join(path_stoiber, 
                                    "build_debug", 
                                    "examples")
    path_rs_rgb = os.path.join(path_executables, "my_tracker_rs_rgb")

    ############################## Create Yaml files ##############################
    # Delete the .yaml files
    delete_all_contents(stoiber_data_dir)

    default_body = default_body.replace("INFER_FROM_NAME", template_cad_path)
    for instance_id in range(num_obj_list[0]):
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


    ################################# Run Tracker #################################
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
        print(f"Error occurred while running XMem script:\n{e.stderr}")

    print("End")



