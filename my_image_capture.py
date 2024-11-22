'''
Capture image data with realsense camera

Is able to capture single image but also series of images/videos
'''

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os


def delete_folder_content(folder_path):
    '''Deletes files in given folder_path'''
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
            
    return 0


def get_intrinsics(frame):
    '''Get intrinsics of a specific stream (color or depth)'''
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    calibration_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])
    return calibration_matrix


def capture_scene(data_dir: str, width=1280, height=720):
    '''
    Captures either a single image or image sequence
    
    Saves the captured visual data in the given folder under
    '''
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Set up a directory to save images and videos
    image_dir = os.path.join(data_dir, "scene", "image")
    video_dir = os.path.join(data_dir, "scene", "video")

    # Manage directories for image and video data
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    else:
        delete_folder_content(image_dir)

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    else:
        delete_folder_content(video_dir)

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Color sensor")
        exit(0)

    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Initialize variables for recording and saving
    recording = False
    frame_list = []
    image_counter = 0
    calibration_matrix = np.array([])

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            #if not depth_frame or not color_frame:
            #    continue

            # Convert images to numpy arrays
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply color map on depth image
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)

            # Get user input
            key = cv2.waitKey(1) & 0xFF

            # Capture an image
            if key == ord('c'):
                image_path = os.path.join(image_dir, "scene.png")
                cv2.imwrite(image_path, color_image)
            # Start recording                
            elif key == ord('r') and not recording:
                recording = True
            # Stop recording
            elif key == ord('r') and recording:
                recording = False
                image_counter = 0
            # Quit camera
            elif key == ord('q'):
                print("Exiting image capture...")
                break

            if recording:
                image_path = os.path.join(video_dir, 
                                          f"scene_{str(image_counter).zfill(6)}.png")
                cv2.imwrite(image_path, color_image)
                image_counter += 1
            
            calibration_matrix = get_intrinsics(color_frame)

            

    finally:
        pipeline.stop()

    return calibration_matrix


if __name__ == "__main__":
    scene_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..")
    capture_scene(scene_dir)
