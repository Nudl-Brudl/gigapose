import os

import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import torch
from segment_anything import SamPredictor, sam_model_registry


class SegmentInterface:
  
    def __init__(self, sam_checkpoint_path, device, image_path):
        
        # Do SAM Stuff
        self.sam_checkpoint_path = sam_checkpoint_path
        self.model_type = "vit_h"
        self.device = device

        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

        # Do Image stuff
        self.img_array_bgr = cv2.imread(image_path)
        img_array_rgb = cv2.cvtColor(self.img_array_bgr, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(img_array_rgb)
        height, width, _ = img_array_rgb.shape

        # Do UI stuff
        self.root = tk.Tk()
        self.root.title("Object Segmentation")

        self.canvas = tk.Canvas(self.root, width=width, height=height)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_click)

        photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo


    def on_click(self, event):
        if self.img_array_bgr is None:
            print("Image is empty")
        
        self.predictor.set_image(self.img_array_bgr)

        x, y = event.x, event.y
        input_point = np.array([[x, y]])
        input_label = np.array([1])


        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        # Display segmentation result
        segmented_image = self.img_array_bgr.copy()
        mask = masks[0]
        segmented_image[mask == False] = [0, 0, 0]

        cv2_seg_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        seg_image = Image.fromarray(cv2_seg_image)
        photo = ImageTk.PhotoImage(seg_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def run(self):
        self.root.mainloop()



        