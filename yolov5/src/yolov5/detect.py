import cv2
import numpy as np
import os
import sys
from numpy.core.numeric import False_
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.utils import check_img_size, non_max_suppression, scale_coords, plot_one_box
from utils.torch_utils import time_synchronized
from models.experimental import attempt_load
from utils.datasets import letterbox

class YoloV5():
    def __init__(self, weights='ycb_v8.pt', img_size=640, conf_thres=0.3, iou_thres=0.5):
        cudnn.benchmark = True
        self.weights = weights
        self.device = 'cuda'
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # half precision only supported on CUDA | Recommend only for gpus > 1660 super
        self.half = self.device != 'cpu'
        
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        export = False
        if export:
            print('save')
            torch.save(self.model, os.getcwd()+"/weights/yolov5n_PY2.pt", _use_new_zipfile_serialization=False)
            exit()
        
        self.imgsz = check_img_size(img_size, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Run inference
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device != 'cpu' else None  # run once
        print('YOLOV5 loaded')

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        #print(names)

    def detect(self, img0):
        view_img = False
        s = '%g: '
        img = letterbox(img0, new_shape=self.imgsz)[0] # letterbox
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t2 = time_synchronized()

        # Process detections
        detections = []
        if len(pred): 
            det = pred[0]
            if not det is None:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for d in det:
                    detections.append(d)
        return detections

def main():
    det = YoloV5(weights="/home/matias/uchile_ws/ros/jaime/high_ws/src/yolov5/yolov5n.pt")
    print("listo")


if __name__=="__main__":
    main()