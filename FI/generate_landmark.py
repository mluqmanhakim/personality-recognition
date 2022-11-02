import cv2
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.functions import draw_landmarks
from utils.render import render
from utils.depth import depth
import matplotlib.pyplot as plt
import os
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX    

def get_landmark(img_path):
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    onnx_flag = True  # or True to use ONNX to speed up

    if onnx_flag:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        tddfa = TDDFA(gpu_mode=False, **cfg)
        face_boxes = FaceBoxes()

    img = cv2.imread(img_path)
    boxes = face_boxes(img)
    
    if (len(boxes) < 1):
        return None
    else:
        param_list, roi_box_list = tddfa(img, boxes)
        landmark_list = tddfa.recon_vers(param_list, roi_box_list, dense_flag=False)
        return landmark_list