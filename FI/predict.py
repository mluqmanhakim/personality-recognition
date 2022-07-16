import os
import numpy as np
import torch
import torch.nn as nn
from model.MEFL import MEFARG
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir,set_env
from PIL import Image
import cv2
from facenet_pytorch import MTCNN
import pickle
import time
from face_alignment import face_alignment

def delete_dir(path):
    command = "rm -r " + path
    os.system(command)

def convert_to_frames(video_path, frames_path):
    command = "/home/luqman/ffmpeg/ffmpeg-git-20220622-amd64-static/ffmpeg -loglevel warning -t 15 -i " + video_path + " -r:v 30 -frames:v 450 " + os.path.join(frames_path, "%d.png") 
    os.system(command)

def run_mtcnn(frames_path, batch_size=32):
    mtcnn = MTCNN(keep_all=True, device='cuda:1')
    frames = []
    landmarks = []
    frames_count = len([x for x in os.listdir(frames_path)])
    
    for i in range(1, frames_count+1):
        filename = str(i) + '.png'
        img_path = os.path.join(frames_path, filename)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(img)
        frames.append(frame)

        if (len(frames) == batch_size or i == frames_count):
            _, _, batch_landmarks = mtcnn.detect(frames, landmarks=True)
            landmarks.extend(batch_landmarks)
            frames = []

    output = {'landmarks': landmarks}
    return output

def generate_aligned_frames(input_dir, output_dir, mtcnn_output):
    for i, landmark in enumerate(mtcnn_output['landmarks']):
        if (landmark is None):
            continue
        else:
            landmark = landmark[0]
            filename = str(i+1) + ".png"
            in_img_path = os.path.join(input_dir, filename)
            out_img_path = os.path.join(output_dir, filename)        
            img = cv2.cvtColor(cv2.imread(in_img_path), cv2.COLOR_BGR2RGB)
            aligned_img = face_alignment(img, landmark)
            img = Image.fromarray(aligned_img)
            img.save(out_img_path)

def predict_action_unit(inputs):
    net = MEFARG(num_classes=12, backbone="swin_transformer_base")
    net_fold1 = load_state_dict(net, "/nas/project_data/B1_Behavior/hakim/ME-GraphAU/pretrained/swin/MEFARG_swin_base_BP4D_fold1.pth")
    net_fold2 = load_state_dict(net, "/nas/project_data/B1_Behavior/hakim/ME-GraphAU/pretrained/swin/MEFARG_swin_base_BP4D_fold2.pth")
    net_fold3 = load_state_dict(net, "/nas/project_data/B1_Behavior/hakim/ME-GraphAU/pretrained/swin/MEFARG_swin_base_BP4D_fold3.pth")

    if torch.cuda.is_available():
        net_fold1 = nn.DataParallel(net_fold1, device_ids = [1,0]).cuda(device=1)
        net_fold2 = nn.DataParallel(net_fold2, device_ids = [1,0]).cuda(device=1)
        net_fold3 = nn.DataParallel(net_fold3, device_ids = [1,0]).cuda(device=1)
    
    with torch.no_grad():
        outputs_fold1, _ = net_fold1(inputs)
        outputs_fold2, _ = net_fold2(inputs)
        outputs_fold3, _ = net_fold3(inputs)
    outputs = (outputs_fold1 + outputs_fold2 + outputs_fold3) / 3
    return outputs

def make_prediction(input_dir, predictions_dir, mtcnn_output, batch_size=32):
    predictions = []
    frames = []
    
    for i, landmark in enumerate(mtcnn_output['landmarks']):        
        if (landmark is None):
            continue
        else: 
            filename = str(i+1) + ".png"
            img_path = os.path.join(input_dir, filename)        
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            frames.append(img)
            
        if (len(frames) == batch_size or i == len(mtcnn_output['landmarks'])-1):
            inputs = np.array(frames)
            inputs = torch.tensor(inputs, dtype=torch.float32).cuda(device=1)
            inputs = inputs.permute(0,3,1,2)
            batch_predictions = predict_action_unit(inputs)
            predictions.extend(batch_predictions)
            frames = []
    
    out_filename = input_dir.rsplit("/")[-1] + ".pickle"
    out_path = os.path.join(predictions_dir, out_filename)
    outfile = open(out_path, 'wb')
    pickle.dump(predictions, outfile)
    outfile.close()
    print(f'Prediction: {out_path}')

def main():
    start_time = time.time()
    video_dir = '/nas/database/Big5/valid'
    frames_dir = "/nas/project_data/B1_Behavior/hakim/FI/frames/val"
    aligned_frames_dir = '/nas/project_data/B1_Behavior/hakim/FI/frames_aligned/val'
    predictions_dir = '/nas/project_data/B1_Behavior/hakim/FI/predictions/val'
    annotation_path = '/nas/database/Big5/gt/annotation_validation.pkl'

    with open(annotation_path, 'rb') as f:
        annotation = pickle.load(f, encoding='latin1')
    videos = list(annotation['extraversion'].keys())

    for i, video in enumerate(videos):
        if (i < 10):
            continue
        print(f'Processing {i} {video}')
        video_path = os.path.join(video_dir, video)
        video_name = video.rsplit(".")[0] + "." + video.rsplit(".")[1]
        frames_path = os.path.join(frames_dir, video_name)
        aligned_frames_path = os.path.join(aligned_frames_dir, video_name)
        os.makedirs(frames_path, exist_ok=True)
        os.makedirs(aligned_frames_path, exist_ok=True)
        convert_to_frames(video_path, frames_path)
        mtcnn_output = run_mtcnn(frames_path, batch_size=128)
        generate_aligned_frames(frames_path, aligned_frames_path, mtcnn_output)
        make_prediction(aligned_frames_path, predictions_dir, mtcnn_output, batch_size=128)
        delete_dir(frames_path)
        delete_dir(aligned_frames_path)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()