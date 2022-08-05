import os
import pickle
import torch
import numpy as np
import time

def main():
    annotation_path = '/nas/database/Big5/gt/annotation_test.pkl'
    preds_dir = '/nas/project_data/B1_Behavior/hakim/FI/predictions/test'
    out_preds_dir = '/nas/project_data/B1_Behavior/hakim/FI/predictions/test_npy'

    with open(annotation_path, 'rb') as f:
        annotation = pickle.load(f, encoding='latin1')
    videos = list(annotation['extraversion'].keys())

    for i, video in enumerate(videos):
        in_name = video.rsplit(".")[0] + "." + video.rsplit(".")[1] + ".pickle"
        pred_path = os.path.join(preds_dir, in_name)
        out_name = video.rsplit(".")[0] + "." + video.rsplit(".")[1] + ".npy"
        out_pred_path = os.path.join(out_preds_dir, out_name)
        prediction = pickle.load(open(pred_path, "rb"))
        data = []
        
        for item in prediction:
            data.append(item.cpu().numpy())
        data = np.array(data)
        
        with open(out_pred_path, 'wb') as f:
            np.save(f, data)
        print(f'Output {i} : {out_pred_path}') 
        

if __name__ == "__main__":
    main()