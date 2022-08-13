import os
import pickle
import torch
import numpy as np
import time

def main():
    train_annotation_path = '/nas/database/Big5/gt/annotation_training.pkl'
    valid_annotation_path = '/nas/database/Big5/gt/annotation_validation.pkl'
    test_annotation_path = '/nas/database/Big5/gt/annotation_test.pkl'

    train_energy_dir = '/home/xiaowei/nipg-mask_cluster_rcnn/Big5_out_train/'
    valid_energy_dir = '/home/xiaowei/nipg-mask_cluster_rcnn/Big5_out_valid/'
    test_energy_dir = '/home/xiaowei/nipg-mask_cluster_rcnn/Big5_out_test/'

    train_output_dir = '/home/luqman/FI_GESTURE_ENERGY/train'
    valid_output_dir = '/home/luqman/FI_GESTURE_ENERGY/valid'
    test_output_dir = '/home/luqman/FI_GESTURE_ENERGY/test'

    with open(test_annotation_path, 'rb') as f:
        annotation = pickle.load(f, encoding='latin1')
    videos = list(annotation['extraversion'].keys())

    for i, video in enumerate(videos):
        in_name = video.rsplit(".")[0] + video.rsplit(".")[1]
        energy_left_hand_path = os.path.join(test_energy_dir, in_name, 'energy_left_hand.npy')
        energy_right_hand_path = os.path.join(test_energy_dir, in_name, 'energy_right_hand.npy')
        energy_body_path = os.path.join(test_energy_dir, in_name, 'energy_body.npy')
        energy_head_path = os.path.join(test_energy_dir, in_name, 'energy_head.npy')
        
        with open(energy_left_hand_path, 'rb') as f:
            energy_left_hand = np.load(f)
        with open(energy_right_hand_path, 'rb') as f:
            energy_right_hand = np.load(f)
        with open(energy_body_path, 'rb') as f:
            energy_body = np.load(f)
        with open(energy_head_path, 'rb') as f:
            energy_head = np.load(f)
        
        concat = np.concatenate((energy_head, energy_body, energy_left_hand, energy_right_hand))
        energy = concat.T
        output_name = video.rsplit(".")[0] + '.' + video.rsplit(".")[1] + '.npy'
        output_path = os.path.join(test_output_dir, output_name)
        
        with open(output_path, 'wb') as f:
            np.save(f, energy)

    print('done')
            

if __name__ == "__main__":
    main()