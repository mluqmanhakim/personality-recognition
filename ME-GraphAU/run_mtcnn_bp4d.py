from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import pickle
import time
import pandas as pd

def main():
    start_time = time.time()
    video_names = ['F001','M007','F018','F008','F002','M004','F010','F009','M012','M001','F016','M014',
                    'F023','M008','M011','F003','M010','M002','F005','F022','M018','M017','F013','M016',
                    'F020','F011','M013','M005','F007','F015','F006','F019','M006','M009','F012','M003',
                    'F004','F021','F017','M015','F014']
    tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
    labels_dir = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/AU_labels'
    input_dir = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/img_raw'
    batch_size = 32
    mtcnn = MTCNN(keep_all=True, device='cuda:0')

    for video_name in video_names:
        print(f'Processing {video_name}')
        
        for task in tasks:
            label_filename = video_name + "_" + task + ".csv"
            label_path = os.path.join(labels_dir, label_filename)
            label_df = pd.read_csv(label_path, header=0, index_col=0, usecols=[0])
            
            frames_idx = []
            frames = []
            boxes = []
            landmarks = []
            probs = []
            img_dir = os.path.join(input_dir, video_name, task)
            files_count = len(label_df)
            counter = 0
        
            for idx, row in label_df.iterrows():
                counter += 1
                filename = str(idx-1) + '.jpg'
                img_path = os.path.join(img_dir, filename)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(img)
                frames.append(frame)
                frames_idx.append(idx)

                if (len(frames) == batch_size or counter == files_count):
                    batch_boxes, batch_probs, batch_landmarks = mtcnn.detect(frames, landmarks=True)
                    boxes.extend(batch_boxes)
                    landmarks.extend(batch_landmarks)
                    probs.extend(batch_probs)
                    frames = []
        
            output = {'frames_idx': frames_idx, 'boxes': boxes, 'probs': probs, 'landmarks': landmarks}
            out_filename = video_name + "_" + task + ".pickle"
            out_path = os.path.join('/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/mtcnn_out', out_filename)
            outfile = open(out_path, 'wb')
            pickle.dump(output, outfile)
            outfile.close()
            print(f'Output is saved on {out_path}')
    
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()