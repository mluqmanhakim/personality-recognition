from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import pickle
import time

def main():
    start_time = time.time()
    video_names = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN016',
                'SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024',
                'SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']
    input_dir = "/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/frames"
    batch_size = 32
    mtcnn = MTCNN(keep_all=True, device='cuda:0')

    for v in video_names:
        print(f'Processing {v}')
        frames = []
        boxes = []
        landmarks = []
        probs = []
        img_dir = os.path.join(input_dir, v)
        files_count = len([name for name in os.listdir(img_dir)])
        
        for i in range(files_count):
            filename = str(i+1) + '.png'
            img_path = os.path.join(img_dir, filename)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(img)
            frames.append(frame)

            if (len(frames) == batch_size or (i+1 == files_count)):
                batch_boxes, batch_probs, batch_landmarks = mtcnn.detect(frames, landmarks=True)
                boxes.extend(batch_boxes)
                landmarks.extend(batch_landmarks)
                probs.extend(batch_probs)
                frames = []
        
        output = {'boxes': boxes, 'probs': probs, 'landmarks': landmarks}
        out_filename = v + ".pickle"
        out_path = os.path.join('/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/mtcnn_output', out_filename)
        outfile = open(out_path, 'wb')
        pickle.dump(output, outfile)
        outfile.close()
        print(f'output is saved on {out_path}')
    
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()