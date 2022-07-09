from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import os
import pickle
import time

def main():
    start_time = time.time()
    input_dir = '/nas/project_data/B1_Behavior/hakim/FI/frames/train'
    output_dir = '/nas/project_data/B1_Behavior/hakim/FI/mtcnn'
    annotation_path = '/nas/database/Big5/gt/annotation_training.pkl'
    batch_size = 32
    mtcnn = MTCNN(keep_all=True, device='cuda:0')
    with open(annotation_path, 'rb') as f:
        annotation = pickle.load(f, encoding='latin1')
    video_names = list(annotation['extraversion'].keys())

    for video_name in video_names:
        print(f'Processing {video_name}')
        frames = []
        boxes = []
        landmarks = []
        probs = []
        dir_name = video_name.rsplit(".")[0] + "." + video_name.rsplit(".")[1]
        img_dir = os.path.join(input_dir, dir_name)
        file_count = len([x for x in os.listdir(img_dir)])
        
        for i in range(1, file_count+1):
            filename = str(i) + '.png'
            img_path = os.path.join(img_dir, filename)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(img)
            frames.append(frame)

            if (len(frames) == batch_size or i == file_count):
                batch_boxes, batch_probs, batch_landmarks = mtcnn.detect(frames, landmarks=True)
                boxes.extend(batch_boxes)
                landmarks.extend(batch_landmarks)
                probs.extend(batch_probs)
                frames = []

        output = {'boxes': boxes, 'probs': probs, 'landmarks': landmarks}
        out_filename = dir_name + ".pickle"
        out_path = os.path.join(output_dir, out_filename)
        outfile = open(out_path, 'wb')
        pickle.dump(output, outfile)
        outfile.close()
        print(f'Output is saved on {out_path}')

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()