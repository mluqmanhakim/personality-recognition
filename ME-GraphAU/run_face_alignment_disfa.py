import numpy as np
import cv2
from PIL import Image
import os
import pickle
import time

def face_alignment(img: np.ndarray,
                   landmarks: np.ndarray,
                   detector: str = 'mtcnn',
                   left_eye_ratio=(0.38, 0.38),
                   output_width=224,
                   output_height=None):
    # expected MTCNN implementation: https://github.com/timesler/facenet-pytorch

    assert landmarks.shape == (5,2), f'Expected: (5,2), got istead: {landmarks.shape}'
    assert detector in {'mtcnn'}, 'Only MTCNN format is supported right now.'

    if output_height is None:
        output_height = output_width

    landmarks = np.rint(landmarks).astype(np.int32)

    left_eye_x, left_eye_y = landmarks[1,:] # participant's left eye
    right_eye_x, right_eye_y = landmarks[0,:] # participant's right eye

    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x

    angle = np.degrees(np.arctan2(dY, dX)) - 180

    center = (int((left_eye_x + right_eye_x) // 2),
                  int((left_eye_y + right_eye_y) // 2))

    right_eye_ratio_x = 1.0 - left_eye_ratio[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    output_dist = (right_eye_ratio_x - left_eye_ratio[0])
    output_dist *= output_width
    scale = output_dist / dist

    M = cv2.getRotationMatrix2D(center, angle, scale)

    t_x = output_width * 0.5
    t_y = output_height * left_eye_ratio[1]
    M[0, 2] += (t_x - center[0])
    M[1, 2] += (t_y - center[1])

    return cv2.warpAffine(img, M, (output_width, output_height), flags=cv2.INTER_CUBIC)

def main():
    start_time = time.time()
    video_names = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN016',
                'SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024',
                'SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']
    input_dir = "/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/frames"
    output_path = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/img_align'
    mtcnn_output_path = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/mtcnn_output'

    for v in video_names:
        print(f'Processing {v}')
        img_dir = os.path.join(input_dir, v)
        mtcnn_output_file = mtcnn_output_path + "/" + v + ".pickle"
        mtcnn_output = pickle.load(open(mtcnn_output_file, "rb"))
        out_dir = os.path.join(output_path, v)
        os.makedirs(out_dir, exist_ok=True)
        
        for i, landmark in enumerate(mtcnn_output['landmarks']):
            if (mtcnn_output['landmarks'][i] is None):
                continue
            else:
                in_filename = str(i+1) + ".png"
                in_img_path = os.path.join(img_dir, in_filename)
                out_filename = str(i) + ".png"
                out_img_path = os.path.join(out_dir, out_filename)
                landmark = mtcnn_output['landmarks'][i][0]        
                img = cv2.cvtColor(cv2.imread(in_img_path), cv2.COLOR_BGR2RGB)
                aligned_img = face_alignment(img, landmark)
                img = Image.fromarray(aligned_img)
                img.save(out_img_path)
                
    print("--- %s seconds ---" % (time.time() - start_time))
    

if __name__ == "__main__":
    main()