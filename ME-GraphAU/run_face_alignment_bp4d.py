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
    video_names = ['F001','M007','F018','F008','F002','M004','F010','F009','M012','M001','F016','M014',
                    'F023','M008','M011','F003','M010','M002','F005','F022','M018','M017','F013','M016',
                    'F020','F011','M013','M005','F007','F015','F006','F019','M006','M009','F012','M003',
                    'F004','F021','F017','M015','F014']
    tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
    input_dir = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/img_raw'
    mtcnn_output_dir = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/mtcnn_out'
    output_path = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/BP4D/img'

    for video_name in video_names:
        print(f'Processing {video_name}')
        
        for task in tasks:
            img_dir = os.path.join(input_dir, video_name, task)
            out_dir = os.path.join(output_path, video_name, task)
            os.makedirs(out_dir, exist_ok=True)
            mtcnn_output_file = mtcnn_output_dir + "/" + video_name + "_" + task + ".pickle"
            mtcnn_output = pickle.load(open(mtcnn_output_file, "rb"))
        
            for i, frame_idx in enumerate(mtcnn_output['frames_idx']):
                in_filename = str(frame_idx-1) + ".jpg"
                in_img_path = os.path.join(img_dir, in_filename)
                out_filename = str(frame_idx) + ".jpg"
                out_img_path = os.path.join(out_dir, out_filename)
                    
                if (mtcnn_output['landmarks'][i] is None):
                    print(f'NO FACE DETECTION - {in_img_path}')
                    continue
                else:
                    landmark = mtcnn_output['landmarks'][i][0]        
                    img = cv2.cvtColor(cv2.imread(in_img_path), cv2.COLOR_BGR2RGB)
                    aligned_img = face_alignment(img, landmark)
                    img = Image.fromarray(aligned_img)
                    img.save(out_img_path)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()