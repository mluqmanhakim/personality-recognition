import numpy as np
import cv2

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
