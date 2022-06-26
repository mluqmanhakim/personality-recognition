from PIL import Image
import os
import pickle
import time

def rescale_box(box, scale):
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + int(w/2)
    cy = box[1] + int(h/2)
    max_len = max(w, h)
    new_w = int(max_len * scale / 2)
    x1 = cx - new_w
    y1 = cy - new_w
    x2 = cx + new_w
    y2 = cy + new_w
    new_box = [x1,y1,x2,y2]
    return new_box

# crop image and resize to 224x224
def crop_img(img_path, coords, out_path):
    image_obj = Image.open(img_path)
    cropped_image = image_obj.crop(coords)
    img = cropped_image.resize((224,224))
    img.save(out_path)

def main():
    start_time = time.time()
    box_path = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/mtcnn'
    input_dir = '/home/luqman/ME-GraphAU/data/DISFA/img'
    output_path = '/nas/project_data/B1_Behavior/hakim/ME-GraphAU/data/DISFA/img'

    video_names = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN016',
              'SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024',
              'SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']

    for v in video_names:
        print(f'Processing {v}')
        f_path = box_path + "/" + v + ".pickle"
        boxes = pickle.load(open(f_path, "rb"))
        img_dir = os.path.join(input_dir, v)        
        out_dir = os.path.join(output_path, v)
        os.makedirs(out_dir, exist_ok=True)
        
        for i, box in enumerate(boxes):
            in_filename = str(i+1) + ".png"
            img_path = os.path.join(img_dir, in_filename)
            out_filename = str(i) + ".png"
            out_img_path = os.path.join(out_dir, out_filename)
            s_box = rescale_box(box[0], 1.3)
            crop_img(img_path, s_box, out_img_path)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()